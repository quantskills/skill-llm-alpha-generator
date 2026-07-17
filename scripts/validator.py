# -*- coding: utf-8 -*-
"""
validator.py — 表达式三层校验器

对一棵表达式树 (expression.Node) 做三层校验，任一层不过即判非法，
并返回 {valid, reason, layer}，layer 指出被哪一层拦下。

三层校验：
    ① 语法 / 白名单：
       - collect_operators(node) 中每个算子必须在 operators.OPERATOR_NAMES；
       - collect_features(node) 中每个特征必须已在 registry.has() 里注册。
       任一越界 → 拦。

    ② 单位推导（infer_node_unit）：
       递归推每个子树的物理单位（price/money/count/dimensionless/bool/unknown）。
       - add / sub：左右子树单位必须兼容（实体单位严格同单位；无量纲之间放宽），
         否则非法（如 price + money、price + 比率）；结果取 merge_units。
       - mul：靠乘法组合表推导（价格×计数→金额；X×无量纲→X；查不到→无量纲）。
       - protected_div：靠除法组合表推导（金额÷计数→价格；同实体相除→无量纲；…）。
       - log / signed_power：要求输入为「连续量」，bool 单位进入这些算子 → 非法；结果无量纲。
       - if_then_else(cond, a, b)：a / b 两分支单位要求兼容（cond 不参与单位约束）。
       推导过程中一旦发现冲突，抛 _DimensionError("单位非法: ...")。

    ③ 前视（未来数据）校验：
       本算子集里的时序算子都是「回看」的，主要防止负窗口 / 未来 shift。
       窗口类算子（ts_*, delay, diff, ts_corr）的窗口参数 n 必须 >= 0；
       n < 0 → 非法（会读到未来数据）。
       单标的时序场景（single_asset=True）额外禁用横截面 rank：rank 本意是
       「同一天一批标的间排名」，用在单标的时序上会退化为「拿整段历史(含未来)排名」，
       等价于前视 → 拦，提示改用滚动、不前视的 ts_rank。

对外接口：
    validate_expression(node, registry, single_asset=False) -> dict
        {valid: bool, reason: str, layer: '语法'|'注册表'|'量纲'|'前视'|None}
    validate_batch(nodes, registry, single_asset=False) -> list[dict]
        批量校验，逐个调用 validate_expression。

说明：layer 取值中，第一层再细分为 '语法'（算子越界）与 '注册表'（特征未注册），
因为二者定位不同、便于排错；通过时 layer=None。
"""
from __future__ import annotations

import numpy as np

from operators import OPERATORS, OPERATOR_NAMES
from expression import Node, collect_operators, collect_features, to_formula_string
from feature_registry import FeatureRegistry
from dimensions import combine_mul, combine_div, add_sub_compatible, merge_units

# 需要「窗口整数常数」作为最后一个参数的时序算子（与 expression._WINDOW_OPS 保持一致）
_WINDOW_OPS = {'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_rank',
               'ts_zscore', 'delay', 'diff', 'ts_decay_linear', 'ts_corr'}

# 「连续量」算子：输入必须是连续数值量纲，bool 单位进入即非法
# （sqrt 不在本算子集里，但按约定一并纳入，未来若补充算子可直接生效）
_CONTINUOUS_ONLY_OPS = {'log', 'sqrt', 'signed_power'}


class _DimensionError(Exception):
    """量纲推导过程中的内部异常，携带中文原因。"""


# ---------------------------------------------------------------------------
# ① 语法 / 白名单
# ---------------------------------------------------------------------------
def _check_whitelist(node: Node, registry: FeatureRegistry) -> dict | None:
    """检查算子白名单与特征注册表。

    返回：不通过时返回 {layer, reason} dict；通过返回 None。
    """
    # 算子越界检查
    for op in collect_operators(node):
        if op not in OPERATOR_NAMES:
            return {'layer': '语法', 'reason': f"算子 {op!r} 不在白名单，允许算子: {sorted(OPERATOR_NAMES)}"}

    # 特征注册检查
    for feat in collect_features(node):
        if not registry.has(feat):
            return {'layer': '注册表', 'reason': f"特征 {feat!r} 未在注册表中注册"}

    return None


# ---------------------------------------------------------------------------
# ② 单位推导
# ---------------------------------------------------------------------------
def infer_node_unit(node: Node, registry: FeatureRegistry) -> str:
    """递归推导一棵子树的物理单位，返回单位标签字符串。

    单位规则：
        - feature 叶子：取 registry 中登记的 unit；
        - const 叶子：常数视作无量纲通配，统一记 'unknown'（可与任意单位兼容）；
        - op 节点：按算子语义推导（见模块 docstring）。

    冲突时抛 _DimensionError。
    """
    # ---- 叶子 ----
    if node.node_type == 'feature':
        name = str(node.value)
        # 走到这里前 _check_whitelist 已保证已注册；稳妥起见仍兜底
        if not registry.has(name):
            raise _DimensionError(f"特征 {name!r} 未注册，无法推导单位")
        return registry.get_unit(name)

    if node.node_type == 'const':
        # 常数无量纲，记 unknown（与任意单位判定时视为兼容通配）
        return 'unknown'

    # ---- 算子节点 ----
    op = node.op

    # 加减：左右子树单位必须兼容（实体严、无量纲宽）
    if op in ('add', 'sub'):
        u_left = infer_node_unit(node.children[0], registry)
        u_right = infer_node_unit(node.children[1], registry)
        if not add_sub_compatible(u_left, u_right):
            raise _DimensionError(
                f"{op} 要求兼容单位，但左={u_left} 右={u_right}"
                f"（子式: {to_formula_string(node)}）")
        return merge_units(u_left, u_right)

    # 乘法：靠乘法组合表推导（价格×计数→金额；X×无量纲→X；查不到→无量纲）
    if op == 'mul':
        u0 = infer_node_unit(node.children[0], registry)
        u1 = infer_node_unit(node.children[1], registry)
        return combine_mul(u0, u1)

    # 除法：靠除法组合表推导（金额÷计数→价格；同实体相除→无量纲；…）
    if op == 'protected_div':
        u0 = infer_node_unit(node.children[0], registry)
        u1 = infer_node_unit(node.children[1], registry)
        return combine_div(u0, u1)

    # 连续量算子：log / sqrt / signed_power，输入不得为 bool 单位
    if op in _CONTINUOUS_ONLY_OPS:
        # signed_power(a, b)：a 为被作用的连续量，b 为指数（不约束单位）
        u_in = infer_node_unit(node.children[0], registry)
        if u_in == 'bool':
            raise _DimensionError(
                f"{op} 要求输入为连续量，但输入单位为 bool"
                f"（子式: {to_formula_string(node)}）")
        if op == 'signed_power':
            # 指数分支仍需递归校验其内部合法性
            infer_node_unit(node.children[1], registry)
        # 连续量算子结果视为无量纲（数值变换）
        return 'dimensionless'

    # 条件：if_then_else(cond, a, b)，a/b 分支单位须兼容，cond 不约束
    if op == 'if_then_else':
        infer_node_unit(node.children[0], registry)  # cond：仅校验内部合法性
        u_a = infer_node_unit(node.children[1], registry)
        u_b = infer_node_unit(node.children[2], registry)
        if not add_sub_compatible(u_a, u_b):
            raise _DimensionError(
                f"if_then_else 的两分支要求兼容单位，但 a={u_a} b={u_b}"
                f"（子式: {to_formula_string(node)}）")
        return merge_units(u_a, u_b)

    # 其余算子（时序类 ts_*、delay、diff、ts_corr、max2/min2、abs/sign/rank）：
    # 递归校验子树后返回一个代表性单位。
    return _infer_generic_op(node, registry)


def _infer_generic_op(node: Node, registry: FeatureRegistry) -> str:
    """处理未单独列出的算子的单位推导。

    - 时序算子（_WINDOW_OPS）：最后一个窗口参数是常数窗口，不参与单位；
      结果单位取第一个序列子树的单位（如 ts_mean(price,5) 仍是 price；
      ts_std/diff 与原序列同单位）。ts_zscore/ts_rank 是标准化算子，输出无量纲。
    - ts_corr(a,b,n)：相关系数无量纲，返回 'dimensionless'。
    - max2/min2：逐元素取值，结果单位取两输入的合并单位（要求兼容）。
    - abs：保留输入单位。
    - sign / rank：输出无量纲，返回 'dimensionless'。
    """
    op = node.op

    if op == 'ts_corr':
        # a, b 序列位递归校验，窗口位是常数不推导
        infer_node_unit(node.children[0], registry)
        infer_node_unit(node.children[1], registry)
        return 'dimensionless'

    if op in _WINDOW_OPS:
        # 仅第一个序列子树参与单位传导，最后一个窗口常数跳过
        u_in = infer_node_unit(node.children[0], registry)
        # zscore / rank 类标准化输出无量纲；其余（ts_mean/ts_std/ts_max/ts_min/
        # delay/diff/ts_decay_linear）保留输入单位
        if op in ('ts_zscore', 'ts_rank'):
            return 'dimensionless'
        return u_in

    if op in ('max2', 'min2'):
        u_a = infer_node_unit(node.children[0], registry)
        u_b = infer_node_unit(node.children[1], registry)
        if not add_sub_compatible(u_a, u_b):
            raise _DimensionError(
                f"{op} 要求两输入兼容单位，但 a={u_a} b={u_b}"
                f"（子式: {to_formula_string(node)}）")
        return merge_units(u_a, u_b)

    if op == 'abs':
        return infer_node_unit(node.children[0], registry)

    if op in ('sign', 'rank'):
        # 递归校验子树内部合法性，输出无量纲
        infer_node_unit(node.children[0], registry)
        return 'dimensionless'

    # 兜底：递归校验全部子树后返回 unknown（理论上不会走到）
    for c in node.children:
        if c.node_type != 'const':
            infer_node_unit(c, registry)
    return 'unknown'


# ---------------------------------------------------------------------------
# ③ 前视校验
# ---------------------------------------------------------------------------
def _check_lookahead(node: Node) -> dict | None:
    """检查是否存在使用未来数据的算子（负窗口 / 负 shift）。

    本算子集里时序算子均为回看；窗口类算子（含 delay/diff）的窗口参数 n
    必须 >= 0，n < 0 会读到未来数据 → 非法。

    返回：不通过时返回 {layer, reason}；通过返回 None。
    """
    if node.node_type != 'op':
        return None

    if node.op in _WINDOW_OPS:
        # 窗口参数是最后一个子节点，通常是 const 整数
        win_child = node.children[-1]
        n = _extract_window_value(win_child)
        if n is not None and n < 0:
            return {'layer': '前视',
                    'reason': f"{node.op} 的窗口参数 n={n} 为负，会读取未来数据"
                              f"（子式: {to_formula_string(node)}）"}

    # 递归子节点
    for c in node.children:
        res = _check_lookahead(c)
        if res is not None:
            return res
    return None


def _extract_window_value(win_node: Node):
    """从窗口子节点里提取窗口数值；无法确定时返回 None（不拦）。"""
    if win_node.node_type == 'const':
        try:
            return float(win_node.value)
        except (TypeError, ValueError):
            return None
    # 窗口位若不是常数（罕见），无法静态判定，交给运行时，这里不拦
    return None


def _check_single_asset_rank(node: Node) -> dict | None:
    """单标的时序场景下禁用横截面 rank（m9）。

    rank 语义是「同一横截面（同一天一批标的）内的百分位排名」。用在单标的时序上，
    实现只能退化为「拿整段序列(含未来样本)一起排名」——要知道今天的分位就得先看到
    未来所有值，等价于前视偏差；且信号可能退化为常数。滚动、只看过去 n 期的 ts_rank
    是单标的场景的正确替代，故这里发现 rank 即拦。

    返回：含 rank 时返回 {layer, reason}；否则 None。
    """
    if 'rank' in collect_operators(node):
        return {'layer': '前视',
                'reason': "单标的时序场景禁用横截面 rank（会退化为含未来数据的整段排名，"
                          "属前视偏差）；请改用滚动、不前视的 ts_rank"
                          f"（公式: {to_formula_string(node)}）"}
    return None


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def validate_expression(node: Node, registry: FeatureRegistry,
                        single_asset: bool = False) -> dict:
    """对单棵表达式树做三层校验。

    参数：
        node:         表达式树根节点
        registry:     特征注册表（提供 has / get_unit）
        single_asset: 是否单标的时序场景。为 True 时前视层额外禁用横截面 rank
                      （见 _check_single_asset_rank）。默认 False（多标的，rank 合法）。

    返回：
        {
            'valid':  bool,
            'reason': str,   # 通过时为 ''
            'layer':  '语法'|'注册表'|'量纲'|'前视'|None,
        }
    """
    # ① 语法 / 白名单
    res = _check_whitelist(node, registry)
    if res is not None:
        return {'valid': False, 'reason': res['reason'], 'layer': res['layer']}

    # ② 单位推导（layer 沿用 '量纲' 名，涵盖单位不兼容）
    try:
        infer_node_unit(node, registry)
    except _DimensionError as e:
        return {'valid': False, 'reason': str(e), 'layer': '量纲'}

    # ③ 前视校验（负窗口）
    res = _check_lookahead(node)
    if res is not None:
        return {'valid': False, 'reason': res['reason'], 'layer': res['layer']}

    # ③b 单标的场景禁用横截面 rank（m9，语义上亦属前视）
    if single_asset:
        res = _check_single_asset_rank(node)
        if res is not None:
            return {'valid': False, 'reason': res['reason'], 'layer': res['layer']}

    # 全部通过
    return {'valid': True, 'reason': '', 'layer': None}


def validate_batch(nodes, registry: FeatureRegistry,
                   single_asset: bool = False) -> list[dict]:
    """批量校验一组表达式树，逐个返回校验结果。

    参数：
        nodes:        可迭代的 Node 列表
        registry:     特征注册表
        single_asset: 透传给 validate_expression（单标的时序禁用 rank）

    返回：list[dict]，每个 dict 结构同 validate_expression 的返回。
    """
    return [validate_expression(n, registry, single_asset=single_asset) for n in nodes]
