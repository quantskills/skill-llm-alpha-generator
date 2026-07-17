# -*- coding: utf-8 -*-
"""
expression.py — 表达式树 + 求值引擎

表达式用一棵 Node 树表示：
    - 叶子节点：特征名（node_type='feature'）或 常数（node_type='const'）
    - 内部节点：算子（node_type='op'，op=算子名，children=子节点列表）

求值时递归下降，算子从 operators.OPERATORS 查表调用。
data 是 {特征名: pd.Series/np.ndarray}，常数叶子在求值时广播成 Series。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from operators import OPERATORS, OPERATOR_NAMES, get_operator

# 时序类算子的窗口参数默认候选（供 random_tree 采样）
_WINDOW_CHOICES = (2, 3, 5, 10, 20, 30, 60)


@dataclass
class Node:
    """表达式树节点。

    属性：
        node_type: 'feature' | 'const' | 'op'
        op:        算子名（仅 node_type='op' 时有值，否则 None）
        value:     叶子的特征名(str) 或 常数值(float/int)（仅叶子时有值）
        children:  子节点列表（仅 op 节点非空）
    """
    node_type: str
    op: str | None = None
    value: object | None = None
    children: list["Node"] = field(default_factory=list)

    # -------- 便捷构造 --------
    @staticmethod
    def feature(name: str) -> "Node":
        """构造特征叶子。"""
        return Node(node_type='feature', value=name)

    @staticmethod
    def const(v) -> "Node":
        """构造常数叶子。"""
        return Node(node_type='const', value=v)

    @staticmethod
    def make_op(op: str, children: list["Node"]) -> "Node":
        """构造算子节点，并校验算子存在性与元数（arity）。"""
        if op not in OPERATOR_NAMES:
            raise KeyError(f"未知算子: {op!r}")
        arity = OPERATORS[op]['arity']
        if len(children) != arity:
            raise ValueError(
                f"算子 {op} 需要 {arity} 个子节点，实际传入 {len(children)} 个")
        return Node(node_type='op', op=op, children=list(children))

    def __repr__(self) -> str:  # 便于调试
        return to_formula_string(self)


# ---------------------------------------------------------------------------
# 求值
# ---------------------------------------------------------------------------
def evaluate(node: Node, data: dict) -> pd.Series:
    """递归求值一棵表达式树，返回 pd.Series。

    参数：
        node: 表达式树根节点
        data: {特征名: pd.Series 或 np.ndarray}
    """
    if node.node_type == 'feature':
        name = node.value
        if name not in data:
            raise KeyError(f"特征 {name!r} 不在 data 中，可用特征: {list(data.keys())}")
        series = data[name]
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series))
        return series

    if node.node_type == 'const':
        # 常数在这里先返回标量，由上层算子按需广播；
        # 若整棵树就是一个常数，则构造一个长度为 1 的 Series。
        return node.value  # type: ignore[return-value]

    if node.node_type == 'op':
        func = get_operator(node.op)['func']
        # 递归求值子节点
        args = [evaluate(child, data) for child in node.children]
        # 标量广播：若参数里同时存在 Series 和标量，把标量按第一个 Series 的索引广播成 Series。
        # 这样时序/相关等算子（需要 .rolling）拿到的序列位不会是裸标量。
        # 例外：时序算子最后一个「窗口」参数应保持标量（int），故不广播最后一个位置。
        ref_index = None
        for a in args:
            if isinstance(a, pd.Series):
                ref_index = a.index
                break
        if ref_index is not None:
            # 仅「窗口类算子」的最后一个参数需保持标量（int 窗口），其余标量位一律广播成 Series。
            keep_last_scalar = node.op in _WINDOW_OPS
            last_i = len(args) - 1
            new_args = []
            for i, a in enumerate(args):
                is_window_pos = keep_last_scalar and i == last_i
                if not isinstance(a, pd.Series) and not is_window_pos:
                    a = pd.Series(np.full(len(ref_index), float(a)), index=ref_index)
                new_args.append(a)
            args = new_args
        return func(*args)

    raise ValueError(f"未知节点类型: {node.node_type!r}")


# ---------------------------------------------------------------------------
# 结构分析
# ---------------------------------------------------------------------------
def count_nodes(node: Node) -> int:
    """统计树中节点总数（含叶子与算子节点）。"""
    if node.node_type != 'op':
        return 1
    return 1 + sum(count_nodes(c) for c in node.children)


def depth(node: Node) -> int:
    """树的深度（单个叶子深度为 1）。"""
    if node.node_type != 'op' or not node.children:
        return 1
    return 1 + max(depth(c) for c in node.children)


def to_formula_string(node: Node) -> str:
    """把表达式树转成可读公式字符串。"""
    if node.node_type == 'feature':
        return str(node.value)
    if node.node_type == 'const':
        v = node.value
        # 整数常数不带小数
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if node.node_type == 'op':
        inner = ", ".join(to_formula_string(c) for c in node.children)
        return f"{node.op}({inner})"
    raise ValueError(f"未知节点类型: {node.node_type!r}")


def collect_features(node: Node) -> set[str]:
    """收集树中用到的所有特征名。"""
    if node.node_type == 'feature':
        return {str(node.value)}
    if node.node_type == 'const':
        return set()
    result: set[str] = set()
    for c in node.children:
        result |= collect_features(c)
    return result


def collect_operators(node: Node) -> set[str]:
    """收集树中用到的所有算子名。"""
    result: set[str] = set()
    if node.node_type == 'op':
        result.add(node.op)  # type: ignore[arg-type]
        for c in node.children:
            result |= collect_operators(c)
    return result


# ---------------------------------------------------------------------------
# 随机树生成（供 GP 随机初始化）
# ---------------------------------------------------------------------------
# 需要「窗口整数常数」作为最后一个参数的算子（时序类）
_WINDOW_OPS = {'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_rank',
               'ts_zscore', 'delay', 'diff', 'ts_decay_linear', 'ts_corr'}


def random_tree(feature_names, max_depth: int, rng=None) -> Node:
    """随机生成一棵合法表达式树。

    参数：
        feature_names: 可用特征名列表/集合
        max_depth:     最大深度（>=1）
        rng:           np.random.Generator（可选，便于复现）

    规则：
        - 到达 max_depth 或随机命中，返回叶子（多数为特征，少量为常数）。
        - 否则随机选一个算子，为其每个参数递归生成子树；
          时序类算子的最后一个参数强制为窗口整数常数。
    """
    if rng is None:
        rng = np.random.default_rng()
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("feature_names 不能为空")

    # 注意：OPERATOR_NAMES 是 set，直接 list() 的顺序随进程 hash 种子变化，
    # 会导致 rng.choice 选中的算子在不同进程间不一致（破坏可复现性）。
    # 必须排序成确定顺序，才能让固定 seed 得到可复现结果。
    op_list = sorted(OPERATOR_NAMES)

    def _make_leaf(require_series: bool = False) -> Node:
        # require_series=True 时只出特征叶子（用于算子的序列位，避免时序算子拿到裸常数）。
        # 否则约 80% 概率生成特征叶子，20% 常数。
        if require_series or rng.random() < 0.8:
            return Node.feature(str(rng.choice(feature_names)))
        return Node.const(float(rng.choice([-2.0, -1.0, 0.5, 1.0, 2.0, 5.0])))

    def _make_window() -> Node:
        return Node.const(int(rng.choice(_WINDOW_CHOICES)))

    def _build(cur_depth: int, require_series: bool = False) -> Node:
        # cur_depth 从 0 起，最终整棵树深度 = 最深叶子的 (cur_depth + 1)。
        # 因此当 cur_depth 已到 max_depth-1 时必须出叶子，才能保证 depth <= max_depth。
        # 另外一定概率提前出叶子（cur_depth>=1 时）。
        if cur_depth >= max_depth - 1 or (cur_depth >= 1 and rng.random() < 0.3):
            return _make_leaf(require_series=require_series)
        op = str(rng.choice(op_list))
        arity = OPERATORS[op]['arity']
        children: list[Node] = []
        for i in range(arity):
            is_last = (i == arity - 1)
            if op in _WINDOW_OPS and is_last:
                # 时序算子最后一个参数必须是窗口整数常数
                children.append(_make_window())
            else:
                # 非窗口位是序列位：递归生成，且要求叶子情形只出特征
                children.append(_build(cur_depth + 1, require_series=True))
        return Node.make_op(op, children)

    return _build(0)
