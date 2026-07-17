# -*- coding: utf-8 -*-
"""
dimensions.py — 量纲体系的唯一枚举与规则源（双维度：物理单位 × 金融语义）

设计动机（替换旧的单一扁平标签 price/return/ratio/volume/oi/bool/unknown）：
    旧体系把「金额(元)」和「成交量(手)」混成同一个 volume 标签，量级差 1e4 却能过
    量纲校验；而且乘除一律拍平成 ratio，丢掉了「价格×数量=金额」这类真实的量纲推导。

本模块把量纲拆成两个正交维度：
    ① 物理单位 unit —— **硬约束**，参与合法性校验。
       price / money / count / dimensionless / bool / unknown
       加减法要求单位兼容（实体单位严格同单位，无量纲之间放宽）；
       乘除法靠「组合表」自动推导/抵消单位（价格×计数→金额，金额÷计数→价格…）。
    ② 金融语义 semantic —— **软标注**，完全不参与校验、不进 GP 搜索。
       只用于喂 LLM 做金融逻辑打分 / 经济解释 / 报告展示。
       price/return/momentum/volatility/volume/turnover/open_interest/liquidity/...

为什么单位是硬约束、语义是软标注：
    - 单位错了（元+手）是量纲错误，运算本身没有物理意义，必须拦。
    - 语义只是「这个因子在捕捉什么行为」，动量因子和反转因子相加不是「非法」，
      只是逻辑上未必优美——这种事交给 LLM 评分和人来判断，不该硬拦，
      硬拦会让双维度过度局限（连 open 价 + close 价都可能因语义细分被拒）。

对外接口：
    combine_mul(u1, u2) -> str        乘法结果单位（可交换）
    combine_div(u1, u2) -> str        除法结果单位（u1 / u2，不可交换）
    add_sub_compatible(u1, u2) -> bool 加减法两侧单位是否兼容
    merge_units(u1, u2) -> str         加减/max2/min2 的结果单位（取更具体者）
"""
from __future__ import annotations

# ===========================================================================
# 枚举：单位（硬约束）与语义（软标注）
# ===========================================================================
# 物理单位：参与合法性校验
UNITS: set[str] = {
    'price',          # 价格（元/股、元/点）
    'money',          # 金额 / 成交额（元）
    'count',          # 计数（手 / 张 / 笔 / 股）
    'dimensionless',  # 无量纲（比率、标准化输出、相关系数、[0,1] 有界量）
    'bool',           # 0-1 信号
    'unknown',        # 常数 / 推断不出（通配，与任意单位兼容）
}

# 金融语义：软标注，不参与校验
SEMANTICS: set[str] = {
    'price', 'return', 'momentum', 'volatility', 'volume', 'turnover',
    'open_interest', 'liquidity', 'intraday', 'correlation', 'unknown',
}

# 实体单位：有明确物理量纲，加减必须严格同单位
_ENTITY_UNITS: set[str] = {'price', 'money', 'count'}
# 无量纲族：加减时彼此兼容（比率、标准化、bool 都是没有物理单位的纯数）
_DIMLESS_UNITS: set[str] = {'dimensionless', 'bool'}


# ===========================================================================
# 乘除组合表
# ===========================================================================
# 乘法（可交换）：用 frozenset 作键，只列有意义的实体组合。
#   价格 × 计数 = 金额（如 成交价 × 成交手数 = 成交额）
_MUL_TABLE: dict[frozenset, str] = {
    frozenset({'price', 'count'}): 'money',
    # 其余实体组合（price×price, price×money, count×count, money×count, money×money）
    # 无清晰金融意义，走 combine_mul 兜底为 dimensionless（乘法放宽，不报错）。
}

# 除法（不可交换）：用有序元组 (被除, 除) 作键。
#   金额 ÷ 计数 = 价格；金额 ÷ 价格 = 计数；同实体相除 = 无量纲。
_DIV_TABLE: dict[tuple, str] = {
    ('money', 'count'): 'price',
    ('money', 'price'): 'count',
    ('price', 'price'): 'dimensionless',
    ('money', 'money'): 'dimensionless',
    ('count', 'count'): 'dimensionless',
}


def combine_mul(u1: str, u2: str) -> str:
    """乘法结果单位（可交换）。

    规则（按优先级）：
        - 任一为 unknown → unknown（通配传播，无法确定就别硬推）
        - 任一为无量纲(dimensionless/bool) → 另一个单位原样保留（X × 无量纲 = X）；
          两个都无量纲 → dimensionless
        - 命中乘法组合表（price × count → money）→ 表中结果
        - 其余实体组合 → 兜底 dimensionless（乘法放宽，不报错）
    """
    if u1 == 'unknown' or u2 == 'unknown':
        return 'unknown'

    d1 = u1 in _DIMLESS_UNITS
    d2 = u2 in _DIMLESS_UNITS
    if d1 and d2:
        return 'dimensionless'
    if d1:
        return u2          # 无量纲 × 实体 = 该实体
    if d2:
        return u1          # 实体 × 无量纲 = 该实体

    key = frozenset({u1, u2})
    if key in _MUL_TABLE:
        return _MUL_TABLE[key]
    # 两个实体单位但无明确组合意义 → 兜底无量纲
    return 'dimensionless'


def combine_div(u1: str, u2: str) -> str:
    """除法结果单位（u1 / u2，不可交换）。

    规则（按优先级）：
        - 任一为 unknown → unknown（通配传播）
        - 除数为无量纲(dimensionless/bool) → 被除数原样保留（X ÷ 无量纲 = X）
        - 被除数为无量纲、除数为实体 → dimensionless（无量纲 ÷ 实体，无清晰意义）
        - 同实体相除 → dimensionless（价格/价格、金额/金额、计数/计数）
        - 命中除法组合表（金额÷计数→价格 等）→ 表中结果
        - 其余 → 兜底 dimensionless（除法放宽，不报错）
    """
    if u1 == 'unknown' or u2 == 'unknown':
        return 'unknown'

    # 除数无量纲：被除数单位不变（X ÷ 比率 = X）
    if u2 in _DIMLESS_UNITS:
        return u1
    # 被除数无量纲、除数是实体：结果无量纲
    if u1 in _DIMLESS_UNITS:
        return 'dimensionless'

    # 走到这里 u1、u2 都是实体单位
    if u1 == u2:
        return 'dimensionless'
    if (u1, u2) in _DIV_TABLE:
        return _DIV_TABLE[(u1, u2)]
    return 'dimensionless'


def add_sub_compatible(u1: str, u2: str) -> bool:
    """加减法两侧单位是否兼容（可相加减）。

    规则（实体严、无量纲宽）：
        - 任一为 unknown → 兼容（常数 / 未标注特征不阻断组合，通配）
        - 两者都是无量纲族(dimensionless/bool) → 兼容（都是纯数，可加减）
        - 否则要求严格相等（price+price ✓；price+money ✗；money+count ✗）
    """
    if u1 == 'unknown' or u2 == 'unknown':
        return True
    if u1 in _DIMLESS_UNITS and u2 in _DIMLESS_UNITS:
        return True
    return u1 == u2


def merge_units(u1: str, u2: str) -> str:
    """合并两个（已判定兼容的）单位，返回结果单位。

    用于 add/sub、max2/min2、if_then_else 分支合并的结果单位判定。
    取更「具体」的那个：
        - 任一 unknown → 另一个（非 unknown 优先，通配让位于具体）
        - 相等 → 原样
        - 两个不同的无量纲(dimensionless vs bool) → 归一到 dimensionless
        - 其余（理论上加减已保证兼容，不会走到实体跨类）→ 取 u1
    """
    if u1 == 'unknown':
        return u2
    if u2 == 'unknown':
        return u1
    if u1 == u2:
        return u1
    if u1 in _DIMLESS_UNITS and u2 in _DIMLESS_UNITS:
        return 'dimensionless'
    return u1
