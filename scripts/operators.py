# -*- coding: utf-8 -*-
"""
operators.py — 算子集

定义 GP / LLM alpha 表达式所使用的全部算子。
每个算子的 func 接收 pd.Series 参数（时序算子的窗口参数为 int），返回 pd.Series。

OPERATORS 结构：
    OPERATORS[算子名] = {
        'func':     callable,        # 求值函数
        'arity':    int,             # 参数个数
        'name':     str,             # 算子名（与 key 相同）
        'category': str,             # '一阶时序' | '二阶组合' | '条件逻辑'
    }

窗口约定：
    时序算子（ts_*, delay, diff）的最后一个（窗口）参数是一个整数常数 n，
    在表达式树里以 const 叶子的形式出现，求值时会被转成 int 传入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 一个极小值，用于除零 / log 保护
_EPS = 1e-9


def winsorize(x, lower_q: float = 0.01, upper_q: float = 0.01):
    """缩尾：先把 inf 视作 NaN，再把超出 [lower_q, 1-upper_q] 分位的极端值截断到边界。

    m8 数值卫生的第二道防线（第一道是 op_signed_power 的指数 clip）。
    signed_power 之外的算子偶发极端值 / inf 也会污染 AlphaEval 的数值型维度
    （robustness 用 std(signal) 加噪，inf 会让 std 变 inf/nan，整维判 0）。
    在信号进 AlphaEval 前统一缩尾，是量化标准做法，量纲无关、不改秩序（rankIC 不受影响）。

    参数：
        x:       pd.Series / array-like
        lower_q: 下侧缩尾分位（默认 1%）
        upper_q: 上侧缩尾分位（默认 1%）
    返回：
        缩尾后的 pd.Series（inf 已转 NaN；全 NaN / 常数序列原样返回不报错）。
    """
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x))
    s = s.replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty or valid.nunique() < 2:
        # 全 NaN 或常数：无分位可算，原样返回（已把 inf 清成 NaN）
        return s
    lo = valid.quantile(lower_q)
    hi = valid.quantile(1.0 - upper_q)
    return s.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _as_int_window(n) -> int:
    """把窗口参数安全转成 int。

    窗口参数在表达式里可能是 python 标量、numpy 标量，或长度为 1 的 Series，
    这里统一转成 >=1 的整数。
    """
    if isinstance(n, (pd.Series, np.ndarray)):
        # 取第一个元素（窗口应当是常数）
        arr = np.asarray(n).ravel()
        val = float(arr[0]) if arr.size else 1.0
    else:
        val = float(n)
    iv = int(round(val))
    return max(1, iv)


def _to_series(x, like: pd.Series | None = None) -> pd.Series:
    """把任意输入转成 pd.Series，尽量对齐 like 的索引。"""
    if isinstance(x, pd.Series):
        return x
    if like is not None:
        # 标量广播到 like 的索引上
        return pd.Series(np.full(len(like), float(x)), index=like.index)
    return pd.Series(x)


# ---------------------------------------------------------------------------
# 一阶时序算子（arity=2，第二参数为窗口 int）
# ---------------------------------------------------------------------------
def op_ts_mean(x: pd.Series, n) -> pd.Series:
    """滚动均值。"""
    n = _as_int_window(n)
    return x.rolling(n, min_periods=1).mean()


def op_ts_std(x: pd.Series, n) -> pd.Series:
    """滚动标准差（样本标准差，ddof=1；窗口不足时用 min_periods=1，单点返回 0）。"""
    n = _as_int_window(n)
    return x.rolling(n, min_periods=1).std().fillna(0.0)


def op_ts_max(x: pd.Series, n) -> pd.Series:
    """滚动最大值。"""
    n = _as_int_window(n)
    return x.rolling(n, min_periods=1).max()


def op_ts_min(x: pd.Series, n) -> pd.Series:
    """滚动最小值。"""
    n = _as_int_window(n)
    return x.rolling(n, min_periods=1).min()


def op_ts_rank(x: pd.Series, n) -> pd.Series:
    """滚动窗口内的时序百分位排名（当前值在最近 n 个值中的分位，取值 0~1）。"""
    n = _as_int_window(n)

    def _rank_last(arr: np.ndarray) -> float:
        # arr[-1] 在窗口内的百分位（含等于），取值区间 (0,1]
        last = arr[-1]
        return float((arr <= last).sum()) / float(len(arr))

    return x.rolling(n, min_periods=1).apply(_rank_last, raw=True)


def op_ts_zscore(x: pd.Series, n) -> pd.Series:
    """滚动 z-score = (x - 滚动均值) / 滚动标准差。"""
    n = _as_int_window(n)
    mean = x.rolling(n, min_periods=1).mean()
    std = x.rolling(n, min_periods=1).std().fillna(0.0)
    return (x - mean) / (std + _EPS)


def op_delay(x: pd.Series, n) -> pd.Series:
    """滞后 n 期（shift）。"""
    n = _as_int_window(n)
    return x.shift(n)


def op_diff(x: pd.Series, n) -> pd.Series:
    """n 期差分 x - delay(x, n)。"""
    n = _as_int_window(n)
    return x - x.shift(n)


def op_ts_decay_linear(x: pd.Series, n) -> pd.Series:
    """线性衰减加权移动平均（权重 n, n-1, ..., 1，越近权重越大，归一化）。"""
    n = _as_int_window(n)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    def _wavg(arr: np.ndarray) -> float:
        w = weights[-len(arr):]
        w = w / w.sum()
        return float(np.dot(arr, w))

    return x.rolling(n, min_periods=1).apply(_wavg, raw=True)


# ---------------------------------------------------------------------------
# 时序相关（arity=3：a, b, 窗口 n）
# ---------------------------------------------------------------------------
def op_ts_corr(a: pd.Series, b: pd.Series, n) -> pd.Series:
    """a 与 b 在滚动窗口 n 内的皮尔逊相关系数。"""
    n = _as_int_window(n)
    b = _to_series(b, like=a)
    corr = a.rolling(n, min_periods=2).corr(b)
    return corr.fillna(0.0)


# ---------------------------------------------------------------------------
# 二元算子（arity=2）
# ---------------------------------------------------------------------------
def op_add(a: pd.Series, b: pd.Series) -> pd.Series:
    """加。"""
    return _to_series(a) + _to_series(b, like=a if isinstance(a, pd.Series) else None)


def op_sub(a: pd.Series, b: pd.Series) -> pd.Series:
    """减。"""
    return _to_series(a) - _to_series(b, like=a if isinstance(a, pd.Series) else None)


def op_mul(a: pd.Series, b: pd.Series) -> pd.Series:
    """乘。"""
    return _to_series(a) * _to_series(b, like=a if isinstance(a, pd.Series) else None)


def op_protected_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """除零保护除法：|b| 很小时返回 1.0（GP 常用约定）。"""
    a = _to_series(a)
    b = _to_series(b, like=a)
    result = a / b.where(b.abs() > _EPS, np.nan)
    # 除零位置填 1.0
    return result.where(b.abs() > _EPS, 1.0)


def op_signed_power(a: pd.Series, b: pd.Series) -> pd.Series:
    """带符号幂：sign(a) * |a|^b。b 可为常数或序列。

    m8 数值卫生：指数 b 先 clip 到 [-3, 3]。金融因子里指数超过 ±3 已无经济意义，
    而 |a|^b 是指数级放大器（a 量级大 + b 稍大即触发 numpy overflow → inf/nan，
    进而让下游 robustness 维被误判 0）。掐住指数从源头防爆，同时保住表达能力。
    """
    a = _to_series(a)
    b = _to_series(b, like=a)
    b = b.clip(lower=-3.0, upper=3.0)
    return np.sign(a) * np.power(a.abs() + _EPS, b)


def op_max2(a: pd.Series, b: pd.Series) -> pd.Series:
    """逐元素取较大值。"""
    a = _to_series(a)
    b = _to_series(b, like=a)
    return pd.Series(np.maximum(a.values, b.values), index=a.index)


def op_min2(a: pd.Series, b: pd.Series) -> pd.Series:
    """逐元素取较小值。"""
    a = _to_series(a)
    b = _to_series(b, like=a)
    return pd.Series(np.minimum(a.values, b.values), index=a.index)


# ---------------------------------------------------------------------------
# 一元算子（arity=1）
# ---------------------------------------------------------------------------
def op_log(x: pd.Series) -> pd.Series:
    """保护对数：log(|x| + eps)。"""
    x = _to_series(x)
    return np.log(x.abs() + _EPS)


def op_abs(x: pd.Series) -> pd.Series:
    """绝对值。"""
    return _to_series(x).abs()


def op_sign(x: pd.Series) -> pd.Series:
    """符号函数（-1/0/1）。"""
    x = _to_series(x)
    return pd.Series(np.sign(x.values), index=x.index)


def op_rank(x: pd.Series) -> pd.Series:
    """截面百分位排序（0~1）。

    单标的时序场景下退化为对整段序列做百分位排名（pct rank），
    横截面场景由上层按时间对齐后调用。
    """
    x = _to_series(x)
    return x.rank(pct=True)


# ---------------------------------------------------------------------------
# 条件算子（arity=3）
# ---------------------------------------------------------------------------
def op_if_then_else(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """条件选择：cond>0 取 a，否则取 b。cond 视作布尔（>0 为真）。"""
    cond = _to_series(cond)
    a = _to_series(a, like=cond)
    b = _to_series(b, like=cond)
    mask = cond > 0
    return a.where(mask, b)


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------
OPERATORS: dict[str, dict] = {
    # ---- 一阶时序 ----
    'ts_mean':        {'func': op_ts_mean,        'arity': 2, 'category': '一阶时序'},
    'ts_std':         {'func': op_ts_std,         'arity': 2, 'category': '一阶时序'},
    'ts_max':         {'func': op_ts_max,         'arity': 2, 'category': '一阶时序'},
    'ts_min':         {'func': op_ts_min,         'arity': 2, 'category': '一阶时序'},
    'ts_rank':        {'func': op_ts_rank,        'arity': 2, 'category': '一阶时序'},
    'ts_zscore':      {'func': op_ts_zscore,      'arity': 2, 'category': '一阶时序'},
    'delay':          {'func': op_delay,          'arity': 2, 'category': '一阶时序'},
    'diff':           {'func': op_diff,           'arity': 2, 'category': '一阶时序'},
    'ts_decay_linear':{'func': op_ts_decay_linear,'arity': 2, 'category': '一阶时序'},
    'ts_corr':        {'func': op_ts_corr,        'arity': 3, 'category': '一阶时序'},
    # ---- 二阶组合 ----
    'add':            {'func': op_add,            'arity': 2, 'category': '二阶组合'},
    'sub':            {'func': op_sub,            'arity': 2, 'category': '二阶组合'},
    'mul':            {'func': op_mul,            'arity': 2, 'category': '二阶组合'},
    'protected_div':  {'func': op_protected_div,  'arity': 2, 'category': '二阶组合'},
    'signed_power':   {'func': op_signed_power,   'arity': 2, 'category': '二阶组合'},
    'max2':           {'func': op_max2,           'arity': 2, 'category': '二阶组合'},
    'min2':           {'func': op_min2,           'arity': 2, 'category': '二阶组合'},
    'log':            {'func': op_log,            'arity': 1, 'category': '二阶组合'},
    'abs':            {'func': op_abs,            'arity': 1, 'category': '二阶组合'},
    'sign':           {'func': op_sign,           'arity': 1, 'category': '二阶组合'},
    'rank':           {'func': op_rank,           'arity': 1, 'category': '二阶组合'},
    # ---- 条件逻辑 ----
    'if_then_else':   {'func': op_if_then_else,   'arity': 3, 'category': '条件逻辑'},
}

# 把 name 字段补齐（等于 key），方便下游取用
for _k, _v in OPERATORS.items():
    _v['name'] = _k

# 全部算子名集合
OPERATOR_NAMES: set[str] = set(OPERATORS.keys())


def get_operator(name: str) -> dict:
    """按名字取算子定义 dict；不存在则抛 KeyError。"""
    if name not in OPERATORS:
        raise KeyError(f"未知算子: {name!r}，已注册算子: {sorted(OPERATOR_NAMES)}")
    return OPERATORS[name]


def list_operators() -> list[str]:
    """返回全部算子名（排序后的列表）。"""
    return sorted(OPERATOR_NAMES)
