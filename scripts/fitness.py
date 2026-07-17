# -*- coding: utf-8 -*-
"""
fitness.py — GP（遗传规划）适应度函数

给 alpha 表达式打分，指导演化搜索。核心思路：
    - 主指标用「预测能力」× 「稳定性折扣」：
        主指标 = |Spearman rankIC(signal, future_return)| × (1 - λ · turnover)
      rankIC 衡量信号对未来收益的单调预测力；turnover 是信号排序的相邻期翻转率，
      换手越高越不稳定（交易成本越大），用 λ 折扣。
    - 多样性惩罚：若给了 elite_signals（已入选精英池的信号），减去
        α · max(|corr(signal, 每个精英)|)
      逼迫种群产出与已有精英「不相关」的新信号，避免同质化。
    - 复杂度惩罚：节点数过大（node_count>30）时轻微扣分，抑制表达式膨胀（bloat）。

最终 fitness = 主指标 - 多样性惩罚 - 复杂度惩罚。

有效样本不足（<30）或计算退化时，返回 0.0（连同 details 里各项置 0），
让这类信号在演化中被自然淘汰。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 最少有效样本数：低于此值认为无法可靠估计 IC，直接判 0
_MIN_VALID = 30
# 数值保护极小值
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 辅助函数：由收盘价构造未来收益
# ---------------------------------------------------------------------------
def compute_future_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """由收盘价序列构造「未来 horizon 期」的收益率。

    未来收益 = close.shift(-horizon) / close - 1
    即：站在 t 时刻，看未来 horizon 期后的价格相对当前的涨跌幅。
    末尾 horizon 个位置因无未来价而为 NaN（由调用方 / 适应度函数处理）。

    参数：
        close:   收盘价 pd.Series
        horizon: 前瞻步长（>=1）

    返回：
        与 close 同索引的未来收益 pd.Series。
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(np.asarray(close))
    horizon = max(1, int(horizon))
    future_price = close.shift(-horizon)
    # 用当前价做分母；价格为 0 处避免除零
    denom = close.replace(0.0, np.nan)
    return future_price / denom - 1.0


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _as_series(x) -> pd.Series:
    """把输入统一成 pd.Series。"""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x))


def _clean_pair(a: pd.Series, b: pd.Series):
    """对齐两个序列并剔除任一侧为 NaN/Inf 的位置，返回对齐后的 (a, b)。"""
    a = _as_series(a)
    b = _as_series(b)
    # 对齐索引（内连接），再把 inf 视作 NaN 一并剔除
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['a'], df['b']


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """计算 Spearman 秩相关；样本不足或退化（常数序列）时返回 0.0。"""
    a2, b2 = _clean_pair(a, b)
    if len(a2) < _MIN_VALID:
        return 0.0
    # 常数序列无法定义秩相关（分母为 0），直接判 0
    if a2.nunique() < 2 or b2.nunique() < 2:
        return 0.0
    rho, _ = spearmanr(a2.values, b2.values)
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


# 公开别名：供 generate.py 复用同口径的 Spearman rankIC 计算
# （算样本内/样本外 IC、top_k 去重的信号相关，都走这个入口，避免各写一份）。
def safe_rank_ic(a: pd.Series, b: pd.Series) -> float:
    """计算 Spearman 秩相关（rankIC）；样本不足或退化时返回 0.0。

    与 fitness / alpha_eval 内部口径完全一致（同 _MIN_VALID 门槛、同常数保护），
    是 _safe_spearman 的对外公开名。
    """
    return _safe_spearman(a, b)


def _compute_turnover(signal: pd.Series) -> float:
    """计算信号排序的相邻期翻转率（turnover）。

    做法：把信号转成百分位秩序列 rank_pct（0~1），相邻期秩变化的平均绝对值
        turnover = mean(|rank_pct_t - rank_pct_{t-1}|)
    衡量信号在时间上的抖动程度。取值区间约 [0, 1]，0 表示排序完全不变、
    1 表示每期都在秩极端间跳变。NaN 位置先剔除后再算相邻差。
    """
    s = _as_series(signal).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return 0.0
    # pct 秩：把信号映射到 [0,1] 的相对排序，量纲无关
    rank_pct = s.rank(pct=True)
    diff = rank_pct.diff().abs().dropna()
    if diff.empty:
        return 0.0
    return float(diff.mean())


def _max_abs_corr_with_elites(signal: pd.Series,
                              elite_signals: list[pd.Series] | None) -> float:
    """计算 signal 与各精英信号的最大 |Spearman 相关|。

    用秩相关而非皮尔逊，与主指标口径一致（关注单调关系而非线性）。
    无精英或全部无法计算时返回 0.0。
    """
    if not elite_signals:
        return 0.0
    max_abs = 0.0
    for elite in elite_signals:
        if elite is None:
            continue
        rho = _safe_spearman(signal, elite)
        max_abs = max(max_abs, abs(rho))
    return float(max_abs)


# ---------------------------------------------------------------------------
# 适应度主函数
# ---------------------------------------------------------------------------
def evaluate_fitness(
    signal: pd.Series,
    future_return: pd.Series,
    elite_signals: list[pd.Series] | None = None,
    node_count: int | None = None,
    *,
    lam: float = 0.15,
    alpha: float = 0.2,
    complexity_threshold: int = 30,
) -> tuple[float, dict]:
    """计算一个 alpha 信号的适应度。

    参数：
        signal:         alpha 信号 pd.Series（表达式在样本上的取值）
        future_return:  对齐的未来收益 pd.Series（见 compute_future_return）
        elite_signals:  精英池信号列表（用于多样性惩罚）；None 表示不惩罚
        node_count:     表达式节点数（用于复杂度惩罚）；None 表示不惩罚
        lam:            换手折扣系数 λ（默认 0.15）
        alpha:          多样性惩罚系数 α（默认 0.2）
        complexity_threshold: 复杂度惩罚起点，节点数超过此值开始扣分（默认 30）

    返回：
        (fitness: float, details: dict)
        details 键：
            'rankic'              —— Spearman rankIC（带符号，便于诊断方向）
            'abs_rankic'          —— |rankIC|
            'turnover'            —— 换手率
            'base_score'          —— |rankIC| × (1 - λ·turnover)（主指标）
            'diversity_penalty'   —— 多样性惩罚（α·max|corr|）
            'complexity_penalty'  —— 复杂度惩罚
            'max_corr_elite'      —— 与精英的最大 |相关|
            'node_count'          —— 传入的节点数
            'n_valid'             —— 参与 IC 计算的有效样本数
            'valid'               —— 是否达到最小有效样本
    """
    # ---- 1) 数据清洗 + 有效样本判断 ----
    sig_clean, fut_clean = _clean_pair(signal, future_return)
    n_valid = int(len(sig_clean))

    details: dict = {
        'rankic': 0.0,
        'abs_rankic': 0.0,
        'turnover': 0.0,
        'base_score': 0.0,
        'diversity_penalty': 0.0,
        'complexity_penalty': 0.0,
        'max_corr_elite': 0.0,
        'node_count': node_count,
        'n_valid': n_valid,
        'valid': False,
    }

    # 有效样本不足：直接判 0（信号无预测价值证据）
    if n_valid < _MIN_VALID:
        return 0.0, details

    # ---- 2) 主指标：|rankIC| × (1 - λ·turnover) ----
    rankic = _safe_spearman(sig_clean, fut_clean)
    abs_rankic = abs(rankic)

    # turnover 用「原始 signal」计算（保留全时间轴的排序抖动信息），
    # 而非清洗后仅与收益对齐的子集
    turnover = _compute_turnover(signal)
    # 折扣因子夹到 [0, 1]，防止极端 turnover 让主指标变负
    discount = float(np.clip(1.0 - lam * turnover, 0.0, 1.0))
    base_score = abs_rankic * discount

    details['rankic'] = float(rankic)
    details['abs_rankic'] = float(abs_rankic)
    details['turnover'] = float(turnover)
    details['base_score'] = float(base_score)
    details['valid'] = True

    # ---- 3) 多样性惩罚：α · max|corr(signal, elite)| ----
    max_corr_elite = _max_abs_corr_with_elites(signal, elite_signals)
    diversity_penalty = float(alpha * max_corr_elite)
    details['max_corr_elite'] = float(max_corr_elite)
    details['diversity_penalty'] = diversity_penalty

    # ---- 4) 复杂度惩罚：节点数超阈值后线性轻微扣分 ----
    complexity_penalty = 0.0
    if node_count is not None and node_count > complexity_threshold:
        # 每超出 1 个节点扣 0.005，封顶 0.1，避免复杂度项压过主指标
        over = node_count - complexity_threshold
        complexity_penalty = float(min(0.005 * over, 0.1))
    details['complexity_penalty'] = complexity_penalty

    # ---- 5) 汇总 ----
    fitness = base_score - diversity_penalty - complexity_penalty
    return float(fitness), details
