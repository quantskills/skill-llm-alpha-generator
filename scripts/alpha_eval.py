# -*- coding: utf-8 -*-
"""
alpha_eval.py — AlphaEval 五维打分

给一个 alpha 信号做「多角度体检」，输出五个各自归一化到 [0,1] 的维度分，
再按权重汇总成一个 weighted_score。相比 fitness.py 只看「预测力 - 惩罚」的
单标量，AlphaEval 更像一张雷达图，用于筛选 / 复盘 / 报告，暴露信号在
「准不准 / 稳不稳 / 抗不抗噪 / 有没有金融逻辑 / 和别人像不像」五个侧面的表现。

五维：
    ① PPS  (Predictive Power Score, 预测能力)
        基于 rankIC 与其信息比率 IR。复用 fitness 口径的 Spearman rankIC，
        把整体 IC 幅度与「按块估计的 IC 稳定性(IR)」映射到 [0,1]。
    ② PFS  (Predictive Fitness Stability, 时序稳定性)
        用「相邻期信号排名一致性」度量：把信号做百分位秩，相邻期秩变化越小
        （换手越低）越稳。用 1 - 平均秩变化 归一化到 [0,1]。
    ③ RRE  (Robustness to Random perturbation Evaluation, 扰动鲁棒性)
        给 signal 叠加多档高斯噪声后，rankIC 相对原始 rankIC 的保持率均值。
        保持率越接近 1 越鲁棒。
    ④ logic (金融逻辑)
        给了 formula_str + llm_client 就调 LLM 对表达式的金融合理性打分
        (1-10 归一化到 [0,1])；否则优雅降级为中性 0.5。LLM 只产数值分与理由，
        遵循下方 LLM_POLICY，绝不回流修改其它维度。
    ⑤ diversity (多样性)
        给了 all_signals 就算「信号相关矩阵特征值分布的谱熵」并归一化到 [0,1]：
        目标信号与其它信号越不相关（相关矩阵越接近单位阵、特征值越均匀），
        谱熵越高，多样性越好；否则降级为中性 0.5。

weighted_score = 0.4·pps + 0.2·pfs + 0.2·rre + 0.15·logic + 0.05·diversity

设计原则：
    - 任意维度在数据退化 / 依赖缺失时都稳健返回中性或 0，绝不抛异常中断 pipeline。
    - LLM 完全可选，无 llm_client 时 logic 优雅降级为 0.5。

对外主入口：
    alpha_eval(signal, future_return, all_signals=None,
               formula_str=None, llm_client=None, config=None) -> dict
        返回 {pps, pfs, rre, logic, diversity, weighted_score, detail}
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 模型名解析统一入口：复用 llm_explainer 的裸别名归一化，避免各写一份。
# LLMError：启用状态下 LLM 硬失败异常（方案 A），由上层传导中断 pipeline。
from llm_explainer import LLMError
# m8 数值卫生：信号进五维前统一缩尾，复用 operators 的唯一实现。
from operators import winsorize

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 最少有效样本数：低于此值认为无法可靠估计 IC（与 fitness.py 口径一致）
_MIN_VALID = 30
# 数值保护极小值
_EPS = 1e-12

# 五维权重（务必和 docstring / 需求保持一致，和为 1.0）
_WEIGHTS = {
    'pps': 0.40,
    'pfs': 0.20,
    'rre': 0.20,
    'logic': 0.15,
    'diversity': 0.05,
}

# 逻辑维在无 LLM / LLM 失败时的中性降级分
_NEUTRAL = 0.5

# PPS 里把 rankIC 幅度映射到 [0,1] 的参考尺度：
#   |rankIC|=IC_SCALE 时预测幅度分约到 0.76（tanh(1)）。
# BTC/期货日频 rankIC 到 0.05 已属不错，取 0.05 作为「良好」参考。
_IC_SCALE = 0.05
# IR（信息比率）映射尺度：IR=IR_SCALE 时稳定性分约 0.76。
_IR_SCALE = 1.0

# RRE 默认噪声档位（相对信号自身标准差的倍数）
_RRE_NOISE_LEVELS = (0.1, 0.25, 0.5)
# RRE 每档噪声重复采样次数（取均值降方差）
_RRE_N_REPEAT = 5
# RRE 基准 IC 门槛：base_ic 低于此值认为信号本无预测力，
#   谈「加噪保持率」无意义（分母是噪声），直接判 rre=0。
#   取 0.02，明显高于大样本随机信号的 IC 噪声地板（~1/sqrt(n)）。
_RRE_MIN_BASE_IC = 0.02


# ---------------------------------------------------------------------------
# Shared LLM policy
# ---------------------------------------------------------------------------
# AlphaEval logic is mandatory and must use the injected shared runtime.
#   - LLM 只产文字与它自己那一维的数值分，绝不改其它维度字段。
_DEFAULT_MODEL = 'opus'

# 强制 tool use 的工具 schema：让 LLM 只能输出结构化的金融逻辑评分
_LOGIC_TOOL = {
    'name': 'report_logic_score',
    'description': '根据 alpha 表达式的金融含义，给它的金融逻辑合理性打分。',
    'input_schema': {
        'type': 'object',
        'properties': {
            'score': {
                'type': 'integer',
                'minimum': 1,
                'maximum': 10,
                'description': '金融逻辑合理性评分，1(牵强/无逻辑)~10(逻辑清晰且有经济学依据)',
            },
            'reason': {
                'type': 'string',
                'description': '简短评分理由（中文，≤120字）',
            },
        },
        'required': ['score', 'reason'],
    },
}


# ===========================================================================
# 内部工具（对齐 / 清洗 / rankIC；与 fitness.py 同口径）
# ===========================================================================
def _as_series(x) -> pd.Series:
    """把输入统一成 pd.Series。"""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x))


def _clean_pair(a: pd.Series, b: pd.Series):
    """对齐两个序列并剔除任一侧为 NaN/Inf 的位置，返回对齐后的 (a, b)。"""
    a = _as_series(a)
    b = _as_series(b)
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['a'], df['b']


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """计算 Spearman 秩相关；样本不足或退化（常数序列）时返回 0.0。"""
    a2, b2 = _clean_pair(a, b)
    if len(a2) < _MIN_VALID:
        return 0.0
    if a2.nunique() < 2 or b2.nunique() < 2:
        return 0.0
    rho, _ = spearmanr(a2.values, b2.values)
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


# ===========================================================================
# ① PPS —— 预测能力（rankIC / IR）
# ===========================================================================
def compute_pps(signal: pd.Series, future_return: pd.Series,
                n_blocks: int = 5) -> dict:
    """预测能力分：综合 rankIC 幅度 与 分块 IC 的信息比率 IR。

    做法：
        - 全样本 |rankIC| 反映整体预测幅度；
        - 把样本按时间切成 n_blocks 段，每段各算一个 rankIC，用
              IR = mean(block_ic) / std(block_ic)
          衡量 IC 的「稳定性 / 一致性」（信息比率）；
        - 幅度分 amp = tanh(|rankIC| / _IC_SCALE) ∈ [0,1)
          稳定分 ir  = tanh(|IR|   / _IR_SCALE) ∈ [0,1)
        - pps = 0.6·amp + 0.4·ir，夹到 [0,1]。

    有效样本不足或退化 → pps=0。

    返回 detail 子字典：{pps, rankic, ir, amp_score, ir_score, n_valid, block_ics}
    """
    sig, fut = _clean_pair(signal, future_return)
    n_valid = int(len(sig))
    out = {
        'pps': 0.0, 'rankic': 0.0, 'ir': 0.0,
        'amp_score': 0.0, 'ir_score': 0.0,
        'n_valid': n_valid, 'block_ics': [],
    }
    if n_valid < _MIN_VALID:
        return out

    rankic = _safe_spearman(sig, fut)
    out['rankic'] = float(rankic)

    # —— 分块 IC 与 IR ——
    n_blocks = max(2, int(n_blocks))
    # 每块至少要有足够样本才算，否则减少块数
    while n_blocks > 2 and n_valid // n_blocks < 10:
        n_blocks -= 1
    block_ics: list[float] = []
    if n_valid // n_blocks >= 5:
        # 按顺序等分（时间序），保留时序结构
        idx_splits = np.array_split(np.arange(n_valid), n_blocks)
        for idx in idx_splits:
            if len(idx) < 5:
                continue
            bi = _safe_spearman(sig.iloc[idx], fut.iloc[idx])
            block_ics.append(float(bi))
    out['block_ics'] = block_ics

    if len(block_ics) >= 2:
        arr = np.asarray(block_ics, dtype=float)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std(ddof=1))
        ir = mean_ic / std_ic if std_ic > _EPS else 0.0
    else:
        ir = 0.0
    out['ir'] = float(ir)

    # —— 映射到 [0,1] ——
    amp_score = float(np.tanh(abs(rankic) / max(_IC_SCALE, _EPS)))
    ir_score = float(np.tanh(abs(ir) / max(_IR_SCALE, _EPS)))
    out['amp_score'] = amp_score
    out['ir_score'] = ir_score

    pps = 0.6 * amp_score + 0.4 * ir_score
    out['pps'] = float(np.clip(pps, 0.0, 1.0))
    return out


# ===========================================================================
# ② PFS —— 时序稳定性（相对排名熵 / 相邻期排名一致性）
# ===========================================================================
def compute_pfs(signal: pd.Series) -> dict:
    """时序稳定性分：相邻期信号排名的一致性，越稳越高。

    做法（相对排名一致性）：
        - 把信号映射为百分位秩 rank_pct ∈ [0,1]（量纲无关）；
        - 相邻期秩变化 |Δrank_pct| 的均值即换手率 turnover ∈ [0,1]；
          turnover 越小说明信号在时间上的排名越稳定；
        - pfs = 1 - turnover，夹到 [0,1]。

    这里的「相对排名熵」用相邻期一致性来近似：完全稳定(turnover=0)→熵最低→pfs=1；
    每期在秩极端间跳变(turnover→1)→混乱→pfs→0。

    样本不足（<2）→ 无从判断稳定性，返回中性 0.5。

    返回 detail 子字典：{pfs, turnover, n_valid}
    """
    s = _as_series(signal).replace([np.inf, -np.inf], np.nan).dropna()
    out = {'pfs': _NEUTRAL, 'turnover': None, 'n_valid': int(len(s))}
    if len(s) < 2:
        return out
    if s.nunique() < 2:
        # 常数信号：排名永不变化，视为「极稳」(turnover=0)→pfs=1
        out['turnover'] = 0.0
        out['pfs'] = 1.0
        return out

    rank_pct = s.rank(pct=True)
    diff = rank_pct.diff().abs().dropna()
    turnover = float(diff.mean()) if not diff.empty else 0.0
    out['turnover'] = turnover
    out['pfs'] = float(np.clip(1.0 - turnover, 0.0, 1.0))
    return out


# ===========================================================================
# ③ RRE —— 扰动鲁棒性（加高斯噪声后 rankIC 保持率）
# ===========================================================================
def compute_rre(signal: pd.Series, future_return: pd.Series,
                noise_levels=_RRE_NOISE_LEVELS,
                n_repeat: int = _RRE_N_REPEAT,
                rng: np.random.Generator | None = None) -> dict:
    """扰动鲁棒性分：给 signal 叠加多档高斯噪声后 rankIC 的保持率均值。

    做法：
        - 先算原始 |rankIC| 作为基准 base_ic；
        - 对每档噪声 level：噪声标准差 = level · std(signal)，
          重复 n_repeat 次「加噪 → 重算 |rankIC|」，取该档保持率
              retain = mean(|rankIC_noisy|) / |base_ic|
          夹到 [0,1]（加噪不应「变强」，>1 截断为 1）；
        - rre = 所有档保持率的均值。

    含义：真正有效的信号排序不该被小扰动轻易破坏；脆弱信号（如过拟合的
    尖锐组合）加一点噪声 IC 就崩，保持率低。

    退化情形：
        - 样本不足 / base_ic 低于门槛（信号本就无预测力）→ 鲁棒性无意义，返回 0.0；
        - signal 为常数（std≈0）→ 无可加噪的尺度，返回中性 0.5。

    返回 detail 子字典：{rre, base_ic, per_level, n_valid}
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    sig, fut = _clean_pair(signal, future_return)
    n_valid = int(len(sig))
    out = {'rre': 0.0, 'base_ic': 0.0, 'per_level': {}, 'n_valid': n_valid}
    if n_valid < _MIN_VALID:
        return out

    base_ic = abs(_safe_spearman(sig, fut))
    out['base_ic'] = float(base_ic)
    # 信号本无预测力（IC 低于门槛）：谈不上「鲁棒」，直接判 0（避免 0/0 与噪声放大）
    if base_ic < _RRE_MIN_BASE_IC:
        return out

    sig_std = float(sig.std(ddof=0))
    if sig_std < _EPS:
        # 常数信号：无尺度可加噪，鲁棒性无意义 → 中性
        out['rre'] = _NEUTRAL
        return out

    per_level: dict[str, float] = {}
    retains: list[float] = []
    sig_vals = sig.values
    for level in noise_levels:
        sigma = level * sig_std
        rep_retains = []
        for _ in range(max(1, int(n_repeat))):
            noise = rng.normal(0.0, sigma, size=n_valid)
            noisy = pd.Series(sig_vals + noise, index=sig.index)
            ic_noisy = abs(_safe_spearman(noisy, fut))
            rep_retains.append(min(ic_noisy / base_ic, 1.0))
        lvl_retain = float(np.mean(rep_retains))
        per_level[f'{level:g}'] = lvl_retain
        retains.append(lvl_retain)

    out['per_level'] = per_level
    out['rre'] = float(np.clip(np.mean(retains), 0.0, 1.0)) if retains else 0.0
    return out


# ===========================================================================
# ④ logic —— 金融逻辑（LLM 可选，遵循 LLM_POLICY）
# ===========================================================================
def _llm_config(config: dict | None) -> dict:
    raise LLMError("AlphaEval has no private LLM configuration path; use the shared runtime")

def _build_llm_client(cfg: dict):
    raise LLMError("AlphaEval has no private LLM client path; use the shared runtime")


def compute_logic(formula_str: str | None, llm_client=None,
                  config: dict | None = None) -> dict:
    """金融逻辑分：LLM 对 alpha 表达式的金融合理性打分（1-10 归一化到 [0,1]）。

    方案 A（默认开启 + 硬失败）：
        - any unavailable or malformed LLM result raises LLMError;
        - 无 formula_str → 无从评判 → 0.5（非 LLM 故障，属正常中性）；
        - 启用状态下以下情况一律抛 LLMError（不降级、不重试）：
            · 既无注入 llm_client 又无法从环境/config 构建客户端（缺 KEY/SDK）
            · LLM 调用异常 / 非 tool_use 回复

    传入 llm_client 时直接用它（便于测试注入 mock）；否则按 config/env 尝试构建。

    LLM 只产它这一维的数值分与文字理由，绝不触碰其它维度。

    返回 detail 子字典：{logic, source, raw_score, reason}
        source = 'llm' on every successful path; failures raise LLMError.
        （启用状态下的客户端缺失 / 调用失败不再返回 neutral_*，而是抛 LLMError）
    """
    out = {'logic': _NEUTRAL, 'source': 'neutral', 'raw_score': None, 'reason': None}

    if not formula_str:
        raise LLMError("AlphaEval logic requires a non-empty formula")

    client = llm_client
    if client is None:
        raise LLMError("AlphaEval logic requires the shared LLM runtime client")

    prompt = (
        "你是量化选股/择时因子的金融逻辑审查专家。给你一个 alpha 因子表达式，"
        "请评估它是否具有清晰、可解释的金融/经济学逻辑（例如动量、反转、"
        "量价背离、波动率、流动性等有据可循的机理），而非纯粹的数据挖掘噪声。\n\n"
        f"因子表达式：{formula_str}\n\n"
        "请调用 report_logic_score 工具打分：1=牵强/纯拟合/无法解释，"
        "10=逻辑清晰且有扎实经济学依据。"
    )

    try:
        model = getattr(client, "_alpha_model", None) or (config or {}).get("llm", {}).get("model")
        if not model:
            model = "current-agent"
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            tools=[_LOGIC_TOOL],
            tool_choice={'type': 'tool', 'name': 'report_logic_score'},
            messages=[{'role': 'user', 'content': prompt}],
        )
        for block in getattr(resp, 'content', []):
            if getattr(block, 'type', None) == 'tool_use':
                data = block.input
                raw = float(data.get('score', 5))
                raw = min(max(raw, 1.0), 10.0)
                # 1-10 → [0,1]：(raw-1)/9
                logic = (raw - 1.0) / 9.0
                out['logic'] = float(np.clip(logic, 0.0, 1.0))
                out['raw_score'] = raw
                out['reason'] = str(data.get('reason', ''))
                out['source'] = 'llm'
                return out
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        # 启用状态下调用异常硬失败，不重试
        raise LLMError(f"LLM 逻辑打分调用失败（model={model}）: {exc!r}") from exc

    # 拿到回复但没有 tool_use → 启用状态下硬失败
    raise LLMError(
        f"LLM 逻辑打分回复无 tool_use（model={model}）；"
        "模型未按要求调用 report_logic_score 工具"
    )


# ===========================================================================
# ⑤ diversity —— 多样性（相关矩阵特征值谱熵）
# ===========================================================================
def compute_diversity(signal: pd.Series,
                      all_signals: list[pd.Series] | None) -> dict:
    """多样性分：目标信号与信号集合相关矩阵特征值分布的谱熵，归一化到 [0,1]。

    直觉：
        - 把 [signal] + all_signals 组成一个信号集，算它们两两的 Spearman
          相关矩阵 C（k×k，对角线为 1）；
        - 若信号彼此高度相关，C 接近全 1 矩阵，特征值集中在一个大值上，
          谱熵低 → 多样性差；
        - 若信号彼此不相关，C 接近单位阵，k 个特征值都≈1（均匀），
          谱熵最高 → 多样性好；
        - 谱熵 H = -Σ pᵢ·log(pᵢ)，pᵢ = λᵢ / Σλ（特征值归一化为分布），
          除以 log(k) 归一化到 [0,1]。

    降级：
        - all_signals 为空 / None → 无可比对象 → 中性 0.5；
        - 有效对比信号 <1（对齐后全退化）→ 中性 0.5；
        - 只有 1 个总信号（k=1）→ 无从谈多样性 → 中性 0.5。

    注意：diversity 反映的是「整个信号集的分散程度」，作为目标信号所处
    组合环境的多样性代理；集合越分散，新增该信号的边际多样性越高。

    返回 detail 子字典：{diversity, k, spectral_entropy, eigen_ratio}
    """
    out = {'diversity': _NEUTRAL, 'k': 0, 'spectral_entropy': None, 'eigen_ratio': None}

    if not all_signals:
        return out

    # 组装信号集：目标信号在前，其余在后
    series_list = [_as_series(signal)]
    for s in all_signals:
        if s is None:
            continue
        series_list.append(_as_series(s))

    if len(series_list) < 2:
        return out

    # 对齐到共同索引，转秩（Spearman = 秩上的 Pearson），再剔除退化列
    df = pd.concat(
        [s.rename(f's{i}') for i, s in enumerate(series_list)],
        axis=1, join='inner',
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < _MIN_VALID:
        return out

    # 丢掉常数列（无法定义相关）
    valid_cols = [c for c in df.columns if df[c].nunique() >= 2]
    if len(valid_cols) < 2:
        return out
    df = df[valid_cols]

    # 秩变换后算相关矩阵（等价于 Spearman 相关）
    ranked = df.rank()
    corr = ranked.corr().values
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    # 对称化并把对角线锁定为 1，保证是合法相关矩阵
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    k = corr.shape[0]
    out['k'] = int(k)

    # 特征值（相关矩阵对称半正定，特征值应 ≥0；数值误差截断到 0）
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals.real, 0.0, None)
    total = float(eigvals.sum())
    if total < _EPS:
        return out

    p = eigvals / total
    p = p[p > _EPS]  # 0 概率不贡献熵
    entropy = float(-np.sum(p * np.log(p)))
    # 归一化：k 个信号时最大熵为 log(k)（特征值全相等 = 完全不相关）
    max_entropy = float(np.log(k)) if k > 1 else 1.0
    spectral_entropy = entropy / max_entropy if max_entropy > _EPS else 0.0

    out['spectral_entropy'] = float(spectral_entropy)
    # 最大特征值占比：越大越集中（越同质），做个诊断字段
    out['eigen_ratio'] = float(eigvals.max() / total)
    out['diversity'] = float(np.clip(spectral_entropy, 0.0, 1.0))
    return out


# ===========================================================================
# 主入口
# ===========================================================================
def alpha_eval(signal: pd.Series,
               future_return: pd.Series,
               all_signals: list[pd.Series] | None = None,
               formula_str: str | None = None,
               llm_client=None,
               config: dict | None = None) -> dict:
    """AlphaEval 五维打分主入口。

    参数：
        signal:        alpha 信号 pd.Series（表达式在样本上的取值）
        future_return: 对齐的未来收益 pd.Series（见 fitness.compute_future_return）
        all_signals:   其它信号列表（用于多样性维）；None → diversity=0.5
        formula_str:   alpha 表达式字符串（供 LLM 评金融逻辑）；None → logic=0.5
        llm_client:    Anthropic 风格客户端（可选，测试可注入 mock）；
                       未传则按 config/env 尝试构建，仍不可用 → logic=0.5
        config:        可选配置，config.llm.{enabled,model,base_url,api_key} 覆盖 LLM 行为

    返回：
        {
            'pps':            float,  # 预测能力 [0,1]
            'pfs':            float,  # 时序稳定性 [0,1]
            'rre':            float,  # 扰动鲁棒性 [0,1]
            'logic':          float,  # 金融逻辑 [0,1]
            'diversity':      float,  # 多样性 [0,1]
            'weighted_score': float,  # 加权综合分 [0,1]
            'detail':         dict,   # 各维子诊断明细
        }
    """
    # —— m8 数值卫生：signal 进五维前统一缩尾 ——
    # signed_power 之外的算子也可能产出 inf / 极端值，污染依赖数值分布的维度
    # （尤其 rre 用 std(signal) 加噪，inf 会让整维判 0）。此处一次缩尾，五维全受益；
    # 缩尾不改秩序，故 rankIC 口径（pps）不受影响。diversity 比对的 all_signals 同口径缩尾。
    signal = winsorize(signal)
    if all_signals:
        all_signals = [winsorize(s) for s in all_signals]

    # —— 逐维计算（每维内部都已做退化 / 降级保护）——
    pps_d = compute_pps(signal, future_return)
    pfs_d = compute_pfs(signal)
    rre_d = compute_rre(signal, future_return)
    logic_d = compute_logic(formula_str, llm_client=llm_client, config=config)
    div_d = compute_diversity(signal, all_signals)

    pps = float(np.clip(pps_d['pps'], 0.0, 1.0))
    pfs = float(np.clip(pfs_d['pfs'], 0.0, 1.0))
    rre = float(np.clip(rre_d['rre'], 0.0, 1.0))
    logic = float(np.clip(logic_d['logic'], 0.0, 1.0))
    diversity = float(np.clip(div_d['diversity'], 0.0, 1.0))

    # —— logic 维降级时的权重重分配 ——
    # LLM 不可用/失败时 logic 是占位中性 0.5（source 以 'neutral' 打头），
    # 若仍以 0.15 权重进 weighted_score 会虚高。此时把 logic 的权重按比例
    # 分摊给其余四维（pps/pfs/rre/diversity）重新归一化到和为 1，不让占位分参与。
    logic_source = str(logic_d.get('source') or '')
    logic_degraded = logic_source.startswith('neutral')

    weights = dict(_WEIGHTS)
    if logic_degraded:
        w_logic = weights['logic']
        others = ('pps', 'pfs', 'rre', 'diversity')
        base_sum = sum(weights[k] for k in others)
        if base_sum > _EPS:
            for k in others:
                weights[k] += w_logic * (weights[k] / base_sum)
        weights['logic'] = 0.0

    weighted_score = (
        weights['pps'] * pps
        + weights['pfs'] * pfs
        + weights['rre'] * rre
        + weights['logic'] * logic
        + weights['diversity'] * diversity
    )
    weighted_score = float(np.clip(weighted_score, 0.0, 1.0))

    return {
        'pps': pps,
        'pfs': pfs,
        'rre': rre,
        'logic': logic,
        'diversity': diversity,
        'weighted_score': weighted_score,
        # logic 维来源透传给 report 展示（是否降级中性）
        'logic_source': logic_source,
        'logic_degraded': logic_degraded,
        'detail': {
            # 实际参与加权的权重（logic 降级时已重分配），供 report 展示与自测校验
            'weights': weights,
            'pps': pps_d,
            'pfs': pfs_d,
            'rre': rre_d,
            'logic': logic_d,
            'diversity': div_d,
        },
    }
