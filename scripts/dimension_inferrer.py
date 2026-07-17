# -*- coding: utf-8 -*-
"""
dimension_inferrer.py — 特征标签自动推断器（双维度：物理单位 × 金融语义）

给任意一列数据（特征），自动判定它的两个正交标签：
    ① unit     物理单位（硬约束，参与校验）: price/money/count/dimensionless/bool/unknown
    ② semantic 金融语义（软标注，只喂 LLM/报告）: price/return/momentum/volatility/...

为什么要它：整套 alpha 生成框架不写死特征白名单，任何新数据列都能动态注册进
FeatureRegistry。但算子组合、单位一致性校验都依赖 unit 标签，因此需要一个能对
陌生特征自动贴标签的推断器——这就是可扩展性的核心。

**推断顺序（关键设计）**：先按名称/单位关键词做定性分类（命中即高置信直接定），
名称定不了才落到数据统计画像兜底。即「名称语义优先，统计取值范围只是兜底」——
不再像旧版那样一上来就纯看取值范围。

unit 推断采用「三票投票」，权重 llm > name > stat：
    ① name_unit_vote —— 名称/单位关键词（元/亿/万/手/张/笔/%/持仓 + 英文），最优先
    ② stat_unit_vote —— 数据统计画像兜底（只区分 bool/无量纲/价格量级；金额vs计数
                        无法从数值区分，落 unknown 低置信交名称定夺）
    ③ llm_vote       —— LLM 判定（可选，llm_client=None 时优雅跳过）

semantic 单独推（semantic_vote，名称 + 统计画像），不参与校验。

对外主入口：
    infer_feature_tags(name, series, close_series=None, llm_client=None) -> dict
        返回 {unit, semantic, unit_confidence, votes, reason}
    infer_and_register(registry, name, series, close_series=None, llm_client=None)
        推断后把 unit/semantic 回填进 FeatureRegistry，带同名缓存。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dimensions import UNITS, SEMANTICS
from llm_runtime import LLMRuntimeError

# 合法单位 / 语义（从 dimensions 单一源引入）
VALID_UNITS = UNITS
VALID_SEMANTICS = SEMANTICS

# 各票权重：LLM 最重，名称次之（名称/单位词是定性依据），统计画像最轻（只兜底）。
# 注意这是相对旧版的**反转**：旧版 stat(1.0) > name(0.6)，现在 name(1.0) > stat(0.6)。
_VOTE_WEIGHTS = {
    'llm': 1.2,
    'name': 1.0,
    'stat': 0.6,
}

# 置信度阈值：低于此值判为 unknown 并告警
_CONFIDENCE_FLOOR = 0.5


# ===========================================================================
# 数据统计画像（供 stat 票 / semantic / llm 复用）
# ===========================================================================
def _profile_series(series: pd.Series, close_series: pd.Series | None = None) -> dict:
    """对一列数据做统计画像，供 stat 票 / semantic / llm 复用。

    参数：
        series:       待推断的特征列
        close_series: 收盘价参照列（可选），用于计算量级比

    返回：一个描述性 dict（各字段都为 Python 原生类型，便于喂给 LLM）。
    """
    s = pd.Series(series).astype('float64')
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    prof: dict = {
        'n': int(len(s)),
        'n_unique': int(s.nunique()) if len(s) else 0,
        'unique_values': None,      # 仅当取值极少时才填
        'min': None, 'max': None, 'mean': None, 'std': None,
        'abs_mean': None, 'abs_median': None,
        'all_nonneg': None,         # 是否恒非负
        'bounded_01': None,         # 是否全部落在 [0,1]
        'is_binary01': None,        # 是否只有 {0,1} 两种取值
        'scale_ratio_to_close': None,  # 与 close 的量级比（中位数之比）
    }
    if len(s) == 0:
        return prof

    vmin, vmax = float(s.min()), float(s.max())
    prof['min'] = vmin
    prof['max'] = vmax
    prof['mean'] = float(s.mean())
    prof['std'] = float(s.std(ddof=0))
    prof['abs_mean'] = float(s.abs().mean())
    prof['abs_median'] = float(s.abs().median())
    prof['all_nonneg'] = bool(vmin >= 0)
    prof['bounded_01'] = bool(vmin >= -1e-9 and vmax <= 1.0 + 1e-9)

    uniq = np.unique(s.values)
    if len(uniq) <= 5:
        prof['unique_values'] = [float(x) for x in uniq]
    # 只有 0 和 1（允许浮点误差）→ 二值
    prof['is_binary01'] = bool(
        set(np.round(uniq, 9)).issubset({0.0, 1.0}) and len(uniq) <= 2
    )

    # 与 close 的量级比：用绝对值中位数之比，避免被符号 / 极值带偏
    if close_series is not None:
        cs = pd.Series(close_series).astype('float64')
        cs = cs.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cs):
            close_scale = float(cs.abs().median())
            self_scale = prof['abs_median']
            if close_scale > 1e-12 and self_scale is not None:
                prof['scale_ratio_to_close'] = float(self_scale / close_scale)

    return prof


# ===========================================================================
# ① 名称/单位关键词票（unit，最优先）
# ===========================================================================
# 关键词 → 单位（按判定优先级排列，先匹配到的胜出）。含中文单位词与英文标识。
_UNIT_NAME_RULES: list[tuple[tuple[str, ...], str]] = [
    # bool / 信号：is_ / flag / signal
    (('is_', 'flag', 'signal'), 'bool'),
    # 无量纲：比率 / 百分比 / 标准化 / 排序 / 相关
    (('ratio', 'pct', 'rate', '%', 'zscore', 'z_score', 'corr', 'rank',
      'return', 'ret', 'chg', 'mom', 'rsi'), 'dimensionless'),
    # 金额（元/亿/万/成交额 amount）
    (('amount', '成交额', 'turnover_amt', 'money', '元', '亿', '万'), 'money'),
    # 计数（手/张/笔/股 + 成交量/持仓量）
    (('手', '张', '笔', '股', 'volume', 'vol', 'oi', 'interest', 'position',
      '持仓', 'count'), 'count'),
    # 价格
    (('close', 'open', 'high', 'low', 'vwap', 'twap', 'price', 'settle',
      'mid', '价'), 'price'),
]


def name_unit_vote(name: str) -> dict:
    """基于变量名里的关键词/单位词投票判定 unit（最优先的定性依据）。

    返回：{'unit': str, 'confidence': float, 'reason': str}
    """
    lname = str(name).lower()
    for keywords, unit in _UNIT_NAME_RULES:
        hit = next((kw for kw in keywords if kw.lower() in lname), None)
        if hit is not None:
            return {'unit': unit, 'confidence': 0.8,
                    'reason': f"变量名 {name!r} 含关键词 {hit!r} → {unit}"}
    return {'unit': 'unknown', 'confidence': 0.15,
            'reason': f"变量名 {name!r} 无已知单位关键词"}


# ===========================================================================
# ② 统计画像票（unit，仅兜底，弱化）
# ===========================================================================
def stat_unit_vote(series: pd.Series, close_series: pd.Series | None = None) -> dict:
    """基于数据统计画像兜底判定 unit（名称定不了时才用）。

    弱化后只区分能从「取值范围」可靠判定的几种：
        - 只有 {0,1}              → bool
        - 全部落在 [0,1]（非二值）→ dimensionless（比率）
        - 均值≈0、有正有负、幅度小 → dimensionless（收益率类也是无量纲）
        - 与 close 同量级(0.1~10) 且恒非负 → price
        - 恒正大数                → count（**注意**：金额 vs 计数无法从纯数值区分，
                                    统一落 count 低置信，交名称/LLM 定夺——
                                    不再靠量级瞎猜金额还是成交量）
        - 其它                    → unknown

    返回：{'unit': str, 'confidence': float, 'reason': str, 'profile': dict}
    """
    prof = _profile_series(series, close_series)
    if prof['n'] == 0:
        return {'unit': 'unknown', 'confidence': 0.0,
                'reason': '空序列，无法统计', 'profile': prof}

    # —— bool：只有 0/1 ——
    if prof['is_binary01']:
        return {'unit': 'bool', 'confidence': 0.9,
                'reason': f"取值只有 {prof['unique_values']}，判为布尔/信号",
                'profile': prof}

    # —— dimensionless：全部在 [0,1] 且非二值（比率）——
    if prof['bounded_01']:
        return {'unit': 'dimensionless', 'confidence': 0.7,
                'reason': f"取值全部落在 [0,1]（min={prof['min']:.4g}, "
                          f"max={prof['max']:.4g}），判为无量纲比率",
                'profile': prof}

    # —— dimensionless：均值≈0、有正有负、幅度小（收益率类）——
    has_negative = prof['min'] < 0
    small_scale = (prof['abs_median'] is not None and prof['abs_median'] < 0.5)
    near_zero_mean = abs(prof['mean']) < max(0.05, 2 * (prof['std'] or 0))
    if has_negative and small_scale and near_zero_mean:
        return {'unit': 'dimensionless', 'confidence': 0.65,
                'reason': f"均值≈0（{prof['mean']:.4g}）、有正有负、幅度小"
                          f"（|中位数|≈{prof['abs_median']:.4g}），判为无量纲",
                'profile': prof}

    ratio_to_close = prof['scale_ratio_to_close']

    # —— price：与 close 同量级 ——
    if ratio_to_close is not None and 0.1 <= ratio_to_close <= 10.0 and prof['all_nonneg']:
        return {'unit': 'price', 'confidence': 0.7,
                'reason': f"与 close 同量级（比值≈{ratio_to_close:.3g}）且恒非负，判为价格",
                'profile': prof}

    # —— count：恒正大数（无法区分金额/计数，统一 count 低置信交名称定夺）——
    if prof['all_nonneg'] and prof['abs_median'] is not None and prof['abs_median'] > 1e4:
        return {'unit': 'count', 'confidence': 0.55,
                'reason': f"恒非负且量级极大（|中位数|≈{prof['abs_median']:.4g}），"
                          f"弱判为计数（金额/计数需名称区分）",
                'profile': prof}

    # —— 兜底：恒正较大数，也弱倾向 count ——
    if prof['all_nonneg'] and prof['abs_median'] is not None and prof['abs_median'] > 100:
        return {'unit': 'count', 'confidence': 0.4,
                'reason': f"恒非负、量级较大（|中位数|≈{prof['abs_median']:.4g}），"
                          f"弱判为计数",
                'profile': prof}

    return {'unit': 'unknown', 'confidence': 0.2,
            'reason': '统计画像无明显特征，无法确定单位', 'profile': prof}


# ===========================================================================
# ③ 金融语义票（semantic，软标注，不校验）
# ===========================================================================
# 语义关键词 → semantic（名称优先命中）
_SEMANTIC_NAME_RULES: list[tuple[tuple[str, ...], str]] = [
    (('open_interest', 'oi', 'interest', 'position', '持仓'), 'open_interest'),
    (('amount', '成交额', 'turnover'), 'turnover'),
    (('volume', 'vol'), 'volume'),
    (('mom', 'rsi', 'momentum'), 'momentum'),
    (('vol10', 'volat', 'std', 'atr', 'range'), 'volatility'),
    (('ret', 'return', 'chg'), 'return'),
    (('liquid', 'vol_ratio'), 'liquidity'),
    (('clv', 'intraday'), 'intraday'),
    (('corr',), 'correlation'),
    (('close', 'open', 'high', 'low', 'vwap', 'price', 'ma', '价'), 'price'),
]


def semantic_vote(name: str, profile: dict | None = None) -> dict:
    """推断金融语义（软标注）。名称关键词优先，辅以统计画像。

    返回：{'semantic': str, 'confidence': float, 'reason': str}
    """
    lname = str(name).lower()
    for keywords, sem in _SEMANTIC_NAME_RULES:
        hit = next((kw for kw in keywords if kw.lower() in lname), None)
        if hit is not None:
            return {'semantic': sem, 'confidence': 0.7,
                    'reason': f"变量名 {name!r} 含 {hit!r} → 语义 {sem}"}

    # 名称无线索：用统计画像兜底推收益率类语义（均值≈0、有负、幅度小）
    if profile:
        has_negative = (profile.get('min') or 0) < 0
        small = (profile.get('abs_median') is not None
                 and profile['abs_median'] < 0.5)
        near_zero = (profile.get('mean') is not None
                     and abs(profile['mean']) < max(0.05, 2 * (profile.get('std') or 0)))
        if has_negative and small and near_zero:
            return {'semantic': 'return', 'confidence': 0.5,
                    'reason': "统计画像（均值≈0、有正有负、幅度小）→ 语义 return"}

    return {'semantic': 'unknown', 'confidence': 0.15,
            'reason': f"变量名 {name!r} 无已知语义关键词"}


# ===========================================================================
# ④ LLM 判定票（可选，产 unit + semantic 双字段）
# ===========================================================================
_LLM_TOOL = {
    'name': 'report_dimension',
    'description': '根据统计画像与变量名，判定该金融特征的物理单位与金融语义。',
    'input_schema': {
        'type': 'object',
        'properties': {
            'unit': {
                'type': 'string',
                'enum': sorted(VALID_UNITS),
                'description': '物理单位（硬约束）',
            },
            'semantic': {
                'type': 'string',
                'enum': sorted(VALID_SEMANTICS),
                'description': '金融语义（软标注）',
            },
            'confidence': {
                'type': 'number',
                'description': '对 unit 判定的置信度 0~1',
            },
            'reason': {
                'type': 'string',
                'description': '简短判定理由（中文）',
            },
        },
        'required': ['unit', 'confidence', 'reason'],
    },
}


def llm_vote(name: str, profile: dict, llm_client=None) -> dict | None:
    """用 LLM 判定 unit + semantic（可选）。

    参数：
        name:       变量名
        profile:    _profile_series 生成的统计画像
        llm_client: Anthropic 风格客户端；为 None 时返回 None（跳过这一票）。

    返回：{'unit', 'semantic', 'confidence', 'reason'} 或 None（缺席 / 调用失败）。

    兼容：若 LLM 只返回旧字段 'dimension'（旧 schema / 测试 mock），
          按旧标签反推 unit + semantic。
    """
    if llm_client is None:
        raise LLMRuntimeError("dimension inference requires the shared LLM runtime client")

    slim = {k: profile.get(k) for k in (
        'n', 'n_unique', 'unique_values', 'min', 'max', 'mean', 'std',
        'abs_median', 'all_nonneg', 'bounded_01', 'is_binary01',
        'scale_ratio_to_close',
    )}
    prompt = (
        "你是金融特征量纲判定专家。给定一列数据的变量名与统计画像，判定它的：\n"
        "① 物理单位 unit（硬约束）：price(价格) / money(金额,元) / "
        "count(计数,手/张/笔/股) / dimensionless(无量纲:比率/标准化) / "
        "bool(0-1信号) / unknown。\n"
        "② 金融语义 semantic（软标注）：price/return/momentum/volatility/"
        "volume/turnover/open_interest/liquidity/intraday/correlation/unknown。\n\n"
        f"变量名：{name}\n"
        f"统计画像(JSON)：{json.dumps(slim, ensure_ascii=False)}\n\n"
        "请调用 report_dimension 工具给出判定。"
    )

    try:
        resp = llm_client.messages.create(
            model=getattr(llm_client, '_dim_model', 'claude-sonnet-4-5'),
            max_tokens=512,
            tools=[_LLM_TOOL],
            tool_choice={'type': 'tool', 'name': 'report_dimension'},
            messages=[{'role': 'user', 'content': prompt}],
        )
        for block in getattr(resp, 'content', []):
            if getattr(block, 'type', None) == 'tool_use':
                data = block.input
                unit, semantic = _parse_llm_unit_semantic(data)
                conf = float(data.get('confidence', 0.5))
                conf = min(max(conf, 0.0), 1.0)
                reason = str(data.get('reason', ''))
                return {'unit': unit, 'semantic': semantic,
                        'confidence': conf, 'reason': f"LLM: {reason}"}
    except Exception as e:  # noqa: BLE001
        raise LLMRuntimeError(f"dimension LLM call failed: {e!r}") from e

    raise LLMRuntimeError("dimension LLM returned no structured result")


# 旧 legacy 标签 → (unit, semantic) 反推（供 LLM 只回 dimension 时兼容）
_LEGACY_TO_UNIT_SEM: dict[str, tuple[str, str]] = {
    'price': ('price', 'price'),
    'volume': ('count', 'volume'),
    'oi': ('count', 'open_interest'),
    'return': ('dimensionless', 'return'),
    'ratio': ('dimensionless', 'unknown'),
    'bool': ('bool', 'unknown'),
    'unknown': ('unknown', 'unknown'),
}


def _parse_llm_unit_semantic(data: dict) -> tuple[str, str]:
    """从 LLM tool 输出里解析 (unit, semantic)。

    优先读新字段 unit/semantic；缺失则回退旧字段 dimension 反推。
    """
    unit = data.get('unit')
    semantic = data.get('semantic')
    if unit in VALID_UNITS:
        sem = semantic if semantic in VALID_SEMANTICS else 'unknown'
        return unit, sem
    # 回退：旧 dimension 反推
    legacy = str(data.get('dimension', 'unknown'))
    return _LEGACY_TO_UNIT_SEM.get(legacy, ('unknown', 'unknown'))


# ===========================================================================
# 三票加权投票（对 unit）
# ===========================================================================
def _combine_unit_votes(votes: dict) -> tuple[str, float, str]:
    """把各票（对 unit 的判定）加权汇总成最终 (unit, confidence, reason)。

    参数：
        votes: {'name': {...}, 'stat': {...}, 'llm': {...} 或缺席}
               每票需含 'unit' 与 'confidence'。

    汇总逻辑同旧版：得分最高的 unit 胜出，置信按一致性微调。
    """
    scores: dict[str, float] = {}
    total_weight = 0.0
    active: list[tuple[str, str, float]] = []  # (票名, unit, conf)

    for vote_name, weight in _VOTE_WEIGHTS.items():
        v = votes.get(vote_name)
        if v is None:
            continue
        unit = v['unit']
        conf = float(v.get('confidence', 0.0))
        if unit == 'unknown':
            # unknown 不占主张，也不计入分母（避免拉低有效票置信度）
            continue
        contribution = weight * conf
        scores[unit] = scores.get(unit, 0.0) + contribution
        total_weight += weight * 1.0
        active.append((vote_name, unit, conf))

    if not scores:
        return 'unknown', 0.0, '所有票均判为 unknown 或缺席'

    best_unit = max(scores, key=scores.get)
    best_score = scores[best_unit]
    base_conf = best_score / total_weight if total_weight > 0 else 0.0

    dims_set = {u for _, u, _ in active}
    unanimous = (len(dims_set) == 1)

    if unanimous and len(active) >= 2:
        # 多票一致 → 加成（封顶 0.98）
        conf = min(0.98, base_conf + 0.15)
        agree_note = f"{len(active)}票一致({best_unit})"
    elif unanimous and len(active) == 1:
        # 仅单票有主张 → 不打折（名称/统计兜底本就是唯一依据，无交叉验证不代表不可信）
        conf = base_conf
        agree_note = f"仅单票主张({best_unit})"
    else:
        # 分歧 → 用「胜出 unit 得分 / 全部投出得分」的占比作为置信度。
        # 名称权重(1.0)高于统计(0.6)，名称胜出时占比自然更高，体现「名称优先」，
        # 不再用固定折扣把高权重的胜出票误伤到降级线下。
        total_score = sum(scores.values())
        conf = best_score / total_score if total_score > 0 else base_conf
        losing = sorted(dims_set - {best_unit})
        agree_note = f"分歧: 胜出={best_unit}, 其它={losing}"

    conf = float(min(max(conf, 0.0), 1.0))

    parts = [agree_note]
    for vote_name, unit, c in active:
        parts.append(f"{vote_name}→{unit}({c:.2f})")
    reason = "; ".join(parts)

    return best_unit, conf, reason


# ===========================================================================
# 主入口
# ===========================================================================
def infer_feature_tags(name: str, series: pd.Series,
                       close_series: pd.Series | None = None,
                       llm_client=None) -> dict:
    """自动推断一列数据的 unit（硬约束）+ semantic（软标注）。

    参数：
        name:         特征名
        series:       特征数据列
        close_series: 收盘价参照列（可选，用于价格量级比）
        llm_client:   LLM 客户端（可选，None 则跳过 LLM 票）

    返回：
        {
            'unit':            str,    # 物理单位，unit_confidence<0.5 时为 'unknown'
            'semantic':        str,    # 金融语义
            'unit_confidence': float,  # 单位判定置信度 0~1
            'votes':           dict,   # 各票明细 {'name':..,'stat':..,'llm':..}
            'reason':          str,    # 汇总理由（含告警）
        }
    """
    prof = _profile_series(series, close_series)

    # —— unit 三票（名称优先，统计兜底）——
    unit_votes: dict = {
        'name': name_unit_vote(name),
        'stat': stat_unit_vote(series, close_series),
    }
    llm_result = llm_vote(name, prof, llm_client)
    if llm_result is not None:
        # llm 票对 unit 的主张（含 confidence）
        unit_votes['llm'] = {'unit': llm_result['unit'],
                             'confidence': llm_result['confidence'],
                             'reason': llm_result['reason']}

    unit, unit_conf, unit_reason = _combine_unit_votes(unit_votes)

    # 低置信兜底：unit 判为 unknown 并告警
    if unit_conf < _CONFIDENCE_FLOOR:
        unit_reason = (f"[告警] 单位置信度 {unit_conf:.2f} < {_CONFIDENCE_FLOOR}，"
                       f"原判定={unit}，降级为 unknown。明细: {unit_reason}")
        unit = 'unknown'

    # —— semantic（软标注，LLM 优先，否则名称/统计）——
    if llm_result is not None and llm_result.get('semantic') in VALID_SEMANTICS \
            and llm_result['semantic'] != 'unknown':
        semantic = llm_result['semantic']
    else:
        semantic = semantic_vote(name, prof)['semantic']

    return {
        'unit': unit,
        'semantic': semantic,
        'unit_confidence': unit_conf,
        'votes': unit_votes,
        'reason': unit_reason,
    }


# ===========================================================================
# 推断并回填注册表（带缓存）
# ===========================================================================
def infer_and_register(registry, name: str, series: pd.Series,
                       close_series: pd.Series | None = None,
                       llm_client=None) -> dict:
    """推断 unit+semantic 并回填进 FeatureRegistry。

    参数：
        registry:     FeatureRegistry 实例（需有 set_unit/set_semantic 方法）
        name:         特征名
        series:       特征数据列
        close_series: 收盘价参照列（可选）
        llm_client:   LLM 客户端（可选）

    返回：infer_feature_tags 的结果 dict（附带 'cached' 字段标记是否命中缓存）。

    缓存：同一 registry 上、同名特征不重复推断（挂在 registry 的私有属性上）。
    """
    cache = getattr(registry, '_dim_infer_cache', None)
    if cache is None:
        cache = {}
        setattr(registry, '_dim_infer_cache', cache)

    if name in cache:
        result = dict(cache[name])
        result['cached'] = True
        return result

    result = infer_feature_tags(name, series, close_series, llm_client)
    result['cached'] = False

    # 回填注册表（unit + semantic）
    registry.set_unit(name, result['unit'])
    registry.set_semantic(name, result['semantic'])

    cache[name] = result
    return result
