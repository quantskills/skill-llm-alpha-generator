# -*- coding: utf-8 -*-
"""
generate.py — skill 主入口，串联 LLM-Alpha 因子挖掘全流程

数据流（严格按方案）：
    ① 数据      —— 合成（造带已知信号的 OHLCV，供离线验收）或 data_loader 拉真实数据
    ② 特征      —— 算一批日频量价特征（ret/ma/rsi/atr/vol_ratio ...），注册进 FeatureRegistry；
                   陌生特征过 dimension_inferrer 贴量纲
    ③ LLM 生成  —— agent provider 消费当前 AI 预先写入的公式 payload；
                   anthropic provider 才构造 SDK 客户端并强制 tool use；
                   parse_formula 解析成 Node；不可用则跳过（warm_start_pop=None）
    ④ 校验      —— validate_expression 过滤候选，被拒的记入 rejected（含原因/层）
    ⑤ GP        —— 合法候选作 warm_start_pop 传给 GPEngine.run()；
                   fitness_fn 包 evaluate_fitness，validator_fn 包 validate_expression
    ⑥ AlphaEval —— 对 top_k 因子调 alpha_eval 五维打分
    ⑦ 解释      —— 对入选因子 explain_factor 生成经济解释（LLM 可选）
    ⑧ 报告      —— build_report 生成自包含 HTML 写到 output_dir/report.html
    ⑨ 返回      —— {factors, trajectory, rejected, report_path, meta}

要点：
    - LLM 全程强制。任一阶段失败直接抛错，不允许纯 GP、空解释或中性评分降级。
      logic 维中性 0.5）。在 Codex/Claude/Cursor 等 agent 环境中，优先使用
      未配置外部模型时使用当前工具宿主 AI，不要求用户配置 API key。
    - 参照 skill-trade-review/scripts/review.py 的 run() 风格：一个 run() 串全程，
      CLI 自检兜底。

对外主入口：
    run(universe, start_date, end_date, config=None) -> dict
    parse_formula(s, registry) -> Node       （公式字符串 → 表达式树）

依赖同级模块（裸模块名 import，运行时需 scripts/ 在 sys.path）。
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 允许 scripts/ 目录直接 import 同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from operators import OPERATORS, OPERATOR_NAMES
from expression import (
    Node,
    evaluate,
    to_formula_string,
    count_nodes,
)
from feature_registry import FeatureRegistry
from fitness import evaluate_fitness, compute_future_return, safe_rank_ic
from dimension_inferrer import infer_and_register
from validator import validate_expression
from gp_engine import GPEngine
from alpha_eval import alpha_eval
from llm_explainer import explain_factor, LLMError
from report_builder import build_report
from llm_runtime import LLMRuntime, LLMRuntimeError
from llm_agent_protocol import AtomicRunOutput

# data_loader 依赖 panda_data，缺库时也不能拖垮纯合成流程 —— 惰性容错 import
try:
    from data_loader import load_ohlcv
except Exception:  # noqa: BLE001
    load_ohlcv = None


BUILD_ID = "G00"
BUILD_NAME = "LLM-Alpha 因子挖掘"

# 前瞻步长：用未来 HORIZON 期收益衡量因子预测力
_HORIZON = 1

# 样本外验证最少有效样本数：train / holdout 任一段不足此值就关闭 oos
# （与 fitness.py / alpha_eval.py 的 _MIN_VALID 口径一致）
_MIN_VALID = 30

# ===========================================================================
# 公式解析：字符串 → 表达式树 Node
# ===========================================================================
def parse_formula(s: str, registry: FeatureRegistry) -> Node:
    """把前缀函数式公式字符串解析成表达式树 Node。

    支持的语法（Python 函数调用形式）：
        - 算子调用：add(close, ts_mean(close, 5))
        - 特征叶子：裸标识符（如 close），须在 registry 中已注册
        - 常数叶子：数字字面量（如 5、-2.0）、一元负号（如 -1）

    映射规则：
        - ast.Call        → Node.make_op(算子名, 子节点...)（算子须在 OPERATOR_NAMES）
        - ast.Name        → Node.feature(名)（须已注册，否则抛错交由上层记 rejected）
        - ast.Constant/Num→ Node.const(数值)
        - 一元负号 -x      → 若 x 为常数则折叠成负常数，否则报错（本算子集无一元减）

    参数：
        s:        公式字符串
        registry: 特征注册表（用于校验特征是否已注册）

    返回：Node 根节点。

    异常：
        ValueError —— 语法非法 / 未知算子 / 未注册特征 / arity 不符 等，
                      统一抛 ValueError（上层捕获后记入 rejected）。
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError("公式字符串为空")

    text = s.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"公式语法错误: {e}") from e

    def _build(node: ast.AST) -> Node:
        # 函数调用 → 算子节点
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("算子调用形式非法（函数名必须是标识符）")
            op = node.func.id
            if op not in OPERATOR_NAMES:
                raise ValueError(f"未知算子: {op!r}")
            if node.keywords:
                raise ValueError(f"算子 {op} 不支持关键字参数")
            children = [_build(arg) for arg in node.args]
            # make_op 内部会校验 arity，不符则抛 ValueError
            return Node.make_op(op, children)

        # 标识符 → 特征叶子
        if isinstance(node, ast.Name):
            name = node.id
            if not registry.has(name):
                raise ValueError(f"特征 {name!r} 未在注册表中注册")
            return Node.feature(name)

        # 数字字面量 → 常数叶子
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"非法常数字面量: {node.value!r}")
            return Node.const(node.value)

        # 一元负号：仅允许作用在常数上（折叠成负常数）
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _build(node.operand)
            if inner.node_type == "const":
                return Node.const(-inner.value)
            raise ValueError("一元负号只能作用于常数")

        raise ValueError(f"不支持的语法节点: {type(node).__name__}")

    return _build(tree.body)


# ===========================================================================
# ① 数据：合成 or 真实
# ===========================================================================
def _make_synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """造一段带「已知信号」的合成 OHLCV，供离线验收。

    构造思路：
        - 收盘价走几何随机游走；
        - 人为在收益里注入一个「短期动量」信号：
          未来收益与 (close 相对其 5 日均线的偏离) 正相关，
          这样 GP / LLM 若挖到 sub(close, ts_mean(close,5)) 一类因子应有正 IC，
          用于验证 pipeline 能端到端产出有效因子。
    """
    rng = np.random.default_rng(seed)
    # 基础对数收益
    base_ret = rng.normal(0.0, 0.012, size=n)

    close = np.empty(n, dtype=float)
    close[0] = 100.0
    for t in range(1, n):
        close[t] = close[t - 1] * (1.0 + base_ret[t])

    close_s = pd.Series(close)
    ma5 = close_s.rolling(5, min_periods=1).mean()
    # 归一化的动量偏离（当前价相对 5 日均线）
    momentum = (close_s - ma5) / (ma5 + 1e-9)

    # 让「未来一期」收益与当前动量偏离正相关（信号强度 0.6）+ 噪声
    signal_ret = 0.6 * momentum.shift(0).fillna(0.0) + rng.normal(0.0, 0.006, size=n)
    # 把注入信号叠加进价格：t 期的注入影响 t+1 期收益
    for t in range(1, n):
        close[t] = close[t - 1] * (1.0 + base_ret[t] + 0.15 * signal_ret.iloc[t - 1])

    close_s = pd.Series(close)
    # 用收盘价反推一套自洽的 OHLCV
    prev = close_s.shift(1).fillna(close_s.iloc[0])
    open_ = prev * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close_s) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
    low = np.minimum(open_, close_s) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
    volume = pd.Series(rng.integers(1_000, 50_000, size=n).astype(float))
    amount = volume * close_s
    open_interest = pd.Series(rng.integers(10_000, 200_000, size=n).astype(float))

    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "trade_date": dates,
        "open": open_.values,
        "high": high.values,
        "low": low.values,
        "close": close_s.values,
        "volume": volume.values,
        "amount": amount.values,
        "open_interest": open_interest.values,
    })
    return df


def _load_real_ohlcv(universe, start_date, end_date, config, warnings) -> pd.DataFrame:
    """拉真实 OHLCV。universe 支持单代码字符串或列表（取第一个标的，本 skill 单标的时序）。"""
    if load_ohlcv is None:
        warnings.append("data_loader 不可用（缺 panda_data），无法拉真实数据")
        return pd.DataFrame()

    ts_code = universe[0] if isinstance(universe, (list, tuple)) and universe else universe
    if not ts_code:
        warnings.append("universe 为空，无法拉真实数据")
        return pd.DataFrame()

    data_cfg = (config or {}).get("data", {})
    try:
        df = load_ohlcv(
            str(ts_code), str(start_date), str(end_date),
            asset_type=data_cfg.get("asset_type", "auto"),
            username=data_cfg.get("username"),
            password=data_cfg.get("password"),
            adj_method=data_cfg.get("adj_method", "close_pcs"),
            frequency=data_cfg.get("frequency"),
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"load_ohlcv 异常: {exc!r}")
        return pd.DataFrame()

    if df is None or df.empty:
        warnings.append(f"load_ohlcv 返回空数据（ts_code={ts_code}）")
        return pd.DataFrame()
    return df


# ===========================================================================
# ② 特征：算一批日频量价特征并注册
# ===========================================================================
def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """基于标准 OHLCV 算一批常用日频特征（约 15-20 个）。

    返回一个特征 DataFrame（列即特征名），不含原始 trade_date。
    所有特征都是「回看」的，无前视。
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float) if "amount" in df.columns else volume * close
    eps = 1e-9

    feats: dict[str, pd.Series] = {}

    # 原始量价（直接可作叶子）
    feats["open"] = open_
    feats["high"] = high
    feats["low"] = low
    feats["close"] = close
    feats["volume"] = volume
    feats["amount"] = amount
    if "open_interest" in df.columns:
        feats["open_interest"] = df["open_interest"].astype(float)

    # 收益率类
    feats["ret"] = close.pct_change().fillna(0.0)                       # 单期收益
    feats["ret5"] = close.pct_change(5).fillna(0.0)                     # 5 期收益
    feats["log_ret"] = np.log(close / close.shift(1)).fillna(0.0)       # 对数收益

    # 均线 / 动量
    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    feats["ma5"] = ma5
    feats["ma10"] = ma10
    feats["ma20"] = ma20
    feats["mom5"] = (close - ma5) / (ma5 + eps)                         # 相对 5 日均线偏离（无量纲）
    feats["mom20"] = (close - ma20) / (ma20 + eps)                      # 相对 20 日均线偏离

    # 波动率
    feats["vol10"] = feats["ret"].rolling(10, min_periods=1).std().fillna(0.0)

    # ATR（14 期真实波幅均值）
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    feats["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    # RSI（14 期）
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + eps)
    feats["rsi14"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    # 量比（当前量相对 20 日均量）
    vma20 = volume.rolling(20, min_periods=1).mean()
    feats["vol_ratio"] = volume / (vma20 + eps)

    # 日内位置 / 振幅
    feats["hl_range"] = (high - low) / (close + eps)                    # 相对振幅（无量纲）
    feats["clv"] = ((close - low) - (high - close)) / ((high - low) + eps)  # 收盘位置 [-1,1]

    out = pd.DataFrame(feats)
    # 用前值/0 兜底残余 NaN/Inf，保证下游求值稳定
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out


# 特征双维度标注：unit（物理单位，硬约束）+ semantic（金融语义，软标注）
# 修复缺陷：amount=money 与 volume=count 分离（旧体系混为 volume，量级差 1e4 却能过校验）；
# 派生的比率/动量/波动率类明确标 dimensionless。
_FEATURE_TAGS: dict[str, tuple[str, str]] = {
    # 价格类（单位 price）
    "open": ("price", "price"), "high": ("price", "price"),
    "low": ("price", "price"), "close": ("price", "price"),
    "ma5": ("price", "price"), "ma10": ("price", "price"),
    "ma20": ("price", "price"),
    "atr14": ("price", "volatility"),        # 真实波幅，单位=价格
    # 金额（单位 money，元）
    "amount": ("money", "turnover"),
    # 计数（单位 count，手/张）
    "volume": ("count", "volume"), "vma20": ("count", "volume"),
    "open_interest": ("count", "open_interest"),
    # 无量纲派生
    "ret": ("dimensionless", "return"), "ret5": ("dimensionless", "return"),
    "log_ret": ("dimensionless", "return"),
    "mom5": ("dimensionless", "momentum"), "mom20": ("dimensionless", "momentum"),
    "vol10": ("dimensionless", "volatility"),
    "vol_ratio": ("dimensionless", "liquidity"),   # 量比 count/count → 无量纲
    "rsi14": ("dimensionless", "momentum"),         # 0-100 有界
    "hl_range": ("dimensionless", "volatility"),
    "clv": ("dimensionless", "intraday"),           # 收盘位置 [-1,1]
}


def _register_features(feat_df: pd.DataFrame, close: pd.Series,
                       llm_client=None, warnings: list | None = None) -> FeatureRegistry:
    """把特征列注册进 FeatureRegistry；已知特征直接标 unit+semantic，陌生特征过推断器。

    m5：跳过常数 / 全 NaN 列——上游 `ffill().fillna(0.0)` 会把全 NaN 列（如某些标的
    的 open_interest）变成恒零常量，这类列进 GP 特征池毫无预测力、还白占搜索空间、
    可能被算子放大成异常，故 nunique<=1（含全 NaN 兜底后为常数）直接跳过注册并 warn。
    """
    reg = FeatureRegistry(seed_defaults=True)
    for col in feat_df.columns:
        col = str(col)
        # m5：常数 / 全 NaN 列跳过注册（不进 GP 池）
        series = feat_df[col]
        if series.nunique(dropna=True) <= 1:
            if warnings is not None:
                warnings.append(f"特征 {col!r} 为常数/全 NaN（无区分度），跳过注册")
            continue
        if False and col in _FEATURE_TAGS:
            unit, semantic = _FEATURE_TAGS[col]
            reg.register(col, unit=unit, semantic=semantic)
        else:
            # 陌生特征：交给 dimension_inferrer 推断 unit+semantic（内部对 LLM 优雅降级）
            try:
                infer_and_register(reg, col, feat_df[col], close_series=close,
                                   llm_client=llm_client)
            except Exception:
                raise
    return reg


# ===========================================================================
# ③ LLM 生成候选表达式（简化版 llm_generator，内联实现）
# ===========================================================================
# 强制 tool use 的 schema：让 LLM 一次性吐出一批候选公式
_GEN_TOOL = {
    "name": "emit_alpha_candidates",
    "description": "产出一批量化 alpha 因子候选表达式（前缀函数式），供后续遗传编程做种子。",
    "input_schema": {
        "type": "object",
        "properties": {
            "formulas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "候选因子公式字符串列表，每个形如 "
                               "sub(ts_mean(close,5), ts_mean(close,20))。",
            },
        },
        "required": ["formulas"],
    },
}


def _gen_system_prompt() -> str:
    """生成候选公式的 system prompt：说清算子集与语法约束。"""
    op_names = ", ".join(sorted(OPERATOR_NAMES))
    return (
        "你是量化因子研究专家。请构造一批**多样化**的 alpha 因子候选表达式，"
        "用于后续遗传编程演化。\n"
        "语法：前缀函数式（Python 调用形式），如 sub(ts_mean(close,5), ts_mean(close,20))。\n"
        f"仅可使用以下算子（严格拼写）：{op_names}。\n"
        "时序算子（ts_mean/ts_std/ts_max/ts_min/ts_rank/ts_zscore/delay/diff/"
        "ts_decay_linear/ts_corr）的最后一个参数必须是整数窗口常数（如 5、10、20）。\n"
        "只能使用给定的特征名，不要臆造特征。\n"
        "注意物理单位：不同单位不能相加减（如价格 price 与计数 count、金额 money 之间"
        "不能 add/sub），只有同单位或无量纲之间可加减；乘除会自动推导新单位"
        "（如 价格×计数=金额）。\n"
        "必须调用 emit_alpha_candidates 工具返回公式列表。"
    )


def _llm_generate_candidates(feature_names, n_candidates, llm_client, config, meta):
    """调 LLM 生成 n_candidates 个候选公式字符串。

    方案 A（硬失败）：只在 LLM 启用（llm_client 非 None）时被调用。调用异常 /
    非 tool_use 回复 → 抛 LLMError 中断，绝不吞掉降级成纯 GP。返回的公式列表
    可以为空（模型合法地返回了空 formulas），此属正常，由上层记 warning。
    """
    if llm_client is None:
        raise LLMRuntimeError("candidate generation requires the shared LLM runtime client")

    # 模型解析统一走 llm_explainer（含裸别名归一化），不各写一份。
    model = ((config or {}).get("llm", {}) or {}).get("model") \
        or getattr(llm_client, "_alpha_model", None) \
        or "current-agent"

    user = (
        f"可用特征名（仅限这些）：{', '.join(sorted(feature_names))}\n"
        f"请给出 {n_candidates} 个不同的候选因子公式，覆盖动量/反转/波动率/量价背离等不同逻辑。\n"
        "调用 emit_alpha_candidates 工具返回。"
    )

    started = time.time()
    try:
        resp = llm_client.messages.create(
            model=model,
            max_tokens=2048,
            system=_gen_system_prompt(),
            tools=[_GEN_TOOL],
            tool_choice={"type": "tool", "name": "emit_alpha_candidates"},
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001 —— 启用状态下调用异常硬失败，不降级
        latency = int((time.time() - started) * 1000)
        meta["llm_generate"] = {"called": True, "skipped_reason": f"call_failed: {exc!r}",
                                "n_formulas": 0, "latency_ms": latency}
        raise LLMError(
            f"LLM 候选生成调用失败（model={model}, latency={latency}ms）: {exc!r}"
        ) from exc

    latency = int((time.time() - started) * 1000)

    tool_use = None
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            tool_use = block
            break

    if tool_use is None:
        meta["llm_generate"] = {"called": True, "skipped_reason": "no_tool_use",
                                "n_formulas": 0, "latency_ms": latency}
        raise LLMError(
            f"LLM 候选生成回复无 tool_use（model={model}, latency={latency}ms）；"
            "模型未按要求调用 emit_alpha_candidates 工具"
        )

    data = tool_use.input or {}
    formulas = data.get("formulas") or []
    formulas = [str(f).strip() for f in formulas if str(f).strip()]
    meta["llm_generate"] = {"called": True, "skipped_reason": None,
                            "n_formulas": len(formulas), "latency_ms": latency,
                            "model": model}
    return formulas


# ===========================================================================
# top_k 相关去重：从 GP 候选里贪心选出彼此低相关的因子（打散同质化）
# ===========================================================================
def _select_diverse(candidates, k, corr_thresh=0.8):
    """从按 fitness 降序的候选里贪心选出彼此低相关的 k 个，抑制 top_k 同质化。

    候选 top5 常同质（如全是波动率族），diversity 偏低。做法：按 fitness 从高到低
    逐个考察，只要与「已选中」的每个信号 |Spearman 相关| 都低于 corr_thresh 就入选；
    若考察完仍不足 k 个（候选普遍高度相关），用剩余 fitness 最高者补齐到 k
    （保证产出个数不因去重而缩水）。信号相关口径与挖掘一致（safe_rank_ic）。

    参数：
        candidates:  list[dict]，每项 {'node','fitness','signal'}，已按 fitness 降序
        k:           目标个数
        corr_thresh: 入选的最大允许 |相关|（默认 0.8）

    返回：
        (selected: list[dict], n_deduped: int)
        n_deduped —— 因高相关被跳过、最终靠补齐才凑够的个数（诊断用，0 表示未触发补齐）
    """
    selected: list[dict] = []
    skipped: list[dict] = []
    for item in candidates:
        if len(selected) >= k:
            break
        sig = item["signal"]
        too_similar = any(
            abs(safe_rank_ic(sig, s["signal"])) >= corr_thresh for s in selected
        )
        if too_similar:
            skipped.append(item)
        else:
            selected.append(item)

    # 不足 k：候选都高度相关，用被跳过里 fitness 最高的补齐（skipped 已按原降序）
    n_deduped = 0
    for item in skipped:
        if len(selected) >= k:
            break
        selected.append(item)
        n_deduped += 1
    return selected, n_deduped


# ===========================================================================
# 主入口
# ===========================================================================
def run(universe, start_date, end_date, config: dict | None = None) -> dict:
    """LLM-Alpha 因子挖掘主入口，串联全流程。

    参数：
        universe:   标的（单代码字符串或列表；本 skill 单标的时序，取第一个）
        start_date: 起始日期（真实数据模式下透传 data_loader，'YYYYMMDD'）
        end_date:   结束日期
        config:     配置 dict，字段：
            llm:  {enabled, model, base_url, api_key}   —— LLM 全程可选
            n_candidates:  LLM 候选公式个数（默认 12）
            gp:   {pop_size, n_gen, max_depth, elite_frac,
                   crossover_rate, mutation_rate, seed}  —— GPEngine 超参
            data: {use_synthetic: bool, asset_type, username, password}
            output_dir:    报告输出目录（默认 scripts/../output）
            top_k:         最终入选因子数（默认 5）

    返回：
        {
            'factors':     list[dict],  # 入选因子（含 formula/alpha_scores/rankic/解释等）
            'trajectory':  list[dict],  # GP 每代轨迹
            'rejected':    list[dict],  # 被校验/解析拒绝的候选（含原因）
            'report_path': str | None,  # 报告 HTML 路径
            'meta':        dict,        # 运行元信息（LLM 状态/数据源/超参 等）
        }
    """
    config = config or {}
    warnings: list[str] = []
    rejected: list[dict] = []
    meta: dict = {
        "build_id": BUILD_ID,
        "build_name": BUILD_NAME,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "warnings": warnings,
    }

    runtime = LLMRuntime.from_config(config)
    config = dict(config)
    config["_run_id"] = runtime.run_id
    llm_client = runtime.client
    llm_enabled = True
    n_candidates = int(config.get("n_candidates", 12))
    top_k = int(config.get("top_k", 5))
    # 未来收益前瞻步长：默认 _HORIZON（日线口径 1），分钟线可经 config.horizon 覆盖
    # （如 15m 线预测未来 1 小时 = 4 根 → horizon=4）。
    horizon = int(config.get("horizon", _HORIZON))
    gp_cfg = config.get("gp", {}) or {}
    data_cfg = config.get("data", {}) or {}
    use_synthetic = bool(data_cfg.get("use_synthetic", False))

    # 样本外验证配置（m10）：后 holdout_frac 作样本外 holdout，GP 只在 train 挖
    oos_cfg = config.get("oos", {}) or {}
    holdout_frac = float(oos_cfg.get("holdout_frac", 0.3))
    enable_oos = bool(oos_cfg.get("enabled", True))
    dedup_corr_thresh = float(oos_cfg.get("dedup_corr_thresh", 0.8))

    # 输出目录
    default_out = str(Path(__file__).resolve().parent.parent / "output")
    output_dir = config.get("output_dir") or default_out
    os.makedirs(output_dir, exist_ok=True)

    # ---- LLM 客户端 / agent payload ----
    meta["llm_enabled"] = llm_enabled
    meta["llm_mode"] = runtime.mode
    meta["llm_available"] = True
    meta["llm_runtime"] = runtime.audit()
    # llm_call_ok：LLM 是否真实调用成功过至少一次（区别于 llm_available=客户端构造成功）。
    # 只要生成/解释/逻辑打分任一环节真正拿到 LLM 结果就置 True。
    meta["llm_call_ok"] = False

    # ---- ① 数据 ----
    if use_synthetic:
        df = _make_synthetic_ohlcv()
        meta["data_source"] = "synthetic"
    else:
        df = _load_real_ohlcv(universe, start_date, end_date, config, warnings)
        meta["data_source"] = "real"

    if df is None or df.empty:
        raise LLMRuntimeError("data unavailable; no formal alpha report was published")
    meta["n_rows"] = int(len(df))

    close = df["close"].astype(float).reset_index(drop=True)
    # 主连换月标记（分钟线主连未复权，换月有跳空）：dominant_id 变化的 bar 记为换月点。
    # 日线后复权 / 合成数据无此列时 dom_id=None，不做换月处理。
    dom_id = None
    if "dominant_id" in df.columns:
        dom_id = df["dominant_id"].reset_index(drop=True)

    # ---- ② 特征 ----
    feat_df = _compute_features(df).reset_index(drop=True)
    registry = _register_features(feat_df, close, llm_client=llm_client,
                                  warnings=warnings)
    runtime.record("report_dimension", status="success",
                   n_features=len(registry.feature_names()))
    feature_names = registry.list_features()
    meta["n_features"] = len(feature_names)

    # 求值用 data 字典：{特征名: Series}
    data_dict = {name: feat_df[name] for name in feat_df.columns}
    # 未来收益（fitness / alpha_eval 的标签），horizon 可配（分钟线用）
    future_return = compute_future_return(close, horizon=horizon)
    meta["horizon"] = horizon

    # ---- 换月跳空处理（分钟线主连）----
    # future_return(t) = close(t+h)/close(t)-1，若窗口 [t, t+h] 跨越换月点，
    # 该收益含跳空（虚假信号）→ 置 NaN（用户选定「换月日收益置零」，此处置 NaN
    # 更严谨：NaN 会被 fitness/alpha_eval 的 _clean_pair 剔除，不参与 IC 估计）。
    if dom_id is not None:
        changed = dom_id.ne(dom_id.shift()).to_numpy()
        changed[0] = False  # 首行不算换月
        n_switch = int(changed.sum())
        if n_switch > 0:
            fr = future_return.to_numpy(dtype=float).copy()
            n = len(fr)
            # 任一换月点 c，凡窗口覆盖它的起点 t ∈ [c-h, c] 的 future_return 置 NaN
            switch_idx = np.flatnonzero(changed)
            for c in switch_idx:
                lo = max(0, c - horizon)
                fr[lo:c + 1] = np.nan
            future_return = pd.Series(fr, index=future_return.index)
            warnings.append(
                f"主连换月 {n_switch} 次，已将跨换月窗口的未来收益置 NaN（不参与 IC）")
            meta["n_contract_switch"] = n_switch

    # ---- 样本外切分（m10）----
    # 按时间顺序切：前 (1-holdout_frac) 作 train（GP 只在此挖掘），后段作 holdout
    # （只算 oos_ic 作样本外体检，不参与挖掘）。任一段有效样本不足则关闭 oos，
    # 退回全样本挖掘并告警（数据太短时样本外 IC 无统计意义）。
    n_rows = len(close)
    split_idx = int(round(n_rows * (1.0 - holdout_frac)))
    if not enable_oos or split_idx < _MIN_VALID or (n_rows - split_idx) < _MIN_VALID:
        if enable_oos:  # 本想开但数据太短
            warnings.append(
                f"数据过短（n={n_rows}，切分点={split_idx}），train 或 holdout 段"
                f"不足 {_MIN_VALID} 有效样本，关闭样本外验证，退回全样本挖掘"
            )
        enable_oos = False
        split_idx = n_rows
    train_slice = slice(0, split_idx)
    meta["oos"] = {
        "enabled": enable_oos,
        "holdout_frac": holdout_frac,
        "split_idx": split_idx,
        "n_train": int(split_idx),
        "n_holdout": int(n_rows - split_idx),
    }

    # GP 挖掘只见 train 段：对特征与标签同步切片，口径一致（不泄漏样本外信息）
    data_dict_train = {name: s.iloc[train_slice] for name, s in data_dict.items()}
    future_return_train = future_return.iloc[train_slice]

    # ---- ③ LLM 生成候选 ----
    warm_start_pop: list[Node] = []
    formulas: list[str] = []
    formula_source = "llm"
    formulas = _llm_generate_candidates(
        feature_names, n_candidates, llm_client,
        {"llm": {"model": runtime.model}}, meta
    )
    runtime.record("emit_alpha_candidates", status="success", n_formulas=len(formulas))
    meta["llm_call_ok"] = True

    # ---- ④ 解析 + 校验 ----
    for f in formulas:
        try:
            node = parse_formula(f, registry)
        except Exception as exc:  # noqa: BLE001 —— 解析失败记 rejected
            rejected.append({"formula": f, "reason": f"解析失败: {exc}",
                             "layer": "解析", "source": formula_source})
            continue
        res = validate_expression(node, registry, single_asset=True)
        if res["valid"]:
            warm_start_pop.append(node)
        else:
            rejected.append({"formula": to_formula_string(node),
                             "reason": res["reason"], "layer": res["layer"],
                             "source": formula_source})

    meta["n_warm_start"] = len(warm_start_pop)

    # ---- ⑤ GP 演化 ----
    # fitness_fn：把 node 求值成信号，再算适应度（异常 → 返回 -inf 由引擎处理）
    # 只在 train 段求值 + 评估，样本外 holdout 完全不参与挖掘。
    def _fitness_fn(node: Node) -> float:
        signal = evaluate(node, data_dict_train)
        if not isinstance(signal, pd.Series):
            # 整棵树退化成常数：无预测力
            return 0.0
        fit, _ = evaluate_fitness(signal, future_return_train,
                                  node_count=count_nodes(node))
        return fit

    def _validator_fn(node: Node) -> bool:
        # 本 skill 是单标的时序（universe 取第一个标的），单标的场景禁用横截面 rank
        # （m9：rank 在单标的上退化为含未来数据的整段排名 → 前视）。GP 生成的含 rank
        # 个体会被此处拦下、不进种群，从源头把演化收敛到无 rank 的可复现空间。
        return validate_expression(node, registry, single_asset=True)["valid"]

    engine = GPEngine(
        feature_names=feature_names,
        fitness_fn=_fitness_fn,
        validator_fn=_validator_fn,
        pop_size=int(gp_cfg.get("pop_size", 80)),
        n_gen=int(gp_cfg.get("n_gen", 15)),
        max_depth=int(gp_cfg.get("max_depth", 6)),
        elite_frac=float(gp_cfg.get("elite_frac", 0.05)),
        crossover_rate=float(gp_cfg.get("crossover_rate", 0.7)),
        mutation_rate=float(gp_cfg.get("mutation_rate", 0.2)),
        seed=int(gp_cfg.get("seed", 42)),
    )
    gp_out = engine.run(warm_start_pop=warm_start_pop or None)
    trajectory = gp_out["trajectory"]
    top_pairs = gp_out["top_k"]  # list[(Node, fitness)]

    # ---- ⑥ AlphaEval 五维 + ⑦ 解释 ----
    # 先对 GP 全部候选各算「整段 signal」（供样本外 IC）与「train 段 signal」
    # （供挖掘口径的评分/去重），再做相关去重选出最终 top_k，抑制同质化。
    cand_nodes = [n for (n, _f) in top_pairs]
    candidates: list[dict] = []
    for i, node in enumerate(cand_nodes):
        try:
            sig_full = evaluate(node, data_dict)
            if not isinstance(sig_full, pd.Series):
                sig_full = pd.Series(np.full(len(close), float(sig_full)))
        except Exception:  # noqa: BLE001
            sig_full = pd.Series(np.zeros(len(close)))
        candidates.append({
            "node": node,
            "fitness": float(top_pairs[i][1]),
            "signal": sig_full.reset_index(drop=True),
            # train 段信号：去重相关与五维评分都用它（挖掘口径，不含样本外）
            "signal_train": sig_full.reset_index(drop=True).iloc[train_slice],
        })

    # 相关去重：从按 fitness 降序的候选里选彼此低相关的 top_k。
    # 用 train 段信号判定相关（挖掘口径），_select_diverse 直接返回选中的完整候选。
    dedup_input = [
        {"node": c["node"], "fitness": c["fitness"], "signal": c["signal_train"],
         "_orig": c}
        for c in candidates
    ]
    picked, n_deduped = _select_diverse(dedup_input, top_k,
                                        corr_thresh=dedup_corr_thresh)
    selected = [p["_orig"] for p in picked]
    meta["n_deduped"] = n_deduped

    # 选中因子的 train 段信号集，供多样性维互相参照
    top_signals_train = [c["signal_train"] for c in selected]

    factors: list[dict] = []
    n_oos_failed = 0      # 样本外失效的因子数
    for i, cand in enumerate(selected):
        node = cand["node"]
        formula = to_formula_string(node)
        signal = cand["signal"]              # 整段信号（信号统计/样本外 IC 用）
        signal_train = cand["signal_train"]  # train 段信号（挖掘口径评分用）
        others = [s for j, s in enumerate(top_signals_train) if j != i]

        # 五维评分只用 train 段（避免样本外信息泄漏进评分）
        scores = alpha_eval(signal_train, future_return_train,
                            all_signals=others or None,
                            formula_str=formula, llm_client=llm_client,
                            config={"llm": {"model": runtime.model}})
        runtime.record("report_logic_score", status="success", formula=formula)

        # 样本内 / 样本外 IC（m10）：train_ic 与五维 pps 的 rankic 同口径；
        # oos_ic 在 holdout 段单独算，enable_oos=False 时为 None。
        train_ic = float(scores["detail"]["pps"].get("rankic") or 0.0)
        oos_ic = None
        oos_failed = False
        oos_note = ""
        if enable_oos:
            sig_hold = signal.iloc[split_idx:]
            fut_hold = future_return.iloc[split_idx:]
            oos_ic = float(safe_rank_ic(sig_hold, fut_hold))
            # 失效判定：train 有预测力（|train_ic|>0.02）时才谈样本外是否延续
            if abs(train_ic) > 0.02:
                if train_ic * oos_ic < 0:
                    oos_failed = True
                    oos_note = "样本外符号翻转"
                elif abs(oos_ic) < 0.3 * abs(train_ic):
                    oos_failed = True
                    decay_pct = (1.0 - abs(oos_ic) / abs(train_ic)) * 100.0
                    oos_note = f"样本外衰减 {decay_pct:.0f}%"
            if oos_failed:
                n_oos_failed += 1

        expl = explain_factor(formula, alpha_scores={"fitness": cand["fitness"],
                                                     "ic": train_ic},
                              llm_client=llm_client,
                              config={"llm": {"model": runtime.model}})
        runtime.record("emit_explanation", status="success", formula=formula)

        logic_degraded = bool(scores.get("logic_degraded"))
        if logic_degraded or scores.get("logic_source") != "llm":
            raise LLMRuntimeError("AlphaEval logic did not return a real LLM score")
        meta["llm_call_ok"] = True

        # 信号统计（供报告展示，用整段信号）
        sig_clean = signal.replace([np.inf, -np.inf], np.nan)
        stats = {
            "mean": float(sig_clean.mean()) if sig_clean.notna().any() else 0.0,
            "std": float(sig_clean.std()) if sig_clean.notna().any() else 0.0,
            "turnover": float(scores["detail"]["pfs"].get("turnover", 0.0)),
            "coverage": float(sig_clean.notna().mean()),
        }

        factors.append({
            "formula": formula,
            "explanation": expl.get("explanation", ""),
            "captures": expl.get("captures", ""),
            "applicable_scenario": expl.get("applicable_scenario", ""),
            "failure_scenario": expl.get("failure_scenario", ""),
            "alpha_scores": {
                "effectiveness": scores["pps"],
                "robustness": scores["rre"],
                "interpretability": scores["logic"],
                "diversity": scores["diversity"],
                "parsimony": scores["pfs"],
                "weighted_score": scores["weighted_score"],
            },
            "alpha_eval": scores,           # 完整五维明细
            "logic_source": scores.get("logic_source"),
            "logic_degraded": logic_degraded,
            # rankic 保留为「train 段 IC」（挖掘口径，与旧字段兼容）
            "rankic": train_ic,
            # m10 样本外验证字段
            "train_ic": train_ic,
            "oos_ic": oos_ic,
            "oos_failed": oos_failed,
            "oos_note": oos_note,
            "fitness": float(cand["fitness"]),
            "node_count": count_nodes(node),
            "signal_stats": stats,
        })

    meta["n_factors"] = len(factors)
    meta["n_oos_failed"] = n_oos_failed

    if not factors:
        raise LLMRuntimeError("no valid factors survived mandatory LLM evaluation")

    # m10：样本外全军覆没 → 顶层告警（强过拟合信号）
    if enable_oos and factors and n_oos_failed == len(factors):
        warnings.append(
            f"全部 {len(factors)} 个因子样本外失效（符号翻转或大幅衰减），"
            "疑似过拟合，样本内 IC 不可信"
        )

    # ---- ⑧ 报告 ----
    meta["llm_runtime"] = runtime.audit()
    report_path = _write_report(factors, trajectory, rejected, output_dir,
                                config, warnings)

    # ---- ⑨ 返回 ----
    return {
        "factors": factors,
        "trajectory": trajectory,
        "rejected": rejected,
        "report_path": report_path,
        "meta": meta,
    }


def _write_report(factors, trajectory, rejected, output_dir, config, warnings) -> str | None:
    """Render into a private staging directory, then publish atomically."""
    run_id = (config or {}).get("_run_id") or "untracked"
    try:
        with AtomicRunOutput(output_dir, run_id) as staged:
            build_report(factors, trajectory, rejected=rejected,
                         output_path=str(staged.stage_dir / "report.html"),
                         config=config)
            return str(staged.publish("report.html"))
    except Exception as exc:  # noqa: BLE001
        raise LLMRuntimeError(f"report construction failed: {exc!r}") from exc


# ===========================================================================
# CLI / 冒烟自检
# ===========================================================================
def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="G00 LLM-Alpha 因子挖掘")
    parser.add_argument("--universe", default="RB.SHF", help="标的代码（真实数据模式）")
    parser.add_argument("--start", default="20230101", help="起始日 YYYYMMDD")
    parser.add_argument("--end", default="20241231", help="结束日 YYYYMMDD")
    parser.add_argument("--synthetic", action="store_true", help="用合成数据（离线验收）")
    parser.add_argument("--out", help="报告输出目录")
    args = parser.parse_args()

    config = {
        "data": {"use_synthetic": args.synthetic},
        "gp": {"pop_size": 80, "n_gen": 15},
    }
    if args.out:
        config["output_dir"] = args.out

    out = run(args.universe, args.start, args.end, config=config)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"因子数={len(out['factors'])} 轨迹代数={len(out['trajectory'])} "
          f"淘汰={len(out['rejected'])}")
    print(f"report={out['report_path']}")
    _print_factors(out["factors"])


def _print_factors(factors):
    """打印每个因子的样本内/样本外 IC 与综合分（样本外失效带标记）。"""
    for f in factors:
        oos = f.get("oos_ic")
        oos_str = f"{oos:+.4f}" if oos is not None else "—"
        flag = f" ⚠{f.get('oos_note')}" if f.get("oos_failed") else ""
        print(f"  - {f['formula']}\n"
              f"      train_IC={f.get('train_ic', f['rankic']):+.4f} "
              f"oos_IC={oos_str}{flag} "
              f"weighted={f['alpha_scores']['weighted_score']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        # 冒烟自检：合成数据 + 关闭 LLM，纯 GP 跑通
        cfg = {
            "data": {"use_synthetic": True},
            "gp": {"pop_size": 60, "n_gen": 10},
            "top_k": 5,
        }
        out = run("SYNTH", "20230101", "20241231", config=cfg)
        sys.stdout.reconfigure(encoding="utf-8")
        print(f"[冒烟] 因子数={len(out['factors'])} "
              f"轨迹代数={len(out['trajectory'])} 淘汰={len(out['rejected'])}")
        print(f"[冒烟] report={out['report_path']}")
        _print_factors(out["factors"])
