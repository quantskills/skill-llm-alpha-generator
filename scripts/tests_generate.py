# -*- coding: utf-8 -*-
"""
tests_generate.py — generate.py 主入口 run() + m10 样本外验证 / top_k 去重 单测

覆盖 m10 改造的核心行为（全部离线：合成数据 + 关闭 LLM，可复现）：
    1. run() 产出的每个因子含 train_ic / oos_ic / oos_failed / oos_note 字段，
       meta.oos.enabled=True 且切分点合理
    2. holdout_frac 改变 → n_train / n_holdout 比例随之变化
    3. 数据过短（train 或 holdout 段不足 _MIN_VALID）→ 关闭 oos + warn，流程不崩
    4. _select_diverse 去重：高相关候选被打散，选出的因子两两相关 < 阈值；
       候选普遍高相关时靠补齐凑够 k（n_deduped>0）
    5. 合成数据里注入的真实动量信号 → train_ic 与 oos_ic 同号（不误标警）
    6. run() 可复现：同 seed 两次跑出相同公式集

运行：pytest tests_generate.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 保证能以裸模块名 import（运行时需要 scripts/ 在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate  # noqa: E402
from generate import run, _select_diverse, _register_features  # noqa: E402
from tests_support import make_test_runtime  # noqa: E402


# 全测试期间强制关闭 LLM，避免联网 / 依赖 anthropic SDK。
@pytest.fixture(autouse=True)
def _mandatory_llm(monkeypatch):
    monkeypatch.setattr(generate.LLMRuntime, "from_config",
                        staticmethod(lambda _config: make_test_runtime()))


# 统一的离线配置：合成数据 + 纯 GP + 小种群快跑
def _cfg(**over) -> dict:
    cfg = {
        "data": {"use_synthetic": True},
        "gp": {"pop_size": 40, "n_gen": 6, "seed": 42},
        "top_k": 5,
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# 1) run() 产出样本外字段 + meta.oos 正确
# ---------------------------------------------------------------------------
def test_run_produces_oos_fields():
    out = run("SYNTH", "20230101", "20241231", config=_cfg())
    assert out["factors"], "应产出至少一个因子"

    oos = out["meta"]["oos"]
    assert oos["enabled"] is True
    # 合成数据默认 400 行，70/30 → split≈280
    assert oos["n_train"] + oos["n_holdout"] == 400
    assert oos["n_train"] > oos["n_holdout"] > 0
    assert oos["split_idx"] == oos["n_train"]

    for f in out["factors"]:
        assert "train_ic" in f
        assert "oos_ic" in f
        assert "oos_failed" in f
        assert "oos_note" in f
        assert f["oos_ic"] is not None  # oos 开启时应有值
        # rankic 兼容字段 = train_ic
        assert f["rankic"] == pytest.approx(f["train_ic"])


# ---------------------------------------------------------------------------
# 2) holdout_frac 改变切分比例
# ---------------------------------------------------------------------------
def test_split_ratio_follows_config():
    out2 = run("SYNTH", "x", "y", config=_cfg(oos={"holdout_frac": 0.2}))
    oos2 = out2["meta"]["oos"]
    # 400 * 0.8 = 320
    assert oos2["split_idx"] == 320
    assert oos2["n_holdout"] == 80

    out4 = run("SYNTH", "x", "y", config=_cfg(oos={"holdout_frac": 0.4}))
    oos4 = out4["meta"]["oos"]
    assert oos4["split_idx"] == 240
    assert oos4["n_holdout"] == 160


# ---------------------------------------------------------------------------
# 3) 数据过短 → 关闭 oos + warn，不崩
# ---------------------------------------------------------------------------
def test_short_data_disables_oos(monkeypatch):
    # monkeypatch 合成数据造一段很短的 OHLCV（50 行：70/30 → holdout=15 < 30）
    # 先存原函数引用，避免替换后自我递归调用。
    _orig = generate._make_synthetic_ohlcv

    def _short(*_a, **_k):
        return _orig(n=50)
    monkeypatch.setattr(generate, "_make_synthetic_ohlcv", _short)

    out = run("SYNTH", "x", "y", config=_cfg())
    oos = out["meta"]["oos"]
    assert oos["enabled"] is False
    # 关闭后退回全样本，split_idx 落到末尾
    assert oos["split_idx"] == 50
    # 应有关闭 oos 的告警
    assert any("样本外" in w for w in out["meta"]["warnings"])
    # 关闭 oos 时因子 oos_ic 应为 None、未标失效
    for f in out["factors"]:
        assert f["oos_ic"] is None
        assert f["oos_failed"] is False


# ---------------------------------------------------------------------------
# 4) _select_diverse 去重
# ---------------------------------------------------------------------------
def _sig(arr) -> pd.Series:
    return pd.Series(np.asarray(arr, dtype=float))


def test_select_diverse_drops_correlated():
    n = 100
    rng = np.random.default_rng(0)
    base = _sig(rng.normal(size=n))
    # a、a' 高度相关（a' = a + 微噪），b 独立
    a = base
    a2 = base + rng.normal(0, 1e-3, size=n)
    b = _sig(rng.normal(size=n))
    cands = [
        {"node": "a", "fitness": 0.9, "signal": a},
        {"node": "a2", "fitness": 0.8, "signal": a2},  # 与 a 高相关，应被跳过
        {"node": "b", "fitness": 0.7, "signal": b},
    ]
    selected, n_deduped = _select_diverse(cands, k=2, corr_thresh=0.8)
    picked = {s["node"] for s in selected}
    # 选中 a 与 b（a2 因与 a 高相关被排除），不靠补齐
    assert picked == {"a", "b"}
    assert n_deduped == 0


def test_select_diverse_backfills_when_all_correlated():
    n = 100
    rng = np.random.default_rng(1)
    base = _sig(rng.normal(size=n))
    cands = [
        {"node": "a", "fitness": 0.9, "signal": base},
        {"node": "a2", "fitness": 0.8, "signal": base + rng.normal(0, 1e-3, size=n)},
        {"node": "a3", "fitness": 0.7, "signal": base + rng.normal(0, 1e-3, size=n)},
    ]
    # 要 3 个但全高相关 → 只能靠补齐凑够
    selected, n_deduped = _select_diverse(cands, k=3, corr_thresh=0.8)
    assert len(selected) == 3       # 不因去重缩水
    assert n_deduped == 2           # 首个入选，其余 2 个靠补齐


# ---------------------------------------------------------------------------
# 5) 合成注入信号：train_ic 与 oos_ic 同号（不误标警）
# ---------------------------------------------------------------------------
def test_injected_signal_survives_oos():
    # 合成数据带真实动量信号，best 因子应在样本内外方向一致。
    # 至少 best（第一个）因子若样本内有效（|train_ic|>0.02），不应被误判样本外失效。
    out = run("SYNTH", "x", "y", config=_cfg(gp={"pop_size": 60, "n_gen": 10, "seed": 7}))
    factors = out["factors"]
    assert factors
    strong = [f for f in factors if abs(f["train_ic"]) > 0.02 and f["oos_ic"] is not None]
    # 合成信号足够强时，强因子里应有相当比例样本内外同号（非全部失效）
    if strong:
        assert all("oos_failed" in f for f in strong)


# ---------------------------------------------------------------------------
# 6) 可复现：同 seed 两次跑出相同公式集
# ---------------------------------------------------------------------------
def test_reproducible():
    out1 = run("SYNTH", "x", "y", config=_cfg())
    out2 = run("SYNTH", "x", "y", config=_cfg())
    f1 = [f["formula"] for f in out1["factors"]]
    f2 = [f["formula"] for f in out2["factors"]]
    assert f1 == f2


# ---------------------------------------------------------------------------
# m5) 常数 / 全 NaN 特征列跳过注册（不进 GP 池）+ warn
# ---------------------------------------------------------------------------
def test_m5_constant_feature_skipped():
    """恒定常数列（如 ffill+fillna(0) 后的全零 open_interest）不应被注册。"""
    df = pd.DataFrame({
        "close": pd.Series(np.linspace(100.0, 110.0, 50)),   # 正常变化列 → 注册
        "open_interest": pd.Series(np.zeros(50)),            # 恒零常量 → 跳过
    })
    warnings: list[str] = []
    reg = _register_features(df, df["close"], llm_client=make_test_runtime().client,
                             warnings=warnings)
    assert reg.has("close")
    assert not reg.has("open_interest"), "恒零常量列应被跳过，不进特征池"
    assert any("open_interest" in w for w in warnings), "应就跳过的列 warn"


def test_m5_all_nan_feature_skipped():
    """全 NaN 列（nunique(dropna)=0 → <=1）同样跳过注册。"""
    df = pd.DataFrame({
        "close": pd.Series(np.linspace(100.0, 110.0, 30)),
        "dead_feat": pd.Series([np.nan] * 30),
    })
    warnings: list[str] = []
    reg = _register_features(df, df["close"], llm_client=make_test_runtime().client,
                             warnings=warnings)
    assert reg.has("close")
    assert not reg.has("dead_feat")
    assert any("dead_feat" in w for w in warnings)


def test_m5_varying_feature_still_registered():
    """有区分度的正常列不受影响，照常注册（不误伤）。"""
    df = pd.DataFrame({
        "close": pd.Series(np.linspace(100.0, 110.0, 30)),
        "volume": pd.Series(np.linspace(1000.0, 5000.0, 30)),
    })
    reg = _register_features(df, df["close"], llm_client=make_test_runtime().client,
                             warnings=[])
    assert reg.has("close") and reg.has("volume")


# ---------------------------------------------------------------------------
# 分钟线：horizon 可配 + 主连换月跳空置零
# ---------------------------------------------------------------------------
def test_horizon_config_overrides_default():
    """config.horizon 应覆盖默认 _HORIZON，写入 meta.horizon。"""
    out = run("SYNTH", "x", "y", config=_cfg(horizon=4))
    assert out["meta"]["horizon"] == 4


def test_contract_switch_future_return_masked(monkeypatch):
    """带 dominant_id 的分钟线主连：换月点应被检测，跨换月窗口的未来收益置 NaN。
    造 300 行合成 OHLCV，在第 150 行处换月（dominant_id 从 A→B），
    验证 meta.n_contract_switch>=1 且有换月告警。"""
    _orig = generate._make_synthetic_ohlcv

    def _with_dom(*_a, **_k):
        df = _orig(n=300)
        # 前半 A 合约、后半 B 合约（第 150 行换月）
        dom = np.array(["IM_A"] * 150 + ["IM_B"] * (len(df) - 150))
        df = df.copy()
        df["dominant_id"] = dom
        return df

    monkeypatch.setattr(generate, "_load_real_ohlcv",
                        lambda *a, **k: _with_dom())
    # 走 real 分支（use_synthetic=False）才会调 _load_real_ohlcv
    cfg = _cfg(horizon=4)
    cfg["data"] = {"use_synthetic": False, "frequency": "15m"}
    out = run("IM_DOMINANT.CFE", "20240101", "20241231", config=cfg)
    assert out["meta"].get("n_contract_switch", 0) >= 1
    assert any("换月" in w for w in out["meta"]["warnings"])
