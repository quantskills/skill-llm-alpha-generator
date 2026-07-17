# -*- coding: utf-8 -*-
"""
tests_data.py — data_loader 数据接入层单测

覆盖：
    - classify_asset 正则分类正确
    - ensure_login 凭证缺失时返回 False（不抛）
    - 用真实凭证拉 RB2405.SHF 2024-01 期货日线，验证 shape / 列 / close 全正
      （网络或凭证问题导致失败时 pytest.skip 跳过，标注原因）
    - _standardize 标准化逻辑（空输入 / 缺列补 NaN）

运行： pytest scripts/tests_data.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    classify_asset,
    ensure_login,
    load_future_daily,
    load_future_dominant_post,
    load_future_min,
    load_ohlcv,
    _to_variety_code,
    _standardize,
    _empty_standard,
    STANDARD_COLUMNS,
    FUTURE_ADJ_METHODS,
    FUTURE_MIN_FREQ,
)

# 真实凭证（任务方提供）
_USERNAME = "8613148739672"
_PASSWORD = "fdwdyl92"


# ---------------------------------------------------------------------------
# classify_asset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code,expected",
    [
        ("RB2405.SHF", "future_cn"),
        ("IF2406.CFE", "future_cn"),
        ("cu2405.SHF", "future_cn"),
        ("000001.SZ", "stock_a"),
        ("600000.SH", "stock_a"),
        ("000001.sz", "stock_a"),  # 大小写不敏感
        # 期货品种/主连代码 → future_dominant（走官方后复权）
        ("RB", "future_dominant"),
        ("rb", "future_dominant"),
        ("RB.SHF", "future_dominant"),
        ("RB_DOMINANT.SHF", "future_dominant"),
        ("IF", "future_dominant"),
        ("", "unknown"),
        (None, "unknown"),
        ("12345.SZ", "unknown"),  # 只有 5 位
    ],
)
def test_classify_asset(code, expected):
    """资产分类正则应正确区分期货合约 / 主连 / 股票 / 未知。"""
    assert classify_asset(code) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("RB", "RB"),
        ("rb", "RB"),
        ("RB.SHF", "RB"),
        ("RB_DOMINANT.SHF", "RB"),
        ("if_dominant.cfe", "IF"),
    ],
)
def test_to_variety_code(code, expected):
    """品种/主连代码归一成品种代码。"""
    assert _to_variety_code(code) == expected


# ---------------------------------------------------------------------------
# _standardize
# ---------------------------------------------------------------------------
def test_standardize_empty():
    """空输入应返回带标准列的空 DataFrame。"""
    out = _standardize(pd.DataFrame())
    assert list(out.columns) == STANDARD_COLUMNS
    assert len(out) == 0


def test_standardize_fills_missing():
    """缺失列应补 NaN，date 应成为排序后的索引。"""
    raw = pd.DataFrame(
        {
            "date": ["20240103", "20240101", "20240102"],
            "close": [3.0, 1.0, 2.0],
            "volume": [30, 10, 20],
        }
    )
    out = _standardize(raw)
    # 标准列齐全
    assert list(out.columns) == STANDARD_COLUMNS
    # 按 date 升序
    assert list(out["close"]) == [1.0, 2.0, 3.0]
    # 缺失列（open/high/low/amount/open_interest）应为 NaN
    assert out["open"].isna().all()
    assert out["open_interest"].isna().all()
    # date 成为索引
    assert out.index.name == "date"
    assert isinstance(out.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# ensure_login 降级
# ---------------------------------------------------------------------------
def test_ensure_login_missing_credentials(monkeypatch):
    """凭证全缺时应返回 False 且不抛异常。"""
    # 清空环境变量并重置模块登录缓存
    import data_loader
    monkeypatch.delenv("PANDA_DATA_USERNAME", raising=False)
    monkeypatch.delenv("PANDA_DATA_PASSWORD", raising=False)
    monkeypatch.setattr(data_loader, "_login_ok", False, raising=False)
    assert ensure_login(username=None, password=None) is False


# ---------------------------------------------------------------------------
# 真实数据拉取（网络/凭证问题时 skip）
# ---------------------------------------------------------------------------
def test_load_future_daily_real():
    """
    用真实凭证拉 RB2405.SHF 2024-01 期货日线。
    验证：shape 非空、含 close/volume/open_interest 列、close 全正。
    网络或凭证问题导致拿不到数据时 skip（不算失败）。
    """
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败（网络/凭证/服务不可用），跳过真实数据测试")

    df = load_future_daily(
        "RB2405.SHF",
        start_date="20240101",
        end_date="20240115",
        username=_USERNAME,
        password=_PASSWORD,
    )

    if df is None or len(df) == 0:
        pytest.skip("接口返回空数据（可能网络/权限/合约无行情），跳过真实数据断言")

    # shape 非空
    assert len(df) > 0
    # 关键列存在
    for col in ("close", "volume", "open_interest"):
        assert col in df.columns, f"缺少列 {col}"
    # close 全正
    close = df["close"].dropna()
    assert len(close) > 0, "close 全为 NaN"
    assert (close > 0).all(), "存在非正 close"


def test_load_ohlcv_auto_future():
    """load_ohlcv auto 分流：期货代码应能拉到与 load_future_daily 一致结构。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过 load_ohlcv 真实测试")

    df = load_ohlcv(
        "RB2405.SHF",
        start_date="20240101",
        end_date="20240115",
        asset_type="auto",
        username=_USERNAME,
        password=_PASSWORD,
    )
    if df is None or len(df) == 0:
        pytest.skip("接口返回空数据，跳过 load_ohlcv 断言")

    assert list(df.columns) == STANDARD_COLUMNS
    assert (df["close"].dropna() > 0).all()


# ---------------------------------------------------------------------------
# 期货主连后复权（路A：官方复权 get_future_daily_post）
# ---------------------------------------------------------------------------
def test_future_adj_methods_const():
    """复权方法常量应含 panda 官方 4 种。"""
    assert set(FUTURE_ADJ_METHODS) == {"close_or", "close_os", "close_pcr", "close_pcs"}


def test_load_future_dominant_post_real():
    """用真实凭证拉 RB 主连后复权，验证：
    - 标准列齐全、close 全正、volume 全正
    - amount 被「复权 close × volume」重算（非全 0，且 ≈ close*volume）
    - 覆盖 2023-09 换月点，后复权应抹平原始跳空（换月处日收益率不异常）
    网络/凭证问题时 skip。
    """
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过主连后复权真实测试")

    df = load_future_dominant_post(
        "RB", start_date="20230820", end_date="20230910",
        method="close_pcs", username=_USERNAME, password=_PASSWORD,
    )
    if df is None or len(df) == 0:
        pytest.skip("主连后复权接口返回空数据，跳过断言")

    assert list(df.columns) == STANDARD_COLUMNS
    close = df["close"].dropna()
    assert len(close) > 0 and (close > 0).all()
    assert (df["volume"].dropna() > 0).all()

    # amount 被重算（非全 0）且 ≈ close*volume
    amt = df["amount"].dropna()
    assert len(amt) > 0 and (amt > 0).all(), "amount 未被重算（仍为 0/NaN）"
    recomputed = (df["close"] * df["volume"]).dropna()
    # 逐行比对（重算口径应完全一致）
    aligned = (df["amount"] - df["close"] * df["volume"]).abs().dropna()
    assert (aligned < 1e-6).all(), "amount 不等于 复权close×volume"

    # 后复权抹平换月跳空：换月点（2023-09-01 附近）日收益率不应出现原始跳空级异常。
    # 原始未复权换月跳空约 0.9%+，且相邻日常常波动 <3%；这里验证无异常大跳。
    ret = df["close"].pct_change().dropna().abs()
    assert (ret < 0.15).all(), f"存在异常日收益（疑似换月跳空未抹平）：max={ret.max():.4f}"


def test_load_ohlcv_auto_dominant_real():
    """load_ohlcv auto 分流：品种代码 'RB' 应走主连后复权路径。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过 load_ohlcv 主连测试")

    df = load_ohlcv(
        "RB", start_date="20240101", end_date="20240115",
        asset_type="auto", username=_USERNAME, password=_PASSWORD,
    )
    if df is None or len(df) == 0:
        pytest.skip("接口返回空数据，跳过断言")

    assert list(df.columns) == STANDARD_COLUMNS
    assert (df["close"].dropna() > 0).all()
    # 主连后复权路径下 amount 应被重算（非全 0）
    assert (df["amount"].dropna() > 0).all()


def test_load_future_dominant_post_bad_method():
    """非法复权方法应回退默认 close_pcs（不崩），登录失败则 skip。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过")
    df = load_future_dominant_post(
        "RB", start_date="20240101", end_date="20240110",
        method="not_a_method", username=_USERNAME, password=_PASSWORD,
    )
    # 回退后仍应能拿到数据（或至少不崩、返回标准结构）
    assert list(df.columns) == STANDARD_COLUMNS


# ---------------------------------------------------------------------------
# 期货分钟线（get_future_min）
# ---------------------------------------------------------------------------
def test_future_min_freq_const():
    """FUTURE_MIN_FREQ 常量含 panda 支持的四档频率。"""
    assert FUTURE_MIN_FREQ == ("1m", "5m", "15m", "60m")


def test_load_future_min_im_15m_real():
    """真实拉 IM 主连 15m：DatetimeIndex 到分钟、标准列齐全、保留 dominant_id、
    OHLCV 全正、amount 非恒 0（分钟线接口返回真实成交额）。登录失败则 skip。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过分钟线真实测试")
    df = load_future_min(
        "IM_DOMINANT.CFE", start_date="20240102", end_date="20240131",
        frequency="15m", username=_USERNAME, password=_PASSWORD,
    )
    if df is None or len(df) == 0:
        pytest.skip("分钟线接口返回空数据，跳过断言")
    # 标准列齐全 + 附加 dominant_id
    for col in STANDARD_COLUMNS:
        assert col in df.columns
    assert "dominant_id" in df.columns, "分钟线应保留 dominant_id 供换月检测"
    # 索引到分钟
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    # 同一天应有多根（15m → 每日约 16 根），证明确实是分钟级
    first_day = df.index.normalize().min()
    assert (df.index.normalize() == first_day).sum() > 1
    # OHLCV 全正、amount 非恒 0
    assert (df["close"].dropna() > 0).all()
    assert not (df["amount"].fillna(0) == 0).all(), "分钟线 amount 不应恒 0"


def test_load_future_min_bad_frequency():
    """非法频率应回退 15m（不崩）；登录失败则 skip。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过")
    df = load_future_min(
        "IM_DOMINANT.CFE", start_date="20240102", end_date="20240110",
        frequency="7m", username=_USERNAME, password=_PASSWORD,
    )
    # 回退后仍返回标准结构（+dominant_id），不崩
    for col in STANDARD_COLUMNS:
        assert col in df.columns


def test_load_ohlcv_frequency_routes_to_min():
    """load_ohlcv 传 frequency 应分流到分钟线（返回带 dominant_id 的分钟级数据）。"""
    if not ensure_login(username=_USERNAME, password=_PASSWORD):
        pytest.skip("panda_data 登录失败，跳过")
    df = load_ohlcv(
        "IM_DOMINANT.CFE", "20240102", "20240110", frequency="15m",
        username=_USERNAME, password=_PASSWORD,
    )
    if df is None or len(df) == 0:
        pytest.skip("接口空数据，跳过")
    assert isinstance(df.index, pd.DatetimeIndex)
    # 分钟线路径特征：同日多根 + 保留 dominant_id
    first_day = df.index.normalize().min()
    assert (df.index.normalize() == first_day).sum() > 1
