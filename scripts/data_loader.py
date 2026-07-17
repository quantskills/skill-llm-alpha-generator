# -*- coding: utf-8 -*-
"""
data_loader.py — panda_data 数据接入层

职责：
    - 惰性登录 panda_data（凭证从参数或环境变量读取，缺失时优雅降级）
    - 拉取期货 / 股票日线，标准化为统一列名的 pd.DataFrame（date 为索引）
    - 提供统一入口 load_ohlcv，根据 ts_code 自动分流期货 / 股票
    - 提供 classify_asset 做资产类型判定

设计原则：
    - 接口失败不抛异常，返回空 DataFrame + warning，由 caller 决定降级策略
    - 登录只做一次（模块级缓存），避免重复握手

依赖：panda_data（第三方数据服务），pandas，numpy
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

try:
    import panda_data
except ImportError:  # pragma: no cover - 环境缺失时降级
    panda_data = None


# ---------------------------------------------------------------------------
# 常量：标准化列名
# ---------------------------------------------------------------------------
# 统一对外输出的列（无论期货还是股票，缺的列一律填 NaN）
STANDARD_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "open_interest"]

# 环境变量键名
ENV_USERNAME = "PANDA_DATA_USERNAME"
ENV_PASSWORD = "PANDA_DATA_PASSWORD"

# 资产类型正则
# 期货具体合约（中国）：字母品种 + 数字合约月 + . + 交易所，如 RB2405.SHF / IF2406.CFE
_RE_FUTURE_CN = re.compile(r"^[A-Za-z]{1,3}\d{3,4}\.[A-Za-z]{3,4}$")
# 期货主连符号：{品种}_DOMINANT.{交易所}，如 RB_DOMINANT.SHF
_RE_FUTURE_DOMINANT = re.compile(r"^[A-Za-z]{1,3}_DOMINANT\.[A-Za-z]{3,4}$", re.IGNORECASE)
# 期货品种代码：纯字母品种（1-3 位），可选带交易所后缀，如 RB / rb / RB.SHF
#   （不含数字合约月，用于触发主连后复权路径 get_future_daily_post）
_RE_FUTURE_VARIETY = re.compile(r"^[A-Za-z]{1,3}(\.[A-Za-z]{3,4})?$")
# A 股：6 位数字 + . + SZ/SH/BJ，如 000001.SZ / 600000.SH
_RE_STOCK_A = re.compile(r"^\d{6}\.(SZ|SH|BJ)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 模块级登录状态缓存（避免重复登录）
# ---------------------------------------------------------------------------
_login_ok = False  # 是否已成功登录


# ---------------------------------------------------------------------------
# 资产分类
# ---------------------------------------------------------------------------
def classify_asset(ts_code: str) -> str:
    """
    根据代码字符串判断资产类型。

    参数:
        ts_code: 证券代码，如 'RB2405.SHF' / 'RB' / 'RB_DOMINANT.SHF' / '000001.SZ'

    返回:
        'future_cn'        —— 中国期货具体合约（如 RB2405.SHF）
        'future_dominant'  —— 期货品种/主连代码（如 RB、RB.SHF、RB_DOMINANT.SHF），
                              走官方主连后复权 get_future_daily_post
        'stock_a'          —— A 股
        'unknown'          —— 无法识别
    """
    if not isinstance(ts_code, str) or not ts_code.strip():
        return "unknown"
    code = ts_code.strip()
    # A 股 / 具体合约先判（更具体），避免被品种代码正则误吞
    if _RE_STOCK_A.match(code):
        return "stock_a"
    if _RE_FUTURE_CN.match(code):
        return "future_cn"
    # 主连符号 或 纯字母品种代码 → 主连后复权路径
    if _RE_FUTURE_DOMINANT.match(code) or _RE_FUTURE_VARIETY.match(code):
        return "future_dominant"
    return "unknown"


def _to_variety_code(ts_code: str) -> str:
    """把品种/主连代码归一成 get_future_daily_post 需要的 underlying_symbol（品种代码）。

    'RB' -> 'RB'；'rb' -> 'RB'；'RB.SHF' -> 'RB'；'RB_DOMINANT.SHF' -> 'RB'。
    """
    code = str(ts_code).strip().upper()
    # 去掉主连后缀 _DOMINANT.XXX
    code = re.sub(r"_DOMINANT\.[A-Z]{3,4}$", "", code)
    # 去掉交易所后缀 .XXX
    code = re.sub(r"\.[A-Z]{3,4}$", "", code)
    return code


# ---------------------------------------------------------------------------
# 惰性登录
# ---------------------------------------------------------------------------
def ensure_login(username: str | None = None, password: str | None = None) -> bool:
    """
    惰性登录 panda_data。

    凭证优先级：显式参数 > 环境变量 PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD。
    两处都拿不到凭证时返回 False（不抛异常，由 caller 决定降级）。
    登录成功后缓存状态，后续调用直接返回 True，不重复握手。

    参数:
        username: 显式用户名（可选）
        password: 显式密码（可选）

    返回:
        bool —— 是否已成功登录
    """
    global _login_ok

    if _login_ok:
        return True

    if panda_data is None:
        warnings.warn("panda_data 未安装，无法登录，caller 请降级处理")
        return False

    # 凭证：参数优先，其次环境变量
    user = username or os.environ.get(ENV_USERNAME)
    pwd = password or os.environ.get(ENV_PASSWORD)

    if not user or not pwd:
        warnings.warn(
            f"panda_data 凭证缺失（参数未传且环境变量 {ENV_USERNAME}/{ENV_PASSWORD} 未设置），"
            "登录跳过，caller 请降级处理"
        )
        return False

    try:
        # 部分版本 init_token 读环境变量，这里同时写入 env 以兼容两种范式
        os.environ[ENV_USERNAME] = user
        os.environ[ENV_PASSWORD] = pwd
        panda_data.init_token(username=user, password=pwd)
        _login_ok = True
        return True
    except Exception as e:  # 登录失败不抛，降级
        warnings.warn(f"panda_data 登录失败：{e!r}，caller 请降级处理")
        return False


# ---------------------------------------------------------------------------
# 内部工具：标准化输出
# ---------------------------------------------------------------------------
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始日线 DataFrame 标准化：
        - 按 date 升序排序，date 转为索引（DatetimeIndex）
        - 保证 STANDARD_COLUMNS 全部存在，缺失列填 NaN
        - 只保留标准列（其余原始列丢弃，避免下游列名不一致）

    传入空 DataFrame 时返回带标准列的空 DataFrame。
    """
    if df is None or len(df) == 0:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        empty.index.name = "date"
        return empty

    df = df.copy()

    # date 处理：转 datetime，排序，设为索引
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
        df = df.sort_values("date").set_index("date")
    else:
        # 没有 date 列时，尽量用已有索引，仅告警
        warnings.warn("原始数据缺少 date 列，无法按日期排序/索引")

    # 补齐标准列
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 仅保留标准列，并转数值类型
    out = df[STANDARD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out.index.name = "date"
    return out


def _empty_standard() -> pd.DataFrame:
    """返回一个带标准列的空 DataFrame。"""
    empty = pd.DataFrame(columns=STANDARD_COLUMNS)
    empty.index.name = "date"
    return empty


# ---------------------------------------------------------------------------
# 期货日线
# ---------------------------------------------------------------------------
def load_future_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """
    拉取期货日线 OHLCV。

    参数:
        symbol: 期货合约代码，如 'RB2405.SHF'
        start_date / end_date: 'YYYYMMDD' 格式
        username / password: 可选，透传给 ensure_login

    返回:
        标准化 DataFrame（date 索引 + STANDARD_COLUMNS）。
        登录失败 / 接口异常 / 无数据时返回空 DataFrame（不抛）。
    """
    if not ensure_login(username, password):
        return _empty_standard()

    try:
        df = panda_data.get_future_daily(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
    except Exception as e:
        warnings.warn(f"get_future_daily 失败（symbol={symbol}）：{e!r}")
        return _empty_standard()

    if df is None or len(df) == 0:
        warnings.warn(f"get_future_daily 返回空数据（symbol={symbol}）")
        return _empty_standard()

    return _standardize(df)


# ---------------------------------------------------------------------------
# 期货主连后复权（路A：全程用 panda 官方复权 get_future_daily_post，绝不自建复权）
# ---------------------------------------------------------------------------
# 合法复权方法（对齐 panda get_future_daily_post 的 method 取值）
FUTURE_ADJ_METHODS = ("close_or", "close_os", "close_pcr", "close_pcs")


def load_future_dominant_post(
    variety: str,
    start_date: str,
    end_date: str,
    method: str = "close_pcs",
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """拉取期货主连**后复权**日线（官方复权，绝不自建）。

    走 panda 官方 `get_future_daily_post(underlying_symbol=品种, method=...)`——
    复权过程完全由 panda 完成，本函数不做任何价格拼接/复权计算。

    已知接口特性（2026-07 实测）：
        - 参数是 underlying_symbol（品种代码，如 'RB'），不是具体合约；
        - 返回主连级别复权 OHLC + volume + open_interest + dominant_id；
        - **amount（成交额）恒为 0**（该接口只复权价格、不返回成交额）——
          本函数用「复权后 close × volume」重算 amount 补上（量纲自洽：
          复权价 × 成交量 = 复权成交额；价格复权仍由 panda 完成，未自建复权）；
        - underlying_symbol 过滤不严，可能返回全品种，需按 variety 再筛。

    参数:
        variety:    品种代码 'RB'（也接受 'RB.SHF' / 'RB_DOMINANT.SHF'，内部归一）
        start_date / end_date: 'YYYYMMDD'
        method:     复权方法，默认 close_pcs（比例后复权）；须 ∈ FUTURE_ADJ_METHODS
        username / password: 可选，透传给 ensure_login

    返回:
        标准化 DataFrame（date 索引 + STANDARD_COLUMNS）。
        登录失败 / 接口异常 / 无数据时返回空 DataFrame（不抛）。
    """
    if method not in FUTURE_ADJ_METHODS:
        warnings.warn(
            f"复权方法 {method!r} 非法，回退默认 close_pcs（合法: {FUTURE_ADJ_METHODS}）")
        method = "close_pcs"

    if not ensure_login(username, password):
        return _empty_standard()

    underlying = _to_variety_code(variety)
    try:
        df = panda_data.get_future_daily_post(
            underlying_symbol=[underlying],
            start_date=start_date, end_date=end_date, method=method,
        )
    except Exception as e:
        warnings.warn(f"get_future_daily_post 失败（variety={underlying}）：{e!r}")
        return _empty_standard()

    if df is None or len(df) == 0:
        warnings.warn(f"get_future_daily_post 返回空数据（variety={underlying}）")
        return _empty_standard()

    # 接口过滤不严：按品种再筛一次
    if "underlying_symbol" in df.columns:
        df = df[df["underlying_symbol"].astype(str).str.upper() == underlying]
    if len(df) == 0:
        warnings.warn(f"get_future_daily_post 无该品种数据（variety={underlying}）")
        return _empty_standard()

    df = df.copy()
    # amount 恒 0 → 用「复权 close × volume」重算（价格复权由 panda 完成，此处仅派生成交额）
    if "close" in df.columns and "volume" in df.columns:
        close_num = pd.to_numeric(df["close"], errors="coerce")
        vol_num = pd.to_numeric(df["volume"], errors="coerce")
        recomputed = close_num * vol_num
        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce")
            # 仅在 amount 缺失/为 0 处用重算值替换（保留接口可能给出的真实值）
            df["amount"] = amt.where((amt.notna()) & (amt != 0), recomputed)
        else:
            df["amount"] = recomputed

    return _standardize(df)


# ---------------------------------------------------------------------------
# 期货分钟线（主连 / 具体合约）
# ---------------------------------------------------------------------------
# 分钟线支持的频率（panda get_future_min）
FUTURE_MIN_FREQ = ("1m", "5m", "15m", "60m")


def load_future_min(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str = "15m",
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """拉取期货分钟线（走 panda 官方 `get_future_min`）。

    与日线不同点：
        - 走 `get_future_min(symbol=..., frequency=...)`，symbol 是**具体合约或主连**
          （如 'IM_DOMINANT.CFE' 主连 / 'IM2506.CFE' 具体合约）；分钟线接口**无官方复权版**，
          主连为未复权、换月有跳空（由上层用 dominant_id 抹掉换月点收益）。
        - 索引用 `datetime`（精确到分钟）而非 date；接口 amount **非恒 0**（真实成交额）。
        - 额外保留 `dominant_id` 列（供上层检测主连换月）——不进 STANDARD_COLUMNS，
          作为附加列挂在返回 DataFrame 上。

    参数:
        symbol:     合约/主连代码，如 'IM_DOMINANT.CFE'
        start_date / end_date: 'YYYYMMDD'
        frequency:  ∈ FUTURE_MIN_FREQ，默认 '15m'
        username / password: 可选，透传 ensure_login

    返回:
        标准化 DataFrame（datetime 索引 + STANDARD_COLUMNS + 附加 dominant_id 列）。
        登录失败 / 接口异常 / 无数据时返回空 DataFrame（不抛）。
    """
    if frequency not in FUTURE_MIN_FREQ:
        warnings.warn(
            f"分钟线频率 {frequency!r} 非法，回退 15m（合法: {FUTURE_MIN_FREQ}）")
        frequency = "15m"

    if not ensure_login(username, password):
        return _empty_standard()

    try:
        df = panda_data.get_future_min(
            symbol=symbol, start_date=start_date, end_date=end_date,
            frequency=frequency,
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"get_future_min 失败（symbol={symbol}）：{e!r}")
        return _empty_standard()

    if df is None or len(df) == 0:
        warnings.warn(f"get_future_min 返回空数据（symbol={symbol}）")
        return _empty_standard()

    df = df.copy()
    # 保留 dominant_id（换月检测用）；接口返回的 datetime 转索引并按时间升序
    dom_id = df["dominant_id"] if "dominant_id" in df.columns else None
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"].astype(str), errors="coerce")
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str), errors="coerce")
    else:
        warnings.warn("分钟线数据缺 datetime/date 列，无法建立时间索引")
        return _empty_standard()

    df = df.assign(_dt=dt)
    if dom_id is not None:
        df = df.assign(dominant_id=dom_id.values)
    df = df.sort_values("_dt")

    # 补齐标准列并转数值
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[STANDARD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(df["_dt"].values, name="date")
    # 附加 dominant_id（非标准列，供换月检测；缺失则不加）
    if "dominant_id" in df.columns:
        out["dominant_id"] = df["dominant_id"].values
    return out


# ---------------------------------------------------------------------------
# 股票日线
# ---------------------------------------------------------------------------
def load_stock_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """
    拉取 A 股日线 OHLCV。

    参数:
        symbol: 股票代码，如 '000001.SZ'
        start_date / end_date: 'YYYYMMDD' 格式
        username / password: 可选，透传给 ensure_login

    返回:
        标准化 DataFrame（date 索引 + STANDARD_COLUMNS）。
        登录失败 / 接口异常 / 无数据时返回空 DataFrame（不抛）。
    """
    if not ensure_login(username, password):
        return _empty_standard()

    try:
        df = panda_data.get_stock_daily(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
    except Exception as e:
        warnings.warn(f"get_stock_daily 失败（symbol={symbol}）：{e!r}")
        return _empty_standard()

    if df is None or len(df) == 0:
        warnings.warn(f"get_stock_daily 返回空数据（symbol={symbol}）")
        return _empty_standard()

    return _standardize(df)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def load_ohlcv(
    ts_code: str,
    start_date: str,
    end_date: str,
    asset_type: str = "auto",
    username: str | None = None,
    password: str | None = None,
    adj_method: str = "close_pcs",
    frequency: str | None = None,
) -> pd.DataFrame:
    """
    统一 OHLCV 入口，根据 ts_code 自动分流期货具体合约 / 期货主连后复权 / 股票。

    参数:
        ts_code: 证券代码：
                 'RB2405.SHF' → 期货具体合约（未复权，具体合约无需复权）
                 'RB' / 'RB.SHF' / 'RB_DOMINANT.SHF' → 期货主连后复权（官方复权）
                 '000001.SZ' → A 股
        start_date / end_date: 'YYYYMMDD'
        asset_type: 'auto' | 'future_cn' | 'future_dominant' | 'stock_a'
                    'auto' 时用 classify_asset 判定
        username / password: 可选，透传给底层 loader
        adj_method: 主连后复权方法（仅 future_dominant 用），默认 close_pcs
        frequency:  非空时走**分钟线**（get_future_min，∈ FUTURE_MIN_FREQ，如 '15m'）——
                    此时 ts_code 直接作 symbol 传入（主连或具体合约均可），
                    不做日线分流；None（默认）走日线路径。

    返回:
        标准化 DataFrame。无法识别 / 接口异常时返回空 DataFrame（不抛）。
    """
    # 分钟线优先分流（frequency 非空即走分钟线，ts_code 直接作 symbol）
    if frequency:
        return load_future_min(
            ts_code, start_date, end_date, frequency=frequency,
            username=username, password=password)

    kind = asset_type
    if kind == "auto":
        kind = classify_asset(ts_code)

    if kind == "future_cn":
        return load_future_daily(ts_code, start_date, end_date, username, password)
    if kind == "future_dominant":
        return load_future_dominant_post(
            ts_code, start_date, end_date, method=adj_method,
            username=username, password=password)
    if kind == "stock_a":
        return load_stock_daily(ts_code, start_date, end_date, username, password)

    warnings.warn(f"无法识别资产类型（ts_code={ts_code}, asset_type={asset_type}），返回空 DataFrame")
    return _empty_standard()


# ---------------------------------------------------------------------------
# 主力合约映射（附带能力，便于回测选主力）
# ---------------------------------------------------------------------------
def load_future_dominant(
    underlying_symbol: str,
    start_date: str,
    end_date: str,
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """
    拉取期货主力合约映射（品种 -> 每日主力合约）。

    参数:
        underlying_symbol: 品种代码，如 'RB'
        start_date / end_date: 'YYYYMMDD'

    返回:
        原始映射 DataFrame（不做列标准化，因其非 OHLCV）。
        登录失败 / 异常时返回空 DataFrame（不抛）。
    """
    if not ensure_login(username, password):
        return pd.DataFrame()

    try:
        df = panda_data.get_future_dominant(
            underlying_symbol=underlying_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        warnings.warn(f"get_future_dominant 失败（{underlying_symbol}）：{e!r}")
        return pd.DataFrame()

    return df if df is not None else pd.DataFrame()
