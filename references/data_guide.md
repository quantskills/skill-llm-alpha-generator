# 数据接入与字段口径（G00 LLM-Alpha 因子挖掘）

## 数据来源

- **真实数据**：由 `data_loader.load_ohlcv` 通过 PandaAI `panda_data` SDK 拉取期货 / 股票日线，按 `ts_code` 自动分流。
- **合成数据**：`config.data.use_synthetic=True` 时用 `_make_synthetic_ohlcv` 造一段**带已知动量信号**的 OHLCV，供离线验收（验证 pipeline 能端到端挖到有效因子），**不得作为正式结论依据**。

正式生产时行情必须来自 panda_data 或项目指定数据源。不得使用来源不明、字段不稳定的临时表作为正式输入。

## 环境变量

```bash
set PANDA_DATA_USERNAME=
set PANDA_DATA_PASSWORD=
```

LLM 相关（可选，启用 LLM 时）：

```bash
# 走当前会话默认，无需单独配置
```

数据凭证只属于 `config.data.username` / `config.data.password`；LLM 外部配置只允许放在唯一的 `config.llm` 入口，并且必须同时提供 `base_url`、`api_key`、`model`、`protocol`。

## 品种自动识别（data_loader.classify_asset）

| ts_code 模式 | asset_type | 说明 |
|---|---|---|
| `\d{6}\.(SZ\|SH\|BJ)` | `stock_a` | A 股，如 `000001.SZ` / `600000.SH` |
| `[A-Za-z]{1,3}\d{3,4}\.[A-Za-z]{3,4}` | `future_cn` | 中国期货合约，如 `RB2405.SHF` / `IF2406.CFE` |
| 其他 | `unknown` | 无法识别，返回空 DataFrame + warning |

`config.data.asset_type` 默认 `auto`（用 classify_asset 判定），也可显式传 `future_cn` / `stock_a`。

## 统一入口

```python
load_ohlcv(ts_code, start_date, end_date,
           asset_type="auto", username=None, password=None) -> pd.DataFrame
```

- `future_cn` → `load_future_daily` → `panda_data.get_future_daily`
- `stock_a`   → `load_stock_daily`  → `panda_data.get_stock_daily`
- 另有 `load_future_dominant`（品种→每日主力合约映射，附带能力）

日期格式统一 `YYYYMMDD`。

## 标准化输出（STANDARD_COLUMNS）

无论期货还是股票，`load_ohlcv` 一律标准化为 **date 索引（DatetimeIndex，升序）+ 以下 7 列**，缺失列填 NaN、转数值类型：

| 列 | 说明 |
|---|---|
| `open` / `high` / `low` / `close` | OHLC |
| `volume` | 成交量 |
| `amount` | 成交额（缺则下游用 volume·close 兜底） |
| `open_interest` | 持仓量（期货） |

## 特征计算口径（generate._compute_features）

基于标准 OHLCV 算约 20 个日频特征，全部「回看」无前视，注册进 `FeatureRegistry`：

| 特征 | 量纲 | 说明 |
|---|---|---|
| `open`/`high`/`low`/`close` | price | 原始价格 |
| `volume`/`amount` | volume | 量 / 额 |
| `open_interest` | oi | 持仓量（期货有则含） |
| `ret`/`ret5`/`log_ret` | return | 单期 / 5 期 / 对数收益 |
| `ma5`/`ma10`/`ma20` | price | 均线 |
| `mom5`/`mom20` | ratio | 相对 5/20 日均线偏离 |
| `vol10` | ratio | 收益 10 日滚动波动 |
| `atr14` | price | 14 期真实波幅均值 |
| `rsi14` | ratio | 14 期 RSI |
| `vol_ratio` | ratio | 量比（当前量 / 20 日均量） |
| `hl_range` | ratio | 相对振幅 (high-low)/close |
| `clv` | ratio | 收盘位置 [-1,1] |

残余 NaN/Inf 用前值 / 0 兜底，保证下游求值稳定。

## 量纲自动推断（dimension_inferrer）

内置量价特征直接按 `_FEATURE_DIMS` 贴量纲；**陌生特征**（不在映射里）过 `infer_and_register`：

- 三票投票：`stat_vote`（统计画像）+ `name_vote`（命名语义）+ `llm_vote`（LLM，可选）
- 权重 llm 1.2 > stat 1.0 > name 0.6
- confidence < 0.5 → 强制 `unknown` 并告警
- 这是「不写死特征白名单」的可扩展性核心：任何新数据列都能动态注册后被表达式引擎使用

## 未来收益标签（fitness.compute_future_return）

因子预测力对齐的标签是**未来 HORIZON 期收益**（默认 HORIZON=1）：

```
future_return = close.shift(-horizon) / close - 1
```

站在 t 时刻看未来 horizon 期后的涨跌幅；末尾 horizon 个位置为 NaN，由 fitness / alpha_eval 剔除后再算 IC。

## 降级与容错

- panda_data 未装 / 凭证缺失 / 登录失败 / 接口异常 → 返回空 DataFrame + warning（**不抛异常**），由 `run()` 记入 `meta.warnings`，无数据时流程终止（返回空 factors + 报告）。
- 登录只做一次（模块级缓存），避免重复握手。

## 正式接入提醒

- 本 skill 单标的时序：`universe` 传列表时**只取第一个**标的。多标的请分别跑 `run()`。
- 只做样本内评估，**不做回测**；严格样本外裁决 / 费率 / 滑点 / 资金费率归回测 skill。
