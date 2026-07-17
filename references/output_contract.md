# Output Contract（G00 LLM-Alpha 因子挖掘）

## run() 返回结构

```python
run(universe, start_date, end_date, config=None) -> dict
```

| 顶层键 | 类型 | 必有 | 说明 |
|---|---|---|---|
| `factors` | list[dict] | yes | 入选因子（≤ top_k）；无数据 / 无产出时为 `[]` |
| `trajectory` | list[dict] | yes | GP 每代演化轨迹 |
| `rejected` | list[dict] | yes | 被解析 / 校验拒绝的候选 |
| `report_path` | str \| None | yes | 自包含 HTML 报告路径；build 失败为 None |
| `meta` | dict | yes | 运行元信息 |

## factor JSON schema（factors 元素）

| 字段 | 类型 | 说明 |
|---|---|---|
| `formula` | str | 因子公式（前缀函数式，如 `sub(ts_mean(close,5), ts_mean(close,20))`） |
| `explanation` | str | 整体经济解释（LLM，降级为占位串） |
| `captures` | str | 一句话概括捕捉的核心信号（LLM，可空串） |
| `applicable_scenario` | str | 适用场景（LLM，可空串） |
| `failure_scenario` | str | 失效场景（LLM，可空串） |
| `alpha_scores` | dict | 五维分（见下） |
| `alpha_eval` | dict | AlphaEval 完整五维明细（含 `detail`，见 fitness_guide） |
| `rankic` | float | Spearman rankIC（带符号） |
| `fitness` | float | GP 适应度 |
| `node_count` | int | 表达式节点数 |
| `signal_stats` | dict | 信号统计（见下） |

### `alpha_scores`

| 键 | 来源 | 说明 |
|---|---|---|
| `effectiveness` | pps | 预测能力 [0,1] |
| `robustness` | rre | 扰动鲁棒性 [0,1] |
| `interpretability` | logic | 金融逻辑 [0,1]（无 LLM 为 0.5） |
| `diversity` | diversity | 多样性 [0,1]（无 others 为 0.5） |
| `parsimony` | pfs | 时序稳定性 [0,1] |
| `weighted_score` | — | `0.4·pps+0.2·pfs+0.2·rre+0.15·logic+0.05·diversity` |

> 注：`alpha_scores` 是给报告的**易读别名层**（effectiveness/robustness/interpretability/diversity/parsimony），原始五维键（pps/pfs/rre/logic/diversity）在 `alpha_eval` 里。

### `signal_stats`

| 键 | 说明 |
|---|---|
| `mean` | 信号均值 |
| `std` | 信号标准差 |
| `turnover` | 换手率（来自 alpha_eval.detail.pfs） |
| `coverage` | 非 NaN 覆盖率 |

## rejected JSON schema（rejected 元素）

| 字段 | 类型 | 说明 |
|---|---|---|
| `formula` | str | 被拒公式字符串 |
| `reason` | str | 拒绝原因 |
| `layer` | str | 被哪层拦下：`解析` / `语法` / `注册表` / `量纲` / `前视` |
| `source` | str | 来源，如 `llm` |

## trajectory JSON schema（trajectory 元素）

GP 每代一条记录，由 `GPEngine.run()` 产出，典型含代号、当代最优 / 均值适应度、最优公式等（详见 `gp_engine.py`）。

## meta 关键字段

| 字段 | 说明 |
|---|---|
| `build_id` / `build_name` | `G00` / `LLM-Alpha 因子挖掘` |
| `universe` / `start_date` / `end_date` | 输入回显 |
| `data_source` | `synthetic` / `real` |
| `n_rows` / `n_features` | 数据行数 / 特征数 |
| `llm_enabled` / `llm_available` | LLM 请求 / 实际可用 |
| `llm_generate` | `{called, skipped_reason, n_formulas, latency_ms, model}` |
| `n_warm_start` | 通过校验的 warm-start 种子数 |
| `n_factors` | 入选因子数 |
| `warnings` | 警告列表 |

## Acceptance

- `run()` 返回 dict，含 `factors` / `trajectory` / `rejected` / `report_path` / `meta` 五键。
- 每个 factor 的 `formula` 可被 `parse_formula` 重新解析成合法表达式树。
- `alpha_scores.weighted_score` ∈ [0,1]；各维分 ∈ [0,1]。
- `rejected` 每条含 `layer`，能定位被哪层校验拦下。
- LLM 关闭时：`factors` 仍产出（纯 GP），`explanation` 为占位串，`interpretability`=0.5。
- 无可用数据时：`factors=[]`，`report_path` 仍尝试生成（空报告），`meta.warnings` 说明原因。
