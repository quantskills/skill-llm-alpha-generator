# LLM 调用边界（G00 LLM-Alpha 因子挖掘）

## Mandatory Runtime Contract

All LLM stages are permanently mandatory and share the single runtime created from `config.llm`. Candidate generation, dimension inference, AlphaEval logic scoring, and economic explanation must each make a structured tool call. Any unavailable model, exception, malformed response, or missing field is a hard failure. There is no downgrade, neutral score, template explanation, environment switch, `enabled` flag, `provider` flag, or alternate configuration path. With no external fields, the current tool host model is used. With external configuration, `base_url`, `api_key`, `model`, and `protocol` are one required set and are used by every stage.

## 硬性原则

> **所有可量化的结论由确定性代码产出（rankIC / fitness / AlphaEval 五维中的 pps/pfs/rre/diversity / 信号数值）；LLM 只产出候选公式、经济解释（自然语言）、以及它自己那一维的金融逻辑评分（logic，1-10）。LLM 输出永远不会回流去修改任何其它数值字段。**

## LLM 参与的三处

| 环节 | 模块 | 工具（强制 tool use） | LLM 产物 |
|---|---|---|---|
| ① 候选公式生成 | `generate._llm_generate_candidates` | `emit_alpha_candidates` | 一批候选公式字符串（GP warm-start 种子） |
| ② 量纲推断投票 | `dimension_inferrer` | LLM 一票（stat/name/llm 三票之一） | 量纲判定（仅参与投票，非唯一裁决） |
| ③ 因子经济解释 | `llm_explainer.explain_factor` | `emit_explanation` | explanation / captures / applicable_scenario / failure_scenario（纯文字） |
| ④ 金融逻辑评分 | `alpha_eval.compute_logic` | `report_logic_score` | logic 维数值分（1-10 归一化到 [0,1]） |

## 默认行为

- LLM 永久强制开启；任何关闭开关、静默降级和中性替代都非法。
- 未配置外部模型时使用当前工具宿主模型，不要求 API key；配置外部模型时唯一入口是 `config.llm`，且 `base_url`、`api_key`、`model`、`protocol` 必须同时提供。

```python
config = {
    "llm": {
        "enabled": True,
        "provider": "agent",      # agent 环境优先；独立 CLI 才用 anthropic
        "agent_payload": {
            "model": "current-agent",
            "formulas": ["ts_corr(ma10, atr14, 30)"],
            "explanations": {}
        },
    }
}
run(universe, start, end, config=config)
```

外部 Anthropic 网关模式：

```python
config = {
    "llm": {
        "enabled": True,
        "provider": "anthropic",
        "model": "用户模型",
        "base_url": "https://your-llm-gateway.example/v1",
        "api_key": "<YOUR_API_KEY>",
        "protocol": "anthropic-tools",
    }
}
run(universe, start, end, config=config)
```

禁止关闭 LLM；调用失败必须直接抛错。

## 八条硬约束

1. **host-first**：未配置外部字段时使用当前宿主 AI 的真实结果，不要求 API key。
2. **single-config**：配置外部模型时只允许 `config.llm`，四个外部字段必须同时存在，所有阶段共用同一客户端。
3. **强制 tool use**：三处调用全部用工具（`emit_alpha_candidates` / `emit_explanation` / `report_logic_score`）强制返回结构化 JSON；纯文本 / 非 tool_use 回复一律降级。
4. **schema 校验**：公式须能被 `parse_formula` 解析成合法表达式树；解释四段字段齐全；逻辑分是 1-10 整数。不合规的部分被丢弃或降级。
5. **禁止 fake LLM**：不得创建 `LocalLLMClient`、mock client、模板解释器或启发式规则来冒充当前工具大模型；没有真实 agent payload 时必须显式标记缺失或关闭 LLM。
6. **纯 GP 可跑**：LLM 关闭 / 不可用时，warm_start=None（纯随机初始化 GP），无经济解释，logic 维中性 0.5，量纲推断只用 stat+name 两票。
7. **审计**：每次 LLM 调用的状态写入 `meta`（`llm_enabled` / `llm_provider` / `llm_available` / `llm_generate.{source, called, skipped_reason, n_formulas, latency_ms, model}`）与各 factor 的 `llm_meta`；解释模块记 `skipped_reason`。
8. **只产文字与自身维度分，绝不改数值**：LLM 不修改 rankIC / fitness / pps / pfs / rre / diversity / 信号统计 / 因子公式的求值结果。

## Agent Payload Schema

```json
{
  "model": "current-agent",
  "formulas": [
    "ts_corr(ma10, atr14, 30)",
    "diff(ts_corr(open_interest, open, 30), 5)"
  ],
  "explanations": {
    "ts_corr(ma10, atr14, 30)": {
      "explanation": "衡量短期均线与真实波幅的滚动相关性，捕捉趋势推进时波动是否同步放大或收敛。",
      "captures": "短趋势与波动扩张的联动。",
      "applicable_scenario": "适合趋势延续和波动放大的小时级行情。",
      "failure_scenario": "震荡市中均线和波动关系易反复切换。"
    }
  }
}
```

缺少解释时，报告应显示“当前 agent 尚未提供经济解释；不得用模板或公式复读代替”。后续 agent 应补充 payload 后重建报告，而不是生成机械解释。

## 候选公式生成 prompt 结构

```
SYSTEM (_gen_system_prompt):
  你是量化因子研究专家。请构造一批多样化的 alpha 因子候选表达式...
  语法：前缀函数式，如 sub(ts_mean(close,5), ts_mean(close,20))。
  仅可使用以下算子（严格拼写）：<sorted(OPERATOR_NAMES)>。
  时序算子最后一个参数必须是整数窗口常数。
  只能使用给定的特征名，不要臆造特征。
  注意量纲：不同量纲不能相加减。
  必须调用 emit_alpha_candidates 工具返回。

USER:
  可用特征名（仅限这些）：<feature_names>
  请给出 N 个不同的候选因子公式，覆盖动量/反转/波动率/量价背离等不同逻辑。
```

## emit_alpha_candidates 工具

```json
{
  "name": "emit_alpha_candidates",
  "input_schema": {
    "type": "object",
    "required": ["formulas"],
    "properties": {
      "formulas": {"type": "array", "items": {"type": "string"}}
    }
  }
}
```

## emit_explanation 工具

```json
{
  "name": "emit_explanation",
  "input_schema": {
    "type": "object",
    "required": ["explanation", "captures", "applicable_scenario", "failure_scenario"],
    "properties": {
      "explanation":          {"type": "string"},
      "captures":             {"type": "string"},
      "applicable_scenario":  {"type": "string"},
      "failure_scenario":     {"type": "string"}
    }
  }
}
```

降级占位：`explanation="(LLM未启用，无解释)"`，其余字段空串。

## report_logic_score 工具（AlphaEval logic 维）

```json
{
  "name": "report_logic_score",
  "input_schema": {
    "type": "object",
    "required": ["score", "reason"],
    "properties": {
      "score":  {"type": "integer", "minimum": 1, "maximum": 10},
      "reason": {"type": "string"}
    }
  }
}
```

score 归一化到 [0,1] 作为 logic 维；无 LLM / 失败时 logic=0.5（中性）。

## 排查 FAQ

- **LLM 请求了但没跑**：这是硬失败。检查唯一运行时的 `mode`、`run_id`、`request_id` 和 `payload_hash`，不得切换配置或降级。
- **候选全被拒**：看 `rejected`（含 layer / reason）；常见算子拼错（语法层）、特征未注册（注册表层）、量纲不一致（量纲层）、负窗口（前视层）。
- **logic 恒为 0.5**：LLM 未启用或调用失败降级为中性，属预期。
- **想省 token**：关 LLM 纯 GP，或调小 `n_candidates`。
