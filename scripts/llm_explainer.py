"""Mandatory economic explanation adapter.

Client construction and configuration belong exclusively to llm_runtime.py.
"""
from __future__ import annotations

import time


class LLMError(RuntimeError):
    """A mandatory LLM call failed or returned an invalid result."""


_EXPLAIN_TOOL = {
    "name": "emit_explanation",
    "description": "Return the economic explanation of one alpha factor.",
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "captures": {"type": "string"},
            "applicable_scenario": {"type": "string"},
            "failure_scenario": {"type": "string"},
        },
        "required": ["explanation", "captures", "applicable_scenario", "failure_scenario"],
    },
}


_SYSTEM_PROMPT = (
    "You are a quantitative factor economics reviewer. Explain the market behavior, "
    "economic intuition, applicable scenario, and failure scenario of the formula. "
    "Use Chinese. Do not invent numeric metrics. Always call emit_explanation."
)


def _build_user_prompt(formula_str: str, alpha_scores: dict | None) -> str:
    lines = [f"因子表达式：{formula_str}"]
    if alpha_scores:
        fields = ("fitness", "ic", "rank_ic", "turnover", "coverage", "node_count")
        picked = {k: alpha_scores[k] for k in fields if k in alpha_scores}
        if picked:
            lines.append(f"历史数值仅供定性参考，不得重新计算：{picked}")
    lines.append("请调用 emit_explanation 工具返回四个完整字段。")
    return "\n".join(lines)


def explain_factor(formula_str: str, alpha_scores: dict | None = None,
                   llm_client=None, config: dict | None = None) -> dict:
    if not formula_str:
        raise LLMError("economic explanation requires a formula")
    if llm_client is None:
        raise LLMError("economic explanation requires the shared LLM runtime client")

    model = getattr(llm_client, "_alpha_model", None)
    if not model:
        model = (config or {}).get("llm", {}).get("model") or "current-agent"
    started = time.time()
    try:
        response = llm_client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_EXPLAIN_TOOL],
            tool_choice={"type": "tool", "name": "emit_explanation"},
            messages=[{"role": "user", "content": _build_user_prompt(formula_str, alpha_scores)}],
        )
    except Exception as exc:
        raise LLMError(f"economic explanation call failed: {exc!r}") from exc

    tool = next((b for b in getattr(response, "content", [])
                 if getattr(b, "type", None) == "tool_use"), None)
    if tool is None or not isinstance(getattr(tool, "input", None), dict):
        raise LLMError("economic explanation response did not contain tool_use")

    data = tool.input
    fields = ("explanation", "captures", "applicable_scenario", "failure_scenario")
    values = {key: str(data.get(key) or "").strip() for key in fields}
    if not all(values.values()):
        raise LLMError("economic explanation response contains empty required fields")
    usage = getattr(response, "usage", None)
    tokens = None
    if usage is not None:
        tokens = {
            "input": getattr(usage, "input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
        }
    return {
        **values,
        "llm_meta": {
            "enabled": True,
            "model": model,
            "called": True,
            "skipped_reason": None,
            "latency_ms": int((time.time() - started) * 1000),
            "tokens": tokens,
        },
    }
