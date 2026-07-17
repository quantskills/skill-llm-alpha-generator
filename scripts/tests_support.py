from __future__ import annotations

import re
from types import SimpleNamespace

from llm_runtime import LLMRuntime


class RecordingLLMClient:
    """Deterministic transport used only by unit tests."""

    _alpha_model = "test-model"

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        tool = kwargs["tool_choice"]["name"]
        text = "\n".join(str(item.get("content", "")) for item in kwargs.get("messages", []))
        self.calls.append({"tool": tool, "model": kwargs.get("model"), "text": text})
        if tool == "emit_alpha_candidates":
            data = {"formulas": ["sub(close, ma5)", "ts_zscore(ret5, 20)", "clv"]}
        elif tool == "report_dimension":
            name = re.search(r"(?:变量名|feature|特征)[：: ]*([A-Za-z_][A-Za-z0-9_]*)", text)
            feature = name.group(1) if name else "feature"
            if feature in {"open", "high", "low", "close", "ma5", "ma10", "ma20"}:
                unit, semantic = "price", "price"
            elif feature in {"volume", "open_interest"}:
                unit, semantic = "count", "volume"
            else:
                unit, semantic = "dimensionless", "return"
            data = {"unit": unit, "semantic": semantic, "confidence": 0.9,
                    "reason": "deterministic test transport"}
        elif tool == "report_logic_score":
            data = {"score": 7, "reason": "test transport supplied a structured logic score"}
        elif tool == "emit_explanation":
            data = {
                "explanation": "测试解释：该因子描述价格状态与短期风险状态的联动。",
                "captures": "测试市场行为",
                "applicable_scenario": "测试趋势阶段",
                "failure_scenario": "测试震荡阶段",
            }
        else:
            raise AssertionError(tool)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=data)],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def make_test_runtime() -> LLMRuntime:
    client = RecordingLLMClient()
    runtime = LLMRuntime(mode="test", model="test-model", client=client, config={"mode": "test"})
    runtime.test_client = client
    return runtime
