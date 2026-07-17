from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_explainer import LLMError, explain_factor
from tests_support import RecordingLLMClient


FORMULA = "sub(close, ma5)"


def test_explanation_requires_shared_client():
    with pytest.raises(LLMError):
        explain_factor(FORMULA)


def test_explanation_requires_all_fields():
    class Empty:
        _alpha_model = "test-model"

        class messages:
            @staticmethod
            def create(**_kwargs):
                return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={})])

    with pytest.raises(LLMError):
        explain_factor(FORMULA, llm_client=Empty())


def test_explanation_uses_tool_and_returns_all_fields():
    client = RecordingLLMClient()
    out = explain_factor(FORMULA, llm_client=client)
    assert set(("explanation", "captures", "applicable_scenario", "failure_scenario")) <= set(out)
    assert out["llm_meta"]["called"] is True
    assert client.calls[0]["tool"] == "emit_explanation"


def test_non_tool_response_fails():
    class Plain:
        _alpha_model = "test-model"

        class messages:
            @staticmethod
            def create(**_kwargs):
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="plain")])

    with pytest.raises(LLMError):
        explain_factor(FORMULA, llm_client=Plain())
