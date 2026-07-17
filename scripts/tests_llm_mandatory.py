from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate
from llm_runtime import LLMRuntime, LLMRuntimeError
from tests_support import make_test_runtime


def test_llm_is_mandatory_without_payload():
    with pytest.raises(LLMRuntimeError):
        LLMRuntime.from_config({})


def test_legacy_disable_and_provider_are_rejected():
    for cfg in (
        {"llm": {"enabled": False}},
        {"llm": {"provider": "anthropic"}},
    ):
        with pytest.raises(LLMRuntimeError):
            LLMRuntime.from_config(cfg)


def test_external_config_is_single_and_complete():
    with pytest.raises(LLMRuntimeError):
        LLMRuntime.from_config({"llm": {"api_key": "secret"}})


def test_agent_payload_runtime_has_one_client():
    payload = {
        "model": "current-agent",
        "formulas": ["sub(close, ma5)"],
        "dimension_votes": {},
        "logic_scores": {},
        "explanations": {},
    }
    runtime = LLMRuntime.from_config({"llm": {"agent_payload": payload}})
    assert runtime.mode == "current-agent"
    assert runtime.client.messages is not None


def test_generate_receives_shared_runtime(monkeypatch):
    runtime = make_test_runtime()
    monkeypatch.setattr(generate.LLMRuntime, "from_config",
                        staticmethod(lambda _config: runtime))
    out = generate.run(
        "SYNTH", "x", "y",
        config={
            "data": {"use_synthetic": True},
            "gp": {"pop_size": 20, "n_gen": 2, "seed": 1},
            "top_k": 2,
        },
    )
    assert out["factors"]
    assert {call["tool"] for call in runtime.test_client.calls} >= {
        "emit_alpha_candidates", "report_dimension", "report_logic_score", "emit_explanation"
    }
