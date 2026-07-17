from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dimension_inferrer import (  # noqa: E402
    infer_and_register,
    infer_feature_tags,
    llm_vote,
    name_unit_vote,
    stat_unit_vote,
)
from feature_registry import FeatureRegistry  # noqa: E402
from llm_runtime import LLMRuntimeError  # noqa: E402
from tests_support import RecordingLLMClient  # noqa: E402


def _client() -> RecordingLLMClient:
    return RecordingLLMClient()


def test_llm_vote_requires_shared_client():
    with pytest.raises(LLMRuntimeError):
        llm_vote("close", {"n": 10}, None)


def test_llm_vote_returns_structured_dimension():
    result = llm_vote("close", {"n": 10, "mean": 100}, _client())
    assert result["unit"] in {"price", "dimensionless", "count", "money", "bool", "unknown"}
    assert result["semantic"]


def test_infer_feature_tags_requires_llm_and_registers_result():
    series = pd.Series(np.linspace(100.0, 110.0, 40))
    result = infer_feature_tags("close", series, llm_client=_client())
    assert result["unit"] == "price"
    assert result["semantic"] == "price"
    registry = FeatureRegistry()
    registered = infer_and_register(registry, "close", series, llm_client=_client())
    assert registered["unit"] == "price"
    assert registry.get_unit("close") == "price"


def test_malformed_llm_response_fails_loudly():
    class BadMessages:
        def create(self, **kwargs):
            return type("Response", (), {"content": []})()

    client = type("Client", (), {"messages": BadMessages()})()
    with pytest.raises(LLMRuntimeError):
        llm_vote("close", {"n": 10}, client)


def test_deterministic_votes_remain_available():
    series = pd.Series(np.linspace(100.0, 110.0, 40))
    assert name_unit_vote("close")["unit"] == "price"
    assert stat_unit_vote(series)["unit"] in {"price", "count", "dimensionless", "unknown"}
