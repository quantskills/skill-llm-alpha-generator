from __future__ import annotations

import json

import pytest

from llm_agent_protocol import (
    AgentProtocolError,
    AtomicRunOutput,
    make_request,
    payload_hash,
    validate_response,
)


def test_request_response_identity_is_checked():
    request = make_request("run-1", "emit_alpha_candidates", {"n": 3})
    response = {**request, "result": {"formulas": ["close"]}}
    assert validate_response(request, response)["formulas"] == ["close"]
    response["request_id"] = "stale"
    with pytest.raises(AgentProtocolError):
        validate_response(request, response)


def test_payload_hash_is_stable():
    assert payload_hash({"b": 1, "a": 2}) == payload_hash({"a": 2, "b": 1})


def test_atomic_output_publishes_only_complete_artifact(tmp_path):
    with AtomicRunOutput(tmp_path, "run-1") as staged:
        staged.write_json("manifest.json", {"run_id": "run-1"})
        published = staged.publish("manifest.json")
    assert json.loads(published.read_text(encoding="utf-8"))["run_id"] == "run-1"
    assert not list(tmp_path.glob(".run-*"))
