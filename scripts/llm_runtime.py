"""Single LLM configuration and client boundary for the alpha skill."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from llm_agent_protocol import new_run_id, payload_hash


class LLMRuntimeError(RuntimeError):
    """Raised when the mandatory LLM boundary cannot produce a valid result."""


_EXTERNAL_FIELDS = {"base_url", "api_key", "model", "protocol"}
_TOOLS = {
    "emit_alpha_candidates",
    "report_dimension",
    "report_logic_score",
    "emit_explanation",
}


class _AgentPayloadMessages:
    def __init__(self, payload: dict):
        self._payload = payload

    def create(self, **kwargs):
        tool_choice = kwargs.get("tool_choice") or {}
        name = tool_choice.get("name")
        if name not in _TOOLS:
            raise LLMRuntimeError(f"unsupported mandatory LLM tool: {name!r}")
        data = self._lookup(name, kwargs.get("messages") or [])
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=data)],
            usage=SimpleNamespace(input_tokens=None, output_tokens=None),
        )

    def _lookup(self, name: str, messages: list[dict]) -> dict:
        payload = self._payload
        if name == "emit_alpha_candidates":
            formulas = payload.get("formulas") or payload.get("candidate_formulas")
            if not isinstance(formulas, list) or not formulas:
                raise LLMRuntimeError("agent payload missing candidate formulas")
            return {"formulas": formulas}

        text = "\n".join(str(m.get("content", "")) for m in messages)
        if name == "report_dimension":
            votes = payload.get("dimension_votes") or {}
            feature = _find_key(text, votes)
            if feature is None:
                raise LLMRuntimeError("agent payload missing dimension vote")
            value = votes[feature]
            if not isinstance(value, dict):
                raise LLMRuntimeError(f"invalid dimension vote for {feature!r}")
            return value

        formulas = payload.get("logic_scores" if name == "report_logic_score" else "explanations") or {}
        formula = _find_key(text, formulas)
        if formula is None:
            raise LLMRuntimeError(f"agent payload missing {name} result")
        value = formulas[formula]
        if not isinstance(value, dict):
            raise LLMRuntimeError(f"invalid {name} result for formula")
        return value


def _find_key(text: str, values: dict) -> str | None:
    matches = [key for key in values if str(key) and str(key) in text]
    return max(matches, key=len) if matches else None


class LLMRuntime:
    """Owns the only LLM configuration and client used by a run."""

    def __init__(self, *, mode: str, model: str, client, config: dict):
        self.mode = mode
        self.model = model
        self.client = client
        self.config = dict(config)
        self.calls: list[dict] = []
        self.run_id = new_run_id()

    @classmethod
    def from_config(cls, config: dict | None) -> "LLMRuntime":
        root = config or {}
        llm = root.get("llm", {})
        if not isinstance(llm, dict):
            raise LLMRuntimeError("config.llm must be an object")
        if "enabled" in llm or "provider" in llm:
            raise LLMRuntimeError(
                "config.llm.enabled/provider are forbidden; LLM is mandatory and provider is automatic"
            )

        configured = set(llm) & _EXTERNAL_FIELDS
        if configured:
            missing = sorted(_EXTERNAL_FIELDS - configured)
            if missing:
                raise LLMRuntimeError(
                    "external LLM configuration is incomplete; missing: " + ", ".join(missing)
                )
            if llm["protocol"] != "anthropic-tools":
                raise LLMRuntimeError(
                    "unsupported LLM protocol; only 'anthropic-tools' is currently implemented"
                )
            return cls._external(llm)

        payload = _load_payload(llm)
        if not payload:
            raise LLMRuntimeError(
                "current-agent LLM output is required; provide config.llm.agent_payload or agent_payload_path"
            )
        model = str(payload.get("model") or "current-agent")
        return cls(mode="current-agent", model=model,
                   client=SimpleNamespace(
                       _alpha_model=model,
                       messages=_AgentPayloadMessages(payload),
                   ),
                   config={"mode": "current-agent", "model": model})

    @classmethod
    def _external(cls, llm: dict) -> "LLMRuntime":
        try:
            from anthropic import Anthropic
        except Exception as exc:  # pragma: no cover - depends on deployment
            raise LLMRuntimeError("anthropic SDK is required for configured LLM") from exc
        try:
            client = Anthropic(api_key=llm["api_key"], base_url=llm["base_url"])
        except Exception as exc:  # pragma: no cover - depends on deployment
            raise LLMRuntimeError("configured LLM client construction failed") from exc
        return cls(mode="configured", model=str(llm["model"]), client=client,
                   config={"mode": "configured", "model": str(llm["model"]),
                           "protocol": llm["protocol"], "base_url": llm["base_url"]})

    def record(self, name: str, *, status: str, **extra) -> None:
        event = {"name": name, "status": status, "model": self.model, **extra}
        event["run_id"] = self.run_id
        event["request_id"] = new_run_id()
        event["payload_hash"] = payload_hash({"stage": name, "details": extra})
        self.calls.append(event)

    def audit(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model,
            "run_id": self.run_id,
            "calls": list(self.calls),
        }


def _load_payload(llm: dict) -> dict:
    payload = llm.get("agent_payload")
    path = llm.get("agent_payload_path")
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            file_payload = json.load(fh)
        if payload is None:
            payload = file_payload
        elif isinstance(file_payload, dict) and isinstance(payload, dict):
            merged = dict(file_payload)
            merged.update(payload)
            payload = merged
    if not isinstance(payload, dict):
        raise LLMRuntimeError("agent payload must be a JSON object")
    return payload
