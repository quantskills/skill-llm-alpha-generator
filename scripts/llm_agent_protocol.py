"""Strict protocol helpers for the current-tool LLM runtime.

The agent transport is intentionally treated as a request/response protocol:
each request carries a run id, request id, and payload hash, and every staged
artifact is published only after its manifest validates.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


class AgentProtocolError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def new_run_id() -> str:
    return uuid.uuid4().hex


def make_request(run_id: str, stage: str, payload: dict) -> dict:
    if not run_id or not stage:
        raise AgentProtocolError("run_id and stage are required")
    return {
        "protocol": "llm-alpha-agent/v1",
        "run_id": run_id,
        "request_id": uuid.uuid4().hex,
        "stage": stage,
        "payload_hash": payload_hash(payload),
        "payload": payload,
    }


def validate_response(request: dict, response: dict) -> dict:
    if not isinstance(response, dict):
        raise AgentProtocolError("agent response must be an object")
    for key in ("protocol", "run_id", "request_id", "stage", "payload_hash", "result"):
        if key not in response:
            raise AgentProtocolError(f"agent response missing {key}")
    if response["protocol"] != request["protocol"]:
        raise AgentProtocolError("agent protocol version mismatch")
    for key in ("run_id", "request_id", "stage", "payload_hash"):
        if response[key] != request[key]:
            raise AgentProtocolError(f"agent response {key} mismatch")
    if not isinstance(response["result"], dict):
        raise AgentProtocolError("agent response result must be an object")
    return response["result"]


class AtomicRunOutput:
    """Write a run into a private directory and publish it atomically."""

    def __init__(self, output_dir: str | os.PathLike[str], run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.stage_dir: Path | None = None

    def __enter__(self) -> "AtomicRunOutput":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = Path(tempfile.mkdtemp(prefix=f".run-{self.run_id}-", dir=self.output_dir))
        return self

    def write_json(self, name: str, value: dict) -> Path:
        if self.stage_dir is None:
            raise AgentProtocolError("atomic output is not active")
        path = self.stage_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def publish(self, name: str) -> Path:
        if self.stage_dir is None:
            raise AgentProtocolError("atomic output is not active")
        source = self.stage_dir / name
        if not source.exists() or source.stat().st_size == 0:
            raise AgentProtocolError(f"staged artifact is missing or empty: {name}")
        target = self.output_dir / name
        os.replace(source, target)
        return target

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stage_dir is not None and self.stage_dir.exists():
            for child in self.stage_dir.iterdir():
                child.unlink(missing_ok=True)
            self.stage_dir.rmdir()
