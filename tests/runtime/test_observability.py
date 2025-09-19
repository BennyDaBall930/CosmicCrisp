"""Tests for the runtime observability helpers."""
from __future__ import annotations

import json
from pathlib import Path

from python.runtime.config import ObservabilityConfig
from python.runtime.observability import Observability


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_record_events_and_metrics(tmp_path):
    log_path = tmp_path / "obs.jsonl"
    cfg = ObservabilityConfig(json_log_path=log_path, helicone_enabled=False, metrics_namespace="test_ns")
    obs = Observability(cfg)

    obs.record_run_started(run_type="goal", run_id="r1", session="s1", goal="demo", model="gpt")
    obs.record_task_started(run_id="r1", task_id="t1", owner="s1", description="demo task")
    obs.record_task_completed(run_id="r1", task_id="t1", owner="s1", result="ok")
    obs.record_tool_usage(tool="search", run_id="r1", task_id="t1")
    obs.record_memory_hit(session="s1", count=2)
    obs.record_token_usage(model="gpt", tokens=10, run_id="r1")

    entries = _read_jsonl(log_path)
    assert any(entry.get("type") == "run_started" for entry in entries)
    assert any(entry.get("type") == "tool_used" for entry in entries)

    metrics = obs.export_metrics().decode()
    assert "test_ns_runs_started_total" in metrics
    assert "test_ns_tool_invocations_total" in metrics


def test_helicone_headers_and_logging(tmp_path):
    log_path = tmp_path / "obs.jsonl"
    cfg = ObservabilityConfig(
        json_log_path=log_path,
        helicone_enabled=True,
        helicone_base_url="https://helicone.dev",
        helicone_api_key="secret-key",
        metrics_namespace="test_ns",
    )
    obs = Observability(cfg)

    headers = obs.record_model_call(
        session="s42",
        run_id="run42",
        model="gpt",
        tokens=33,
        tool="planner",
        metadata={"task_id": "t42"},
    )

    assert headers["Helicone-Auth"] == "Bearer secret-key"
    assert headers["Helicone-Property-Session"] == "s42"
    assert headers["Helicone-Property-Tool"] == "planner"
    assert headers["Helicone-Property-task-id"] == "t42"

    entries = _read_jsonl(log_path)
    assert any(entry.get("type") == "model_call" for entry in entries)
