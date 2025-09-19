"""Observability utilities for Apple Zero runtime."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import CollectorRegistry, Counter, generate_latest

from .config import ObservabilityConfig


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class Observability:
    """Collects metrics, emits JSON logs, and prepares Helicone headers."""

    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self.registry = CollectorRegistry()
        self._lock = threading.Lock()
        self._log_path: Optional[Path] = None
        if self.config.json_log_path:
            path = Path(self.config.json_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path = path

        self._records_enabled = bool(self._log_path)
        self._helicone_enabled = bool(config.helicone_enabled and config.helicone_base_url)

        if self._helicone_enabled and config.helicone_base_url:
            os.environ.setdefault("OPENAI_BASE_URL", config.helicone_base_url)
            os.environ.setdefault("HELICONE_BASE_URL", config.helicone_base_url)
        if self._helicone_enabled and config.helicone_api_key:
            os.environ.setdefault("HELICONE_API_KEY", config.helicone_api_key)

        # Prometheus counters
        ns = config.metrics_namespace
        self._runs_started = Counter(f"{ns}_runs_started_total", "Runs started", ["type"], registry=self.registry)
        self._runs_completed = Counter(f"{ns}_runs_completed_total", "Runs completed", ["type"], registry=self.registry)
        self._runs_failed = Counter(f"{ns}_runs_failed_total", "Runs failed", ["type"], registry=self.registry)
        self._tasks_started = Counter(f"{ns}_tasks_started_total", "Tasks started", ["owner"], registry=self.registry)
        self._tasks_completed = Counter(f"{ns}_tasks_completed_total", "Tasks completed", ["owner"], registry=self.registry)
        self._tools_used = Counter(f"{ns}_tool_invocations_total", "Tool invocations", ["tool"], registry=self.registry)
        self._memory_hits = Counter(f"{ns}_memory_hits_total", "Memory hits", registry=self.registry)
        self._memory_misses = Counter(f"{ns}_memory_misses_total", "Memory misses", registry=self.registry)
        self._tokens_emitted = Counter(f"{ns}_tokens_emitted_total", "Tokens emitted", ["model"], registry=self.registry)
        self._subagents_spawned = Counter(f"{ns}_subagents_spawned_total", "Sub-agents spawned", ["tool"], registry=self.registry)
        self._model_calls = Counter(f"{ns}_model_calls_total", "Model calls", ["model"], registry=self.registry)

    # ------------------------------------------------------------------
    def record_run_started(self, *, run_type: str, run_id: str, session: str, goal: Optional[str], model: Optional[str]) -> None:
        self._runs_started.labels(run_type).inc()
        self._log_event(
            "run_started",
            run_type=run_type,
            run_id=run_id,
            session=session,
            goal=goal,
            model=model,
        )

    def record_run_completed(self, *, run_type: str, run_id: str, session: str) -> None:
        self._runs_completed.labels(run_type).inc()
        self._log_event("run_completed", run_type=run_type, run_id=run_id, session=session)

    def record_run_failed(self, *, run_type: str, run_id: str, session: str, error: str) -> None:
        self._runs_failed.labels(run_type).inc()
        self._log_event("run_failed", run_type=run_type, run_id=run_id, session=session, error=error)

    # ------------------------------------------------------------------
    def record_task_started(self, *, run_id: str, task_id: str, owner: str, description: str) -> None:
        self._tasks_started.labels(owner).inc()
        self._log_event("task_started", run_id=run_id, task_id=task_id, owner=owner, description=description)

    def record_task_completed(self, *, run_id: str, task_id: str, owner: str, result: str) -> None:
        self._tasks_completed.labels(owner).inc()
        self._log_event("task_completed", run_id=run_id, task_id=task_id, owner=owner, result=result)

    # ------------------------------------------------------------------
    def record_tool_usage(self, *, tool: str, run_id: Optional[str] = None, task_id: Optional[str] = None) -> None:
        self._tools_used.labels(tool).inc()
        self._log_event("tool_used", tool=tool, run_id=run_id, task_id=task_id)

    def record_memory_hit(self, *, session: str, count: int) -> None:
        self._memory_hits.inc(count)
        self._log_event("memory_hit", session=session, count=count)

    def record_memory_miss(self, *, session: str) -> None:
        self._memory_misses.inc()
        self._log_event("memory_miss", session=session)

    def record_token_usage(self, *, model: str, tokens: int, run_id: Optional[str] = None) -> None:
        if tokens <= 0:
            return
        self._tokens_emitted.labels(model).inc(tokens)
        self._log_event("tokens_emitted", model=model, tokens=tokens, run_id=run_id)

    def record_subagent_spawn(self, *, tool: str, depth: int, parent_session: str) -> None:
        self._subagents_spawned.labels(tool).inc()
        self._log_event("subagent_spawned", tool=tool, depth=depth, session=parent_session)

    def record_model_call(
        self,
        *,
        session: str,
        run_id: str,
        model: Optional[str],
        tokens: int = 0,
        tool: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        model_name = model or "unknown"
        self._model_calls.labels(model_name).inc()
        if tokens > 0:
            self._tokens_emitted.labels(model_name).inc(tokens)
        payload = {
            "session": session,
            "run_id": run_id,
            "model": model_name,
            "tokens": tokens,
            "tool": tool,
            **(metadata or {}),
        }
        self._log_event("model_call", **payload)
        if not self._helicone_enabled:
            return {}
        headers = {
            "Helicone-Auth": f"Bearer {self.config.helicone_api_key}" if self.config.helicone_api_key else "",
            "Helicone-Property-Session": session,
            "Helicone-Property-Run-Id": run_id,
            "Helicone-Property-Tool": tool or "",
        }
        if tokens:
            headers["Helicone-Property-Tokens"] = str(tokens)
        if metadata:
            for key, value in metadata.items():
                header_key = f"Helicone-Property-{key.replace('_', '-')}"
                headers[header_key] = str(value)
        return {k: v for k, v in headers.items() if v}

    # ------------------------------------------------------------------
    def export_metrics(self) -> bytes:
        return generate_latest(self.registry)

    # ------------------------------------------------------------------
    def _log_event(self, event_type: str, **payload: Any) -> None:
        if not self._records_enabled or self._log_path is None:
            return
        record = {"ts": _ts(), "type": event_type, **{k: v for k, v in payload.items() if v is not None}}
        line = json.dumps(record)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


__all__ = ["Observability"]
