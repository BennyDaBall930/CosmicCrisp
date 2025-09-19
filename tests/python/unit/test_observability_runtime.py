from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from python.runtime.config import ObservabilityConfig
from python.runtime.observability import Observability


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ObservabilityConfig:
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('HELICONE_BASE_URL', raising=False)
    monkeypatch.delenv('HELICONE_API_KEY', raising=False)
    cfg = ObservabilityConfig()
    cfg.helicone_enabled = True
    cfg.helicone_base_url = 'https://helicone.local'
    cfg.helicone_api_key = 'redacted'
    cfg.json_log_path = tmp_path / 'observability.jsonl'
    cfg.metrics_namespace = 'test_ns'
    return cfg


def test_record_model_call_sets_headers_and_logs(config: ObservabilityConfig):
    obs = Observability(config)
    headers = obs.record_model_call(
        session='session-1',
        run_id='run-42',
        model='gpt-4o',
        tokens=123,
        tool='search',
        metadata={'latency_ms': 42},
    )

    assert headers['Helicone-Auth'] == 'Bearer redacted'
    assert headers['Helicone-Property-Run-Id'] == 'run-42'
    assert headers['Helicone-Property-Tokens'] == '123'

    log_path = Path(config.json_log_path)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert any(json.loads(line)['type'] == 'model_call' for line in lines)


def test_export_metrics_returns_prometheus_payload(config: ObservabilityConfig):
    obs = Observability(config)
    obs.record_run_started(run_type='agent', run_id='run-1', session='sess', goal='mission', model='gpt')
    payload = obs.export_metrics()
    assert payload.startswith(b'# HELP test_ns_runs_started_total')


def test_model_call_without_helicone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv('HELICONE_API_KEY', raising=False)
    monkeypatch.delenv('HELICONE_BASE_URL', raising=False)
    cfg = ObservabilityConfig()
    cfg.helicone_enabled = False
    cfg.json_log_path = tmp_path / 'obs.jsonl'
    cfg.metrics_namespace = 'noheli'
    obs = Observability(cfg)
    headers = obs.record_model_call(session='s', run_id='r', model=None, tokens=0)
    assert headers == {}
    assert os.environ.get('HELICONE_API_KEY') is None
