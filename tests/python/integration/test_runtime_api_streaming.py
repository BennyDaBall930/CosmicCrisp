from __future__ import annotations

import importlib
from typing import Any, AsyncIterator, Dict

import pytest
from fastapi.testclient import TestClient


class DummyOrchestrator:
    async def stream_run(self, payload) -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover - generator used in test
        yield {"event": "START", "data": {"goal": payload.goal}}
        yield {"event": "TICK", "data": {"step": 1}}
        yield {"event": "DONE", "data": {"status": "ok"}}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app_module = importlib.import_module('python.runtime.api.app')
    monkeypatch.setattr(app_module, 'get_orchestrator', lambda: DummyOrchestrator())
    return TestClient(app_module.app)


def test_run_endpoint_streams_events(client: TestClient):
    payload = {"session_id": "s", "goal": "Test streaming"}
    with client.stream('POST', '/runtime/run', json=payload) as response:
        body = ''.join(response.iter_text())
    assert response.status_code == 200
    assert 'event: START' in body
    assert 'event: DONE' in body
    assert 'Test streaming' in body
