import importlib

import pytest
from fastapi.testclient import TestClient

from python.runtime.api import app as exported_app


class StubOrchestrator:
    async def stream_run(self, payload):  # pragma: no cover - simple stub
        yield {"event": "START", "data": payload.goal}


@pytest.fixture()
def client(monkeypatch):
    app_module = importlib.import_module('python.runtime.api.app')
    monkeypatch.setattr(app_module, 'get_orchestrator', lambda: StubOrchestrator())
    return TestClient(app_module.app)


def test_streaming_run(client: TestClient):
    with client.stream('POST', '/runtime/run', json={'goal': 'find docs', 'session_id': 's'}) as response:
        text = ''.join(response.iter_text())
    assert response.status_code == 200
    assert 'event: START' in text
