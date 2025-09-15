from fastapi.testclient import TestClient

from cosmiccrisp.api.app import app


def test_streaming_run():
    client = TestClient(app)
    with client.stream("POST", "/run", json={"goal": "find docs"}) as resp:
        text = b"".join(resp.iter_raw()).decode()
    assert "START" in text
    assert "EXECUTE" in text
