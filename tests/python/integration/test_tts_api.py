from __future__ import annotations

import base64
import importlib
import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from python.runtime.audio.neutts_provider import NeuttsProvider


def _sine_wave(duration: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    return 0.35 * np.sin(2 * np.pi * 180.0 * t).astype(np.float32)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    app_module = importlib.import_module("python.runtime.api.app")
    provider = NeuttsProvider(
        backbone_repo="stub/backbone",
        codec_repo="stub/codec",
        backbone_device="cpu",
        codec_device="cpu",
        model_cache_dir=tmp_path / "models",
        data_root=tmp_path / "cache",
        stream_chunk_seconds=0.2,
        sample_rate=24_000,
    )

    class _DummyEngine:
        sample_rate = 24_000

        def encode_reference(self, *_args, **_kwargs):
            return torch.arange(64, dtype=torch.int32)

        def infer(self, text: str, ref_codes: torch.Tensor, _ref_text: str) -> np.ndarray:
            duration = max(len(text), 1) * 240  # ~0.01 s per char
            return np.linspace(-0.2, 0.2, duration, dtype=np.float32)

        def infer_stream(self, text: str, ref_codes: torch.Tensor, _ref_text: str):
            del text, ref_codes
            yield np.linspace(-0.2, 0.2, 2_400, dtype=np.float32)

    dummy_engine = _DummyEngine()

    monkeypatch.setattr(provider, "_get_engine", lambda: dummy_engine)

    monkeypatch.setattr(app_module, "get_tts_provider", lambda: provider)
    return TestClient(app_module.app)


def test_create_and_list_voices(client: TestClient):
    sample_rate = 24_000
    audio = _sine_wave(3.2, sample_rate)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    response = client.post(
        "/tts/voices",
        json={"name": "Integration Voice", "ref_text": "Test reference text.", "audio_base64": audio_b64},
    )
    assert response.status_code == 200
    voice_id = response.json()["voice_id"]

    list_resp = client.get("/tts/voices")
    assert list_resp.status_code == 200
    voices = list_resp.json()["voices"]
    assert any(v["id"] == voice_id for v in voices)


def test_stream_speak_returns_audio(client: TestClient):
    sample_rate = 24_000
    audio = _sine_wave(3.0, sample_rate)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    create_resp = client.post(
        "/tts/voices",
        json={"name": "Streaming Voice", "ref_text": "Streaming text.", "audio_base64": audio_b64},
    )
    voice_id = create_resp.json()["voice_id"]

    with client.stream(  # type: ignore[attr-defined]
        "POST",
        "/tts/speak",
        json={"text": "Streaming test text for NeuTTS.", "voice_id": voice_id, "stream": True},
    ) as response:
        assert response.status_code == 200
        chunks = list(response.iter_bytes())

    # Set default voice via API
    default_resp = client.post(
        "/tts/default",
        json={"voice_id": voice_id},
    )
    assert default_resp.status_code == 200
    assert default_resp.json()["default_voice_id"] == voice_id

    # Request synthesis without specifying voice_id (should use default)
    fallback_resp = client.post(
        "/tts/speak",
        json={"text": "Default voice invocation", "stream": False},
    )
    assert fallback_resp.status_code == 200
    assert fallback_resp.headers.get("X-NeuTTS-Voice-Id") == voice_id

    assert chunks
    assert chunks[0].startswith(b"RIFF")
