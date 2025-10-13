from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from python.runtime.audio.neutts_provider import NeuttsProvider


def _sine_wave(duration_sec: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    return 0.4 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)


@pytest.fixture()
def provider(tmp_path: Path) -> NeuttsProvider:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return NeuttsProvider(
        backbone_repo="neuphonic/neutts-air-q4-gguf",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        backbone_device="mps",
        codec_device="cpu",
        model_cache_dir=models_dir,
        data_root=tmp_path / "cache",
        stream_chunk_seconds=0.2,
        sample_rate=24_000,
        quality_default="q4",
    )


def test_register_voice_and_list(provider: NeuttsProvider, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    audio = _sine_wave(4.0, provider.sample_rate)
    sf.write(wav_path, audio, provider.sample_rate)

    voice_id = provider.register_voice("Legacy Lane", str(wav_path), "Reference passage.")
    voices = provider.list_voices()
    assert any(v["id"] == voice_id for v in voices)

    voice_dir = provider.voices_dir / voice_id
    assert voice_dir.exists()
    assert (voice_dir / "ref.codes.npy").exists()
    assert (voice_dir / "meta.json").exists()


def test_synthesize_non_streaming_returns_wav_bytes(provider: NeuttsProvider, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    audio = _sine_wave(4.0, provider.sample_rate)
    sf.write(wav_path, audio, provider.sample_rate)
    voice_id = provider.register_voice("Sample Voice", str(wav_path), "Reference text")

    pcm_bytes = provider.synthesize("Hello from NeuTTS-Air.", voice_id, stream=False)
    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) > 0
    meta = provider.last_output_metadata
    assert meta.get("voice_id") == voice_id
    assert meta.get("watermarked") is True
    assert meta.get("sample_rate") == provider.sample_rate


def test_synthesize_streaming_chunks(provider: NeuttsProvider, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    audio = _sine_wave(3.5, provider.sample_rate)
    sf.write(wav_path, audio, provider.sample_rate)
    voice_id = provider.register_voice("Stream Voice", str(wav_path), "Stream text")

    generator = provider.synthesize("Streaming output test.", voice_id, stream=True)
    chunks = list(generator)
    assert chunks, "Expected streaming chunks"
    assert all(isinstance(chunk, bytes) and len(chunk) > 0 for chunk in chunks)
    first_chunk = chunks[0]
    # Each chunk should roughly align with configured chunk duration
    bytes_per_chunk = provider.sample_rate * provider.stream_chunk_seconds * 2  # mono int16
    assert len(first_chunk) <= bytes_per_chunk * 1.5


def test_delete_voice(provider: NeuttsProvider, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    audio = _sine_wave(3.2, provider.sample_rate)
    sf.write(wav_path, audio, provider.sample_rate)
    voice_id = provider.register_voice("Delete Voice", str(wav_path), "Delete text")
    provider.delete_voice(voice_id)
    assert voice_id not in {v["id"] for v in provider.list_voices()}
    assert not (provider.voices_dir / voice_id).exists()


def test_default_voice_fallback(provider: NeuttsProvider, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    audio = _sine_wave(4.0, provider.sample_rate)
    sf.write(wav_path, audio, provider.sample_rate)
    voice_id = provider.register_voice("Fallback Voice", str(wav_path), "Ref text")

    provider.set_default_voice(voice_id)
    pcm_bytes = provider.synthesize("Default voice usage", None, stream=False)
    assert isinstance(pcm_bytes, bytes)
    meta = provider.last_output_metadata
    assert meta.get("voice_id") == voice_id
