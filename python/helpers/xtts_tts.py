"""Coqui XTTS text-to-speech backend integration."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torchaudio as ta

try:
    from TTS.api import TTS as CoquiTTS
    try:  # torch >= 2.6 requires allow-listing custom configs when weights_only=True
        from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig  # type: ignore
        from TTS.tts.models.xtts import XttsArgs  # type: ignore
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
    except Exception:
        pass
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise RuntimeError(
        "TTS is required for the Coqui XTTS backend. Install with `pip install TTS`"
    ) from exc


logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


def _pick_device(prefer: Optional[str] = None) -> str:
    if prefer and prefer not in {"", "auto"}:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(eq=True)
class XTTSConfig:
    model_id: str = _DEFAULT_MODEL_ID
    device: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = "en"
    speaker_wav_path: Optional[str] = None
    sample_rate: int = 24_000
    max_chars: int = 400
    join_silence_ms: int = 80


class XTTSBackend:
    """Adapter around Coqui XTTS."""

    def __init__(self, cfg: XTTSConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self._model_lock = threading.Lock()
        self._model = self._load_model()
        self.sample_rate = getattr(getattr(self._model, "synthesizer", None), "output_sample_rate", cfg.sample_rate)

    def _load_model(self):
        logger.info(
            "loading XTTS backend",
            extra={"model_id": self.cfg.model_id, "device": self.device},
        )
        model = CoquiTTS(model_name=self.cfg.model_id, progress_bar=False)
        target = self.device
        if target == "mps":  # torch.compile not yet supported; fall back gracefully
            try:
                model.to("mps")
            except Exception:
                logger.warning("XTTS MPS placement failed; falling back to CPU")
                target = "cpu"
        if target in {"cuda", "cpu"}:
            with contextlib.suppress(Exception):
                model.to(target)
        return model

    def update_config(self, cfg: XTTSConfig) -> None:
        """Update runtime configuration without reloading the model."""
        self.cfg = replace(
            self.cfg,
            speaker=cfg.speaker,
            language=cfg.language,
            speaker_wav_path=cfg.speaker_wav_path,
            sample_rate=cfg.sample_rate,
            max_chars=cfg.max_chars,
            join_silence_ms=cfg.join_silence_ms,
        )

    def _chunk_text(self, text: str, *, max_chars: Optional[int] = None) -> list[str]:
        limit = max(int(max_chars or self.cfg.max_chars), 1)
        sentences = [s.strip() for s in text.strip().split(". ") if s.strip()]
        chunks: list[str] = []
        buffer = ""

        def flush() -> None:
            nonlocal buffer
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = ""

        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= limit:
                buffer = candidate
            else:
                flush()
                if len(sentence) <= limit:
                    buffer = sentence
                else:
                    for start in range(0, len(sentence), limit):
                        part = sentence[start : start + limit].strip()
                        if part:
                            chunks.append(part)
        flush()
        return chunks or [text]

    def synthesize(self, text: str, **style: Any) -> bytes:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = text.strip()
        if not text:
            raise ValueError("text is empty")

        speaker = style.get("speaker") or self.cfg.speaker
        language = style.get("language") or self.cfg.language
        speaker_wav = style.get("speaker_wav_path") or self.cfg.speaker_wav_path

        chunks = self._chunk_text(text)
        wavs: list[torch.Tensor] = []

        logger.debug(
            "xtts synthesize start",
            extra={
                "text_len": len(text),
                "chunks": len(chunks),
                "speaker": speaker,
                "language": language,
                "speaker_wav": bool(speaker_wav),
            },
        )

        with self._model_lock:
            for segment in chunks:
                if not segment:
                    continue
                audio = self._model.tts(
                    segment,
                    speaker=speaker,
                    language=language,
                    speaker_wav=speaker_wav,
                )
                tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
                if tensor.dim() > 1:
                    tensor = tensor.squeeze()
                wavs.append(torch.clamp(tensor, -1.0, 1.0))

        if not wavs:
            raise RuntimeError("XTTS returned no audio segments")

        if len(wavs) == 1:
            final = wavs[0]
        else:
            gap_len = int(self.sample_rate * max(self.cfg.join_silence_ms, 0) / 1000)
            gap = torch.zeros(gap_len, dtype=torch.float32) if gap_len else None
            pieces: list[torch.Tensor] = []
            for idx, wav in enumerate(wavs):
                pieces.append(wav)
                if gap is not None and idx < len(wavs) - 1:
                    pieces.append(gap)
            final = torch.cat(pieces) if pieces else wavs[0]

        buffer = io.BytesIO()
        ta.save(buffer, final.unsqueeze(0), self.sample_rate, format="WAV")
        logger.debug(
            "xtts synthesize complete",
            extra={
                "text_len": len(text),
                "segments": len(wavs),
                "samples": int(final.numel()),
                "duration_ms": int(final.numel() / max(self.sample_rate, 1) * 1000),
            },
        )
        return buffer.getvalue()

    def stream_chunks(
        self,
        text: str,
        *,
        speaker: Optional[str],
        language: Optional[str],
        speaker_wav_path: Optional[str],
        max_chars: int,
        join_silence_ms: int,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        chunks = self._chunk_text(text, max_chars=max_chars)

        gap_bytes = b""
        if join_silence_ms > 0:
            silence_samples = int(self.sample_rate * join_silence_ms / 1000)
            if silence_samples > 0:
                gap_bytes = np.zeros(silence_samples, dtype=np.int16).tobytes()

        with self._model_lock:
            for idx, segment in enumerate(chunks):
                if not segment:
                    continue
                audio = self._model.tts(
                    segment,
                    speaker=speaker or self.cfg.speaker,
                    language=language or self.cfg.language,
                    speaker_wav=speaker_wav_path or self.cfg.speaker_wav_path,
                )
                tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
                if tensor.dim() > 1:
                    tensor = tensor.squeeze()
                pcm_bytes = _to_pcm16_bytes(tensor)
                if idx > 0 and gap_bytes:
                    yield gap_bytes
                yield pcm_bytes

    def cleanup(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        if self.device == "mps" and hasattr(torch, "mps"):
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16, data_bytes: Optional[int] = None) -> bytes:
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = data_bytes if data_bytes is not None else 0xFFFFFFFF
    riff_size = 36 + data_size
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(int.to_bytes(riff_size & 0xFFFFFFFF, 4, "little"))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(int.to_bytes(16, 4, "little"))
    buf.write(int.to_bytes(1, 2, "little"))
    buf.write(int.to_bytes(channels, 2, "little"))
    buf.write(int.to_bytes(sample_rate, 4, "little"))
    buf.write(int.to_bytes(byte_rate, 4, "little"))
    buf.write(int.to_bytes(block_align, 2, "little"))
    buf.write(int.to_bytes(bits_per_sample, 2, "little"))
    buf.write(b"data")
    buf.write(int.to_bytes(data_size & 0xFFFFFFFF, 4, "little"))
    return buf.getvalue()


def _to_pcm16_bytes(wave: torch.Tensor) -> bytes:
    tensor = wave.detach().to("cpu", dtype=torch.float32)
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor * 32767.0).to(torch.int16).numpy().tobytes()


def config_from_dict(raw: Mapping[str, Any] | None) -> XTTSConfig:
    raw = raw or {}
    device = raw.get("device")
    if isinstance(device, str) and device.lower() == "auto":
        device = None
    return XTTSConfig(
        model_id=str(raw.get("model_id", _DEFAULT_MODEL_ID) or _DEFAULT_MODEL_ID),
        device=device or None,
        speaker=(raw.get("speaker") or None),
        language=str(raw.get("language", "en") or "en"),
        speaker_wav_path=(raw.get("speaker_wav_path") or None),
        sample_rate=int(raw.get("sample_rate", 24_000) or 24_000),
        max_chars=int(raw.get("max_chars", 400) or 400),
        join_silence_ms=int(raw.get("join_silence_ms", 80) or 80),
    )


_backend_lock = threading.Lock()
_backend: Optional[XTTSBackend] = None


def get_backend(cfg: XTTSConfig) -> XTTSBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = XTTSBackend(cfg)
            return _backend

        if (
            _backend.cfg.model_id != cfg.model_id
            or (_backend.cfg.device or "auto") != (cfg.device or "auto")
        ):
            _backend = XTTSBackend(cfg)
            return _backend

        _backend.update_config(cfg)
        return _backend


async def synthesize_base64(
    text: str,
    cfg: XTTSConfig,
    style: Optional[Dict[str, Any]] = None,
) -> str:
    backend = get_backend(cfg)
    style = style or {}
    wav_bytes = await asyncio.to_thread(backend.synthesize, text, **style)
    try:
        return base64.b64encode(wav_bytes).decode("utf-8")
    finally:
        backend.cleanup()


async def synthesize_wav(
    text: str,
    cfg: XTTSConfig,
    style: Optional[Dict[str, Any]] = None,
) -> bytes:
    backend = get_backend(cfg)
    style = style or {}
    wav_bytes = await asyncio.to_thread(backend.synthesize, text, **style)
    try:
        return wav_bytes
    finally:
        backend.cleanup()
