"""Chatterbox text-to-speech backend integration."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import os
import platform
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _ensure_ffmpeg_lib_path() -> None:
    if platform.system() != "Darwin":
        return
    candidate = os.environ.get("COSMIC_FFMPEG_LIB_DIR") or "/opt/homebrew/opt/ffmpeg@6/lib"
    lib_dir = Path(candidate)
    if not lib_dir.is_dir():
        return

    def _prepend(var: str) -> None:
        existing = os.environ.get(var, "")
        paths = [p for p in existing.split(":") if p]
        lib_str = str(lib_dir)
        if lib_str in paths:
            return
        os.environ[var] = lib_str if not existing else f"{lib_str}:{existing}"

    _prepend("DYLD_LIBRARY_PATH")
    _prepend("DYLD_FALLBACK_LIBRARY_PATH")

    ldflags = os.environ.get("LDFLAGS", "")
    lib_flag = f"-L{lib_dir}"
    if lib_flag not in ldflags.split():
        os.environ["LDFLAGS"] = f"{ldflags} {lib_flag}".strip()


_ensure_ffmpeg_lib_path()


import numpy as np
import torch
import torchaudio as ta

try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError as exc:  # pragma: no cover - surfaced in runtime logs
    raise RuntimeError(
        "chatterbox-tts is required for the Chatterbox backend. Install with "
        "`pip install chatterbox-tts torchaudio`"
    ) from exc

try:
    torch.set_float32_matmul_precision("medium")
except Exception:  # pragma: no cover - torch < 2.0
    pass

try:
    torch.set_num_threads(1)
except Exception:  # pragma: no cover - restricted builds
    pass


logger = logging.getLogger(__name__)

def _pick_device(prefer: Optional[str] = None) -> str:
    """Resolve the most appropriate torch device."""
    if prefer and prefer != "auto":
        return prefer
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(eq=True)
class ChatterboxConfig:
    device: Optional[str] = None
    multilingual: bool = False
    sample_rate: int = 24_000
    exaggeration: float = 0.5
    cfg: float = 0.35
    audio_prompt_path: Optional[str] = None
    language_id: Optional[str] = "en"
    max_chars: int = 600
    join_silence_ms: int = 120


class ChatterboxBackend:
    """Thin adapter around the chatterbox-tts models."""

    def __init__(self, cfg: ChatterboxConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        if cfg.multilingual:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        else:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        if hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except Exception:
                pass
        self.sr = getattr(self.model, "sr", cfg.sample_rate)
        logger.info(
            "chatterbox backend loaded",
            extra={
                "device": self.device,
                "multilingual": cfg.multilingual,
                "sample_rate": self.sr,
            },
        )

    def _chunk_text(self, text: str, *, target_chars: Optional[int] = None) -> list[str]:
        if not text:
            return [""]

        max_chars = target_chars or self.cfg.max_chars
        target = max(80, min(max_chars, 400))
        hard_cap = max(target * 2, max_chars)
        sentences = re.split(r"(?<=[.!?…])\s+", text.strip())

        chunks: list[str] = []
        buffer = ""

        def flush():
            nonlocal buffer
            val = buffer.strip()
            if val:
                chunks.append(val)
            buffer = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > hard_cap:
                flush()
                for start in range(0, len(sentence), hard_cap):
                    part = sentence[start : start + hard_cap].strip()
                    if part:
                        chunks.append(part)
                continue

            tentative = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(tentative) <= target:
                buffer = tentative
            else:
                flush()
                buffer = sentence

        flush()

        return chunks or [text]

    def synthesize(self, text: str, **style: Any) -> bytes:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = text.strip()
        if not text:
            raise ValueError("text is empty")

        exaggeration = float(style.get("exaggeration", self.cfg.exaggeration))
        cfg = float(style.get("cfg", self.cfg.cfg))
        ref = style.get("audio_prompt_path", self.cfg.audio_prompt_path) or None
        lang = style.get("language_id", self.cfg.language_id) or None

        chunks = self._chunk_text(text)
        wavs: list[torch.Tensor] = []

        logger.debug(
            "chatterbox synthesize start",
            extra={
                "text_len": len(text),
                "chunks": len(chunks),
                "exaggeration": exaggeration,
                "cfg": cfg,
                "language": lang,
                "audio_prompt": bool(ref),
            },
        )

        ref_arg = ref
        if ref:
            try:
                self.model.prepare_conditionals(ref, exaggeration=exaggeration)
                ref_arg = None
            except Exception:
                ref_arg = ref

        for segment in chunks:
            if not segment:
                continue
            kwargs: Dict[str, Any] = {}
            if self.cfg.multilingual:
                if lang is not None:
                    kwargs["language_id"] = lang
            with torch.inference_mode():
                with _maybe_autocast(self.device):
                    audio = self.model.generate(
                        segment,
                        audio_prompt_path=ref_arg,
                        exaggeration=exaggeration,
                        cfg_weight=cfg,
                        **kwargs,
                    )
            if isinstance(audio, tuple) and len(audio) == 2:
                _, audio = audio  # some builds return (sr, audio_tensor)
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            audio = audio.detach().to("cpu", dtype=torch.float32)
            if audio.dim() > 1:
                audio = audio.squeeze()
            wavs.append(audio)

        if not wavs:
            raise RuntimeError("Chatterbox returned no audio segments")

        if len(wavs) == 1:
            final = wavs[0]
        else:
            gap_len = int(self.sr * max(self.cfg.join_silence_ms, 0) / 1000)
            gap = torch.zeros(gap_len, dtype=torch.float32) if gap_len else None
            pieces: list[torch.Tensor] = []
            for idx, wav in enumerate(wavs):
                pieces.append(wav)
                if gap is not None and idx < len(wavs) - 1:
                    pieces.append(gap)
            final = torch.cat(pieces) if pieces else wavs[0]

        buffer = io.BytesIO()
        ta.save(buffer, final.unsqueeze(0), self.sr, format="WAV")
        logger.debug(
            "chatterbox synthesize complete",
            extra={
                "text_len": len(text),
                "segments": len(wavs),
                "samples": int(final.numel()),
                "duration_ms": int(final.numel() / max(self.sr, 1) * 1000),
            },
        )
        return buffer.getvalue()

    def stream_chunks(
        self,
        text: str,
        *,
        exaggeration: float,
        cfg: float,
        audio_prompt_path: Optional[str],
        language_id: Optional[str],
        target_chars: int,
        join_silence_ms: int,
        first_chunk_chars: Optional[int] = None,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        segments = self._chunk_text(text, target_chars=target_chars)

        start_time = time.perf_counter()
        chunk_count = 0
        total_bytes = 0
        logger.debug(
            (
                "chatterbox stream start text_len=%d segments=%d target_chars=%d "
                "join_ms=%d first_chunk=%d language=%s"
            ),
            len(text),
            len(segments),
            target_chars,
            join_silence_ms,
            first_chunk_chars or 0,
            language_id or "-",
        )

        requested_first_chunk = max(int(first_chunk_chars or 0), 0)
        if requested_first_chunk > max(target_chars, 0) and len(segments) > 1:
            max_allowed = int(max(target_chars, 0) * 1.5) or requested_first_chunk
            requested_first_chunk = min(requested_first_chunk, max_allowed)
            max_merge_segments = 2  # keep early warm-up manageable
            first_segment = segments[0].strip()
            consumed = 1
            while (
                consumed < len(segments)
                and len(first_segment) < requested_first_chunk
                and consumed < max_merge_segments
            ):
                candidate = segments[consumed].strip()
                if candidate:
                    first_segment = f"{first_segment} {candidate}".strip()
                consumed += 1
            if consumed > 1 and first_segment:
                logger.debug(
                    "chatterbox stream merge merged_segments=%d merged_chars=%d",
                    consumed,
                    len(first_segment),
                )
                segments = [first_segment] + segments[consumed:]

        ref_arg = audio_prompt_path or self.cfg.audio_prompt_path
        if ref_arg:
            try:
                self.model.prepare_conditionals(ref_arg, exaggeration=exaggeration)
                ref_arg = None
            except Exception:
                pass

        gap_bytes = b""
        if join_silence_ms > 0:
            gap = torch.zeros(int(self.sr * join_silence_ms / 1000.0), dtype=torch.float32)
            gap_bytes = _to_pcm16_bytes(gap)

        for idx, segment in enumerate(segments):
            if not segment:
                continue
            chunk_start = time.perf_counter()
            kwargs: Dict[str, Any] = {}
            if self.cfg.multilingual and language_id:
                kwargs["language_id"] = language_id
            with torch.inference_mode():
                with _maybe_autocast(self.device):
                    audio = self.model.generate(
                        segment,
                        audio_prompt_path=ref_arg,
                        exaggeration=exaggeration,
                        cfg_weight=cfg,
                        **kwargs,
                    )
            if isinstance(audio, tuple) and len(audio) == 2:
                _, audio = audio
            if not isinstance(audio, torch.Tensor):
                audio = torch.from_numpy(np.asarray(audio))
            audio = audio.detach().to("cpu", dtype=torch.float32)
            if audio.dim() > 1:
                audio = audio.squeeze()
            sample_count = int(audio.numel())
            elapsed_ms = int((time.perf_counter() - chunk_start) * 1000)
            peak = float(audio.abs().max().item()) if sample_count else 0.0
            gain = 1.0
            if peak > 0.0:
                target_peak = 0.9
                if peak < target_peak:
                    gain = min(target_peak / peak, 12.0)
                    audio = torch.clamp(audio * gain, -1.0, 1.0)
                else:
                    audio = torch.clamp(audio, -1.0, 1.0)
            else:
                audio = torch.zeros_like(audio)
            peak = float(audio.abs().max().item()) if sample_count else 0.0
            rms = float(torch.sqrt(torch.mean(audio.square())).item()) if sample_count else 0.0
            pcm_bytes = _to_pcm16_bytes(audio)
            logger.debug(
                (
                    "chatterbox stream chunk index=%d chars=%d samples=%d bytes=%d "
                    "gen_ms=%d peak=%.4f rms=%.4f gain=%.2f"
                ),
                idx,
                len(segment),
                sample_count,
                len(pcm_bytes),
                elapsed_ms,
                peak,
                rms,
                gain,
            )
            chunk_count += 1
            total_bytes += len(pcm_bytes)
            yield pcm_bytes
            if gap_bytes and idx < len(segments) - 1:
                logger.debug(
                    "chatterbox stream gap index=%d gap_bytes=%d",
                    idx,
                    len(gap_bytes),
                )
                yield gap_bytes

        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(
            "chatterbox stream complete chunks=%d bytes=%d duration_ms=%d",
            chunk_count,
            total_bytes,
            total_ms,
        )

    def cleanup(self):
        if self.device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


_backend_lock = threading.Lock()
_backend: Optional[ChatterboxBackend] = None
_backend_cfg: Optional[ChatterboxConfig] = None


def get_backend(cfg: ChatterboxConfig) -> ChatterboxBackend:
    global _backend, _backend_cfg
    if _backend and _backend_cfg == cfg:
        return _backend
    with _backend_lock:
        if _backend and _backend_cfg == cfg:
            return _backend
        _backend = ChatterboxBackend(cfg)
        _backend_cfg = cfg
        return _backend


def _maybe_autocast(device: str):
    if device == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _to_pcm16_bytes(wave: torch.Tensor) -> bytes:
    tensor = wave.detach().to("cpu", dtype=torch.float32)
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor * 32767.0).to(torch.int16).numpy().tobytes()


def wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16, data_bytes: Optional[int] = None) -> bytes:
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = data_bytes if data_bytes is not None else 0xFFFFFFFF
    riff_size = 36 + data_size
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size & 0xFFFFFFFF))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size & 0xFFFFFFFF))
    return buf.getvalue()


def config_from_dict(raw: Mapping[str, Any] | None) -> ChatterboxConfig:
    raw = raw or {}
    device = raw.get("device")
    if isinstance(device, str) and device.lower() == "auto":
        device = None

    return ChatterboxConfig(
        device=device or None,
        multilingual=bool(raw.get("multilingual", False)),
        sample_rate=int(raw.get("sample_rate", 24_000) or 24_000),
        exaggeration=float(raw.get("exaggeration", 0.5) or 0.5),
        cfg=float(raw.get("cfg", 0.35) or 0.35),
        audio_prompt_path=raw.get("audio_prompt_path") or None,
        language_id=raw.get("language_id") or None,
        max_chars=int(raw.get("max_chars", 600) or 600),
        join_silence_ms=int(raw.get("join_silence_ms", 120) or 120),
    )


async def synthesize_base64(
    text: str,
    cfg: ChatterboxConfig,
    style: Optional[Dict[str, Any]] = None,
) -> str:
    backend = get_backend(cfg)
    style = style or {}
    filtered_style = {k: v for k, v in style.items() if v is not None}
    wav_bytes = await asyncio.to_thread(backend.synthesize, text, **filtered_style)
    try:
        return base64.b64encode(wav_bytes).decode("utf-8")
    finally:
        backend.cleanup()


async def synthesize_wav(
    text: str,
    cfg: ChatterboxConfig,
    style: Optional[Dict[str, Any]] = None,
) -> bytes:
    backend = get_backend(cfg)
    style = style or {}
    filtered_style = {k: v for k, v in style.items() if v is not None}
    wav_bytes = await asyncio.to_thread(backend.synthesize, text, **filtered_style)
    try:
        return wav_bytes
    finally:
        backend.cleanup()
