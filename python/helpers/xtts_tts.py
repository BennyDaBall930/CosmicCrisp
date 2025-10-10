"""Coqui XTTS text-to-speech backend integration with sidecar fallback.

If the local Coqui TTS import fails (e.g., Python 3.12 incompatibility),
this module will transparently fall back to calling a local sidecar
running under Python 3.11 via HTTP. Configure the sidecar URL via
`TTS_SIDECAR_URL` (default: http://127.0.0.1:7055).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import threading
import time
import psutil
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional
import os
import json
import urllib.request
import urllib.error

import numpy as np
import torch
import torchaudio as ta

COQUI_TTS_AVAILABLE = False
try:
    from TTS.api import TTS as CoquiTTS  # type: ignore

    try:  # torch >= 2.6 requires allow-listing custom configs when weights_only=True
        from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig  # type: ignore
        from TTS.tts.models.xtts import XttsArgs  # type: ignore
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
    except Exception:
        pass

    COQUI_TTS_AVAILABLE = True
except Exception:
    COQUI_TTS_AVAILABLE = False


logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


def _get_resource_usage() -> Dict[str, float]:
    """Get current system resource usage statistics."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        # Get GPU memory if available
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # MPS memory tracking is limited, but we can try
                gpu_memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024) if hasattr(torch, 'mps') else 0.0
            except Exception:
                pass

        return {
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "gpu_memory_mb": gpu_memory_mb,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.warning(f"Failed to collect resource usage: {e}")
        return {
            "memory_rss_mb": 0.0,
            "memory_vms_mb": 0.0,
            "cpu_percent": 0.0,
            "gpu_memory_mb": 0.0,
            "timestamp": time.time(),
        }


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


def _post_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"sidecar HTTP {resp.status}")
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid sidecar response: {e}")


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
        if not COQUI_TTS_AVAILABLE:
            raise RuntimeError("Coqui TTS is not available in this environment")
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

        start_resources = _get_resource_usage()

        logger.debug(
            "xtts synthesize start",
            extra={
                "text_len": len(text),
                "chunks": len(chunks),
                "speaker": speaker,
                "language": language,
                "speaker_wav": bool(speaker_wav),
                "initial_memory_mb": start_resources["memory_rss_mb"],
                "initial_cpu_percent": start_resources["cpu_percent"],
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


class SidecarXTTSBackend:
    """HTTP client backend for XTTS via local sidecar."""

    def __init__(self, cfg: "XTTSConfig"):
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self._base_url = os.environ.get("TTS_SIDECAR_URL", "http://127.0.0.1:7055")
        self._attempted_autostart = False
        # Do a quick non-blocking health probe; full autostart handled on first request
        try:
            self._health_check(timeout=0.5)
        except Exception:
            pass

    def _repo_root(self) -> str:
        try:
            from pathlib import Path

            here = Path(__file__).resolve()
            for p in [*here.parents]:
                if (p / "run.sh").exists() and (p / "sidecar" / "app.py").exists():
                    return str(p)
        except Exception:
            pass
        return os.getcwd()

    def _health_check(self, timeout: float = 1.5) -> bool:
        url = f"{self._base_url.rstrip('/')}/healthz"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200

    def _ensure_running(self) -> None:
        try:
            if self._health_check(timeout=1.0):
                return
        except Exception:
            pass

        if os.environ.get("A0_DISABLE_SIDECAR_AUTOSTART") in {"1", "true", "TRUE", "True"}:
            raise RuntimeError(
                "TTS sidecar is not running at "
                f"{self._base_url}. Set TTS engine to 'browser' or start the sidecar: "
                "./setup.sh && ./run.sh"
            )

        if self._attempted_autostart:
            # Avoid repeated spawns
            raise RuntimeError(
                f"Failed to reach TTS sidecar at {self._base_url} after autostart"
            )

        self._attempted_autostart = True

        # Try to start the sidecar in background using the project script
        root = self._repo_root()
        run_sh = os.path.join(root, "run.sh")
        if not os.path.exists(run_sh):
            raise RuntimeError(
                "TTS sidecar not reachable, and run.sh not found to auto-start it"
            )
        try:
            import subprocess, sys

            # Launch detached to persist beyond current request
            subprocess.Popen(
                ["bash", "-lc", "./run.sh"],
                cwd=root,
                stdout=open(os.path.join(root, "logs", "tts_sidecar.out"), "ab"),
                stderr=open(os.path.join(root, "logs", "tts_sidecar.err"), "ab"),
                start_new_session=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to auto-start sidecar: {e}")

        # Wait for health to go green
        import time as _time

        deadline = _time.time() + 90.0
        last_err: Optional[Exception] = None
        while _time.time() < deadline:
            try:
                if self._health_check(timeout=1.5):
                    return
            except Exception as he:
                last_err = he
            _time.sleep(1.0)
        msg = (
            f"Sidecar failed to become healthy at {self._base_url} within timeout. "
            "Check logs/tts_sidecar.err and run ./run.sh manually."
        )
        if last_err:
            msg += f" Last error: {last_err}"
        raise RuntimeError(msg)

    def update_config(self, cfg: "XTTSConfig") -> None:
        self.cfg = replace(
            self.cfg,
            speaker=cfg.speaker,
            language=cfg.language,
            speaker_wav_path=cfg.speaker_wav_path,
            sample_rate=cfg.sample_rate,
            max_chars=cfg.max_chars,
            join_silence_ms=cfg.join_silence_ms,
        )
        self.sample_rate = self.cfg.sample_rate

    def _request(self, text: str, speaker: Optional[str], language: Optional[str], speaker_wav_path: Optional[str]) -> bytes:
        payload = {
            "text": text,
            "language": language or self.cfg.language or "en",
            "speaker": speaker or self.cfg.speaker,
            "speaker_wav_path": speaker_wav_path or self.cfg.speaker_wav_path,
            "model_id": self.cfg.model_id,
            "device": self.cfg.device or None,
            "sample_rate": self.cfg.sample_rate,
            "max_chars": self.cfg.max_chars,
            "join_silence_ms": self.cfg.join_silence_ms,
        }
        url = f"{self._base_url.rstrip('/')}/api/xtts/synthesize"
        try:
            # Ensure service is alive (and auto-start if permitted)
            self._ensure_running()
            resp = _post_json(url, payload)
        except urllib.error.URLError as e:
            # Connection refused / no route to host etc.
            # Try one-time autostart if not yet attempted, then retry once
            try:
                self._ensure_running()
                resp = _post_json(url, payload)
            except Exception:
                raise RuntimeError(
                    "Unable to reach TTS sidecar at "
                    f"{self._base_url}: {e}. If the issue persists, run ./setup.sh and ./run.sh"
                )
        audio_b64 = resp.get("audio_b64")
        if not isinstance(audio_b64, str):
            raise RuntimeError("sidecar did not return audio_b64")
        try:
            return base64.b64decode(audio_b64)
        except Exception as e:
            raise RuntimeError(f"failed to decode sidecar audio: {e}")

    def synthesize(self, text: str, **style: Any) -> bytes:
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        speaker = style.get("speaker") or self.cfg.speaker
        language = style.get("language") or self.cfg.language
        speaker_wav = style.get("speaker_wav_path") or self.cfg.speaker_wav_path
        return self._request(text, speaker, language, speaker_wav)

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
        # Implement client-side chunking and concatenation; request WAV per chunk and return PCM16 blocks
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        chunks = self._chunk_text(text, max_chars=max_chars)
        gap = b""
        if join_silence_ms > 0:
            silence_samples = int(self.sample_rate * join_silence_ms / 1000)
            if silence_samples > 0:
                gap = np.zeros(silence_samples, dtype=np.int16).tobytes()

        for idx, segment in enumerate(chunks):
            wav_bytes = self._request(segment, speaker, language, speaker_wav_path)
            # Strip WAV header and yield PCM16
            pcm = self._wav_to_pcm16(wav_bytes)
            if idx > 0 and gap:
                yield gap
            yield pcm

    def _chunk_text(self, text: str, *, max_chars: int) -> list[str]:
        limit = max(int(max_chars or self.cfg.max_chars), 1)
        sentences = [s.strip() for s in text.strip().split(". ") if s.strip()]
        chunks: list[str] = []
        buf = ""
        for s in sentences:
            cand = (f"{buf} {s}").strip() if buf else s
            if len(cand) <= limit:
                buf = cand
            else:
                if buf:
                    chunks.append(buf)
                if len(s) <= limit:
                    buf = s
                else:
                    for i in range(0, len(s), limit):
                        seg = s[i : i + limit].strip()
                        if seg:
                            chunks.append(seg)
                    buf = ""
        if buf:
            chunks.append(buf)
        return chunks or [text]

    def _wav_to_pcm16(self, wav_bytes: bytes) -> bytes:
        # Minimal WAV header parsing; assume mono PCM16
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore

        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="int16")
        if sr != self.sample_rate:
            # Resample if needed
            try:
                import librosa  # type: ignore

                data_f = librosa.resample(data.astype(np.float32) / 32767.0, orig_sr=sr, target_sr=self.sample_rate)
                data = (np.clip(data_f, -1.0, 1.0) * 32767.0).astype(np.int16)
            except Exception:
                pass
        if data.ndim > 1:
            data = data[:, 0]
        return data.tobytes()

    def cleanup(self) -> None:
        return None


def wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16, data_bytes: Optional[int] = None) -> bytes:
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    # For streaming (data_bytes is None), use placeholder values
    if data_bytes is None:
        riff_size = 0xFFFFFFFF  # Unknown length for streaming
        data_size = 0xFFFFFFFF  # Unknown data size for streaming
    else:
        data_size = data_bytes
        riff_size = 36 + data_size  # RIFF chunk size = header size + data size

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(int.to_bytes(riff_size & 0xFFFFFFFF, 4, "little"))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(int.to_bytes(16, 4, "little"))  # Format chunk size
    buf.write(int.to_bytes(1, 2, "little"))   # Audio format (PCM)
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
_backend: Optional[Any] = None


def get_backend(cfg: XTTSConfig) -> Any:
    global _backend
    with _backend_lock:
        # Decide backend: native Coqui if available, else sidecar
        force_sidecar = os.environ.get("TTS_FORCE_SIDECAR") not in (None, "", "0", "false", "False")
        use_sidecar = force_sidecar or not COQUI_TTS_AVAILABLE
        BackendCls = SidecarXTTSBackend if use_sidecar else XTTSBackend
        if _backend is None or not isinstance(_backend, BackendCls):
            _backend = BackendCls(cfg)
            return _backend

        if (
            _backend.cfg.model_id != cfg.model_id
            or (_backend.cfg.device or "auto") != (cfg.device or "auto")
        ):
            _backend = BackendCls(cfg)
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
