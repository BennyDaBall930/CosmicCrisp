"""Kokoro TTS integration for Apple Zero."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import sys
from typing import Iterable

import numpy as np
import soundfile as sf
from python.helpers.print_style import PrintStyle

logger = logging.getLogger(__name__)

_pipeline = None
_lock = threading.Lock()
_DEFAULT_VOICE = "am_puck,am_onyx"
_DEFAULT_SPEED = 1.1
_DEFAULT_SAMPLE_RATE = 24_000


def _ensure_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        original_exit = sys.exit

        def _exit_guard(code: object = None):  # type: ignore[override]
            raise RuntimeError(f"kokoro initialization attempted sys.exit({code})")

        try:
            sys.exit = _exit_guard
            # Work around misaki/espeak expecting an older phonemizer API.
            # Some releases call EspeakWrapper.set_data_path(...), which does not
            # exist in newer phonemizer versions. Provide a no-op shim so import succeeds.
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
                import espeakng_loader  # type: ignore
                import os
                # Ensure library and data path env hints are present before kokoro/misaki import
                os.environ.setdefault(
                    "PHONEMIZER_ESPEAK_LIBRARY", str(espeakng_loader.get_library_path())
                )
                os.environ.setdefault(
                    "ESPEAKNG_DATA_PATH", str(espeakng_loader.get_data_path())
                )
                if not hasattr(EspeakWrapper, "set_data_path"):
                    def _shim_set_data_path(path):
                        os.environ["ESPEAKNG_DATA_PATH"] = str(path)
                    EspeakWrapper.set_data_path = staticmethod(_shim_set_data_path)  # type: ignore[attr-defined]

                # Patch data path resolver to prefer ESPEAKNG_DATA_PATH or loader path
                try:
                    import pathlib
                    def _patched_fetch_version_and_path(self):  # type: ignore[no-redef]
                        try:
                            version, _ = self._espeak.info()
                            dp = os.environ.get("ESPEAKNG_DATA_PATH", str(espeakng_loader.get_data_path()))
                            self._data_path = pathlib.Path(dp)
                            version = version.decode().strip().split(' ')[0].replace('-dev', '')
                            self._version = tuple(int(v) for v in version.split('.'))
                        except Exception:
                            self._data_path = pathlib.Path(espeakng_loader.get_data_path())
                            self._version = (1, 52, 0)
                    EspeakWrapper._fetch_version_and_path = _patched_fetch_version_and_path  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass
            from kokoro import KPipeline
        except SystemExit as exc:
            logger.error("Kokoro initialization triggered SystemExit", exc_info=exc)
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to import kokoro", exc_info=exc)
            # Surface the underlying cause to aid debugging in preload/logs
            raise RuntimeError(str(exc)) from exc
            pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            _pipeline = pipeline
            PrintStyle(level=logging.DEBUG).print("Kokoro TTS model loaded")
            return _pipeline
        finally:
            sys.exit = original_exit


def synthesize_sentences_sync(
    sentences: Iterable[str],
    voice: str | None = None,
    speed: float | None = None,
    sample_rate: int | None = None,
) -> str:
    pipeline = _ensure_pipeline()
    effective_voice = voice or _DEFAULT_VOICE
    effective_speed = float(speed if speed is not None else _DEFAULT_SPEED)
    sr = int(sample_rate if sample_rate is not None else _DEFAULT_SAMPLE_RATE)

    audio_chunks: list[np.ndarray] = []
    for sentence in sentences:
        text = (sentence or "").strip()
        if not text:
            continue
        try:
            segments = pipeline(text, voice=effective_voice, speed=effective_speed)
        except Exception as exc:
            logger.exception("Kokoro synthesis failed", extra={"text_len": len(text)})
            raise RuntimeError(f"Kokoro TTS synthesis failed: {exc}") from exc
        for segment in segments:
            tensor = getattr(segment, "audio", None)
            if tensor is None:
                continue
            array = tensor.detach().cpu().numpy()  # type: ignore[attr-defined]
            if array.ndim > 1:
                array = np.squeeze(array, axis=0)
            audio_chunks.append(array.astype(np.float32))

    if not audio_chunks:
        raise RuntimeError("Kokoro TTS returned no audio segments")

    combined = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    sf.write(buffer, combined, sr, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def synthesize_sentences(
    sentences: Iterable[str],
    voice: str | None = None,
    speed: float | None = None,
    sample_rate: int | None = None,
) -> str:
    return await asyncio.to_thread(
        synthesize_sentences_sync, sentences, voice, speed, sample_rate
    )


async def preload() -> None:
    try:
        await asyncio.to_thread(_ensure_pipeline)
    except SystemExit as exc:
        PrintStyle().error(f"Kokoro preload skipped: {exc}")
    except Exception as exc:
        PrintStyle().error(f"Kokoro preload skipped: {exc}")
