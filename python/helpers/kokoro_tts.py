"""Kokoro TTS integration for Apple Zero."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import sys
import types
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

# PREVENT espeakng_loader from loading problematic CI paths
# Monkey patch BEFORE any kokoro/phonemizer imports can trigger it
if 'espeakng_loader' not in sys.modules:
    import os
    # First set environment to override any CI paths
    os.environ["ESPEAKNG_DATA_PATH"] = "/opt/homebrew/share/espeak-ng-data"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = "/opt/homebrew/bin/espeak-ng"

    # Create mock module to intercept espeakng_loader imports
    mock_espeakng_loader = types.ModuleType('espeakng_loader')
    mock_espeakng_loader.get_data_path = lambda: "/opt/homebrew/share/espeak-ng-data"
    mock_espeakng_loader.get_library_path = lambda: "/opt/homebrew/lib/libespeak-ng.dylib"
    sys.modules['espeakng_loader'] = mock_espeakng_loader


def _ensure_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        # CRITICAL: Force correct espeak paths BEFORE any kokoro/phonemizer imports
        # This must be done first, before any imports trigger the CI path lookup
        import os
        os.environ.setdefault("ESPEAKNG_DATA_PATH", "/opt/homebrew/share/espeak-ng-data")
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
        os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/bin/espeak-ng")

        # Monkey patch the problematic paths at the sys.modules level BEFORE any imports
        import sys
        if 'espeakng_loader' not in sys.modules:
            import types
            mock_module = types.ModuleType('espeakng_loader')
            mock_module.get_data_path = staticmethod(lambda: "/opt/homebrew/share/espeak-ng-data")
            mock_module.get_library_path = staticmethod(lambda: "/opt/homebrew/lib/libespeak-ng.dylib")
            # Add to sys.modules BEFORE phonemizer gets imported
            sys.modules['espeakng_loader'] = mock_module

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
                import os
                # Try multiple espeak-ng data path locations
                possible_data_paths = [
                    "/opt/homebrew/share/espeak-ng-data",
                    "/opt/homebrew/Cellar/espeak-ng/1.52.0/share/espeak-ng-data",
                    "/usr/local/share/espeak-ng-data",
                    "/usr/share/espeak-ng-data",
                    os.environ.get("ESPEAKNG_DATA_PATH"),  # From macOS run script
                ]

                # Find the first valid espeak-ng data path
                data_path = None
                for path in possible_data_paths:
                    if path and os.path.exists(path):
                        data_path = path
                        break

                if data_path:
                    os.environ.setdefault("ESPEAKNG_DATA_PATH", data_path)
                    # Also try setting PHONEMIZER_ESPEAK_LIBRARY if not set
                    if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
                        # Try common espeak-ng library locations
                        possible_lib_paths = [
                            "/opt/homebrew/lib/libespeak-ng.dylib",
                            "/usr/local/lib/libespeak-ng.dylib",
                            "/usr/lib/libespeak-ng.dylib",
                        ]
                        for lib_path in possible_lib_paths:
                            if os.path.exists(lib_path):
                                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
                                break

                if not hasattr(EspeakWrapper, "set_data_path"):
                    def _shim_set_data_path(path):
                        os.environ["ESPEAKNG_DATA_PATH"] = str(path)
                    EspeakWrapper.set_data_path = staticmethod(_shim_set_data_path)  # type: ignore[attr-defined]

                # Patch data path resolver to use our validated path - more aggressive patching
                try:
                    import pathlib
                    def _patched_fetch_version_and_path(self):  # type: ignore[no-redef]
                        try:
                            # Force our validated data path
                            if hasattr(self, '_espeak') and self._espeak:
                                try:
                                    # Set data path directly before getting info
                                    data_path = "/opt/homebrew/share/espeak-ng-data"  # Direct brew path
                                    os.environ["ESPEAKNG_DATA_PATH"] = data_path
                                    self._data_path = pathlib.Path(data_path)
                                    # Try to get version from espeak
                                    version, _ = self._espeak.info()
                                    version = version.decode().strip().split(' ')[0].replace('-dev', '')
                                    self._version = tuple(int(v) for v in version.split('.'))
                                except Exception:
                                    # If that fails, set basic fallback
                                    self._version = (1, 52, 0)
                                return
                        except Exception:
                            pass
                        # Ultimate fallback
                        self._version = (1, 52, 0)

                    # Try to patch the method
                    if hasattr(EspeakWrapper, '_fetch_version_and_path'):
                        EspeakWrapper._fetch_version_and_path = _patched_fetch_version_and_path
                    else:
                        # Alternative patching approach
                        setattr(EspeakWrapper, '_fetch_version_and_path', _patched_fetch_version_and_path)
                except Exception as e:
                    logger.warning(f"Failed to patch EspeakWrapper: {e}")
                    pass
            except Exception as e:
                # If espeak setup fails completely, log but continue - Kokoro might still work
                logger.warning(f"Espeak setup failed, Kokoro may not work: {e}")
                pass

            # Force the correct espeak data path before importing Kokoro
            # This is critical to prevent the CI path error
            os.environ["ESPEAKNG_DATA_PATH"] = "/opt/homebrew/share/espeak-ng-data"
            if os.path.exists("/opt/homebrew/lib/libespeak-ng.dylib"):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"

            # Try to monkey patch the data path lookup at the module level
            try:
                import espeakng_loader  # This is causing the issue
                # Replace the problematic functions with our own
                espeakng_loader.get_data_path = lambda: "/opt/homebrew/share/espeak-ng-data"
                espeakng_loader.get_library_path = lambda: "/opt/homebrew/lib/libespeak-ng.dylib"
                logger.info("Successfully patched espeakng_loader paths")
            except Exception as e:
                logger.warning(f"Could not patch espeakng_loader: {e}")

            from kokoro import KPipeline
            pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            _pipeline = pipeline
            PrintStyle(level=logging.DEBUG).print("Kokoro TTS model loaded")
            return _pipeline
        except SystemExit as exc:
            logger.error("Kokoro initialization triggered SystemExit", exc_info=exc)
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to import kokoro", exc_info=exc)
            # Surface the underlying cause to aid debugging in preload/logs
            raise RuntimeError(str(exc)) from exc
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
