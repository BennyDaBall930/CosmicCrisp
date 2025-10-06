import logging
import time

import torch
from flask import Response, stream_with_context

from python.helpers.api import ApiHandler, Request
from python.helpers import settings
from typing import Any

logger = logging.getLogger(__name__)


def _tts_settings() -> dict:
    all_settings = settings.get_settings()
    tts_settings = all_settings.get("tts")
    if not isinstance(tts_settings, dict):
        tts_settings = settings.get_default_settings()["tts"]
    return tts_settings


def _chatterbox_settings(tts_settings: dict):
    from python.helpers.chatterbox_tts import config_from_dict as _cfg
    chatterbox = tts_settings.get("chatterbox")
    if not isinstance(chatterbox, dict):
        chatterbox = settings.get_default_settings()["tts"]["chatterbox"]
    return _cfg(chatterbox)


def _xtts_settings(tts_settings: dict):
    from python.helpers.xtts_tts import config_from_dict as _cfg
    xtts = tts_settings.get("xtts")
    if not isinstance(xtts, dict):
        xtts = settings.get_default_settings()["tts"]["xtts"]
    return _cfg(xtts)


def _gap_bytes(sample_rate: int, join_ms: int) -> bytes:
    if join_ms <= 0:
        return b""
    samples = int(sample_rate * join_ms / 1000)
    if samples <= 0:
        return b""
    silence = torch.zeros(samples, dtype=torch.float32)
    return (silence * 32767.0).to(torch.int16).numpy().tobytes()


class SynthesizeStream(ApiHandler):
    async def process(self, input: dict, request: Request) -> Response:
        text = input.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return Response("text is empty", status=400, mimetype="text/plain")

        tts_settings = _tts_settings()
        engine = str(tts_settings.get("engine", "chatterbox")).lower()
        style = input.get("style", {}) or {}
        filtered_style = {k: v for k, v in style.items() if v is not None}

        if engine == "browser":
            return Response("browser TTS engine does not support server streaming", status=400, mimetype="text/plain")

        if engine == "xtts":
            try:
                return self._stream_xtts(text, filtered_style, input, tts_settings)
            except Exception as e:
                return Response(str(e), status=500, mimetype="text/plain")

        try:
            return self._stream_chatterbox(text, filtered_style, input, tts_settings)
        except Exception as e:
            return Response(str(e), status=500, mimetype="text/plain")

    def _stream_chatterbox(self, text: str, style: dict, payload: dict, tts_settings: dict) -> Response:
        try:
            from python.helpers.chatterbox_tts import get_backend as get_chatterbox_backend, wav_header
        except Exception as e:
            raise RuntimeError(f"Chatterbox unavailable: {e}")
        backend = get_chatterbox_backend(_chatterbox_settings(tts_settings))
        target_chars = int(payload.get("target_chars", backend.cfg.max_chars))
        join_ms = int(payload.get("join_silence_ms", backend.cfg.join_silence_ms))
        first_chunk_chars = int(payload.get("first_chunk_chars", 0) or 0)

        exaggeration = float(style.get("exaggeration", backend.cfg.exaggeration))
        cfg = float(style.get("cfg", backend.cfg.cfg))
        audio_prompt = style.get("audio_prompt_path") or backend.cfg.audio_prompt_path
        language_id = style.get("language_id") or backend.cfg.language_id

        sr = getattr(backend, "sr", backend.cfg.sample_rate)

        start_time = time.perf_counter()
        chunk_count = 0
        bytes_sent = 0
        logger.info(
            (
                "synthesize_stream request (chatterbox) text_len=%d target_chars=%d join_ms=%d "
                "first_chunk=%d language=%s"
            ),
            len(text),
            target_chars,
            join_ms,
            first_chunk_chars,
            language_id or "-",
        )

        def generator():
            nonlocal chunk_count, bytes_sent
            try:
                header = wav_header(sr, channels=1, bits_per_sample=16, data_bytes=None)
                bytes_sent += len(header)
                logger.debug(
                    "synthesize_stream header bytes=%d sample_rate=%d",
                    len(header),
                    sr,
                )
                yield header
                for block in backend.stream_chunks(
                    text,
                    exaggeration=exaggeration,
                    cfg=cfg,
                    audio_prompt_path=audio_prompt,
                    language_id=language_id,
                    target_chars=target_chars,
                    join_silence_ms=join_ms,
                    first_chunk_chars=first_chunk_chars,
                ):
                    if not block:
                        continue
                    chunk_count += 1
                    bytes_sent += len(block)
                    logger.debug(
                        "synthesize_stream chunk index=%d bytes=%d total_bytes=%d",
                        chunk_count,
                        len(block),
                        bytes_sent,
                    )
                    yield block
            except Exception:
                logger.exception("synthesize_stream generator error")
                raise
            finally:
                backend.cleanup()
                total_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    (
                        "synthesize_stream complete text_len=%d chunks=%d bytes=%d duration_ms=%d"
                    ),
                    len(text),
                    chunk_count,
                    bytes_sent,
                    total_ms,
                )

        return Response(stream_with_context(generator()), mimetype="audio/wav")

    def _stream_xtts(self, text: str, style: dict, payload: dict, tts_settings: dict) -> Response:
        try:
            from python.helpers.xtts_tts import get_backend as get_xtts_backend, wav_header as xtts_wav_header
        except Exception as e:
            raise RuntimeError(f"XTTS unavailable: {e}")
        backend = get_xtts_backend(_xtts_settings(tts_settings))
        join_ms = int(payload.get("join_silence_ms", backend.cfg.join_silence_ms))
        target_chars = int(payload.get("target_chars", backend.cfg.max_chars))

        speaker = style.get("speaker") or backend.cfg.speaker
        language = style.get("language") or backend.cfg.language
        speaker_wav_path = style.get("speaker_wav_path") or backend.cfg.speaker_wav_path

        sr = getattr(backend, "sample_rate", backend.cfg.sample_rate)

        start_time = time.perf_counter()
        chunk_count = 0
        bytes_sent = 0
        logger.info(
            "synthesize_stream request (xtts) text_len=%d join_ms=%d",
            len(text),
            join_ms,
        )

        def generator():
            nonlocal chunk_count, bytes_sent
            try:
                header = xtts_wav_header(sr, channels=1, bits_per_sample=16, data_bytes=None)
                bytes_sent += len(header)
                yield header
                for block in backend.stream_chunks(
                    text,
                    speaker=speaker,
                    language=language,
                    speaker_wav_path=speaker_wav_path,
                    max_chars=target_chars,
                    join_silence_ms=join_ms,
                ):
                    if not block:
                        continue
                    chunk_count += 1
                    bytes_sent += len(block)
                    logger.debug(
                        "synthesize_stream chunk index=%d bytes=%d total_bytes=%d",
                        chunk_count,
                        len(block),
                        bytes_sent,
                    )
                    yield block
            except Exception:
                logger.exception("synthesize_stream generator error")
                raise
            finally:
                backend.cleanup()
                total_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    (
                        "synthesize_stream complete (xtts) text_len=%d chunks=%d bytes=%d duration_ms=%d"
                    ),
                    len(text),
                    chunk_count,
                    bytes_sent,
                    total_ms,
                )

        return Response(stream_with_context(generator()), mimetype="audio/wav")
