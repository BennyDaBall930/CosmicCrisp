import logging
import time

from flask import Response, stream_with_context

from python.helpers.api import ApiHandler, Request
from python.helpers import settings
from python.helpers.chatterbox_tts import (
    ChatterboxConfig,
    config_from_dict,
    get_backend,
    wav_header,
)

logger = logging.getLogger(__name__)

_backend = None


def _chatterbox_settings() -> ChatterboxConfig:
    all_settings = settings.get_settings()
    tts_settings = all_settings.get("tts")
    if not isinstance(tts_settings, dict):
        tts_settings = settings.get_default_settings()["tts"]
    chatterbox = tts_settings.get("chatterbox")
    if not isinstance(chatterbox, dict):
        chatterbox = settings.get_default_settings()["tts"]["chatterbox"]
    return config_from_dict(chatterbox)


def _ensure_backend():
    global _backend
    cfg = _chatterbox_settings()
    if _backend is None or _backend.cfg != cfg:
        logger.info(
            "synthesize_stream backend refresh",
            extra={
                "cache_hit": _backend is not None,
                "device": cfg.device,
                "multilingual": cfg.multilingual,
            },
        )
        _backend = get_backend(cfg)
    return _backend


class SynthesizeStream(ApiHandler):
    async def process(self, input: dict, request: Request) -> Response:
        backend = _ensure_backend()
        text = input.get("text", "")
        style = input.get("style", {}) or {}
        target_chars = int(input.get("target_chars", backend.cfg.max_chars))
        join_ms = int(input.get("join_silence_ms", backend.cfg.join_silence_ms))
        first_chunk_chars = int(input.get("first_chunk_chars", 0) or 0)

        if not isinstance(text, str) or not text.strip():
            return Response("text is empty", status=400, mimetype="text/plain")

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
                "synthesize_stream request text_len=%d target_chars=%d join_ms=%d "
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
