import base64
import io
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from typing import Any, Iterator

import numpy as np
import torch
from flask import Response, stream_with_context

from python.helpers.api import ApiHandler, Request
from python.helpers import settings

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None

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
    silence = np.zeros(samples, dtype=np.float32)
    return (silence * 32767.0).astype(np.int16).tobytes()


def _piper_vc_settings(tts_settings: dict):
    p = tts_settings.get("piper_vc")
    if not isinstance(p, dict):
        p = {}
    return {
        "piper_bin": p.get("piper_bin", "piper"),
        "piper_model": p.get("piper_model", os.environ.get("PIPER_MODEL", "")),
        "sample_rate": int(p.get("sample_rate", 22050)),
        "chunk_chars": int(p.get("max_chars", 280)),
        "join_silence_ms": int(p.get("join_silence_ms", 80)),
        "target_voice_wav": p.get("target_voice_wav") or "",
    }


def _best_device() -> str:
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class SynthesizeStream(ApiHandler):
    async def process(self, input: dict, request: Request) -> Response:
        text = input.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return Response("text is empty", status=400, mimetype="text/plain")

        tts_settings = _tts_settings()
        engine = str(tts_settings.get("engine", "chatterbox")).lower()
        style = input.get("style", {}) or {}
        filtered_style = {k: v for k, v in style.items() if v is not None}
        if "voice_wav_path" in filtered_style and "speaker_wav_path" not in filtered_style:
            filtered_style["speaker_wav_path"] = filtered_style["voice_wav_path"]

        if engine == "browser":
            return Response("browser TTS engine does not support server streaming", status=400, mimetype="text/plain")

        if engine == "xtts":
            try:
                return self._stream_xtts(text, filtered_style, input, tts_settings)
            except Exception as e:
                return Response(str(e), status=500, mimetype="text/plain")

        if engine == "kokoro":
            try:
                return self._stream_kokoro(text, filtered_style, input, tts_settings)
            except Exception as e:
                return Response(str(e), status=500, mimetype="text/plain")

        if engine == "piper_vc":
            try:
                return self._stream_piper_vc(text, filtered_style, input, tts_settings)
            except Exception as e:
                return Response(str(e), status=500, mimetype="text/plain")

        try:
            return self._stream_chatterbox(text, filtered_style, input, tts_settings)
        except Exception as e:
            return Response(str(e), status=500, mimetype="text/plain")

    def _chunk_text(self, text: str, max_len: int) -> list[str]:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            projected = cur_len + (1 if cur else 0) + len(part)
            if projected > max_len and cur:
                chunks.append(" ".join(cur))
                cur = [part]
                cur_len = len(part)
            else:
                cur.append(part)
                cur_len = projected if cur else len(part)
        if cur:
            chunks.append(" ".join(cur))
        return chunks

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

    def _stream_kokoro(self, text: str, style: dict, payload: dict, tts_settings: dict) -> Response:
        try:
            from python.helpers import kokoro_tts
        except Exception as exc:
            raise RuntimeError(f"Kokoro unavailable: {exc}")

        kokoro_cfg = tts_settings.get("kokoro", {}) if isinstance(tts_settings, dict) else {}
        voice = (
            style.get("speaker")
            or style.get("voice")
            or kokoro_cfg.get("voice")
            or "am_puck,am_onyx"
        )
        speed_raw = style.get("speed", kokoro_cfg.get("speed", 1.1))
        try:
            speed = float(speed_raw)
        except Exception:
            speed = 1.1
        sr_raw = style.get("sample_rate", kokoro_cfg.get("sample_rate", 24_000))
        try:
            sample_rate = int(sr_raw)
        except Exception:
            sample_rate = 24_000

        start_time = time.perf_counter()
        bytes_sent = 0
        chunk_count = 0

        try:
            audio_b64 = kokoro_tts.synthesize_sentences_sync([text], voice=voice, speed=speed, sample_rate=sample_rate)
        except Exception as exc:
            raise RuntimeError(str(exc))
        audio_bytes = base64.b64decode(audio_b64.encode("ascii"))
        buffer = io.BytesIO(audio_bytes)
        with wave.open(buffer, "rb") as wf:
            sr = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())

        def generator() -> Iterator[bytes]:
            nonlocal bytes_sent, chunk_count
            try:
                header = self._wav_header(sr)
                bytes_sent += len(header)
                yield header
                if pcm_data:
                    bytes_sent += len(pcm_data)
                    chunk_count += 1
                    yield pcm_data
            finally:
                total_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "synthesize_stream complete (kokoro) text_len=%d chunks=%d bytes=%d duration_ms=%d",
                    len(text),
                    chunk_count,
                    bytes_sent,
                    total_ms,
                )

        return Response(stream_with_context(generator()), mimetype="audio/wav")

    def _stream_piper_vc(self, text: str, style: dict, payload: dict, tts_settings: dict) -> Response:
        cfg = _piper_vc_settings(tts_settings)
        piper_bin = cfg["piper_bin"]
        piper_model = cfg["piper_model"]
        if not piper_model:
            raise RuntimeError("Piper model path is not configured (tts.piper_vc.piper_model)")
        target_voice_wav = (
            style.get("speaker_wav_path")
            or style.get("voice_wav_path")
            or cfg["target_voice_wav"]
        )
        if not target_voice_wav:
            raise RuntimeError("Target voice WAV is required (style.speaker_wav_path or tts.piper_vc.target_voice_wav)")

        join_ms = int(payload.get("join_silence_ms", cfg["join_silence_ms"]))
        max_chars = int(payload.get("target_chars", cfg["chunk_chars"]))

        if CoquiTTS is None:
            raise RuntimeError("coqui-tts is not installed; run: pip install TTS==0.22.0")

        device = _best_device()
        vc = CoquiTTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2").to(device)

        chunks = self._chunk_text(text, max_chars)
        if not chunks:
            return Response("text is empty", status=400, mimetype="text/plain")

        start_time = time.perf_counter()
        bytes_sent = 0
        chunk_count = 0

        def generator() -> Iterator[bytes]:
            nonlocal bytes_sent, chunk_count
            header_written = False
            silence_gap = b""
            try:
                for idx, chunk in enumerate(chunks):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                        in_path = tmp_in.name
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                        out_path = tmp_out.name
                    try:
                        subprocess.run(
                            [piper_bin, "--model", piper_model, "--output_file", in_path, "--sentence_silence", "0.0"],
                            input=chunk.encode("utf-8"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                    except subprocess.CalledProcessError as cpe:
                        raise RuntimeError(f"Piper failed: {cpe.stderr.decode('utf-8', errors='ignore')}")

                    try:
                        vc.voice_conversion_to_file(
                            source_wav=in_path,
                            target_wav=target_voice_wav,
                            file_path=out_path,
                        )
                    except Exception as exc:
                        raise RuntimeError(f"OpenVoice VC failed: {exc}")

                    with wave.open(out_path, "rb") as wf:
                        sr = wf.getframerate()
                        data = wf.readframes(wf.getnframes())
                        if not header_written:
                            header = self._wav_header(sr)
                            bytes_sent += len(header)
                            yield header
                            silence_gap = _gap_bytes(sr, join_ms)
                            header_written = True
                        if idx > 0 and silence_gap:
                            bytes_sent += len(silence_gap)
                            yield silence_gap
                        bytes_sent += len(data)
                        chunk_count += 1
                        yield data
                    for path in (in_path, out_path):
                        try:
                            if path and os.path.exists(path):
                                os.remove(path)
                        except Exception:
                            pass
            finally:
                total_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "synthesize_stream complete (piper_vc) text_len=%d chunks=%d bytes=%d duration_ms=%d",
                    len(text),
                    chunk_count,
                    bytes_sent,
                    total_ms,
                )

        return Response(stream_with_context(generator()), mimetype="audio/wav")

    def _wav_header(self, sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
        try:
            from python.helpers.chatterbox_tts import wav_header as cb_hdr

            return cb_hdr(sample_rate, channels=channels, bits_per_sample=bits_per_sample, data_bytes=None)
        except Exception:
            pass
        try:
            from python.helpers.xtts_tts import wav_header as xtts_hdr

            return xtts_hdr(sample_rate, channels=channels, bits_per_sample=bits_per_sample, data_bytes=None)
        except Exception:
            pass
        import struct

        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        hdr = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<I", 16)
        hdr += struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
        hdr += b"data" + struct.pack("<I", 0xFFFFFFFF)
        return hdr
