from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import torch/torchaudio lazily in model loaders to reduce startup cost.


def _pick_device(prefer: Optional[str] = None) -> str:
    try:
        import torch  # type: ignore

        if prefer and prefer not in {"", "auto"}:
            return prefer
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and getattr(torch.backends, "mps").is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------
# Request/Response Models
# ---------------------------


class XTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field("en", description="Language code, e.g., 'en'")
    speaker: Optional[str] = Field(None, description="Speaker ID (if the model supports it)")
    speaker_wav_path: Optional[str] = Field(None, description="Path to reference voice WAV on disk")
    speaker_wav_b64: Optional[str] = Field(None, description="Base64-encoded reference voice WAV")

    model_id: str = Field(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        description="Coqui TTS model ID",
    )
    device: Optional[str] = Field(None, description="'auto', 'cpu', 'cuda', or 'mps'")
    sample_rate: int = Field(24000, description="Output sample rate")
    max_chars: int = Field(400, description="Max characters per chunk")
    join_silence_ms: int = Field(80, description="Silence between chunks in ms")


class XTTSResponse(BaseModel):
    audio_b64: str
    sample_rate: int


class VCRequest(BaseModel):
    source_wav_path: Optional[str] = None
    source_wav_b64: Optional[str] = None
    target_wav_path: Optional[str] = None
    target_wav_b64: Optional[str] = None

    model_id: str = Field(
        "voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
        description="Coqui VC model ID",
    )
    device: Optional[str] = Field(None, description="'auto', 'cpu', 'cuda', or 'mps'")


class VCResponse(BaseModel):
    audio_b64: str


# ---------------------------
# App and state
# ---------------------------


app = FastAPI(title="TTS Sidecar", version="1.0")


@dataclass
class _XTTSState:
    model = None
    model_id: Optional[str] = None
    device: Optional[str] = None


@dataclass
class _VCState:
    model = None
    model_id: Optional[str] = None
    device: Optional[str] = None


xtts_state = _XTTSState()
vc_state = _VCState()


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _chunk_text(text: str, limit: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    limit = max(1, int(limit or 400))
    # Simple sentence-ish split
    parts = [s.strip() for s in text.split(". ") if s.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for s in parts:
        candidate = (f"{buf} {s}").strip() if buf else s
        if len(candidate) <= limit:
            buf = candidate
        else:
            flush()
            if len(s) <= limit:
                buf = s
            else:
                for i in range(0, len(s), limit):
                    seg = s[i : i + limit].strip()
                    if seg:
                        chunks.append(seg)
    flush()
    return chunks or [text]


def _load_xtts(model_id: str, device: Optional[str]):
    if xtts_state.model is not None and xtts_state.model_id == model_id and xtts_state.device == device:
        return xtts_state.model
    # Lazy import
    from TTS.api import TTS as CoquiTTS  # type: ignore

    target = _pick_device(device or "auto")
    model = CoquiTTS(model_name=model_id, progress_bar=False)
    # Place device
    if target == "mps":
        try:
            model.to("mps")
        except Exception:
            target = "cpu"
    if target in {"cpu", "cuda"}:
        try:
            model.to(target)
        except Exception:
            pass
    xtts_state.model = model
    xtts_state.model_id = model_id
    xtts_state.device = target
    return model


def _load_vc(model_id: str, device: Optional[str]):
    if vc_state.model is not None and vc_state.model_id == model_id and vc_state.device == device:
        return vc_state.model
    from TTS.api import TTS as CoquiTTS  # type: ignore

    target = _pick_device(device or "auto")
    model = CoquiTTS(model_id).to(target)  # voice conversion models accept .to()
    vc_state.model = model
    vc_state.model_id = model_id
    vc_state.device = target
    return model


def _decode_wav_b64_to_path(b64_str: str) -> str:
    data = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/xtts/synthesize", response_model=XTTSResponse)
def xtts_synthesize(req: XTTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    model = _load_xtts(req.model_id, req.device)

    speaker_wav_path = req.speaker_wav_path
    tmp_path: Optional[str] = None
    try:
        if not speaker_wav_path and req.speaker_wav_b64:
            tmp_path = _decode_wav_b64_to_path(req.speaker_wav_b64)
            speaker_wav_path = tmp_path

        chunks = _chunk_text(text, req.max_chars)
        pieces: List[np.ndarray] = []
        for idx, segment in enumerate(chunks):
            if not segment:
                continue
            audio = model.tts(
                segment,
                speaker=req.speaker,
                language=req.language or "en",
                speaker_wav=speaker_wav_path,
            )
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()
            pieces.append(np.clip(arr, -1.0, 1.0))

        if not pieces:
            raise HTTPException(status_code=500, detail="Model returned no audio")

        if len(pieces) == 1:
            final = pieces[0]
        else:
            gap_samples = int(max(0, req.join_silence_ms) * (req.sample_rate / 1000.0))
            if gap_samples > 0:
                gap = np.zeros(gap_samples, dtype=np.float32)
                final = np.concatenate([p for t in enumerate(pieces) for p in ([t[1]] + ([gap] if t[0] < len(pieces) - 1 else []))])
            else:
                final = np.concatenate(pieces)

        import soundfile as sf  # type: ignore

        buf = io.BytesIO()
        sf.write(buf, final, req.sample_rate, subtype="PCM_16", format="WAV")
        data_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return XTTSResponse(audio_b64=data_b64, sample_rate=req.sample_rate)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/api/vc/convert", response_model=VCResponse)
def vc_convert(req: VCRequest):
    src_path = req.source_wav_path
    tgt_path = req.target_wav_path
    tmp_src: Optional[str] = None
    tmp_tgt: Optional[str] = None

    try:
        if not src_path and req.source_wav_b64:
            tmp_src = _decode_wav_b64_to_path(req.source_wav_b64)
            src_path = tmp_src
        if not tgt_path and req.target_wav_b64:
            tmp_tgt = _decode_wav_b64_to_path(req.target_wav_b64)
            tgt_path = tmp_tgt

        if not src_path or not tgt_path:
            raise HTTPException(status_code=400, detail="source and target WAV are required")

        model = _load_vc(req.model_id, req.device)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            model.voice_conversion_to_file(source_wav=src_path, target_wav=tgt_path, file_path=out_path)
            with open(out_path, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("ascii")
            return VCResponse(audio_b64=data_b64)
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass
    finally:
        for p in (tmp_src, tmp_tgt):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

