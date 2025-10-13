"""NeuTTS-Air provider implementation for Apple Zero."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import shutil

log = logging.getLogger(__name__)

try:  # pragma: no cover - runtime optional dependency
    from python.third_party.neuttsair.neutts import NeuTTSAir  # type: ignore
except Exception:  # pragma: no cover
    NeuTTSAir = None  # type: ignore[assignment]

DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_STREAM_CHUNK_SECONDS = 0.32
MIN_REF_DURATION = 3.0
MAX_REF_DURATION = 15.0
_ESPEAK_ENV_INITIALIZED = False
_ESPEAK_ENV_LOCK = threading.Lock()


def _as_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype("<i2")
    return int16.tobytes()


def _ensure_espeak_environment() -> None:
    global _ESPEAK_ENV_INITIALIZED
    with _ESPEAK_ENV_LOCK:
        if _ESPEAK_ENV_INITIALIZED:
            return
        _ESPEAK_ENV_INITIALIZED = True

        lib_env = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        data_env = os.environ.get("ESPEAKNG_DATA_PATH")

        def _path_ok(path_str: Optional[str]) -> bool:
            if not path_str:
                return False
            return Path(path_str).expanduser().exists()

        lib_configured = _path_ok(lib_env)
        if lib_env and not lib_configured:
            log.debug(
                "Ignoring missing espeak library override at %s; attempting auto-detect.",
                lib_env,
            )
            os.environ.pop("PHONEMIZER_ESPEAK_LIBRARY", None)
            lib_env = None

        data_configured = _path_ok(data_env)
        if data_env and not data_configured:
            log.debug(
                "Ignoring missing espeak data override at %s; attempting auto-detect.",
                data_env,
            )
            os.environ.pop("ESPEAKNG_DATA_PATH", None)
            data_env = None

        if lib_configured and data_configured:
            return

        prefixes: list[Path] = []
        for key in ("ESPEAK_PREFIX", "HOMEBREW_PREFIX"):
            val = os.environ.get(key)
            if val:
                prefixes.append(Path(val))
        for exe_name in ("espeak-ng", "espeak"):
            exe_path = shutil.which(exe_name)
            if exe_path:
                prefixes.append(Path(exe_path).resolve().parent.parent)
        prefixes.extend(
            Path(p)
            for p in ("/opt/homebrew", "/usr/local", "/usr", "/opt/local")
            if Path(p).exists()
        )

        seen: set[Path] = set()
        unique_prefixes: list[Path] = []
        for prefix in prefixes:
            if prefix not in seen and prefix.exists():
                seen.add(prefix)
                unique_prefixes.append(prefix)

        lib_candidates = [
            Path(prefix) / "lib" / name
            for prefix in unique_prefixes
            for name in (
                "libespeak-ng.dylib",
                "libespeak-ng.so",
                "libespeak.dylib",
                "libespeak.so",
            )
        ]

        data_candidates = [
            Path(prefix) / name
            for prefix in unique_prefixes
            for name in ("share/espeak-ng-data", "share/espeak-data")
        ]

        if not lib_configured:
            for candidate in lib_candidates:
                if candidate.exists():
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(candidate)
                    lib_configured = True
                    log.debug("Auto-detected espeak library at %s", candidate)
                    break

        if not data_configured:
            for candidate in data_candidates:
                if candidate.exists():
                    os.environ["ESPEAKNG_DATA_PATH"] = str(candidate)
                    data_configured = True
                    log.debug("Auto-detected espeak data at %s", candidate)
                    break

        if not lib_configured:
            log.warning(
                "Could not locate an espeak/espeak-ng library. Install `espeak-ng` "
                "or set PHONEMIZER_ESPEAK_LIBRARY to the path of libespeak-ng."
            )
        if not data_configured:
            log.debug(
                "espeak data directory not found automatically. Set ESPEAKNG_DATA_PATH "
                "if phonemizer reports missing language issues."
            )


@dataclass(slots=True)
class VoiceMetadata:
    id: str
    name: str
    created_at: float
    ref_text: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    quality: str = "q4"
    watermarked: bool = True

    def to_json(self) -> dict:
        return asdict(self)


class NeuttsProvider:
    """NeuTTS-Air backed implementation satisfying the runtime TTSProvider protocol."""

    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air-q4-gguf",
        codec_repo: str = "neuphonic/neucodec-onnx-decoder",
        backbone_device: str = "mps",
        codec_device: str = "cpu",
        model_cache_dir: Optional[Path] = None,
        *,
        stream_chunk_seconds: float = DEFAULT_STREAM_CHUNK_SECONDS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        quality_default: str = "q4",
        default_voice_id: Optional[str] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.backbone_repo = backbone_repo
        if codec_repo == "neuphonic/neucodec":
            log.info("Switching codec repo to the ONNX decoder for streaming support.")
            codec_repo = "neuphonic/neucodec-onnx-decoder"
        self.codec_repo = codec_repo
        self.backbone_device = backbone_device
        self.codec_device = codec_device
        self.stream_chunk_seconds = max(min(stream_chunk_seconds, 1.0), 0.1)
        self.sample_rate = max(int(sample_rate), 8_000)
        self.quality_default = quality_default if quality_default in {"q4", "q8"} else "q4"
        self.default_voice_id = default_voice_id

        root = Path(data_root or Path("data/tts/neutts"))
        self.data_root = root
        self.models_dir = Path(model_cache_dir or root / "models")
        self.voices_dir = root / "voices"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_lock = threading.RLock()
        self._voices: dict[str, VoiceMetadata] = {}
        self._load_voice_cache()
        if not self.default_voice_id and self._voices:
            fallback = min(self._voices.values(), key=lambda meta: meta.created_at)
            self.default_voice_id = fallback.id

        self._engine_lock = threading.Lock()
        self._engine: NeuTTSAir | None = None
        self._last_output_metadata: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Engine helpers
    # ------------------------------------------------------------------
    def _get_engine(self) -> NeuTTSAir:
        if NeuTTSAir is None:  # pragma: no cover
            raise RuntimeError(
                "NeuTTSAir dependencies are not available. Ensure phonemizer, librosa, "
                "onnxruntime, resemble-perth, transformers, and llama-cpp-python are installed."
            )
        if self._engine is not None:
            return self._engine

        with self._engine_lock:
            if self._engine is not None:
                return self._engine

            hf_cache = self.models_dir / "hf_cache"
            _ensure_espeak_environment()
            os.environ.setdefault("HF_HOME", str(hf_cache))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
            os.environ.setdefault("NEUTTS_CACHE_DIR", str(hf_cache))

            backbone_device = self.backbone_device.lower()
            if backbone_device in {"mps", "gpu"}:
                backbone_device = "gpu"
            elif backbone_device not in {"cpu"}:
                log.warning("Unsupported backbone device '%s'; defaulting to CPU.", backbone_device)
                backbone_device = "cpu"

            codec_device = self.codec_device.lower()
            if codec_device != "cpu":
                log.info("NeuTTS codec currently runs on CPU; overriding codec device from %s to cpu", codec_device)
                codec_device = "cpu"

            log.info(
                "Initialising NeuTTSAir (backbone=%s on %s, codec=%s on %s)",
                self.backbone_repo,
                backbone_device,
                self.codec_repo,
                codec_device,
            )
            self._engine = NeuTTSAir(
                backbone_repo=self.backbone_repo,
                backbone_device=backbone_device,
                codec_repo=self.codec_repo,
                codec_device=codec_device,
            )
            return self._engine

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------
    def _load_voice_cache(self) -> None:
        for meta_path in self.voices_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                voice = VoiceMetadata(**data)
                self._voices[voice.id] = voice
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to load voice metadata %s: %s", meta_path, exc)

    def _voice_path(self, voice_id: str) -> Path:
        return self.voices_dir / voice_id

    def list_voices(self) -> list[dict]:
        with self._metadata_lock:
            return [voice.to_json() for voice in self._voices.values()]

    def register_voice(self, name: str, ref_wav_path: str, ref_text: str) -> str:
        if not name.strip():
            raise ValueError("Voice name must be provided")
        if not ref_text.strip():
            raise ValueError("Reference text must be provided")
        if self.codec_repo == "neuphonic/neucodec-onnx-decoder":
            raise ValueError(
                "NeuTTS is configured with the ONNX decoder, which cannot encode reference audio. "
                "Use pre-encoded reference files (e.g. from the NeuTTS-Air samples) instead."
            )
        
        voice_id = uuid.uuid4().hex
        voice_dir = self._voice_path(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load audio - librosa handles various formats better than soundfile
            ref_audio, sr = librosa.load(ref_wav_path, sr=None, mono=True)
            ref_audio = ref_audio.astype(np.float32)
            
            # Check duration constraints
            duration = len(ref_audio) / float(sr)
            if duration < MIN_REF_DURATION:
                raise ValueError(
                    f"Reference audio too short: {duration:.1f}s. "
                    f"Provide at least {MIN_REF_DURATION:.0f}s for stable voice cloning."
                )
            
            # Trim to max duration if needed
            if duration > MAX_REF_DURATION:
                max_len = int(MAX_REF_DURATION * sr)
                ref_audio = ref_audio[:max_len]
                log.info("Trimmed reference audio from %.1fs to %.1fs", duration, MAX_REF_DURATION)
            
            # Normalize audio
            max_abs = np.max(np.abs(ref_audio))
            if max_abs > 0:
                ref_audio = ref_audio / max_abs
            else:
                raise ValueError("Reference audio is silent or contains only zeros")
            
            # CRITICAL FIX: Resample to 16kHz for encoding
            # The NeuTTS encoder expects 16kHz input, not 24kHz
            ENCODING_SAMPLE_RATE = 16000
            if sr != ENCODING_SAMPLE_RATE:
                log.debug("Resampling reference audio from %d Hz to %d Hz for encoding", sr, ENCODING_SAMPLE_RATE)
                ref_audio_16k = librosa.resample(ref_audio, orig_sr=sr, target_sr=ENCODING_SAMPLE_RATE)
            else:
                ref_audio_16k = ref_audio
            
            # Save the 16kHz version for encoding
            sf.write(voice_dir / "ref.wav", ref_audio_16k, ENCODING_SAMPLE_RATE)
            (voice_dir / "ref.txt").write_text(ref_text.strip(), encoding="utf-8")
            
            # Encode reference with proper error handling
            engine = self._get_engine()
            with self._engine_lock:
                try:
                    ref_codes_tensor = engine.encode_reference(str(voice_dir / "ref.wav"))
                    ref_codes = ref_codes_tensor.detach().cpu().numpy().astype(np.int32)
                except Exception as encode_err:
                    log.error("Failed to encode reference audio: %s", encode_err)
                    raise ValueError(f"Failed to encode reference audio: {encode_err}") from encode_err
            
            # Validate encoded codes
            if ref_codes.size == 0:
                raise ValueError("Reference encoding produced empty codes - audio may be invalid")
            
            np.save(voice_dir / "ref.codes.npy", ref_codes)
            log.info("Encoded reference: %d codes from %.1fs audio", ref_codes.size, len(ref_audio_16k) / ENCODING_SAMPLE_RATE)
            
            # Create metadata
            meta = VoiceMetadata(
                id=voice_id,
                name=name.strip(),
                created_at=time.time(),
                ref_text=ref_text.strip(),
                quality=self.quality_default,
            )
            (voice_dir / "meta.json").write_text(json.dumps(meta.to_json(), indent=2), encoding="utf-8")
            
            with self._metadata_lock:
                self._voices[voice_id] = meta
            
            log.info("Successfully registered NeuTTS voice '%s' (%s)", name, voice_id)
            return voice_id
            
        except Exception as e:
            # Clean up on failure
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)
            log.error("Failed to register voice: %s", e)
            raise

    def delete_voice(self, voice_id: str) -> None:
        voice_dir = self._voice_path(voice_id)
        if voice_dir.exists():
            import shutil

            shutil.rmtree(voice_dir, ignore_errors=True)
        with self._metadata_lock:
            self._voices.pop(voice_id, None)
        if self.default_voice_id == voice_id:
            self.default_voice_id = None
        log.info("Deleted NeuTTS voice %s", voice_id)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _resolve_reference(self, voice_id: Optional[str]) -> tuple[np.ndarray, VoiceMetadata]:
        if not voice_id and self.default_voice_id:
            voice_id = self.default_voice_id
        if not voice_id:
            raise FileNotFoundError("No NeuTTS voices are registered; add one via Voice Lab.")

        voice_dir = self._voice_path(voice_id)
        meta_path = voice_dir / "meta.json"
        
        # Check for .pt file first (official format from standalone encoder)
        codes_pt_path = voice_dir / "ref.codes.pt"
        codes_npy_path = voice_dir / "ref.codes.npy"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' metadata not found at {meta_path}")
        
        # Load codes from .pt or .npy
        if codes_pt_path.exists():
            # Load PyTorch tensor and convert to numpy
            ref_codes_tensor = torch.load(codes_pt_path, map_location='cpu')
            ref_codes = ref_codes_tensor.detach().cpu().numpy().astype(np.int32)
            log.debug(f"Loaded voice codes from {codes_pt_path}")
        elif codes_npy_path.exists():
            # Legacy numpy format
            ref_codes = np.load(codes_npy_path).astype(np.int32)
            log.debug(f"Loaded voice codes from {codes_npy_path}")
        else:
            raise FileNotFoundError(
                f"Voice '{voice_id}' codes not found. Expected either:\n"
                f"  - {codes_pt_path} (official format)\n"
                f"  - {codes_npy_path} (legacy format)"
            )
        
        meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = VoiceMetadata(**meta_dict)
        return ref_codes, meta

    def _render(self, text: str, voice_id: Optional[str]) -> np.ndarray:
        """Render audio from text using the NeuTTS model (non-streaming)."""
        ref_codes, meta = self._resolve_reference(voice_id)
        engine = self._get_engine()
        
        # Create tensor with proper device placement
        # Note: For GGUF models, the backbone handles device placement internally
        ref_tensor = torch.as_tensor(ref_codes, dtype=torch.long)
        
        try:
            with self._engine_lock:
                audio = engine.infer(text, ref_tensor, meta.ref_text)
            
            # Validate output
            if audio is None or len(audio) == 0:
                raise RuntimeError("Model produced empty audio output")
            
            # Check for silent output (common sign of model failure)
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude < 1e-6:
                log.warning("Generated audio is nearly silent (max amplitude: %.2e)", max_amplitude)
            
            self._last_output_metadata = {
                "voice_id": meta.id,
                "voice_name": meta.name,
                "watermarked": True,
                "sample_rate": self.sample_rate,
                "duration": len(audio) / self.sample_rate,
                "quality": meta.quality,
            }
            
            return audio.astype(np.float32)
            
        except Exception as e:
            log.error("Synthesis failed for text '%s' with voice %s: %s", 
                     text[:50] + "..." if len(text) > 50 else text,
                     meta.id, e)
            raise RuntimeError(f"Speech synthesis failed: {e}") from e

    def synthesize(self, text: str, voice_id: str | None = None, *, stream: bool = False):
        """
        Synthesize speech from text using voice cloning.
        
        Args:
            text: Input text to synthesize
            voice_id: Voice ID to use (uses default if None)
            stream: If True, returns an iterator of audio chunks; if False, returns complete audio
            
        Returns:
            bytes (non-streaming) or Iterator[bytes] (streaming) containing PCM16 audio data
        """
        if not text.strip():
            raise ValueError("Text must not be empty for synthesis")
        
        # Validate voice availability
        engine = self._get_engine()
        ref_codes, meta = self._resolve_reference(voice_id)
        
        # Create reference tensor
        ref_tensor = torch.as_tensor(ref_codes, dtype=torch.long)
        
        if not stream:
            # Non-streaming path: render complete audio
            audio = self._render(text, voice_id)
            return _as_pcm16(audio)
        
        # Streaming path: return iterator
        def iterator() -> Iterator[bytes]:
            """Generator for streaming audio chunks."""
            try:
                with self._engine_lock:
                    chunk_count = 0
                    total_samples = 0
                    
                    for chunk in engine.infer_stream(text, ref_tensor, meta.ref_text):
                        if chunk.size == 0:
                            continue
                        
                        chunk_count += 1
                        total_samples += chunk.size
                        
                        # Convert to PCM16 and yield
                        pcm_chunk = _as_pcm16(chunk.astype(np.float32))
                        yield pcm_chunk
                    
                    if chunk_count == 0:
                        log.warning("Streaming synthesis produced zero chunks")
                    else:
                        log.debug("Streamed %d chunks, %d total samples (~%.2fs)", 
                                 chunk_count, total_samples, total_samples / self.sample_rate)
                        
            except Exception as stream_err:
                log.error("Streaming synthesis failed: %s", stream_err)
                raise RuntimeError(f"Streaming synthesis failed: {stream_err}") from stream_err
        
        # Set metadata for streaming (duration unknown)
        self._last_output_metadata = {
            "voice_id": meta.id,
            "voice_name": meta.name,
            "watermarked": True,
            "sample_rate": self.sample_rate,
            "duration": None,  # Unknown for streaming
            "quality": meta.quality,
        }
        
        return iterator()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def set_default_voice(self, voice_id: Optional[str]) -> None:
        self.default_voice_id = voice_id or None

    @property
    def last_output_metadata(self) -> dict[str, object]:
        return dict(self._last_output_metadata)

    def reset_voices(self) -> None:
        """Remove all registered voices (utility for UI reset button)."""
        for voice_id in list(self._voices.keys()):
            self.delete_voice(voice_id)
