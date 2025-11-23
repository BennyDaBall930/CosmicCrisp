"""Supertonic TTS provider implementation."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import soundfile as sf

from .supertonic_utils import TextToSpeech, load_text_to_speech, load_voice_style, Style

log = logging.getLogger(__name__)

# Default sample rate for Supertonic
DEFAULT_SAMPLE_RATE = 24000

class SupertonicProvider:
    """Supertonic-backed implementation satisfying the runtime TTSProvider protocol."""

    def __init__(
        self,
        model_dir: Path,
        default_voice_id: str = "M1.json",
        speed: float = 1.05,
        total_step: int = 5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.model_dir = model_dir
        self.onnx_dir = model_dir / "onnx"
        self.voice_styles_dir = model_dir / "voice_styles"
        self.default_voice_id = default_voice_id
        self.speed = speed
        self.total_step = total_step
        self.sample_rate = sample_rate

        self._engine_lock = threading.Lock()
        self._engine: TextToSpeech | None = None
        self._voice_cache: dict[str, Style] = {}

    def _get_engine(self) -> TextToSpeech:
        if self._engine is not None:
            return self._engine

        with self._engine_lock:
            if self._engine is not None:
                return self._engine

            log.info("Initializing Supertonic TTS engine from %s", self.onnx_dir)
            if not self.onnx_dir.exists():
                 raise FileNotFoundError(f"Supertonic ONNX models not found at {self.onnx_dir}")
            
            self._engine = load_text_to_speech(str(self.onnx_dir), use_gpu=False)
            return self._engine

    def _resolve_voice(self, voice_id: str | None) -> str:
        if not voice_id:
            voice_id = self.default_voice_id
        
        if not voice_id.endswith(".json"):
            voice_id += ".json"
            
        return voice_id

    def _get_style(self, voice_id: str) -> Style:
        voice_path = self.voice_styles_dir / voice_id
        if not voice_path.exists():
            # Fallback to default if specified voice not found
            log.warning("Voice style %s not found, falling back to %s", voice_id, self.default_voice_id)
            voice_id = self.default_voice_id
            if not voice_id.endswith(".json"):
                voice_id += ".json"
            voice_path = self.voice_styles_dir / voice_id
            
        if str(voice_path) in self._voice_cache:
            return self._voice_cache[str(voice_path)]

        if not voice_path.exists():
             raise FileNotFoundError(f"Voice style not found: {voice_path}")
             
        style = load_voice_style([str(voice_path)])
        self._voice_cache[str(voice_path)] = style
        return style

    def synthesize(self, text: str, voice_id: str | None = None, *, stream: bool = False) -> bytes | Iterable[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            voice_id: Voice ID (filename in voice_styles directory)
            stream: If True, returns generator (NOTE: Supertonic doesn't support true streaming yet, 
                   so this will generate full audio and yield chunks)
        """
        engine = self._get_engine()
        voice_id = self._resolve_voice(voice_id)
        style = self._get_style(voice_id)
        
        # Supertonic currently only supports non-streaming inference
        # It generates the full audio array
        wav, _ = engine(text, style, self.total_step, self.speed)
        
        # Convert to PCM16
        audio_data = wav.squeeze()
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2").tobytes()

        if stream:
            # Simulate streaming by yielding chunks
            chunk_size = 4096
            def iterator():
                for i in range(0, len(pcm16), chunk_size):
                    yield pcm16[i:i+chunk_size]
            return iterator()
        
        return pcm16

    @property
    def last_output_metadata(self) -> dict:
        return {
            "voice_id": self.default_voice_id,
            "watermarked": False,
            "quality": "high",
            "duration": 0.0, # TODO: calculate from audio length
            "sample_rate": self.sample_rate
        }

    def register_voice(self, name: str, ref_wav_path: str, ref_text: str) -> str:
        # Supertonic currently implies voice cloning requires training/finetuning or an encoder not present in inference code.
        # We will log a warning and return a placeholder ID, or raise NotImplementedError.
        # For now, we'll raise to indicate it's not supported in this version.
        raise NotImplementedError("Voice cloning is not currently supported by Supertonic integration.")

    def list_voices(self) -> list[dict]:
        voices = []
        if self.voice_styles_dir.exists():
            for path in self.voice_styles_dir.glob("*.json"):
                 voices.append({
                     "id": path.name,
                     "name": path.stem,
                     "created_at": path.stat().st_mtime,
                     "ref_text": "", # Not available from just the style file
                     "preview_audio": "",
                 })
        return voices

    def delete_voice(self, voice_id: str) -> None:
        voice_id = self._resolve_voice(voice_id)
        voice_path = self.voice_styles_dir / voice_id
        if voice_path.exists():
            # Prevent deleting stock voices if desired, but for now allow deletion if it's in the dir
            # Maybe check if it's one of the default ones? M1, M2, F1, F2
            if voice_path.stem in {"M1", "M2", "F1", "F2"}:
                log.warning("Attempted to delete stock voice %s - ignoring", voice_id)
                return
            os.remove(voice_path)
            
    def set_default_voice(self, voice_id: str | None) -> None:
        if voice_id:
             self.default_voice_id = voice_id

    def reset_voices(self) -> None:
        # Reset would theoretically delete all non-stock voices
        pass
