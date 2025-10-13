"""Audio subsystems for the Apple Zero runtime."""
from __future__ import annotations

from typing import Iterable, Protocol


class TTSProvider(Protocol):
    """Protocol describing the runtime-facing TTS provider interface."""

    def synthesize(self, text: str, voice_id: str | None = None, *, stream: bool = False) -> bytes | Iterable[bytes]:
        ...

    def register_voice(self, name: str, ref_wav_path: str, ref_text: str) -> str:
        ...

    def list_voices(self) -> list[dict]:
        ...

    def delete_voice(self, voice_id: str) -> None:
        ...


from .neutts_provider import NeuttsProvider  # noqa: E402

PROVIDERS = {"neutts": NeuttsProvider}

__all__ = ["TTSProvider", "NeuttsProvider", "PROVIDERS"]
