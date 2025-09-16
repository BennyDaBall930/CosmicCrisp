"""Utilities for composing prompts used by the orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from .prompts import PERSONA_MODIFIERS, PROMPT_LIBRARY, SAFETY_GUARDRAILS


class PromptManager:
    """Loads prompt templates and applies persona/override customisations."""

    def __init__(
        self,
        *,
        library: Optional[Dict[str, Dict[str, str]]] = None,
        persona: str = "default",
        overrides: Optional[Dict[str, Dict[str, str]]] = None,
        extra_safety: Optional[Iterable[str]] = None,
    ) -> None:
        self._library = library or PROMPT_LIBRARY
        self._overrides = overrides or {}
        self._persona = persona if persona in PERSONA_MODIFIERS else "default"
        self._extra_safety = list(extra_safety or [])

    # ------------------------------------------------------------------
    def set_persona(self, persona: str) -> None:
        if persona in PERSONA_MODIFIERS:
            self._persona = persona
        else:
            self._persona = "default"

    # ------------------------------------------------------------------
    def get_prompt(
        self,
        profile: str,
        stage: str,
        *,
        persona: Optional[str] = None,
        tool_hint: Optional[str] = None,
    ) -> str:
        template = self._lookup(profile, stage)
        persona = persona or self._persona
        persona_blurb = PERSONA_MODIFIERS.get(persona, PERSONA_MODIFIERS["default"])

        prompt_parts = [template.strip(), SAFETY_GUARDRAILS, persona_blurb]
        if tool_hint:
            prompt_parts.append(f"Available tool hint: {tool_hint}.")
        if self._extra_safety:
            prompt_parts.extend(self._extra_safety)
        return "\n\n".join(part for part in prompt_parts if part)

    # ------------------------------------------------------------------
    def _lookup(self, profile: str, stage: str) -> str:
        stage = stage.lower()
        profile = profile.lower()
        # override order: explicit override > library profile > general fallback
        if profile in self._overrides and stage in self._overrides[profile]:
            return self._overrides[profile][stage]
        if profile in self._library and stage in self._library[profile]:
            return self._library[profile][stage]
        general = self._library.get("general", {})
        if stage in general:
            return general[stage]
        raise KeyError(f"Prompt stage '{stage}' not found for profile '{profile}'")

    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        path: Path,
        *,
        persona: str = "default",
        base_library: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> "PromptManager":
        library = dict(base_library or PROMPT_LIBRARY)
        overrides: Dict[str, Dict[str, str]] = {}
        if path.is_dir():
            for profile_dir in path.iterdir():
                if not profile_dir.is_dir():
                    continue
                profile_key = profile_dir.name
                overrides[profile_key] = {}
                for stage_file in profile_dir.glob("*.txt"):
                    overrides[profile_key][stage_file.stem] = stage_file.read_text().strip()
                for json_file in profile_dir.glob("*.json"):
                    data = json.loads(json_file.read_text())
                    if isinstance(data, dict):
                        overrides[profile_key].update(
                            {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float))}
                        )
        return cls(library=library, persona=persona, overrides=overrides)


__all__ = ["PromptManager"]
