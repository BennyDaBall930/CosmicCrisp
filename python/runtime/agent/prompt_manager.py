"""Utilities for composing prompts used by the orchestrator."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from .prompts import (
    DEFAULT_LIBRARY_PATH,
    PERSONA_MODIFIERS,
    SAFETY_GUARDRAILS,
    load_prompt_library,
    load_prompt_overrides,
)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - fallback
        return "{" + key + "}"


class PromptManager:
    """Loads prompt templates, applies overrides, and tracks adaptive hints."""

    def __init__(
        self,
        *,
        library_path: Optional[Path] = None,
        override_path: Optional[Path] = None,
        persona: str = "default",
        extra_safety: Optional[Iterable[str]] = None,
    ) -> None:
        self._library_path = library_path or DEFAULT_LIBRARY_PATH
        self._library = load_prompt_library(self._library_path)
        self._overrides = load_prompt_overrides(override_path)
        self._runtime_overrides: Dict[str, Dict[str, str]] = {}
        self._persona = persona if persona in PERSONA_MODIFIERS else "default"
        self._extra_safety = list(extra_safety or [])
        self._session_state: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"failures": 0, "tool_misuse": set()}
        )

    # ------------------------------------------------------------------
    def set_persona(self, persona: str) -> None:
        self._persona = persona if persona in PERSONA_MODIFIERS else "default"

    @contextmanager
    def persona_context(self, persona: Optional[str]) -> Iterator[None]:
        if persona is None:
            yield
            return
        previous = self._persona
        self.set_persona(persona)
        try:
            yield
        finally:
            self._persona = previous

    @contextmanager
    def runtime_overrides(self, overrides: Optional[Dict[str, Dict[str, str]]]) -> Iterator[None]:
        if not overrides:
            yield
            return
        old = self._runtime_overrides.copy()
        merged = {**old}
        for profile, stages in overrides.items():
            profile_key = profile.lower()
            merged.setdefault(profile_key, {}).update({k.lower(): v for k, v in stages.items()})
        self._runtime_overrides = merged
        try:
            yield
        finally:
            self._runtime_overrides = old

    # ------------------------------------------------------------------
    def get_prompt(
        self,
        profile: str,
        stage: str,
        *,
        persona: Optional[str] = None,
        tool_hint: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        session: Optional[str] = None,
    ) -> str:
        template = self._lookup(profile, stage)
        formatted = self._apply_variables(template, variables)

        persona_key = persona or self._persona
        persona_blurb = PERSONA_MODIFIERS.get(persona_key, PERSONA_MODIFIERS["default"])

        prompt_parts = [formatted.strip(), SAFETY_GUARDRAILS, persona_blurb]
        if tool_hint:
            prompt_parts.append(f"Available tool hint: {tool_hint}.")
        if self._extra_safety:
            prompt_parts.extend(self._extra_safety)
        adaptive = self._adaptive_instructions(session)
        if adaptive:
            prompt_parts.append(adaptive)
        return "\n\n".join(part for part in prompt_parts if part)

    # ------------------------------------------------------------------
    def record_failure(self, session: str, *, tool: Optional[str] = None) -> None:
        state = self._session_state[session]
        state["failures"] += 1
        if tool:
            state["tool_misuse"].add(tool)

    def record_success(self, session: str, *, reset: bool = False) -> None:
        state = self._session_state[session]
        if reset:
            state["failures"] = 0
            state["tool_misuse"].clear()
        else:
            state["failures"] = max(0, state["failures"] - 1)
            state["tool_misuse"].clear()

    def reset_session(self, session: str) -> None:
        self._session_state.pop(session, None)

    # ------------------------------------------------------------------
    def _lookup(self, profile: str, stage: str) -> str:
        profile_key = profile.lower()
        stage_key = stage.lower()
        if profile_key in self._runtime_overrides and stage_key in self._runtime_overrides[profile_key]:
            return self._runtime_overrides[profile_key][stage_key]
        if profile_key in self._overrides and stage_key in self._overrides[profile_key]:
            return self._overrides[profile_key][stage_key]
        if profile_key in self._library and stage_key in self._library[profile_key]:
            return self._library[profile_key][stage_key]
        general = self._library.get("general", {})
        if stage_key in general:
            return general[stage_key]
        raise KeyError(f"Prompt stage '{stage}' not found for profile '{profile}'")

    def _apply_variables(self, template: str, variables: Optional[Dict[str, Any]]) -> str:
        if not variables:
            return template
        return template.format_map(_SafeDict(**variables))

    def _adaptive_instructions(self, session: Optional[str]) -> str:
        if not session or session not in self._session_state:
            return ""
        state = self._session_state[session]
        segments = []
        if state["failures"] >= 2:
            segments.append("You have encountered recent issues; reflect on mistakes before continuing and validate assumptions.")
        if state["tool_misuse"]:
            tools = ", ".join(sorted(state["tool_misuse"]))
            segments.append(f"Avoid repeating misuse of tools: {tools}. Articulate why a tool is appropriate before invoking it.")
        return "\n".join(segments)


__all__ = ["PromptManager"]
