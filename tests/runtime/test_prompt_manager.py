"""Prompt manager behaviour tests."""
from __future__ import annotations

from pathlib import Path

from python.runtime.agent.prompt_manager import PromptManager
from python.runtime.agent.prompts import PERSONA_MODIFIERS, SAFETY_GUARDRAILS

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "python" / "runtime" / "prompts" / "library"


def test_prompt_includes_guardrails_and_persona():
    manager = PromptManager(persona="concise", library_path=LIBRARY_PATH)
    prompt = manager.get_prompt("general", "system", session="sess-1")
    assert SAFETY_GUARDRAILS in prompt
    assert PERSONA_MODIFIERS["concise"] in prompt


def test_prompt_runtime_overrides_and_adaptive_instructions(tmp_path):
    manager = PromptManager(library_path=LIBRARY_PATH)
    manager.record_failure("run-1", tool="browser")
    manager.record_failure("run-1", tool="browser")

    overrides = {"general": {"system": "Override base system prompt"}}
    with manager.runtime_overrides(overrides):
        prompt = manager.get_prompt("general", "system", session="run-1")
    assert "Override base system prompt" in prompt
    assert "reflect on mistakes" in prompt
    assert "browser" in prompt

    manager.record_success("run-1", reset=True)
    prompt_after_reset = manager.get_prompt("general", "system", session="run-1")
    assert "reflect" not in prompt_after_reset


def test_persona_context_restores_previous_persona():
    manager = PromptManager(persona="detailed", library_path=LIBRARY_PATH)
    base_prompt = manager.get_prompt("general", "system", session="demo")
    with manager.persona_context("concise"):
        concise_prompt = manager.get_prompt("general", "system", session="demo")
    restored_prompt = manager.get_prompt("general", "system", session="demo")

    assert PERSONA_MODIFIERS["concise"] in concise_prompt
    assert PERSONA_MODIFIERS["detailed"] in base_prompt
    assert PERSONA_MODIFIERS["detailed"] in restored_prompt
