"""Prompt inspection tool exposing current templates."""
from __future__ import annotations

from typing import Any

from .base import Tool
from .registry import register_tool
from ..agent.prompt_manager import PromptManager
from ..container import get_config


class PromptTool(Tool):
    name = "prompt"
    description = "Inspect active prompt templates for the orchestrator."

    def __init__(self) -> None:
        config = get_config()
        self.manager = PromptManager(persona=config.agent.persona)

    async def run(self, profile: str = "general", stage: str = "system", **_: Any) -> str:
        try:
            prompt = self.manager.get_prompt(profile, stage)
            return f"Prompt [{profile}/{stage}]:\n{prompt}"
        except KeyError as exc:
            return f"prompt: {exc}"


register_tool(PromptTool)
