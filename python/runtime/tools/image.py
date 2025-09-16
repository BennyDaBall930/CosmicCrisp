"""Image generation tool placeholder."""
from __future__ import annotations

from typing import Any

from .base import Tool
from .registry import register_tool


class ImageTool(Tool):
    name = "image"

    async def run(self, prompt: str = "", **_: Any) -> str:
        return "image-generation-not-implemented"


register_tool(ImageTool)
