"""Image generation tool placeholder."""
from __future__ import annotations

from typing import Any

from .base import Tool


class ImageTool(Tool):
    name = "image"

    async def run(self, prompt: str = "", **_: Any) -> str:
        return "image-generation-not-implemented"


from .registry import registry

registry.register(ImageTool())
