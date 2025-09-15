"""Code generation tool stub."""
from __future__ import annotations

from typing import Any

from .base import Tool


class CodeTool(Tool):
    name = "code"

    async def run(self, prompt: str = "", **_: Any) -> str:
        return f"code:{prompt}"


from .registry import registry

registry.register(CodeTool())
