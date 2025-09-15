"""Browser-use wrapper (read-only)."""
from __future__ import annotations

from typing import Any

from .base import Tool


class BrowserTool(Tool):
    name = "browser"

    async def run(self, url: str, **_: Any) -> str:
        return f"browse:{url}"


from .registry import registry

registry.register(BrowserTool())
