"""Base Tool interface."""
from __future__ import annotations

from typing import Any, Dict, Protocol


class Tool(Protocol):
    name: str

    async def run(self, **kwargs: Any) -> str:
        ...
