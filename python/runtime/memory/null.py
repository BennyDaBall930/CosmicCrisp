"""A no-op memory implementation."""
from __future__ import annotations

from typing import Dict, List

from .interface import AgentMemory


class NullMemory(AgentMemory):
    async def enter(self, session_id: str) -> None:
        return None

    async def add(self, item: Dict) -> str:
        return "0"

    async def similar(self, query: str, k: int = 5) -> List[Dict]:
        return []

    async def reset(self) -> None:
        return None
