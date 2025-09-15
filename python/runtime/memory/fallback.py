"""Memory wrapper that falls back to a secondary store on failure."""
from __future__ import annotations

from typing import Dict, List

from .interface import AgentMemory


class MemoryWithFallback(AgentMemory):
    def __init__(self, primary: AgentMemory, secondary: AgentMemory) -> None:
        self.primary = primary
        self.secondary = secondary

    async def enter(self, session_id: str) -> None:
        try:
            await self.primary.enter(session_id)
        except Exception:
            await self.secondary.enter(session_id)

    async def add(self, item: Dict) -> str:
        try:
            return await self.primary.add(item)
        except Exception:
            return await self.secondary.add(item)

    async def similar(self, query: str, k: int = 5) -> List[Dict]:
        try:
            return await self.primary.similar(query, k)
        except Exception:
            return await self.secondary.similar(query, k)

    async def reset(self) -> None:
        try:
            await self.primary.reset()
        except Exception:
            await self.secondary.reset()
