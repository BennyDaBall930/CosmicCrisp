"""Memory abstractions."""
from __future__ import annotations

from typing import Dict, List, Protocol


class AgentMemory(Protocol):
    async def enter(self, session_id: str) -> None:
        ...

    async def add(self, item: Dict) -> str:
        ...

    async def similar(self, query: str, k: int = 5) -> List[Dict]:
        ...

    async def reset(self) -> None:
        ...
