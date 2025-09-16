"""Memory store interfaces."""
from __future__ import annotations

from typing import Dict, List, Protocol

from .schema import MemoryItem


class MemoryStore(Protocol):
    async def add(self, item: MemoryItem | Dict) -> str:
        ...

    async def similar(self, query: str, k: int = 6) -> List[Dict]:
        ...

    async def recent(self, k: int = 6) -> List[Dict]:
        ...

    async def delete(self, item_id: str) -> None:
        ...

    async def count(self) -> int:
        ...


__all__ = ["MemoryStore", "MemoryItem"]
