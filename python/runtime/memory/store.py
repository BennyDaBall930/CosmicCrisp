"""Memory store interfaces."""
from __future__ import annotations

from typing import Dict, List, Protocol

from .schema import MemoryItem


class MemoryStore(Protocol):
    async def enter(self, session_id: str) -> None:
        ...

    async def add(self, item: MemoryItem | Dict) -> str:
        ...

    async def similar(self, query: str, k: int = 6) -> List[Dict]:
        ...

    async def recent(self, k: int = 6) -> List[Dict]:
        ...

    async def similar_paginated(self, query: str, k: int = 6, offset: int = 0) -> List[Dict]:
        ...

    async def recent_paginated(self, k: int = 6, offset: int = 0) -> List[Dict]:
        ...

    async def get_many(self, ids: List[str]) -> List[Dict]:
        ...

    async def all(self) -> List[Dict]:
        ...

    async def delete(self, item_id: str) -> None:
        ...

    async def count(self) -> int:
        ...

    async def bulk_delete(self, ids: List[str]) -> int:
        ...

    async def sessions(self) -> List[str]:
        ...

    async def stats(self) -> Dict:
        ...


__all__ = ["MemoryStore", "MemoryItem"]
