"""Optional Mem0-backed memory adapter."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..schema import MemoryItem
from ..store import MemoryStore

try:  # pragma: no cover - optional dependency
    from mem0 import Mem0
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Mem0 = None  # type: ignore

logger = logging.getLogger(__name__)


class Mem0Adapter(MemoryStore):
    """Thin wrapper around the mem0 client with optional local fallback."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        namespace: str = "applezero",
        fallback: Optional[MemoryStore] = None,
    ) -> None:
        self._fallback = fallback
        if Mem0 is None:
            logger.warning("mem0 client not installed; using fallback memory store")
            self._client = None
        else:
            self._client = Mem0(api_key=api_key, base_url=base_url, namespace=namespace)

    async def enter(self, session_id: str) -> None:  # type: ignore[override]
        if self._fallback is not None and hasattr(self._fallback, "enter"):
            await getattr(self._fallback, "enter")(session_id)  # type: ignore[misc]

    async def add(self, item: MemoryItem | dict) -> str:  # type: ignore[override]
        if self._client is None:
            if self._fallback is None:
                raise RuntimeError("Mem0 unavailable and no fallback configured")
            return await self._fallback.add(item)
        payload = item.model_dump() if isinstance(item, MemoryItem) else dict(item)
        result = await asyncio.to_thread(self._client.add, payload)
        item_id = result.get("id") if isinstance(result, dict) else None
        return str(item_id or "mem0")

    async def similar(self, query: str, k: int = 6):  # type: ignore[override]
        if self._client is None:
            if self._fallback is None:
                return []
            return await self._fallback.similar(query, k)
        result = await asyncio.to_thread(self._client.search, query, k)
        items = []
        if isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("text"):
                    items.append(MemoryItem(**entry).model_dump())
        return items

    async def recent(self, k: int = 6):  # type: ignore[override]
        if self._client is None:
            if self._fallback is None:
                return []
            return await self._fallback.recent(k)
        result = await asyncio.to_thread(self._client.recent, k)
        items = []
        if isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("text"):
                    items.append(MemoryItem(**entry).model_dump())
        return items

    async def delete(self, item_id: str) -> None:  # type: ignore[override]
        if self._client is None:
            if self._fallback is not None:
                await self._fallback.delete(item_id)
            return
        await asyncio.to_thread(self._client.delete, item_id)

    async def count(self) -> int:  # type: ignore[override]
        if self._client is None:
            if self._fallback is None:
                return 0
            return await self._fallback.count()
        result = await asyncio.to_thread(self._client.count)
        return int(result or 0)


__all__ = ["Mem0Adapter"]
