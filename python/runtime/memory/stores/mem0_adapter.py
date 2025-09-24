"""Optional Mem0-backed memory adapter."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from ..schema import MemoryItem
from ..store import MemoryStore

try:  # pragma: no cover - optional dependency
    from mem0 import Mem0 as _LegacyMem0
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    _LegacyMem0 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from mem0.client.main import MemoryClient as _Mem0Client
except Exception:  # pragma: no cover - optional dependency
    _Mem0Client = None  # type: ignore

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
        self._client_mode: Optional[str] = None
        self._namespace = namespace
        self._local_mode = os.getenv("MEM0_LOCAL_MODE", "").lower() in ("true", "1", "yes")
        client = None

        # Log initialization state
        if self._local_mode:
            logger.info("Mem0Adapter initialized in local mode with namespace: %s", namespace)
        else:
            logger.info("Mem0Adapter initialized in cloud mode with namespace: %s", namespace)

        if _LegacyMem0 is not None:
            try:
                client = _LegacyMem0(api_key=api_key, base_url=base_url, namespace=namespace)
                self._client_mode = "legacy"
            except TypeError:
                try:
                    client = _LegacyMem0(api_key=api_key, base_url=base_url)
                    self._client_mode = "legacy"
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("mem0 legacy client init failed: %s", exc)
                    client = None
        if client is None and _Mem0Client is not None:
            try:
                client = _Mem0Client(api_key=api_key, host=base_url or None)
                self._client_mode = "client"
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("mem0 client init failed: %s", exc)
                client = None

        if client is None:
            logger.warning("mem0 client not installed; using fallback memory store")
            self._client = None
        else:
            self._client = client

    async def enter(self, session_id: str) -> None:  # type: ignore[override]
        if self._fallback is not None and hasattr(self._fallback, "enter"):
            await getattr(self._fallback, "enter")(session_id)  # type: ignore[misc]

    async def add(self, item: MemoryItem | dict) -> str:  # type: ignore[override]
        if self._client is None or self._client_mode is None:
            if self._fallback is None:
                raise RuntimeError("Mem0 unavailable and no fallback configured")
            return await self._fallback.add(item)

        payload = item.model_dump() if isinstance(item, MemoryItem) else dict(item)

        if self._client_mode == "legacy":
            result = await asyncio.to_thread(self._client.add, payload)
        else:
            text = payload.get("text") or ""
            metadata = dict(payload.get("meta", {}))
            if payload.get("tags"):
                metadata.setdefault("tags", list(payload["tags"]))
            if payload.get("kind"):
                metadata.setdefault("kind", payload["kind"])
            if payload.get("ts"):
                metadata.setdefault("ts", payload["ts"].isoformat() if hasattr(payload["ts"], "isoformat") else payload["ts"])
            try:
                result = await asyncio.to_thread(
                    self._client.add,
                    [{"role": "user", "content": text}],
                    user_id=self._namespace or None,
                    metadata=metadata or None,
                )
            except Exception as exc:
                logger.warning("mem0 add failed, using fallback: %s", exc)
                if self._fallback is None:
                    raise
                return await self._fallback.add(item)

        item_id = None
        if isinstance(result, dict):
            item_id = result.get("id") or result.get("memory_id") or result.get("memoryId")
            if item_id is None and isinstance(result.get("memory"), dict):
                item_id = result["memory"].get("id") or result["memory"].get("memory_id")
        return str(item_id or "mem0")

    async def similar(self, query: str, k: int = 6):  # type: ignore[override]
        if self._client is None or self._client_mode is None:
            if self._fallback is None:
                return []
            return await self._fallback.similar(query, k)

        if self._client_mode == "legacy":
            result = await asyncio.to_thread(self._client.search, query, k)
        else:
            try:
                result = await asyncio.to_thread(
                    self._client.search,
                    query,
                    limit=k,
                    user_id=self._namespace or None,
                )
            except Exception as exc:
                logger.warning("mem0 search failed, using fallback: %s", exc)
                if self._fallback is None:
                    return []
                return await self._fallback.similar(query, k)

        items = []
        if isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("text"):
                    items.append(MemoryItem(**entry).model_dump())
                elif isinstance(entry, dict) and entry.get("memory"):
                    items.append({"text": entry.get("memory"), "meta": {k: v for k, v in entry.items() if k != "memory"}})
        return items

    async def recent(self, k: int = 6):  # type: ignore[override]
        if self._client is None or self._client_mode is None:
            if self._fallback is None:
                return []
            return await self._fallback.recent(k)

        if self._client_mode == "legacy":
            result = await asyncio.to_thread(self._client.recent, k)
        else:
            try:
                result = await asyncio.to_thread(
                    self._client.get_all,
                    limit=k,
                    user_id=self._namespace or None,
                )
            except Exception as exc:
                logger.warning("mem0 recent failed, using fallback: %s", exc)
                if self._fallback is None:
                    return []
                return await self._fallback.recent(k)

        items = []
        if isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("text"):
                    items.append(MemoryItem(**entry).model_dump())
                elif isinstance(entry, dict) and entry.get("memory"):
                    items.append({"text": entry.get("memory"), "meta": {k: v for k, v in entry.items() if k != "memory"}})
        return items

    async def delete(self, item_id: str) -> None:  # type: ignore[override]
        if self._client is None or self._client_mode is None:
            if self._fallback is not None:
                await self._fallback.delete(item_id)
            return

        if self._client_mode == "legacy":
            await asyncio.to_thread(self._client.delete, item_id)
            return

        try:
            await asyncio.to_thread(self._client.delete, item_id)
        except Exception as exc:
            logger.warning("mem0 delete failed: %s", exc)
            if self._fallback is not None:
                await self._fallback.delete(item_id)

    async def count(self) -> int:  # type: ignore[override]
        if self._client is None or self._client_mode is None:
            if self._fallback is None:
                return 0
            return await self._fallback.count()

        if self._client_mode == "legacy":
            result = await asyncio.to_thread(self._client.count)
            return int(result or 0)

        try:
            result = await asyncio.to_thread(
                self._client.get_all,
                user_id=self._namespace or None,
            )
        except Exception as exc:
            logger.warning("mem0 count failed, using fallback: %s", exc)
            if self._fallback is None:
                return 0
            return await self._fallback.count()
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
            return len(result["results"])
        return 0

    async def reindex(self) -> None:
        if self._fallback is not None:
            reindex = getattr(self._fallback, "reindex", None)
            if callable(reindex):
                await reindex()


__all__ = ["Mem0Adapter"]
