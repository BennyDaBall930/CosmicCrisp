"""Simple asynchronous event bus for runtime notifications."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Set


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: Dict[str, Any]) -> None:
        if not self._subscribers:
            return
        async with self._lock:
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop if subscriber is slow
                    pass

    async def subscribe(self) -> AsyncIterator[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)


__all__ = ["EventBus"]
