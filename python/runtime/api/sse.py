"""Utilities for Server-Sent Events streaming."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict


async def sse_iter(source: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[bytes]:
    try:
        async for event in source:
            name = event.get("event", "event")
            payload = event.get("data", {})
            if isinstance(payload, (dict, list)):
                data_str = json.dumps(payload)
            else:
                data_str = str(payload)
            chunk = f"event: {name}\n" f"data: {data_str}\n\n"
            yield chunk.encode("utf-8")
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        # cooperative shutdown
        raise


__all__ = ["sse_iter"]
