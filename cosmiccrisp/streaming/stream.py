"""Utilities for streaming responses."""
from __future__ import annotations

from typing import AsyncGenerator, Iterable


async def stream_generator(gen: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    async for chunk in gen:
        yield chunk.encode()
