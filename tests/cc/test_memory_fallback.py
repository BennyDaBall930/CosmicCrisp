import pytest

from cosmiccrisp.memory.fallback import MemoryWithFallback
from cosmiccrisp.memory.null import NullMemory
from cosmiccrisp.memory.interface import AgentMemory


class BadMemory(AgentMemory):
    async def enter(self, session_id: str) -> None:
        raise RuntimeError("boom")

    async def add(self, item):
        raise RuntimeError("boom")

    async def similar(self, query: str, k: int = 5):
        raise RuntimeError("boom")

    async def reset(self):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_fallback_add():
    mem = MemoryWithFallback(BadMemory(), NullMemory())
    await mem.enter("x")
    assert await mem.add({"a": 1}) == "0"
