"""Runtime memory exports."""
from __future__ import annotations

from typing import Optional

from ..config import MemoryConfig, RuntimeConfig
from .fallback import MemoryWithFallback
from .interface import AgentMemory
from .null import NullMemory
from .schema import MemoryItem
from .sqlite_faiss import SQLiteFAISSMemory
from .store import MemoryStore


def _resolve_config(config: RuntimeConfig | MemoryConfig | None) -> MemoryConfig:
    if isinstance(config, RuntimeConfig):
        return config.memory
    if isinstance(config, MemoryConfig):
        return config
    return MemoryConfig()


def get_memory(
    config: RuntimeConfig | MemoryConfig | None = None,
    *,
    embeddings: Optional[object] = None,
) -> MemoryStore:
    memory_config = _resolve_config(config)
    store: MemoryStore = SQLiteFAISSMemory(
        path=str(memory_config.db_path), embeddings=embeddings
    )
    return store


__all__ = [
    "AgentMemory",
    "MemoryItem",
    "MemoryStore",
    "MemoryWithFallback",
    "NullMemory",
    "SQLiteFAISSMemory",
    "get_memory",
]
