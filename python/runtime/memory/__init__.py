"""Runtime memory exports."""
from __future__ import annotations

import os
from typing import Optional

from ..config import MemoryConfig, RuntimeConfig
from .fallback import MemoryWithFallback
from .interface import AgentMemory
from .null import NullMemory
from .schema import MemoryItem
from .sqlite_faiss import SQLiteFAISSMemory
from .store import MemoryStore
from .stores.mem0_adapter import Mem0Adapter


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
    if memory_config.mem0_enabled:
        api_key = os.getenv("MEM0_API_KEY")
        base_url = os.getenv("MEM0_BASE_URL")
        store = Mem0Adapter(api_key=api_key, base_url=base_url, fallback=store)
    return store


__all__ = [
    "AgentMemory",
    "MemoryItem",
    "MemoryStore",
    "MemoryWithFallback",
    "Mem0Adapter",
    "NullMemory",
    "SQLiteFAISSMemory",
    "get_memory",
]
