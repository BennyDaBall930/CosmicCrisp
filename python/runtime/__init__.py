"""Unified Apple Zero runtime package.

Phase 1 of the consolidation exposes container-managed singletons so the rest
of the application can resolve embeddings, memory, tokens, and tools from a
single place.
"""

from .agent.prompt_manager import PromptManager
from .agent.service import AgentService
from .container import (
    embeddings,
    get_config,
    get_embeddings_service,
    get_memory_store,
    get_model_router,
    get_observability,
    get_token_service,
    get_tts_provider,
    memory,
    model_router,
    observability,
    tokens,
    tts,
    tool_registry,
)
from .tokenizer.token_service import TokenService

__all__ = [
    "AgentService",
    "PromptManager",
    "TokenService",
    "embeddings",
    "memory",
    "observability",
    "model_router",
    "tts",
    "tokens",
    "tool_registry",
    "get_config",
    "get_embeddings_service",
    "get_memory_store",
    "get_model_router",
    "get_observability",
    "get_token_service",
    "get_tts_provider",
]
