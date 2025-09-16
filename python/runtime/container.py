"""Lazy dependency container for runtime singletons."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Generic, TypeVar

from .config import RuntimeConfig, load_runtime_config

T = TypeVar("T")


class _LazyProxy(Generic[T]):
    """Proxy object that initializes the underlying dependency on access."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory

    def __call__(self) -> T:  # pragma: no cover - convenience hook
        return self._factory()

    def __getattr__(self, item: str):  # pragma: no cover - dispatch attribute access
        return getattr(self._factory(), item)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<LazyProxy factory={self._factory!r}>"


@lru_cache(maxsize=1)
def get_config() -> RuntimeConfig:
    return load_runtime_config()


@lru_cache(maxsize=1)
def get_event_bus():
    from .event_bus import EventBus

    return EventBus()


@lru_cache(maxsize=1)
def _get_embeddings_service():
    from .embeddings import get_embeddings

    return get_embeddings(get_config())


@lru_cache(maxsize=1)
def get_embeddings_service():
    return _get_embeddings_service()


@lru_cache(maxsize=1)
def get_memory_store():
    from .memory import get_memory

    return get_memory(get_config(), embeddings=get_embeddings_service())


@lru_cache(maxsize=1)
def get_token_service():
    from .tokenizer.token_service import TokenService

    return TokenService(get_config())


@lru_cache(maxsize=1)
def get_tool_registry():
    from .tools.registry import registry

    return registry


@lru_cache(maxsize=1)
def get_prompt_manager():
    from .agent.prompt_manager import PromptManager

    return PromptManager(persona=get_config().agent.persona)


@lru_cache(maxsize=1)
def get_orchestrator():
    from .agent.orchestrator import AgentOrchestrator

    return AgentOrchestrator(
        memory=get_memory_store(),
        token_service=get_token_service(),
        tool_registry=get_tool_registry(),
        config=get_config(),
        prompt_manager=get_prompt_manager(),
        embeddings=get_embeddings_service(),
        event_bus=get_event_bus(),
    )


@lru_cache(maxsize=1)
def get_agent_service():
    from .agent.service import AgentService

    return AgentService(
        memory=get_memory_store(),
        token_service=get_token_service(),
        tool_registry=get_tool_registry(),
        config=get_config(),
        orchestrator=get_orchestrator(),
        prompt_manager=get_prompt_manager(),
        embeddings=get_embeddings_service(),
    )


embeddings = _LazyProxy(get_embeddings_service)
memory = _LazyProxy(get_memory_store)
tokens = _LazyProxy(get_token_service)
tool_registry = get_tool_registry
event_bus = _LazyProxy(get_event_bus)
orchestrator = _LazyProxy(get_orchestrator)
agent_service = _LazyProxy(get_agent_service)


__all__ = [
    "embeddings",
    "memory",
    "tokens",
    "tool_registry",
    "event_bus",
    "orchestrator",
    "agent_service",
    "get_config",
    "get_embeddings_service",
    "get_memory_store",
    "get_token_service",
    "get_tool_registry",
    "get_event_bus",
    "get_orchestrator",
    "get_agent_service",
    "get_prompt_manager",
]
