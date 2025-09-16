"""Enhanced tool registry with plugin loading and enable/disable controls."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..config import RuntimeConfig, ToolsConfig, load_runtime_config
from .base import Tool

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    name: str
    factory: Callable[[], Tool]
    enabled: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)
    instance: Optional[Tool] = None

    def resolve(self) -> Tool:
        if self.instance is None:
            self.instance = self.factory()
        return self.instance


class ToolRegistry:
    def __init__(self, config: Optional[ToolsConfig] = None) -> None:
        self._config = config or load_runtime_config().tools
        self._entries: Dict[str, ToolEntry] = {}
        self._loaded_modules: set[str] = set()

    # ------------------------------------------------------------------
    def register(
        self,
        tool: Tool | Callable[[], Tool] | type,
        *,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        entry = self._build_entry(tool, name=name, description=description, tags=tags)
        enabled = self._is_enabled(entry.name)
        entry.enabled = enabled
        self._entries[entry.name] = entry
        logger.debug("Registered tool %s (enabled=%s)", entry.name, enabled)

    def _build_entry(
        self,
        tool: Tool | Callable[[], Tool] | type,
        *,
        name: Optional[str],
        description: str,
        tags: Optional[Iterable[str]],
    ) -> ToolEntry:
        factory: Callable[[], Tool]
        if isinstance(tool, type):
            tool_cls = tool
            tool_name = name or getattr(tool_cls, "name", tool_cls.__name__.lower())

            def _factory() -> Tool:
                return tool_cls()  # type: ignore[call-arg]

            factory = _factory
        elif callable(tool) and not hasattr(tool, "run"):
            factory = tool  # type: ignore[assignment]
            probe = factory()
            tool_name = name or getattr(probe, "name", probe.__class__.__name__.lower())
            return ToolEntry(
                name=tool_name,
                factory=factory,
                description=description or getattr(probe, "description", ""),
                tags=list(tags or getattr(probe, "tags", [])),
                instance=probe,
            )
        else:
            instance = tool if hasattr(tool, "run") else tool()  # type: ignore[call-arg]
            tool_name = name or getattr(instance, "name", instance.__class__.__name__.lower())
            return ToolEntry(
                name=tool_name,
                factory=lambda: instance,  # type: ignore[misc]
                description=description or getattr(instance, "description", ""),
                tags=list(tags or getattr(instance, "tags", [])),
                instance=instance,
            )

        return ToolEntry(
            name=tool_name,
            factory=factory,
            description=description,
            tags=list(tags or []),
        )

    # ------------------------------------------------------------------
    def load_from_modules(self, modules: Iterable[str]) -> None:
        for module_path in modules:
            if not module_path or module_path in self._loaded_modules:
                continue
            try:
                importlib.import_module(module_path)
                self._loaded_modules.add(module_path)
                logger.debug("Loaded tool module %s", module_path)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to load tool module %s: %s", module_path, exc)

    def load_plugins(self) -> None:
        self.load_from_modules(self._config.plugins)

    # ------------------------------------------------------------------
    def _is_enabled(self, name: str) -> bool:
        enabled_list = {entry.lower() for entry in self._config.enabled}
        disabled_list = {entry.lower() for entry in self._config.disabled}
        if enabled_list and name.lower() not in enabled_list:
            return False
        if name.lower() in disabled_list:
            return False
        return True

    def enable(self, name: str) -> None:
        entry = self._entries.get(name)
        if entry:
            entry.enabled = True

    def disable(self, name: str) -> None:
        entry = self._entries.get(name)
        if entry:
            entry.enabled = False

    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[Tool]:
        entry = self._entries.get(name)
        if not entry or not entry.enabled:
            return None
        return entry.resolve()

    def items(self) -> Iterable[tuple[str, ToolEntry]]:
        return self._entries.items()


_PENDING: List[tuple[Any, Dict[str, Any]]] = []
_GLOBAL_REGISTRY: Optional[ToolRegistry] = None


def register_tool(
    tool: Tool | Callable[[], Tool] | type,
    *,
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[Iterable[str]] = None,
) -> None:
    if _GLOBAL_REGISTRY is None:
        _PENDING.append((tool, {"name": name, "description": description, "tags": tags}))
    else:
        _GLOBAL_REGISTRY.register(tool, name=name, description=description, tags=tags)


def _create_registry() -> ToolRegistry:
    global _GLOBAL_REGISTRY
    config: RuntimeConfig = load_runtime_config()
    registry = ToolRegistry(config.tools)
    _GLOBAL_REGISTRY = registry
    registry.load_from_modules(config.tools.auto_modules)
    registry.load_plugins()
    while _PENDING:
        tool, kwargs = _PENDING.pop(0)
        registry.register(tool, **kwargs)
    return registry


registry = _create_registry()

__all__ = ["ToolRegistry", "ToolEntry", "registry", "register_tool"]
