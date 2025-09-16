"""Unified tool registry for the runtime."""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {"registry", "ToolRegistry", "ToolEntry", "register_tool"}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = importlib.import_module('.registry', __name__)
        return getattr(module, name)
    raise AttributeError(name)


__all__ = sorted(_EXPORTS)
