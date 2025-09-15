"""Compatibility layer for legacy ``cosmiccrisp`` imports.

The runtime has been merged into :mod:`python.runtime`.  To avoid breaking
external imports we lazily alias the old module path to the new location.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Dict


_ALIASES: Dict[str, str] = {
    "cosmiccrisp.agent": "python.runtime.agent",
    "cosmiccrisp.agent.helpers": "python.runtime.agent.helpers",
    "cosmiccrisp.agent.prompts": "python.runtime.agent.prompts",
    "cosmiccrisp.agent.service": "python.runtime.agent.service",
    "cosmiccrisp.agent.task_parser": "python.runtime.agent.task_parser",
    "cosmiccrisp.api": "python.runtime.api",
    "cosmiccrisp.api.app": "python.runtime.api.app",
    "cosmiccrisp.memory": "python.runtime.memory",
    "cosmiccrisp.memory.fallback": "python.runtime.memory.fallback",
    "cosmiccrisp.memory.interface": "python.runtime.memory.interface",
    "cosmiccrisp.memory.null": "python.runtime.memory.null",
    "cosmiccrisp.memory.sqlite_faiss": "python.runtime.memory.sqlite_faiss",
    "cosmiccrisp.streaming": "python.runtime.streaming",
    "cosmiccrisp.streaming.stream": "python.runtime.streaming.stream",
    "cosmiccrisp.tokenizer": "python.runtime.tokenizer",
    "cosmiccrisp.tokenizer.token_service": "python.runtime.tokenizer.token_service",
    "cosmiccrisp.tools": "python.runtime.tools",
    "cosmiccrisp.tools.base": "python.runtime.tools.base",
    "cosmiccrisp.tools.browser": "python.runtime.tools.browser",
    "cosmiccrisp.tools.code": "python.runtime.tools.code",
    "cosmiccrisp.tools.registry": "python.runtime.tools.registry",
    "cosmiccrisp.tools.search": "python.runtime.tools.search",
}


def _ensure_alias(alias: str, target: str) -> ModuleType:
    module = importlib.import_module(target)
    sys.modules.setdefault(alias, module)
    if alias.count(".") == 1:
        setattr(sys.modules[__name__], alias.split(".")[1], module)
    return module


def __getattr__(name: str) -> ModuleType:
    target = _ALIASES.get(f"cosmiccrisp.{name}")
    if not target:
        raise AttributeError(name)
    module = _ensure_alias(f"cosmiccrisp.{name}", target)
    return module


for alias, target in _ALIASES.items():
    _ensure_alias(alias, target)

__all__ = []
