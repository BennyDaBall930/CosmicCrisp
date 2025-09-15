"""Registry for tools."""
from __future__ import annotations

from typing import Dict

from .base import Tool


class ToolRegistry(dict):
    def register(self, tool: Tool) -> None:
        self[tool.name] = tool


registry = ToolRegistry()
