"""File management tool offering read/write/patch operations."""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from .base import Tool
from .registry import register_tool
from ..config import load_runtime_config


class FilesTool(Tool):
    name = "files"
    description = "Read, write, and patch files with diff previews."

    def __init__(self, *, workspace: Optional[Path] = None) -> None:
        config = load_runtime_config()
        self.workspace = Path(workspace or config.tools.workspace).resolve()

    def _resolve(self, path: str) -> Path:
        target = (self.workspace / path).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError("Path escapes workspace")
        return target

    async def run(self, action: str, path: str, content: str = "", **options: Any) -> str:
        if action == "read":
            return self._read(path)
        if action == "write":
            return self._write(path, content, overwrite=options.get("overwrite", True))
        if action == "patch":
            return self._patch(path, content)
        return f"files: unknown action '{action}'"

    def _read(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"files: '{path}' not found"
        return target.read_text()

    def _write(self, path: str, content: str, *, overwrite: bool) -> str:
        target = self._resolve(path)
        if target.exists() and not overwrite:
            return f"files: '{path}' exists (overwrite disabled)"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"files: wrote {len(content)} chars to '{path}'"

    def _patch(self, path: str, replacement: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"files: '{path}' not found"
        original = target.read_text()
        diff = "\n".join(difflib.unified_diff(original.splitlines(), replacement.splitlines(), lineterm=""))
        target.write_text(replacement)
        return f"files: applied patch to '{path}'\n{diff}"


register_tool(FilesTool)
