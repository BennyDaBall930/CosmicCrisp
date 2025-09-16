"""Guarded shell execution tool."""
from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any, Iterable, Optional

from .base import Tool
from .registry import register_tool
from ..config import load_runtime_config


class ShellTool(Tool):
    name = "shell"
    description = "Execute whitelisted shell commands inside the workspace."

    def __init__(self, *, allow_list: Optional[Iterable[str]] = None, workspace: Optional[Path] = None) -> None:
        config = load_runtime_config()
        self.allow_list = set(allow_list or config.tools.shell_allow_list)
        self.workspace = Path(workspace or config.tools.workspace).resolve()

    async def run(self, command: str, **_: Any) -> str:
        tokens = shlex.split(command)
        if not tokens:
            return "shell: empty command"
        if tokens[0] not in self.allow_list:
            return f"shell: command '{tokens[0]}' not permitted"
        process = await asyncio.create_subprocess_exec(
            *tokens,
            cwd=str(self.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        output = stdout.decode().strip()
        return output or f"shell: command exited {process.returncode}"


register_tool(ShellTool)
