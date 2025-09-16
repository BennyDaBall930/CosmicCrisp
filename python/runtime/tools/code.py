"""Code execution tool providing a lightweight async REPL."""
from __future__ import annotations

import asyncio
import contextlib
import io
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Tool
from .registry import register_tool
from ..config import load_runtime_config


class CodeTool(Tool):
    name = "code"
    description = "Execute Python code snippets in a temporary sandbox and capture output."

    def __init__(self, *, workspace: Optional[Path] = None) -> None:
        config = load_runtime_config()
        self.workspace = Path(workspace or config.tools.workspace).resolve()

    async def run(self, code: str, *, filename: Optional[str] = None, run_tests: bool = False, **_: Any) -> str:
        if not code:
            return "code: empty snippet"
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._execute, code, filename, run_tests)
        return result

    def _execute(self, code: str, filename: Optional[str], run_tests: bool) -> str:
        buffer = io.StringIO()
        namespace: Dict[str, Any] = {}
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                compiled = compile(code, filename or "<snippet>", "exec")
                exec(compiled, namespace, namespace)
                if run_tests and callable(namespace.get("main")):
                    namespace["main"]()
            except Exception as exc:  # pragma: no cover
                print(f"Execution error: {exc}")
        return buffer.getvalue().strip() or "code: execution completed with no output"


register_tool(CodeTool)
