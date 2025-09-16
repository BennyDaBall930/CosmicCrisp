"""Browser automation tool leveraging browser-use when available."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import Tool
from .registry import register_tool

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    from browser_use import Agent as BrowserAgent  # type: ignore
    from browser_use import BrowserConfig
except Exception:  # pragma: no cover - fallback when browser_use missing
    BrowserAgent = None  # type: ignore
    BrowserConfig = None  # type: ignore


class BrowserTool(Tool):
    name = "browser"
    description = "Automated browser interactions with human-in-loop fallback."

    def __init__(self) -> None:
        self._failures: Dict[str, int] = {}

    async def run(
        self,
        goal: str,
        *,
        session_id: Optional[str] = None,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if BrowserAgent is None:
            return "browser: browser-use not installed"
        task = instructions or goal
        session_key = session_id or "default"
        config = BrowserConfig(headless=True, disable_automation=True)
        agent = BrowserAgent(goal=task, config=config)

        try:
            result = await agent.run()
        except Exception as exc:  # pragma: no cover
            count = self._failures.get(session_key, 0) + 1
            self._failures[session_key] = count
            if count >= 2:
                return (
                    "browser: CAPTCHA or block detected. Please intervene manually "
                    "and resume the session."
                )
            logger.warning("browser-use failure: %s", exc)
            return f"browser: retry suggested after error: {exc}"
        return str(result)


register_tool(BrowserTool)
