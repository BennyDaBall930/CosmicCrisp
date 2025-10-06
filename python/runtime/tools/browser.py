"""Browser automation tool leveraging browser-use when available."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from .base import Tool
from .registry import register_tool
from python.helpers.settings import get_settings

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
        settings = get_settings()
        config_kwargs: Dict[str, Any] = {
            "headless": settings.get("browser_use_headless", True),
            "stealth": settings.get("browser_use_disable_automation", True),
        }

        user_agent = settings.get("browser_use_user_agent", "")
        if user_agent:
            config_kwargs["user_agent"] = user_agent

        headers = settings.get("browser_use_extra_headers", {}) or {}
        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except json.JSONDecodeError:
                headers = {}
        if headers:
            config_kwargs["headers"] = headers

        timeout_secs = max(int(settings.get("browser_use_timeout_secs", 90)), 0)
        if timeout_secs:
            config_kwargs["timeout"] = timeout_secs * 1000

        config = BrowserConfig(**config_kwargs)
        agent = BrowserAgent(goal=task, config=config)

        max_attempts = 2
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = await agent.run()
                self._failures[session_key] = 0
                return str(result)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning("browser-use attempt %s failed: %s", attempt, exc)
                if attempt < max_attempts:
                    await asyncio.sleep(min(2 * attempt, 5))
                    continue

        count = self._failures.get(session_key, 0) + 1
        self._failures[session_key] = count
        if count >= 2:
            return (
                "browser: human intervention required. "
                "Use the continue endpoint once manual steps are complete."
            )
        return f"browser: retry suggested after error: {last_error}"


register_tool(BrowserTool)
