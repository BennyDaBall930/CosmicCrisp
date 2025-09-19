"""Browser tool fallbacks and retry behaviour."""
from __future__ import annotations

import types

import pytest

from python.runtime.tools import browser as browser_tool


class FailingAgent:
    def __init__(self, goal, config):  # pragma: no cover - simple init
        self.goal = goal
        self.config = config

    async def run(self):
        raise RuntimeError("blocked")


class SucceedingAgent:
    def __init__(self, goal, config):
        self.goal = goal
        self.config = config

    async def run(self):
        return "navigated"


@pytest.mark.asyncio
async def test_browser_tool_retries_and_escalates(monkeypatch):
    monkeypatch.setattr(browser_tool, "BrowserConfig", types.SimpleNamespace)
    monkeypatch.setattr(browser_tool, "BrowserAgent", FailingAgent)

    tool = browser_tool.BrowserTool()
    first = await tool.run("visit site", session_id="sess")
    second = await tool.run("visit site", session_id="sess")

    assert "retry suggested" in first
    assert "CAPTCHA" in second


@pytest.mark.asyncio
async def test_browser_tool_success_path(monkeypatch):
    monkeypatch.setattr(browser_tool, "BrowserConfig", types.SimpleNamespace)
    monkeypatch.setattr(browser_tool, "BrowserAgent", SucceedingAgent)

    tool = browser_tool.BrowserTool()
    result = await tool.run("check news")
    assert result == "navigated"
