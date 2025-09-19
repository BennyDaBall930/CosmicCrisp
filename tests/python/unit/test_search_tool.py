from __future__ import annotations

import asyncio
import os
from typing import List

import pytest
from hypothesis import given, strategies as st

from python.runtime.tools.search import SearchResult, SearchTool


@pytest.mark.asyncio
async def test_run_empty_query_returns_hint():
    tool = SearchTool()
    result = await tool.run("")
    assert result == "search: empty query"


@pytest.mark.asyncio
async def test_run_handles_missing_results(monkeypatch):
    monkeypatch.delenv('SEARXNG_URL', raising=False)

    async def fake_search(_self, _query: str, _limit: int) -> List[SearchResult]:
        return []

    tool = SearchTool()
    tool._duckduckgo_search = fake_search.__get__(tool, SearchTool)  # type: ignore[attr-defined]
    tool._searx_search = fake_search.__get__(tool, SearchTool)  # type: ignore[attr-defined]
    result = await tool.run("cosmic crisp", limit=5)
    assert result == "search: no results for 'cosmic crisp'"


@pytest.mark.asyncio
async def test_run_formats_results(monkeypatch):
    async def fake_search(_url: str, _query: str, _limit: int) -> List[SearchResult]:
        return [
            SearchResult(title="Doc", url="https://example.com", snippet="Snippet"),
            SearchResult(title="Note", url="https://docs", snippet="Note snippet"),
        ]

    tool = SearchTool()
    monkeypatch.setattr(tool, "_searx_search", fake_search)
    result = await tool.run("documentation", limit=2)
    assert "Search results for 'documentation'" in result
    assert "- Doc (https://example.com)" in result
    assert "Snippet" in result


@given(st.text(min_size=1, max_size=60))
def test_no_results_message_contains_query(query: str):
    os.environ.pop('SEARXNG_URL', None)
    tool = SearchTool()

    async def fake_search(_self, __: str, ___: int) -> List[SearchResult]:
        return []

    async def run() -> str:
        tool._duckduckgo_search = fake_search.__get__(tool, SearchTool)  # type: ignore[attr-defined]
        tool._searx_search = fake_search.__get__(tool, SearchTool)  # type: ignore[attr-defined]
        return await tool.run(query, limit=1)

    message = asyncio.run(run())
    assert f"'{query}'" in message
