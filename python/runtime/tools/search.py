"""Asynchronous web search tool with simple streaming summary."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

try:  # pragma: no cover - optional dependency
    import aiohttp
except ModuleNotFoundError:  # pragma: no cover
    aiohttp = None  # type: ignore

from urllib.parse import quote

from .base import Tool
from .registry import register_tool

DEFAULT_TIMEOUT = 10


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_bullet(self) -> str:
        return f"- {self.title} ({self.url})\n  {self.snippet}"


class SearchTool(Tool):
    name = "search"
    description = "General web search returning top snippets."

    async def run(self, query: str, *, limit: int = 3, **_: Any) -> str:
        if not query:
            return "search: empty query"
        if aiohttp is None:
            return "search: aiohttp not available"
        searx_url = os.getenv("SEARXNG_URL")
        if searx_url:
            results = await self._searx_search(searx_url, query, limit)
        else:
            results = await self._duckduckgo_search(query, limit)
        if not results:
            return f"search: no results for '{query}'"
        bullets = "\n".join(item.to_bullet() for item in results)
        return f"Search results for '{query}':\n{bullets}"

    async def _searx_search(self, base_url: str, query: str, limit: int) -> List[SearchResult]:
        params = {"q": query, "format": "json"}
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base_url, params=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        results = []
        for item in payload.get("results", [])[:limit]:
            results.append(
                SearchResult(
                    title=item.get("title", "untitled"),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:240],
                )
            )
        return results

    async def _duckduckgo_search(self, query: str, limit: int) -> List[SearchResult]:
        url = "https://duckduckgo.com/?q=" + quote(query) + "&format=json&pretty=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.text()
        results: List[SearchResult] = []
        for line in data.splitlines():
            if 'FirstURL' in line and len(results) < limit:
                url_part = line.split(':', 1)[1].strip().strip(',')
                url_clean = url_part.strip('"')
                if url_clean:
                    results.append(
                        SearchResult(
                            title=url_clean,
                            url=url_clean,
                            snippet="Result preview unavailable",
                        )
                    )
        return results


register_tool(SearchTool)
