"""Search tool using SearXNG/Serper with graceful fallback."""
from __future__ import annotations

import os
from typing import Any

import requests

from .base import Tool


class SearchTool(Tool):
    name = "search"

    async def run(self, query: str, **_: Any) -> str:
        url = os.getenv("SEARXNG_URL")
        if not url:
            return f"search unavailable: {query}"
        try:
            resp = requests.get(url, params={"q": query}, timeout=5)
            data = resp.json()
            if data.get("results"):
                first = data["results"][0]
                return first.get("title", "no result")
        except Exception:
            pass
        return f"search unavailable: {query}"


# Register
from .registry import registry

registry.register(SearchTool())
