import os
import pytest

from cosmiccrisp.tools.search import SearchTool


@pytest.mark.asyncio
async def test_search_offline():
    os.environ.pop("SEARXNG_URL", None)
    tool = SearchTool()
    result = await tool.run("test query")
    assert "unavailable" in result
