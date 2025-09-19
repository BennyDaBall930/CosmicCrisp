import pytest

from python.runtime.tools.search import SearchTool


@pytest.mark.asyncio
async def test_search_offline(monkeypatch):
    monkeypatch.setattr('python.runtime.tools.search.aiohttp', None, raising=False)
    tool = SearchTool()
    result = await tool.run('test query')
    assert result == 'search: aiohttp not available'
