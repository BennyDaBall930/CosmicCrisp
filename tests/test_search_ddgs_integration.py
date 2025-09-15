import pytest
import types


@pytest.mark.asyncio
async def test_duckduckgo_unavailable_message(capsys, monkeypatch):
    # Import module fresh
    import importlib
    sp = importlib.import_module('python.helpers.search_providers')

    # Ensure cache is empty and DDGS is treated as unavailable
    monkeypatch.setattr(sp, 'DDGS', None, raising=False)

    # Call function
    result = await sp.search_duckduckgo("test query", max_results=1)

    # Should gracefully return None and print notice about ddgs
    captured = capsys.readouterr()
    assert "`ddgs` is not installed" in captured.out
    assert result is None


@pytest.mark.asyncio
async def test_duckduckgo_uses_ddgs_with_backoff(monkeypatch):
    import importlib
    sp = importlib.import_module('python.helpers.search_providers')

    # Avoid real sleeping to keep tests fast
    async def no_sleep(_):
        return None
    monkeypatch.setattr(sp.asyncio, 'sleep', no_sleep)

    # Custom exception to simulate ddgs error
    class MyDDGSException(Exception):
        pass

    # Fake DDGS class implementing context manager and text method
    class FakeDDGS:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5, **kwargs):
            # Fail first two attempts, then succeed
            FakeDDGS.calls += 1
            if FakeDDGS.calls < 3:
                raise MyDDGSException("temporary ddgs failure")
            return [
                {"title": "Result 1", "href": "https://example.com/1", "body": "Body 1"},
                {"title": "Result 2", "href": "https://example.com/2", "body": "Body 2"},
            ][:max_results]

    # Patch DDGS and the corresponding exception in the module
    monkeypatch.setattr(sp, 'DDGS', FakeDDGS)
    monkeypatch.setattr(sp, 'DDGSException', MyDDGSException)

    # Run the search; should succeed after a couple retries
    results = await sp.search_duckduckgo("some query", max_results=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(set(r.keys()) >= {"title", "href", "body"} for r in results)
