import pytest

from python.runtime.tools.search import SearchResult


def test_search_result_bullet_format():
    result = SearchResult(title='Example', url='https://example.com', snippet='Snippet text')
    bullet = result.to_bullet()
    assert bullet.startswith('- Example (https://example.com)')
    assert 'Snippet text' in bullet


@pytest.mark.parametrize('snippet', ['A short summary', ''])
def test_search_result_handles_missing_snippet(snippet):
    result = SearchResult(title='Title', url='https://example', snippet=snippet)
    assert result.to_bullet().endswith(snippet or '')
