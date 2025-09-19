from python.runtime.tokenizer.token_service import TokenService


def test_trim_respects_available_budget():
    service = TokenService(max_total=100)
    prompt = 'word ' * 45
    remaining = service.trim(prompt, requested=30)
    assert remaining <= 30
    assert remaining == service.trim(prompt, requested=remaining)
