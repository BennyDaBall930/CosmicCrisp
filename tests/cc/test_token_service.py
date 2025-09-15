from cosmiccrisp.tokenizer.token_service import TokenService


def test_trim_adjusts():
    ts = TokenService(max_total=100)
    prompt = "word " * 90
    assert ts.trim(prompt, 20) == 10


def test_trim_zero_when_exceeded():
    ts = TokenService(max_total=5)
    prompt = "word " * 5
    assert ts.trim(prompt, 10) == 0
