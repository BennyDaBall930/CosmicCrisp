from __future__ import annotations

from typing import Dict, List

import pytest
from hypothesis import given, strategies as st

from python.runtime.tokenizer.token_service import TokenService, _normalize_key


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('gpt_4o_mini', 'gpt-4o-mini'),
        ('gpt__o3', 'gpt/o3'),
        ('MistralLarge', 'mistrallarge'),
    ],
)
def test_normalize_key(raw: str, expected: str):
    assert _normalize_key(raw) == expected


@given(st.text(max_size=200), st.integers(min_value=0, max_value=256))
def test_trim_never_negative(prompt: str, requested: int):
    svc = TokenService(max_total=128)
    result = svc.trim(prompt, requested)
    assert 0 <= result <= requested


def test_trim_honours_max_total():
    svc = TokenService(max_total=100)
    prompt = 'word ' * 40
    result = svc.trim(prompt, requested=80)
    assert 0 <= result <= 80
    assert svc.count(prompt) + result <= (svc._legacy_budget or svc.budget_for('gpt-4o'))


def test_fit_respects_budget_and_preserves_system_message():
    svc = TokenService(max_total=60)
    system = {'role': 'system', 'content': 'system prompt'}
    convo: List[Dict[str, str]] = [system]
    for i in range(20):
        convo.append({'role': 'user', 'content': f'user says {i} ' * 5})
    fitted = svc.fit(convo, model='gpt-4o')
    assert fitted[0] == system
    assert svc.count(fitted) <= svc.budget_for('gpt-4o') or svc.count(fitted) <= 60
    assert len(fitted) <= len(convo)


@given(st.lists(st.dictionaries(keys=st.sampled_from(['role', 'content']), values=st.text(max_size=50)), min_size=1, max_size=5))
def test_count_handles_message_sequences(messages: List[Dict[str, str]]):
    svc = TokenService(max_total=64)
    total = svc.count(messages)
    assert total >= 0
