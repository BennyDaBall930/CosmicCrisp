"""Token budgeting behaviour tests."""
from __future__ import annotations

import pytest

from python.runtime.config import TokenConfig
from python.runtime.tokenizer.token_service import TokenService


def _make_service(budget: int = 20) -> TokenService:
    config = TokenConfig(budgets={"test-model": budget}, summarizer_model="test-model")
    service = TokenService(config)
    service._count_text = lambda text: max(1, len((text or "").split()))  # type: ignore[attr-defined]
    return service


def test_fit_includes_memory_and_stays_within_budget():
    service = _make_service(budget=18)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "message one two"},
        {"role": "assistant", "content": "reply three four"},
    ]
    fitted = service.fit(messages, "test-model", ["important detail", "secondary fact"])

    assert any(m["content"].startswith("Relevant memory") for m in fitted)
    assert service.count(fitted) <= service.budget_for("test-model")


def test_fit_summarizes_history_when_over_budget():
    service = _make_service(budget=15)
    sequence = [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    conversation = [{"role": "system", "content": "system prompt"}] + [
        {"role": role, "content": f"{role} message {idx} with several words"}
        for idx, role in enumerate(sequence, start=1)
    ]

    original_count = service.count(conversation)
    fitted = service.fit(conversation, "test-model", [])
    joined_roles = [msg["role"] for msg in fitted]
    assert service.count(fitted) < original_count
    assert any(
        "Conversation summary" in msg["content"]
        for msg in fitted
        if msg["role"] == "system"
    )
    # ensure the oldest user message was dropped when summarizing
    assert joined_roles.count("user") < 3
