"""Model-aware token budgeting and context fitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore

from ..config import RuntimeConfig, TokenConfig, load_runtime_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


def _normalize_key(key: str) -> str:
    return key.replace("__", "/").replace("_", "-").lower()


@dataclass(slots=True)
class _EncodingCache:
    name: str
    encoding: object


class TokenService:
    """Token counting and window fitting helper."""

    def __init__(
        self,
        config: RuntimeConfig | TokenConfig | None = None,
        *,
        max_total: int | None = None,
    ) -> None:
        tokens_config = self._resolve_config(config)
        budgets: Dict[str, int] = {
            _normalize_key(model): budget for model, budget in tokens_config.budgets.items()
        }
        if max_total is not None:
            default_key = _normalize_key(tokens_config.summarizer_model or DEFAULT_MODEL)
            budgets[default_key] = max_total
        self.budgets = budgets
        self.summarizer_model = tokens_config.summarizer_model or next(iter(self.budgets), DEFAULT_MODEL)
        self.default_model = self.summarizer_model
        self._legacy_budget = max_total
        self._encodings: Dict[str, _EncodingCache] = {}

    @staticmethod
    def _resolve_config(config: RuntimeConfig | TokenConfig | None) -> TokenConfig:
        if isinstance(config, RuntimeConfig):
            return config.tokens
        if isinstance(config, TokenConfig):
            return config
        runtime = load_runtime_config()
        return runtime.tokens

    # ------------------------------------------------------------------
    def budget_for(self, model: str) -> int:
        return self.budgets.get(_normalize_key(model), 0)

    # ------------------------------------------------------------------
    def count(self, text_or_messages) -> int:
        if isinstance(text_or_messages, str):
            return self._count_text(text_or_messages)
        if isinstance(text_or_messages, Sequence):
            total = 0
            for entry in text_or_messages:
                if isinstance(entry, str):
                    total += self._count_text(entry)
                elif isinstance(entry, MutableMapping):
                    total += self._count_text(str(entry.get("content", "")))
                else:
                    total += self._count_text(str(entry))
            return total
        return self._count_text(str(text_or_messages))

    def _count_text(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        if tiktoken is not None:
            try:
                encoding = self._get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:  # pragma: no cover - safety net when encoding fails
                pass
        return max(1, len(text.split()))

    def _get_encoding(self, name: str):
        cache = self._encodings.get(name)
        if cache is not None:
            return cache.encoding
        if tiktoken is None:
            raise RuntimeError("tiktoken not available")
        encoding = tiktoken.get_encoding(name)
        self._encodings[name] = _EncodingCache(name=name, encoding=encoding)
        return encoding

    # ------------------------------------------------------------------
    def fit(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        memory_snippets: Sequence[str] | None = None,
    ) -> List[Dict[str, str]]:
        """Trim and summarize context to fit model window."""

        messages = list(messages)
        budget = self.budget_for(model) or self.budget_for(self.summarizer_model)
        if budget <= 0:
            return messages

        system_message = messages[0] if messages and messages[0].get("role") == "system" else None
        conversation = messages[1:] if system_message else list(messages)
        memory_snippets = list(memory_snippets or [])

        memory_summary = ""
        if memory_snippets:
            memory_summary = "\n".join(f"- {snippet}" for snippet in memory_snippets)

        def build_context(
            recent_messages: List[Dict[str, str]],
            memory_text: str,
            summary_text: str,
        ) -> List[Dict[str, str]]:
            assembled: List[Dict[str, str]] = []
            if system_message:
                assembled.append(dict(system_message))
            if summary_text:
                assembled.append(
                    {
                        "role": "system",
                        "content": f"Conversation summary:\n{summary_text}",
                    }
                )
            if memory_text:
                assembled.append(
                    {
                        "role": "system",
                        "content": f"Relevant memory:\n{memory_text}",
                    }
                )
            assembled.extend(recent_messages)
            return assembled

        initial = build_context(conversation, memory_summary, "")
        if self.count(initial) <= budget:
            return initial

        keep_recent = 6
        recent = (
            conversation[-keep_recent:]
            if len(conversation) > keep_recent
            else list(conversation)
        )
        history = conversation[:-keep_recent] if len(conversation) > keep_recent else []
        summary_text = self._summarize([msg.get("content", "") for msg in history]) if history else ""

        memory_text = memory_summary
        fitted = build_context(recent, memory_text, summary_text)
        if self.count(fitted) <= budget:
            return fitted

        if memory_text:
            memory_text = self._summarize(list(memory_snippets or [])) or memory_text
            fitted = build_context(recent, memory_text, summary_text)
            if self.count(fitted) <= budget:
                return fitted

        while len(recent) > 1 and self.count(fitted) > budget:
            recent = recent[1:]
            fitted = build_context(recent, memory_text, summary_text)

        if self.count(fitted) > budget:
            fitted = self._truncate_messages(fitted, budget)

        return fitted

    # ------------------------------------------------------------------
    def trim(self, prompt: str, requested: int, model: Optional[str] = None) -> int:
        """Compatibility helper matching the legacy API."""

        budget = self._legacy_budget
        if budget is None:
            budget = self.budget_for(model or self.default_model)
        used = self.count(prompt)
        available = max(budget - used, 0)
        return max(0, min(requested, available))

    # ------------------------------------------------------------------
    def _truncate_messages(self, messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
        truncated: List[Dict[str, str]] = []
        used = 0
        for msg in messages:
            content = msg.get("content", "")
            remaining = max(budget - used, 0)
            if remaining <= 0:
                break
            allowed_content = self._truncate_text(content, remaining)
            new_msg = dict(msg)
            new_msg["content"] = allowed_content
            truncated.append(new_msg)
            used = self.count(truncated)
        return truncated

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if self._count_text(text) <= max_tokens:
            return text
        words = text.split()
        truncated: List[str] = []
        total = 0
        for word in words:
            token_count = self._count_text(word)
            if total + token_count > max_tokens:
                break
            truncated.append(word)
            total += token_count
        if not truncated:
            return text[: max_tokens * 4] + "…"
        return " ".join(truncated) + " …"

    def _summarize(self, texts: Iterable[str]) -> str:
        joined = " ".join(t.strip() for t in texts if t)
        if not joined:
            return ""
        max_chars = 320
        if len(joined) <= max_chars:
            return joined
        summary = joined[: max_chars].rsplit(" ", 1)[0]
        logger.info("Summarizing overflow context (%d chars)", len(joined))
        return summary + " …"
