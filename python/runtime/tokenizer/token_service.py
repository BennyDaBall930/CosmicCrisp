"""Simple token budgeting service."""
from __future__ import annotations

import os


class TokenService:
    def __init__(self, max_total: int | None = None) -> None:
        self.max_total = max_total or int(os.getenv("TOKEN_BUDGET", 4000))

    def count(self, text: str) -> int:
        return len(text.split())

    def trim(self, prompt: str, requested: int) -> int:
        used = self.count(prompt)
        available = self.max_total - used
        if available <= 0:
            return 0
        return min(requested, available)
