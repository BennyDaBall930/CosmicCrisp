"""Model routing utilities for selecting LLM endpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import RouterConfig, RouterModelConfig

if TYPE_CHECKING:  # pragma: no cover - optional import for type checking only
    from ..observability import Observability


_LATENCY_ORDER = {
    "ultra": 0,
    "fast": 1,
    "medium": 2,
    "slow": 3,
}


def _latency_rank(value: str) -> int:
    return _LATENCY_ORDER.get(value.lower(), _LATENCY_ORDER["medium"])


def _context_score(max_context: int) -> int:
    return max_context if max_context > 0 else 0


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Concrete model option the router can return."""

    name: str
    provider: str
    priority: int
    max_context: int
    cost_per_1k: float
    latency: str
    capabilities: Tuple[str, ...]
    tags: Tuple[str, ...]
    is_local: bool
    metadata: Dict[str, str]

    def supports(self, required: Iterable[str]) -> bool:
        required_set = {cap.lower() for cap in required if cap}
        if not required_set:
            return True
        if not self.capabilities:
            return True
        available = {cap.lower() for cap in self.capabilities}
        return required_set.issubset(available)

    def has_context(self, minimum: int) -> bool:
        return self.max_context <= 0 or self.max_context >= minimum

    def latency_rank(self) -> int:
        return _latency_rank(self.latency)


class ModelRouter:
    """Simple policy-driven model selector."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        observability: Optional["Observability"] = None,
    ) -> None:
        self.config = config
        self._observability = observability
        self._candidates: List[ModelCandidate] = [
            self._build_candidate(entry) for entry in config.models
        ]
        seen = {candidate.name for candidate in self._candidates}
        if config.default_model and config.default_model not in seen:
            self._candidates.append(
                self._build_candidate(RouterModelConfig(name=config.default_model))
            )
        self._candidate_map = {candidate.name: candidate for candidate in self._candidates}

    # ------------------------------------------------------------------
    def available(self) -> List[ModelCandidate]:
        return list(self._candidates)

    def get(self, name: str) -> Optional[ModelCandidate]:
        return self._candidate_map.get(name)

    # ------------------------------------------------------------------
    def select(
        self,
        *,
        task_type: str = "default",
        required_capabilities: Optional[Sequence[str]] = None,
        min_context: int = 0,
        strategy: Optional[str] = None,
        allow_local: bool = True,
    ) -> ModelCandidate:
        required = tuple(required_capabilities or ())
        strategy_key = (
            strategy
            or self.config.strategy_overrides.get(task_type.lower())
            or self.config.default_strategy
        ).lower()

        pool = [
            candidate
            for candidate in self._candidates
            if (allow_local or not candidate.is_local)
        ]
        pool = [candidate for candidate in pool if candidate.has_context(min_context)]
        if required:
            pool = [candidate for candidate in pool if candidate.supports(required)]

        if not pool:
            if allow_local:
                pool = [
                    candidate
                    for candidate in self._candidates
                    if candidate.supports(required) and candidate.has_context(min_context)
                ]
        if not pool:
            fallback = self._candidate_map.get(self.config.default_model)
            if fallback is not None:
                return fallback
            raise ValueError("No model candidates satisfy the requested constraints")

        ranking = _strategy_key(strategy_key)
        pool.sort(key=ranking)
        return pool[0]

    # ------------------------------------------------------------------
    def telemetry_headers(
        self,
        *,
        session: str,
        run_id: str,
        model: str,
        tokens: int = 0,
        tool: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if not self._observability:
            return {}
        return self._observability.record_model_call(
            session=session,
            run_id=run_id,
            model=model,
            tokens=tokens,
            tool=tool,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_candidate(entry: RouterModelConfig) -> ModelCandidate:
        return ModelCandidate(
            name=entry.name,
            provider=entry.provider or "openai",
            priority=entry.priority,
            max_context=entry.max_context,
            cost_per_1k=entry.cost_per_1k,
            latency=entry.latency or "medium",
            capabilities=tuple(entry.capabilities),
            tags=tuple(entry.tags),
            is_local=entry.is_local,
            metadata=dict(entry.metadata),
        )


def _strategy_key(strategy: str):
    strategy = strategy.lower()
    if strategy == "cost":
        return lambda c: (
            c.cost_per_1k if c.cost_per_1k > 0 else float("inf"),
            c.priority,
            c.latency_rank(),
            -_context_score(c.max_context),
        )
    if strategy == "speed":
        return lambda c: (
            c.latency_rank(),
            c.priority,
            c.cost_per_1k,
            -_context_score(c.max_context),
        )
    if strategy == "context":
        return lambda c: (
            -_context_score(c.max_context),
            c.priority,
            c.cost_per_1k,
            c.latency_rank(),
        )
    return lambda c: (
        c.priority,
        c.cost_per_1k,
        c.latency_rank(),
        -_context_score(c.max_context),
    )


__all__ = ["ModelRouter", "ModelCandidate"]
