"""Model router policy tests."""
from __future__ import annotations

from python.runtime.config import RouterConfig, RouterModelConfig
from python.runtime.model.router import ModelRouter


class StubObservability:
    def __init__(self) -> None:
        self.calls = []

    def record_model_call(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        return {"Helicone-Property-Session": kwargs.get("session", "")}


def _build_router(overrides: dict[str, str] | None = None, with_observability: bool = False) -> ModelRouter:
    config = RouterConfig(
        default_model="gpt-4o-mini",
        default_strategy="balanced",
        strategy_overrides=overrides or {},
        models=(
            RouterModelConfig(
                name="fast-model",
                provider="openai",
                priority=1,
                max_context=80_000,
                cost_per_1k=0.5,
                latency="fast",
                capabilities=("general", "code"),
            ),
            RouterModelConfig(
                name="cheap-model",
                provider="openai",
                priority=2,
                max_context=32_000,
                cost_per_1k=0.2,
                latency="slow",
                capabilities=("general",),
            ),
            RouterModelConfig(
                name="context-model",
                provider="openai",
                priority=3,
                max_context=200_000,
                cost_per_1k=0.8,
                latency="medium",
                capabilities=("general", "code"),
            ),
        ),
    )
    obs = StubObservability() if with_observability else None
    return ModelRouter(config, observability=obs)


def test_router_balanced_strategy_prefers_priority():
    router = _build_router()
    candidate = router.select(required_capabilities=("code",))
    assert candidate.name == "fast-model"


def test_router_cost_strategy_prefers_cheapest():
    router = _build_router()
    candidate = router.select(strategy="cost")
    assert candidate.name == "cheap-model"


def test_router_context_strategy_via_override():
    router = _build_router(overrides={"coding": "context"})
    candidate = router.select(task_type="coding", min_context=150_000, required_capabilities=("code",))
    assert candidate.name == "context-model"


def test_router_fallbacks_to_default_model_when_no_match():
    router = _build_router()
    candidate = router.select(required_capabilities=("vision",))
    assert candidate.name == "gpt-4o-mini"


def test_router_telemetry_headers_use_observability():
    router = _build_router(with_observability=True)
    headers = router.telemetry_headers(session="sess", run_id="run", model="fast-model", tokens=123)
    assert headers["Helicone-Property-Session"] == "sess"
