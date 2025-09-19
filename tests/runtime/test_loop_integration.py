"""Integration-oriented tests for the orchestrator loop."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from python.runtime.agent.orchestrator import AgentOrchestrator
from python.runtime.agent.task_parser import AnalyzeOutput
from python.runtime.config import AgentConfig, RuntimeConfig
from python.runtime.memory.schema import MemoryItem


class StubMemory:
    def __init__(self) -> None:
        self.session = None
        self.items: list[MemoryItem] = []
        self.similar_queries: list[str] = []

    async def enter(self, session_id: str) -> None:
        self.session = session_id

    async def add(self, item):
        if not isinstance(item, MemoryItem):
            item = MemoryItem(**item)
        self.items.append(item)
        return item.id or "stub"

    async def similar(self, query: str, k: int = 6):
        self.similar_queries.append(query)
        return [{"text": "historic insight"}]

    async def recent(self, k: int = 6):
        return []


class StubTokenService:
    default_model = "test-model"

    def __init__(self) -> None:
        self.memory_snippets = None

    def fit(self, messages, model, memory_snippets):
        self.memory_snippets = tuple(memory_snippets)
        return list(messages)

    def count(self, messages) -> int:
        return len(messages)


class StubTool:
    def __init__(self, name: str, result: str) -> None:
        self.name = name
        self.result = result
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class StubToolRegistry:
    def __init__(self) -> None:
        self.tools = {"search": StubTool("search", "search-result")}

    def get(self, name: str):
        return self.tools.get(name)


class StubPromptManager:
    def __init__(self) -> None:
        self.success_calls = 0
        self.failure_calls = 0

    def get_prompt(self, profile: str, stage: str, **_: dict) -> str:
        return f"{profile}:{stage}"

    @contextmanager
    def runtime_overrides(self, overrides):
        yield

    @contextmanager
    def persona_context(self, persona):
        yield

    def record_success(self, session: str, *, reset: bool = False) -> None:
        self.success_calls += 1

    def record_failure(self, session: str, *, tool: str | None = None) -> None:
        self.failure_calls += 1


@pytest.mark.asyncio
async def test_orchestrator_loop_writes_summary_and_uses_memory():
    memory = StubMemory()
    tokens = StubTokenService()
    tools = StubToolRegistry()
    prompts = StubPromptManager()

    config = RuntimeConfig()
    config.agent = AgentConfig(max_loops=2, subagent_max_depth=1, subagent_timeout=5.0)

    orchestrator = AgentOrchestrator(
        memory=memory,
        token_service=tokens,
        tool_registry=tools,
        config=config,
        prompt_manager=prompts,
        embeddings=None,
        model_router=None,
        event_bus=None,
        observability=None,
    )

    async def fake_analyze(self, task: str, goal: str) -> AnalyzeOutput:
        return AnalyzeOutput(chosen_tool="search", args={"query": task}, rationale="stub")

    orchestrator._analyze_task = fake_analyze.__get__(orchestrator, AgentOrchestrator)

    chunks = []
    async for chunk in orchestrator.run_goal("Research new features"):
        chunks.append(chunk)

    assert any(chunk.startswith("SUMMARY") for chunk in chunks)
    assert memory.similar_queries and memory.similar_queries[0] == "Research new features"
    assert tokens.memory_snippets == ("historic insight",)
    summary_items = [item for item in memory.items if item.kind == "summary"]
    assert summary_items, "Summary item should be persisted"
    assert tools.tools["search"].calls, "Tool should be invoked"
