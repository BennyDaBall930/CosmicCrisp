"""Core agent loop for CosmicCrisp."""
from __future__ import annotations

from typing import AsyncGenerator, Optional

from .prompts import (
    analyze_task_prompt,
    create_tasks_prompt,
    start_goal_prompt,
    summarize_prompt,
)
from .task_parser import AnalyzeOutput
from ..memory.interface import AgentMemory
from ..memory.null import NullMemory
from ..tools.registry import registry
from ..tokenizer.token_service import TokenService


class AgentService:
    """High level interface driving the agent loop."""

    def __init__(
        self,
        memory: Optional[AgentMemory] = None,
        token_service: Optional[TokenService] = None,
    ) -> None:
        self.memory = memory or NullMemory()
        self.token_service = token_service or TokenService()

    async def run(self, goal: str) -> AsyncGenerator[str, None]:
        """Run the agent loop for a single goal, streaming text chunks."""
        yield f"START: {goal}\n"
        # Analyze step – naive heuristic for demo purposes
        tool_name = "search" if "search" in goal.lower() or "find" in goal.lower() else "code"
        analysis = AnalyzeOutput(
            chosen_tool=tool_name,
            args={"query": goal},
            rationale="heuristic",
        )
        yield f"ANALYZE: {analysis.model_dump_json()}\n"
        # Execute
        tool = registry.get(analysis.chosen_tool)
        result = await tool.run(**analysis.args) if tool else "no tool"
        await self.memory.add({"goal": goal, "result": result})
        yield f"EXECUTE: {result}\n"
        yield "SUMMARIZE: done\n"

    async def chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        await self.memory.enter(session_id)
        await self.memory.add({"message": message})
        yield f"ECHO: {message}\n"
