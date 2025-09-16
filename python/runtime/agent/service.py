"""High-level agent service delegating to the orchestrator."""
from __future__ import annotations

from typing import AsyncGenerator, Optional

from ..config import RuntimeConfig, load_runtime_config
from ..embeddings import Embeddings
from ..memory.interface import AgentMemory
from ..memory.null import NullMemory
from ..tokenizer.token_service import TokenService
from ..tools.registry import ToolRegistry, registry as default_registry
from .orchestrator import AgentOrchestrator
from .prompt_manager import PromptManager


class AgentService:
    """Surface area for launching the agent loop."""

    def __init__(
        self,
        memory: Optional[AgentMemory] = None,
        token_service: Optional[TokenService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[RuntimeConfig] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        prompt_manager: Optional[PromptManager] = None,
        embeddings: Optional[Embeddings] = None,
        model_router: Optional[object] = None,
    ) -> None:
        self.config = config or load_runtime_config()
        self.memory = memory or NullMemory()
        self.token_service = token_service or TokenService(self.config)
        self.tool_registry = tool_registry or default_registry
        self.prompt_manager = prompt_manager or PromptManager(persona=self.config.agent.persona)
        self.embeddings = embeddings
        self.model_router = model_router
        self.orchestrator = orchestrator or AgentOrchestrator(
            memory=self.memory,
            token_service=self.token_service,
            tool_registry=self.tool_registry,
            config=self.config,
            prompt_manager=self.prompt_manager,
            embeddings=self.embeddings,
            model_router=self.model_router,
        )

    async def run(self, goal: str) -> AsyncGenerator[str, None]:
        async for chunk in self.orchestrator.run_goal(goal):
            yield chunk

    async def chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        async for chunk in self.orchestrator.chat(session_id, message):
            yield chunk
