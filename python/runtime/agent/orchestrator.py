"""Agent orchestrator integrating planning, memory, and subagents."""
from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterable, List, Optional

from ..config import RuntimeConfig, load_runtime_config
from ..embeddings import Embeddings
from ..event_bus import EventBus
from ..memory import MemoryStore
from ..memory.schema import MemoryItem
from ..observability import Observability
from ..tokenizer.token_service import TokenService
from ..tools.registry import ToolRegistry, registry as default_registry
from .prompt_manager import PromptManager
from .task_parser import AnalyzeOutput, ToolCall

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Task:
    priority: int
    created_at: float = field(default_factory=time.time, compare=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8], compare=False)
    description: str = field(default="", compare=False)
    parent_id: Optional[str] = field(default=None, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    status: str = field(default="pending", compare=False)
    result: Optional[str] = field(default=None, compare=False)


class TaskPlanner:
    """Maintain a priority queue of tasks and simple heuristic planning."""

    def __init__(self, *, prompt_manager: PromptManager, profile: str) -> None:
        self._queue: List[Task] = []
        self._prompt_manager = prompt_manager
        self._profile = profile

    def bootstrap(self, goal: str) -> None:
        # naive split on conjunctions to seed the backlog
        fragments = [frag.strip() for frag in goal.replace(";", ".").split(" and ") if frag.strip()]
        if not fragments:
            fragments = [goal]
        for idx, fragment in enumerate(fragments):
            self.add_task(fragment, priority=idx)

    def add_task(
        self,
        description: str,
        *,
        priority: int = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        task = Task(priority=priority, description=description, parent_id=parent_id, metadata=metadata or {})
        heapq.heappush(self._queue, task)
        return task

    def has_tasks(self) -> bool:
        return bool(self._queue)

    def next_task(self) -> Optional[Task]:
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def consider_followups(self, task: Task, observation: str) -> List[Task]:
        followups: List[Task] = []
        if not observation:
            return followups
        observation_lower = observation.lower()
        if "research" in observation_lower or "investigate" in observation_lower:
            followups.append(
                self.add_task(
                    f"Research findings based on '{task.description}'",
                    priority=task.priority + 1,
                    parent_id=task.id,
                )
            )
        if "todo" in observation_lower or "follow-up" in observation_lower:
            followups.append(
                self.add_task(
                    f"Clarify outstanding items from task {task.id}",
                    priority=task.priority + 2,
                    parent_id=task.id,
                )
            )
        return followups


class SubAgentManager:
    """Spawn specialised subagents with isolated context."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        token_service: TokenService,
        prompt_manager: PromptManager,
        tool_registry: ToolRegistry,
        max_depth: Optional[int],
        timeout: Optional[float],
        observability: Optional[Observability] = None,
    ) -> None:
        self.memory = memory
        self.token_service = token_service
        self.prompt_manager = prompt_manager
        self.tool_registry = tool_registry
        self.max_depth = max_depth
        self.timeout = timeout
        self.observability = observability

    def should_spawn(self, tool_name: str) -> bool:
        return tool_name in {"browser", "code"}

    async def run(
        self,
        *,
        parent_session: str,
        depth: int,
        task: Task,
        analysis: AnalyzeOutput,
        orchestrator: "AgentOrchestrator",
        run_id: Optional[str] = None,
    ) -> str:
        if self.max_depth is not None and depth >= self.max_depth:
            return await orchestrator._invoke_tool(
                parent_session,
                analysis,
                run_id=run_id,
                task_id=task.id,
            )
        if self.observability:
            self.observability.record_subagent_spawn(
                tool=analysis.chosen_tool,
                depth=depth + 1,
                parent_session=parent_session,
            )
        sub_session = f"{parent_session}::sub-{analysis.chosen_tool}-{uuid.uuid4().hex[:4]}"
        enter = getattr(self.memory, "enter", None)
        if enter:
            await enter(sub_session)
        try:
            if self.timeout is None:
                return await orchestrator._invoke_tool(
                    sub_session,
                    analysis,
                    run_id=run_id,
                    task_id=task.id,
                )
            return await asyncio.wait_for(
                orchestrator._invoke_tool(
                    sub_session,
                    analysis,
                    run_id=run_id,
                    task_id=task.id,
                ),
                timeout=self.timeout,
            )
        finally:
            if enter:
                await enter(parent_session)


class AgentOrchestrator:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        token_service: TokenService,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[RuntimeConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
        embeddings: Optional[Embeddings] = None,
        model_router: Optional[Any] = None,
        event_bus: Optional[EventBus] = None,
        observability: Optional[Observability] = None,
    ) -> None:
        self.config = config or load_runtime_config()
        self.memory = memory
        self.token_service = token_service
        self.tool_registry = tool_registry or default_registry
        self.embeddings = embeddings
        self.prompt_manager = prompt_manager or PromptManager(persona=self.config.agent.persona)
        self.model_router = model_router
        self.event_bus = event_bus
        self.observability = observability
        self.memory_top_k = self.config.memory.top_k
        self.max_loops = self.config.agent.max_loops
        self.subagents = SubAgentManager(
            memory=memory,
            token_service=token_service,
            prompt_manager=self.prompt_manager,
            tool_registry=self.tool_registry,
            max_depth=self.config.agent.subagent_max_depth,
            timeout=self.config.agent.subagent_timeout,
            observability=self.observability,
        )
        self._active_runs: Dict[str, asyncio.Event] = {}
        self._hil_waits: Dict[str, asyncio.Future[Any]] = {}

    def _format_event(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"event": event, "data": data}

    def _tokenize(self, text: str) -> Iterable[str]:
        if not text:
            return []
        parts = text.split(" ")
        for idx, part in enumerate(parts):
            suffix = " " if idx < len(parts) - 1 else ""
            yield part + suffix

    async def _publish_bus(self, payload: Dict[str, Any]) -> None:
        if self.event_bus is not None:
            await self.event_bus.publish(payload)

    async def cancel_run(self, run_id: str) -> bool:
        event = self._active_runs.get(run_id)
        if not event:
            return False
        event.set()
        return True

    async def resume_run(self, run_id: str, **_: Any) -> bool:
        return run_id in self._active_runs

    async def browser_continue(self, session_id: str, context_id: str, payload: Dict[str, Any]) -> bool:
        future = self._hil_waits.pop(context_id, None)
        if not future:
            return False
        if not future.done():
            future.set_result({"session_id": session_id, "payload": payload})
        return True

    # ------------------------------------------------------------------
    async def run_goal(
        self,
        goal: str,
        *,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        session = session_id or f"goal:{uuid.uuid4().hex[:8]}"
        run_id = uuid.uuid4().hex
        model_name = model or self.token_service.default_model
        if self.observability:
            self.observability.record_run_started(
                run_type="goal",
                run_id=run_id,
                session=session,
                goal=goal,
                model=model_name,
            )
        success = False
        try:
            await self._enter_session(session)
            await self._log_memory(session, kind="goal", text=goal, tags=["goal"], meta={"goal": goal, "run_id": run_id})
            yield f"START: {goal}\n"

            _ = await self._embed_text(goal)
            memory_items = await self._retrieve_memory(goal, session=session)
            memory_snippets = [item.get("text", "") for item in memory_items if item.get("text")]
            if memory_snippets:
                yield f"MEMORY: recalled {len(memory_snippets)} items\n"

            messages = self._base_messages(goal, session)
            fitted = self.token_service.fit(messages, model_name, memory_snippets)
            token_count = self.token_service.count(fitted)
            yield f"CONTEXT: {token_count} tokens after fit\n"
            if self.observability:
                self.observability.record_token_usage(model=model_name, tokens=token_count, run_id=run_id)

            planner = TaskPlanner(prompt_manager=self.prompt_manager, profile=self.config.agent.planner_profile)
            planner.bootstrap(goal)

            completed: List[Task] = []
            loop_count = 0
            while planner.has_tasks() and (
                self.max_loops is None or loop_count < self.max_loops
            ):
                task = planner.next_task()
                if task is None:
                    break
                loop_count += 1
                yield f"TASK[{task.id}]: {task.description}\n"
                if self.observability:
                    self.observability.record_task_started(
                        run_id=run_id,
                        task_id=task.id,
                        owner=session,
                        description=task.description,
                    )
                analysis = await self._analyze_task(task.description, goal)
                yield f"ANALYZE[{task.id}]: {analysis.model_dump_json()}\n"
                result = await self._execute_task(
                    session,
                    task,
                    analysis,
                    memory_snippets,
                    depth=0,
                    run_id=run_id,
                )
                task.status = "completed"
                task.result = result
                completed.append(task)
                yield f"RESULT[{task.id}]: {result}\n"
                if self.observability:
                    self.observability.record_task_completed(
                        run_id=run_id,
                        task_id=task.id,
                        owner=session,
                        result=self._clip_text(result),
                    )
                planner.consider_followups(task, result)

            summary = self._summarize(goal, completed)
            await self._log_memory(
                session,
                kind="summary",
                text=summary,
                tags=["summary", session],
                meta={"goal": goal, "completed_tasks": [task.id for task in completed], "run_id": run_id},
            )
            yield summary + "\n"
            success = True
        except Exception as exc:
            if self.observability:
                self.observability.record_run_failed(
                    run_type="goal",
                    run_id=run_id,
                    session=session,
                    error=str(exc),
                )
            raise
        finally:
            if self.observability and success:
                self.observability.record_run_completed(
                    run_type="goal",
                    run_id=run_id,
                    session=session,
                )

    # ------------------------------------------------------------------
    async def chat(
        self,
        session_id: str,
        message: str,
        *,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        session = session_id or f"chat:{uuid.uuid4().hex[:6]}"
        run_id = uuid.uuid4().hex
        model_name = model or self.token_service.default_model
        if self.observability:
            self.observability.record_run_started(
                run_type="chat",
                run_id=run_id,
                session=session,
                goal=message,
                model=model_name,
            )
        success = False
        try:
            await self._enter_session(session)
            await self._log_memory(
                session,
                kind="message",
                text=message,
                tags=[session, "user"],
                meta={"role": "user", "run_id": run_id},
            )

            _ = await self._embed_text(message)
            memory_items = await self._retrieve_memory(message, session=session)
            memory_snippets = [item.get("text", "") for item in memory_items if item.get("text")]
            messages = self._base_messages(message, session)
            fitted = self.token_service.fit(messages, model_name, memory_snippets)
            token_count = self.token_service.count(fitted)
            yield f"CONTEXT: {token_count} tokens after fit\n"
            if self.observability:
                self.observability.record_token_usage(model=model_name, tokens=token_count, run_id=run_id)

            response = f"ECHO: {message}"
            await self._log_memory(
                session,
                kind="decision",
                text=response,
                tags=[session, "assistant"],
                meta={"role": "assistant", "run_id": run_id},
            )

            yield response + "\n"
            success = True
        except Exception as exc:
            if self.observability:
                self.observability.record_run_failed(
                    run_type="chat",
                    run_id=run_id,
                    session=session,
                    error=str(exc),
                )
            raise
        finally:
            if self.observability and success:
                self.observability.record_run_completed(
                    run_type="chat",
                    run_id=run_id,
                    session=session,
                )

    async def stream_chat(self, request: Any) -> AsyncIterator[Dict[str, Any]]:
        run_id = uuid.uuid4().hex
        session = getattr(request, "session_id", "default") or "default"
        model = getattr(request, "model", None) or self.token_service.default_model
        messages = getattr(request, "messages", []) or []
        user_content = messages[-1].get("content", "") if messages else ""
        await self._enter_session(session)
        await self._log_memory(
            session,
            kind="message",
            text=user_content,
            tags=[session, "user"],
            meta={"role": "user", "run_id": run_id},
        )

        if self.observability:
            self.observability.record_run_started(
                run_type="chat_stream",
                run_id=run_id,
                session=session,
                goal=user_content,
                model=model,
            )

        ui_prefs = getattr(request, "ui_prefs", {}) or {}
        overrides = ui_prefs.get("prompt_overrides")
        persona_override = ui_prefs.get("persona")

        with self.prompt_manager.runtime_overrides(overrides):
            with self.prompt_manager.persona_context(persona_override):
                completed = False
                try:
                    started = {"type": "chat_started", "run_id": run_id, "session_id": session, "model": model}
                    yield self._format_event("event", started)
                    await self._publish_bus(started)

                    memory_items = await self._retrieve_memory(user_content, session=session)
                    if memory_items:
                        memory_payload = {
                            "type": "memory",
                            "items": memory_items,
                            "run_id": run_id,
                            "session_id": session,
                        }
                        yield self._format_event("event", memory_payload)
                        await self._publish_bus(memory_payload)

                    response = f"ECHO: {user_content}" if user_content else "ACK"
                    for token in self._tokenize(response):
                        yield self._format_event(
                            "token",
                            {"text": token, "run_id": run_id, "node_id": session, "model": model},
                        )

                    summary_payload = {"type": "summary", "run_id": run_id, "text": response}
                    yield self._format_event("event", summary_payload)
                    await self._publish_bus(summary_payload)

                    await self._log_memory(
                        session,
                        kind="decision",
                        text=response,
                        tags=[session, "assistant"],
                        meta={"role": "assistant", "run_id": run_id},
                    )
                    if self.observability:
                        self.observability.record_token_usage(
                            model=model,
                            tokens=self.token_service.count(response),
                            run_id=run_id,
                        )
                    yield self._format_event("done", {"ok": True, "run_id": run_id})
                    completed = True
                    self.prompt_manager.record_success(session, reset=True)
                except Exception as exc:
                    self.prompt_manager.record_failure(session)
                    error_payload = {"message": str(exc), "where": "orchestrator", "run_id": run_id}
                    yield self._format_event("error", error_payload)
                    await self._publish_bus({"type": "error", **error_payload})
                    if self.observability:
                        self.observability.record_run_failed(
                            run_type="chat_stream",
                            run_id=run_id,
                            session=session,
                            error=str(exc),
                        )
                finally:
                    if self.observability and completed:
                        self.observability.record_run_completed(
                            run_type="chat_stream",
                            run_id=run_id,
                            session=session,
                        )

    async def stream_run(self, request: Any) -> AsyncIterator[Dict[str, Any]]:
        run_id = uuid.uuid4().hex
        cancel_event = asyncio.Event()
        self._active_runs[run_id] = cancel_event
        session = getattr(request, "session_id", "default") or "default"
        goal = getattr(request, "goal", "")
        model = getattr(request, "model", None) or self.token_service.default_model
        await self._enter_session(session)
        await self._log_memory(session, kind="goal", text=goal, tags=["goal", session], meta={"run_id": run_id})

        if self.observability:
            self.observability.record_run_started(
                run_type="run_stream",
                run_id=run_id,
                session=session,
                goal=goal,
                model=model,
            )

        ui_prefs = getattr(request, "ui_prefs", {}) or {}
        overrides = ui_prefs.get("prompt_overrides")
        persona_override = ui_prefs.get("persona")

        with self.prompt_manager.runtime_overrides(overrides):
            with self.prompt_manager.persona_context(persona_override):
                run_failed = False
                completed = False
                try:
                    start_payload = {
                        "type": "run_started",
                        "run_id": run_id,
                        "session_id": session,
                        "goal": goal,
                        "model": model,
                    }
                    yield self._format_event("event", start_payload)
                    await self._publish_bus(start_payload)

                    memory_items = await self._retrieve_memory(goal, session=session)
                    if memory_items:
                        memory_payload = {
                            "type": "memory",
                            "items": memory_items,
                            "run_id": run_id,
                            "session_id": session,
                        }
                        yield self._format_event("event", memory_payload)
                        await self._publish_bus(memory_payload)

                    planner = TaskPlanner(prompt_manager=self.prompt_manager, profile=self.config.agent.planner_profile)
                    planner.bootstrap(goal)

                    completed: List[Task] = []
                    loop_count = 0
                    while planner.has_tasks() and (
                        self.max_loops is None or loop_count < self.max_loops
                    ):
                        if cancel_event.is_set():
                            yield self._format_event(
                                "error",
                                {"message": "run cancelled", "where": "orchestrator", "run_id": run_id},
                            )
                            await self._publish_bus({"type": "run_cancelled", "run_id": run_id})
                            if self.observability and not run_failed:
                                self.observability.record_run_failed(
                                    run_type="run_stream",
                                    run_id=run_id,
                                    session=session,
                                    error="cancelled",
                                )
                            run_failed = True
                            break
                        task = planner.next_task()
                        if task is None:
                            break
                        loop_count += 1
                        task_start = {
                            "type": "task_started",
                            "task_id": task.id,
                            "description": task.description,
                            "run_id": run_id,
                        }
                        yield self._format_event("event", task_start)
                        await self._publish_bus(task_start)
                        if self.observability:
                            self.observability.record_task_started(
                                run_id=run_id,
                                task_id=task.id,
                                owner=session,
                                description=task.description,
                            )

                        analysis = await self._analyze_task(task.description, goal)
                        tool_start = {
                            "type": "tool_start",
                            "tool": analysis.chosen_tool,
                            "task_id": task.id,
                            "run_id": run_id,
                        }
                        yield self._format_event("event", tool_start)
                        await self._publish_bus(tool_start)

                        if analysis.chosen_tool == "browser":
                            context_id = uuid.uuid4().hex
                            loop = asyncio.get_running_loop()
                            future: asyncio.Future[Any] = loop.create_future()
                            self._hil_waits[context_id] = future
                            hil_payload = {
                                "type": "browser_hil_required",
                                "session_id": session,
                                "context_id": context_id,
                                "task_id": task.id,
                                "run_id": run_id,
                            }
                            yield self._format_event("hil", hil_payload)
                            await self._publish_bus(hil_payload)
                            try:
                                await future
                            finally:
                                self._hil_waits.pop(context_id, None)

                        result = await self._execute_task(
                            session,
                            task,
                            analysis,
                            [],
                            depth=0,
                            run_id=run_id,
                        )

                        tool_end = {
                            "type": "tool_end",
                            "tool": analysis.chosen_tool,
                            "task_id": task.id,
                            "run_id": run_id,
                        }
                        yield self._format_event("event", tool_end)
                        await self._publish_bus(tool_end)

                        for token in self._tokenize(result):
                            yield self._format_event(
                                "token",
                                {"text": token, "run_id": run_id, "node_id": task.id, "model": model},
                            )
                        task.status = "completed"
                        task.result = result
                        completed.append(task)
                        if self.observability:
                            self.observability.record_task_completed(
                                run_id=run_id,
                                task_id=task.id,
                                owner=session,
                                result=self._clip_text(result),
                            )
                        planner.consider_followups(task, result)

                    summary = self._summarize(goal, completed)
                    summary_payload = {"type": "summary", "run_id": run_id, "text": summary}
                    yield self._format_event("event", summary_payload)
                    await self._publish_bus(summary_payload)
                    await self._log_memory(
                        session,
                        kind="summary",
                        text=summary,
                        tags=["summary", session],
                        meta={"run_id": run_id, "goal": goal},
                    )
                    yield self._format_event("done", {"ok": not run_failed, "run_id": run_id})
                    if not run_failed:
                        completed = True
                        self.prompt_manager.record_success(session, reset=True)
                except Exception as exc:
                    self.prompt_manager.record_failure(session)
                    error_payload = {"message": str(exc), "where": "orchestrator", "run_id": run_id}
                    yield self._format_event("error", error_payload)
                    await self._publish_bus({"type": "error", **error_payload})
                    if self.observability:
                        self.observability.record_run_failed(
                            run_type="run_stream",
                            run_id=run_id,
                            session=session,
                            error=str(exc),
                        )
                    run_failed = True
                finally:
                    self._active_runs.pop(run_id, None)
                    if self.observability and completed and not run_failed:
                        self.observability.record_run_completed(
                            run_type="run_stream",
                            run_id=run_id,
                            session=session,
                        )

    # ------------------------------------------------------------------
    def _base_messages(self, content: str, session: str) -> List[Dict[str, str]]:
        system_prompt = self.prompt_manager.get_prompt(
            self.config.agent.execution_profile, "system", session=session, variables={"session_id": session}
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    async def _analyze_task(self, task: str, goal: str) -> AnalyzeOutput:
        # Prepare simple structured suggestion and validate via Pydantic
        heuristic = "browser" if any(word in task.lower() for word in ("website", "browse", "search")) else "code" if "code" in task.lower() else "search"
        payload = {
            "chosen_tool": heuristic,
            "args": {"query": task, "goal": goal},
            "rationale": f"Heuristic selection based on task: {task}",
        }
        return AnalyzeOutput.model_validate(payload)

    async def _execute_task(
        self,
        session: str,
        task: Task,
        analysis: AnalyzeOutput,
        memory_snippets: Iterable[str],
        *,
        depth: int,
        run_id: Optional[str] = None,
    ) -> str:
        clean_call = ToolCall.model_validate(
            {
                "tool": analysis.chosen_tool if analysis.chosen_tool != "none" else "search",
                "args": analysis.args,
            }
        )
        clean_analysis = AnalyzeOutput(
            chosen_tool=clean_call.tool,
            args=clean_call.args,
            rationale=analysis.rationale,
        )

        if self.subagents.should_spawn(clean_analysis.chosen_tool):
            result = await self.subagents.run(
                parent_session=session,
                depth=depth,
                task=task,
                analysis=clean_analysis,
                orchestrator=self,
                run_id=run_id,
            )
        else:
            result = await self._invoke_tool(
                session,
                clean_analysis,
                run_id=run_id,
                task_id=task.id,
            )
        await self._log_memory(
            session,
            kind="fact",
            text=str(result),
            tags=[session, clean_analysis.chosen_tool, "result"],
            meta={"task": task.description, "tool": clean_analysis.chosen_tool},
        )
        if isinstance(result, str) and result.startswith("tool-error"):
            self.prompt_manager.record_failure(session, tool=clean_analysis.chosen_tool)
        else:
            self.prompt_manager.record_success(session)
        return result

    async def _invoke_tool(
        self,
        session: str,
        analysis: AnalyzeOutput,
        *,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        tool = self.tool_registry.get(analysis.chosen_tool)
        if not tool:
            logger.warning("No tool registered named '%s'", analysis.chosen_tool)
            return "no tool"
        if self.observability:
            self.observability.record_tool_usage(
                tool=analysis.chosen_tool,
                run_id=run_id,
                task_id=task_id,
            )
        try:
            result = await tool.run(**analysis.args)
        except Exception as exc:  # pragma: no cover
            logger.error("Tool '%s' failed: %s", analysis.chosen_tool, exc)
            self.prompt_manager.record_failure(session, tool=analysis.chosen_tool)
            result = f"tool-error: {exc}"
        return str(result)

    def _summarize(self, goal: str, tasks: List[Task]) -> str:
        completed = [task for task in tasks if task.status == "completed"]
        if not completed:
            return f"SUMMARY: Goal '{goal}' yielded no task results."
        bullet_points = "\n".join(f"- ({task.id}) {task.description}: {task.result}" for task in completed)
        return f"SUMMARY: Completed {len(completed)} tasks.\n{bullet_points}"

    def _clip_text(self, text: Any, limit: int = 400) -> str:
        value = "" if text is None else str(text)
        if len(value) <= limit:
            return value
        return value[: limit - 1] + "…"

    async def _enter_session(self, session_id: str) -> None:
        enter = getattr(self.memory, "enter", None)
        if enter:
            try:
                await enter(session_id)
            except Exception:  # pragma: no cover
                logger.debug("Memory store enter failed", exc_info=True)

    async def _retrieve_memory(self, query: str, *, session: str) -> List[Dict]:
        items: List[Dict] = []
        try:
            items = await self.memory.similar(query, k=self.memory_top_k)
        except AttributeError:
            items = []
        finally:
            if self.observability:
                if items:
                    self.observability.record_memory_hit(session=session, count=len(items))
                else:
                    self.observability.record_memory_miss(session=session)
        return items

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        if not self.embeddings or not text:
            return None
        try:
            vectors = await self.embeddings.embed([text])
            return vectors[0] if vectors else None
        except Exception:  # pragma: no cover
            logger.debug("Embedding failed", exc_info=True)
            return None

    async def _log_memory(
        self,
        session: str,
        *,
        kind: str,
        text: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = MemoryItem(
            kind=kind,
            text=text,
            tags=list(tags or []) + [session],
            meta=meta or {},
        )
        try:
            await self.memory.add(payload)
        except Exception:  # pragma: no cover
            logger.debug("Memory add failed", exc_info=True)


__all__ = ["AgentOrchestrator", "Task", "TaskPlanner", "SubAgentManager"]
