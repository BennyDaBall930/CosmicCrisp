"""Pydantic request/response models for the runtime HTTP API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TaskKind = Literal["chat", "code", "browse", "summarize", "long_context", "data", "custom"]


class ChatRequest(BaseModel):
    session_id: str = "default"
    model: Optional[str] = None
    task_kind: TaskKind = "chat"
    messages: List[Dict[str, Any]]
    require_human_approval: bool = False
    ui_prefs: Dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    session_id: str = "default"
    goal: str
    model: Optional[str] = None
    task_kind: TaskKind = "chat"
    autonomy: Dict[str, Any] = Field(default_factory=dict)
    ui_prefs: Dict[str, Any] = Field(default_factory=dict)


class ToolInvokeRequest(BaseModel):
    name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


class MemoryQuery(BaseModel):
    session: str = "default"
    query: Optional[str] = None
    top_k: int = 6
    recent: int = 0
    offset: int = 0
    limit: int = 50


class MemoryBulkDeleteRequest(BaseModel):
    ids: List[str] = Field(default_factory=list)


class CancelRequest(BaseModel):
    run_id: str


class ResumeRequest(BaseModel):
    run_id: str
    model_override: Optional[str] = None


class BrowserContinueRequest(BaseModel):
    session_id: str
    context_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "TaskKind",
    "ChatRequest",
    "RunRequest",
    "ToolInvokeRequest",
    "MemoryQuery",
    "MemoryBulkDeleteRequest",
    "CancelRequest",
    "ResumeRequest",
    "BrowserContinueRequest",
]
