"""Pydantic models for tool calls and analysis outputs."""
from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a function call request from the model."""

    tool: Literal["search", "code", "image", "browser"]
    args: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeOutput(BaseModel):
    """Model response from the analyze step."""

    chosen_tool: Literal["search", "code", "image", "browser", "none"]
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
