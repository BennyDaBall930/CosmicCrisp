"""Typed schema for memory items."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    id: Optional[str] = None
    kind: str = "note"
    ts: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "ts": self.ts,
            "tags": list(self.tags),
            "text": self.text,
            "meta": dict(self.meta),
            "vector": list(self.vector) if self.vector is not None else None,
        }


__all__ = ["MemoryItem"]
