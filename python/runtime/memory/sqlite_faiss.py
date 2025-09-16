"""SQLite-backed memory store with vector similarity search."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
import uuid
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .interface import AgentMemory
from .schema import MemoryItem
from .store import MemoryStore

try:  # Optional typing support
    from ..embeddings import Embeddings
except Exception:  # pragma: no cover - avoid circular import at type-check time
    Embeddings = None  # type: ignore

logger = logging.getLogger(__name__)


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    arr = array("f", vector)
    return arr.tobytes()


def _blob_to_vector(blob: bytes | memoryview | None) -> List[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(bytes(blob))
    return list(arr)


def _cosine_similarity(query: Sequence[float], candidate: Sequence[float], candidate_norm: float) -> float:
    if not query or not candidate:
        return 0.0
    dot = sum(a * b for a, b in zip(query, candidate))
    query_norm = math.sqrt(sum(a * a for a in query))
    denom = query_norm * (candidate_norm or 1e-9)
    if denom == 0:
        return 0.0
    return dot / denom


class SQLiteFAISSMemory(AgentMemory, MemoryStore):
    """Store memories in SQLite and perform cosine similarity queries."""

    def __init__(
        self,
        path: str = ":memory:",
        *,
        embeddings: Optional["Embeddings"] = None,
    ) -> None:
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                session TEXT NOT NULL,
                kind TEXT NOT NULL,
                ts REAL NOT NULL,
                tags TEXT NOT NULL,
                text TEXT NOT NULL,
                meta TEXT NOT NULL,
                vector BLOB,
                vector_norm REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_session_ts ON memory(session, ts DESC)"
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self._session = "default"
        self._embeddings = embeddings

    # AgentMemory interface -------------------------------------------------
    async def enter(self, session_id: str) -> None:
        self._session = session_id

    async def add(self, item: Dict | MemoryItem) -> str:  # type: ignore[override]
        memory_item = self._coerce_item(item)
        if memory_item.vector is None and self._embeddings is not None:
            vectors = await self._embeddings.embed([memory_item.text])
            memory_item.vector = vectors[0] if vectors else []
        vector = memory_item.vector or []
        vector_norm = math.sqrt(sum(v * v for v in vector))
        payload = (
            memory_item.id,
            self._session,
            memory_item.kind,
            memory_item.ts.timestamp(),
            json.dumps(memory_item.tags),
            memory_item.text,
            json.dumps(memory_item.meta),
            _vector_to_blob(vector),
            vector_norm,
        )
        await asyncio.to_thread(self._insert, payload)
        return memory_item.id or ""

    async def similar(self, query: str, k: int = 6) -> List[Dict]:  # type: ignore[override]
        if not self._embeddings:
            return [item.model_dump() for item in await self.recent(k)]
        vectors = await self._embeddings.embed([query])
        query_vector = vectors[0] if vectors else []
        rows = await asyncio.to_thread(self._fetch_session_rows, self._session)
        scored: List[tuple[float, MemoryItem]] = []
        for item, vector, vector_norm in rows:
            score = _cosine_similarity(query_vector, vector, vector_norm)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item.model_dump() for _, item in scored[:k]]

    async def recent(self, k: int = 6) -> List[Dict]:  # type: ignore[override]
        rows = await asyncio.to_thread(self._fetch_recent_rows, self._session, k)
        return [item.model_dump() for item in rows]

    async def reset(self) -> None:
        await asyncio.to_thread(self._delete_session, self._session)

    # MemoryStore interface --------------------------------------------------
    async def delete(self, item_id: str) -> None:
        await asyncio.to_thread(self._delete_item, item_id)

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_session, self._session)

    # Internal helpers -------------------------------------------------------
    def _insert(self, payload: Sequence) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory
                (id, session, kind, ts, tags, text, meta, vector, vector_norm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            self._conn.commit()

    def _fetch_session_rows(self, session: str) -> List[tuple[MemoryItem, List[float], float]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, kind, ts, tags, text, meta, vector, vector_norm FROM memory WHERE session=?",
                (session,),
            )
            rows = cur.fetchall()
        return [self._row_to_item(row) for row in rows]

    def _fetch_recent_rows(self, session: str, limit: int) -> List[MemoryItem]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, kind, ts, tags, text, meta, vector, vector_norm
                FROM memory
                WHERE session=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (session, limit),
            )
            rows = cur.fetchall()
        return [self._row_to_item(row)[0] for row in rows]

    def _delete_session(self, session: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE session=?", (session,))
            self._conn.commit()

    def _delete_item(self, item_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE id=?", (item_id,))
            self._conn.commit()

    def _count_session(self, session: str) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM memory WHERE session=?", (session,))
            (count,) = cur.fetchone()
        return int(count)

    def _row_to_item(self, row: Sequence) -> tuple[MemoryItem, List[float], float]:
        (
            item_id,
            kind,
            ts,
            tags_json,
            text,
            meta_json,
            vector_blob,
            vector_norm,
        ) = row
        tags = json.loads(tags_json) if tags_json else []
        meta = json.loads(meta_json) if meta_json else {}
        meta.setdefault("session", self._session)
        vector = _blob_to_vector(vector_blob)
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        item = MemoryItem(id=item_id, kind=kind, ts=ts_dt, tags=tags, text=text, meta=meta, vector=vector)
        return item, vector, float(vector_norm or 0.0)

    def _coerce_item(self, item: Dict | MemoryItem) -> MemoryItem:
        if isinstance(item, MemoryItem):
            result = item
        else:
            data = dict(item)
            data.setdefault("ts", datetime.utcnow())
            data.setdefault("kind", "note")
            data.setdefault("tags", [])
            data.setdefault("meta", {})
            result = MemoryItem(**data)
        if not result.id:
            result.id = str(uuid.uuid4())
        result.meta.setdefault("session", self._session)
        if self._session not in result.tags:
            result.tags.append(self._session)
        return result


__all__ = ["SQLiteFAISSMemory"]
