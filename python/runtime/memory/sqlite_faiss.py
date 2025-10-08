"""SQLite-backed memory store with vector similarity search."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
import time
import uuid
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .interface import AgentMemory
from .schema import MemoryItem
from .store import MemoryStore

try:  # Optional typing support
    from ..embeddings import Embeddings
except Exception:  # pragma: no cover - avoid circular import at type-check time
    Embeddings = None  # type: ignore

logger = logging.getLogger(__name__)

_LOCAL_ENCODERS: Dict[str, Any] = {}
_LOCAL_ENCODER_LOCK = threading.Lock()

SCHEMA_VERSION = 1


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
        self._init_metadata_storage()
        self._metadata_checked = False
        self._needs_reindex = False
        self._fallback_signature: Optional[Dict[str, str]] = None
        self._init_lock = asyncio.Lock()
        self._reindex_lock = asyncio.Lock()

    # AgentMemory interface -------------------------------------------------
    async def enter(self, session_id: str) -> None:
        await self._ensure_ready()
        self._session = session_id

    async def add(self, item: Dict | MemoryItem) -> str:  # type: ignore[override]
        await self._ensure_ready()
        memory_item = self._coerce_item(item)
        existing_vector: List[float] = memory_item.vector or []
        if not existing_vector:
            embedded = await self._embed_texts([memory_item.text])
            memory_item.vector = embedded[0] if embedded else []
        vector = memory_item.vector or []
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
        if vector:
            self._maybe_record_vector_dim(len(vector))
        return memory_item.id or ""

    async def similar(self, query: str, k: int = 6) -> List[Dict]:  # type: ignore[override]
        await self._ensure_ready()
        return await self.similar_paginated(query=query, k=k, offset=0)

    async def recent(self, k: int = 6) -> List[Dict]:  # type: ignore[override]
        await self._ensure_ready()
        return await self.recent_paginated(k=k, offset=0)

    async def similar_paginated(self, query: str, k: int = 6, offset: int = 0) -> List[Dict]:
        await self._ensure_ready()
        query_embeddings = await self._embed_texts([query])
        query_vector = query_embeddings[0] if query_embeddings else []
        if not query_vector:
            # If we still cannot embed, fall back to recent items and mark perfect score.
            recent = await self.recent_paginated(k=k, offset=offset)
            for item in recent:
                item["score"] = 1.0
            return recent
        rows = await asyncio.to_thread(self._fetch_session_rows, self._session)
        scored: List[tuple[float, MemoryItem]] = []
        for item, vector, vector_norm in rows:
            score = _cosine_similarity(query_vector, vector, vector_norm)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        start = max(offset, 0)
        end = start + max(k, 0)
        results: List[Dict[str, Any]] = []
        for score, item in scored[start:end]:
            payload = item.model_dump()
            payload["score"] = float(score)
            results.append(payload)
        return results

    async def recent_paginated(self, k: int = 6, offset: int = 0) -> List[Dict]:
        await self._ensure_ready()
        rows = await asyncio.to_thread(self._fetch_recent_rows_paginated, self._session, k, offset)
        return [item.model_dump() for item in rows]

    async def get_many(self, ids: Sequence[str]) -> List[Dict]:
        await self._ensure_ready()
        if not ids:
            return []
        rows = await asyncio.to_thread(self._fetch_by_ids, ids)
        return [item.model_dump() for item, _, _ in rows]

    async def all(self) -> List[Dict]:
        await self._ensure_ready()
        rows = await asyncio.to_thread(self._fetch_session_rows, self._session)
        return [item.model_dump() for item, _, _ in rows]

    async def reset(self) -> None:
        await self._ensure_ready()
        await asyncio.to_thread(self._delete_session, self._session)

    # MemoryStore interface --------------------------------------------------
    async def delete(self, item_id: str) -> None:
        await self._ensure_ready()
        await asyncio.to_thread(self._delete_item, item_id)

    async def count(self) -> int:
        await self._ensure_ready()
        return await asyncio.to_thread(self._count_session, self._session)

    async def bulk_delete(self, ids: Sequence[str]) -> int:
        await self._ensure_ready()
        if not ids:
            return 0
        return await asyncio.to_thread(self._bulk_delete_items, ids)

    async def sessions(self) -> List[str]:
        await self._ensure_ready()
        return await asyncio.to_thread(self._fetch_sessions)

    async def stats(self) -> Dict[str, Any]:
        await self._ensure_ready()
        stats = await asyncio.to_thread(self._fetch_stats)
        metadata = self._get_all_metadata()
        stats["embedding_provider"] = metadata.get("embedding_provider")
        stats["embedding_model"] = metadata.get("embedding_model")
        stats["vector_dim"] = self._as_int(metadata.get("vector_dim"))
        stats["last_reindex_ts"] = self._as_float(metadata.get("last_reindex_ts"))
        stats["schema_version"] = self._as_int(metadata.get("schema_version"), default=SCHEMA_VERSION)
        return stats

    async def reindex(self) -> None:
        await self._ensure_ready()
        async with self._reindex_lock:
            await self._perform_reindex()
            self._needs_reindex = False

    # Internal helpers -------------------------------------------------------
    def _init_metadata_storage(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def _set_metadata(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_metadata (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def _get_metadata(self, key: str) -> Any:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM memory_metadata WHERE key=?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

    def _get_all_metadata(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute("SELECT key, value FROM memory_metadata")
            rows = cur.fetchall()
        metadata: Dict[str, Any] = {}
        for key, value in rows:
            try:
                metadata[str(key)] = json.loads(value)
            except Exception:
                metadata[str(key)] = value
        return metadata

    def _has_rows(self) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT 1 FROM memory LIMIT 1")
            return cur.fetchone() is not None

    def _maybe_record_vector_dim(self, dimension: int) -> None:
        if dimension <= 0:
            return
        current = self._get_metadata("vector_dim")
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            current_value = 0
        if current_value > 0:
            return
        self._set_metadata("vector_dim", dimension)

    def _infer_vector_dim(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT vector FROM memory WHERE vector IS NOT NULL AND length(vector) > 0 LIMIT 1"
            )
            row = cur.fetchone()
        if row and row[0]:
            vector = _blob_to_vector(row[0])
            return len(vector)
        return 0

    def _current_embedding_signature(self) -> Dict[str, str]:
        provider = "local_fallback"
        model = "sentence-transformers/all-MiniLM-L6-v2"
        if self._fallback_signature:
            provider = self._fallback_signature.get("provider", provider)
            model = self._fallback_signature.get("model", model)
            return {"provider": provider, "model": model}
        if self._embeddings is not None and getattr(self._embeddings, "provider", None):
            provider_cfg = getattr(self._embeddings.provider, "config", None)
            if provider_cfg and getattr(provider_cfg, "provider", None):
                provider = provider_cfg.provider
            else:
                provider = self._embeddings.provider.__class__.__name__
            try:
                model = self._embeddings.provider.model
            except Exception:
                model = self._embeddings.provider.__class__.__name__
        return {"provider": provider, "model": model}

    async def _ensure_ready(self) -> None:
        if self._metadata_checked:
            await self._run_reindex_if_needed()
            return
        async with self._init_lock:
            if not self._metadata_checked:
                metadata = self._get_all_metadata()
                stored_version = self._as_int(metadata.get("schema_version"))
                if stored_version < SCHEMA_VERSION:
                    self._set_metadata("schema_version", SCHEMA_VERSION)
                signature = self._current_embedding_signature()
                stored_provider = metadata.get("embedding_provider")
                stored_model = metadata.get("embedding_model")
                has_rows = self._has_rows()
                if stored_provider and stored_provider != signature["provider"]:
                    self._needs_reindex = True
                if stored_model and stored_model != signature["model"]:
                    self._needs_reindex = True
                if has_rows and (stored_provider is None or stored_model is None):
                    self._needs_reindex = True
                if stored_provider != signature["provider"]:
                    self._set_metadata("embedding_provider", signature["provider"])
                if stored_model != signature["model"]:
                    self._set_metadata("embedding_model", signature["model"])
                current_dim = self._as_int(metadata.get("vector_dim"))
                if current_dim <= 0:
                    inferred = self._infer_vector_dim()
                    if inferred:
                        self._set_metadata("vector_dim", inferred)
                self._metadata_checked = True
        await self._run_reindex_if_needed()

    async def _run_reindex_if_needed(self) -> None:
        if not self._needs_reindex:
            return
        async with self._reindex_lock:
            if not self._needs_reindex:
                return
            await self._perform_reindex()
            self._needs_reindex = False

    async def _perform_reindex(self) -> None:
        rows = await asyncio.to_thread(self._fetch_all_rows)
        signature = self._current_embedding_signature()
        if not rows:
            self._set_metadata("embedding_provider", signature["provider"])
            self._set_metadata("embedding_model", signature["model"])
            self._set_metadata("vector_dim", 0)
            self._set_metadata("last_reindex_ts", time.time())
            logger.info("Memory reindex skipped; no rows present")
            return
        texts = [text for _, text in rows]
        vectors = await self._embed_texts(texts)
        if len(vectors) != len(rows):
            raise RuntimeError("Embedding count mismatch during reindex")
        updates = []
        dimension = 0
        for (item_id, _), vector in zip(rows, vectors):
            vector_norm = math.sqrt(sum(v * v for v in vector))
            updates.append((_vector_to_blob(vector), vector_norm, item_id))
            if not dimension and vector:
                dimension = len(vector)
        await asyncio.to_thread(self._bulk_update_vectors, updates)
        if not dimension:
            dimension = self._infer_vector_dim()
        self._set_metadata("vector_dim", dimension)
        self._set_metadata("last_reindex_ts", time.time())
        self._set_metadata("embedding_provider", signature["provider"])
        self._set_metadata("embedding_model", signature["model"])
        logger.info("Memory reindex completed with %d items", len(rows))

    def _as_int(self, value: Any, *, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_float(self, value: Any, *, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
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

    def _fetch_recent_rows_paginated(self, session: str, limit: int, offset: int) -> List[MemoryItem]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, kind, ts, tags, text, meta, vector, vector_norm
                FROM memory
                WHERE session=?
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (session, max(limit, 0), max(offset, 0)),
            )
            rows = cur.fetchall()
        return [self._row_to_item(row)[0] for row in rows]

    def _fetch_by_ids(self, ids: Sequence[str]) -> List[tuple[MemoryItem, List[float], float]]:
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        query = (
            "SELECT id, kind, ts, tags, text, meta, vector, vector_norm "
            f"FROM memory WHERE id IN ({placeholders})"
        )
        with self._lock:
            cur = self._conn.execute(query, tuple(ids))
            rows = cur.fetchall()
        return [self._row_to_item(row) for row in rows]

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

    def _fetch_all_rows(self) -> List[tuple[str, str]]:
        with self._lock:
            cur = self._conn.execute("SELECT id, text FROM memory")
            rows = cur.fetchall()
        return [(str(item_id), str(text)) for item_id, text in rows]

    def _bulk_update_vectors(self, updates: Sequence[tuple[bytes, float, str]]) -> None:
        if not updates:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memory SET vector=?, vector_norm=? WHERE id=?",
                updates,
            )
            self._conn.commit()

    def _bulk_delete_items(self, ids: Sequence[str]) -> int:
        with self._lock:
            cur = self._conn.executemany(
                "DELETE FROM memory WHERE id=?",
                [(item_id,) for item_id in ids],
            )
            self._conn.commit()
            return cur.rowcount if cur else 0

    def _fetch_sessions(self) -> List[str]:
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT session FROM memory ORDER BY session ASC")
            rows = cur.fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

    def _fetch_stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT
                    COUNT(*) as total_count,
                    SUM(CASE WHEN vector IS NOT NULL AND length(vector) > 0 THEN 1 ELSE 0 END) as with_vectors,
                    SUM(CASE WHEN vector IS NULL OR length(vector) = 0 THEN 1 ELSE 0 END) as without_vectors,
                    MAX(ts) as last_ts
                FROM memory
                """
            )
            (total_count, with_vectors, without_vectors, last_ts) = cur.fetchone()
        return {
            "total_count": int(total_count or 0),
            "with_vectors": int(with_vectors or 0),
            "without_vectors": int(without_vectors or 0),
            "last_ts": float(last_ts or 0.0),
        }

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

    async def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        used_primary = False
        if self._embeddings is not None:
            try:
                vectors = await self._embeddings.embed(list(texts))
                used_primary = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Primary embeddings failed: %s", exc)
                vectors = []
        if vectors and not any(len(vec) == 0 for vec in vectors):
            if used_primary:
                self._fallback_signature = None
            return vectors
        vectors = await asyncio.to_thread(self._local_embed, list(texts))
        return vectors

    def _local_embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if self._embeddings is not None:
            try:
                configured = getattr(self._embeddings.provider, "model", None)
                if configured:
                    model_name = str(configured)
            except Exception:
                pass
        encoder = self._ensure_local_encoder(model_name)
        if encoder is None:
            return [[] for _ in texts]
        try:
            embeddings = encoder.encode(list(texts), convert_to_numpy=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Local sentence-transformer embedding failed: %s", exc)
            return [[] for _ in texts]
        vectors = [vec.tolist() for vec in embeddings]
        # Record that fallback embeddings were used so metadata stays accurate
        self._fallback_signature = {"provider": "sentence_transformers", "model": model_name}
        try:
            self._set_metadata("embedding_provider", "sentence_transformers")
            self._set_metadata("embedding_model", model_name)
        except Exception:  # pragma: no cover - metadata updates are best-effort
            pass
        return vectors

    def _ensure_local_encoder(self, model_name: str):
        with _LOCAL_ENCODER_LOCK:
            encoder = _LOCAL_ENCODERS.get(model_name)
            if encoder is not None:
                return encoder
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                encoder = SentenceTransformer(model_name)
                _LOCAL_ENCODERS[model_name] = encoder
                logger.info(
                    "Loaded local sentence-transformers embeddings for memory store (model=%s)",
                    model_name,
                )
                return encoder
            except Exception as exc:  # pragma: no cover - fallback
                logger.error("Failed to load local sentence-transformers model '%s': %s", model_name, exc)
                return None


__all__ = ["SQLiteFAISSMemory"]
