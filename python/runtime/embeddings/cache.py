"""SQLite-backed embeddings cache."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    arr = array("f", vector)
    return arr.tobytes()


def _blob_to_vector(blob: bytes | memoryview | None) -> List[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(bytes(blob))
    return list(arr)


class SQLiteEmbeddingsCache:
    """Minimal async-friendly cache storing embeddings vectors by hash key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        _ensure_parent(path)
        self._conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                cache_key TEXT PRIMARY KEY,
                created REAL NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def make_cache_key(model: str, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(b"::")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _get_many(self, keys: Sequence[str]) -> Dict[str, List[float]]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT cache_key, vector FROM embeddings_cache WHERE cache_key IN ({placeholders})"
        with self._lock:
            cur = self._conn.execute(query, tuple(keys))
            rows = cur.fetchall()
        result: Dict[str, List[float]] = {}
        for cache_key, blob in rows:
            result[cache_key] = _blob_to_vector(blob)
        return result

    async def get_many(self, keys: Sequence[str]) -> Dict[str, List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_many, keys)

    def _set_many(self, entries: Mapping[str, Sequence[float]]) -> None:
        if not entries:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "REPLACE INTO embeddings_cache(cache_key, created, vector) VALUES (?, ?, ?)",
                [
                    (cache_key, now, _vector_to_blob(vector))
                    for cache_key, vector in entries.items()
                ],
            )
            self._conn.commit()

    async def set_many(self, entries: Mapping[str, Sequence[float]]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._set_many, entries)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embeddings_cache")
            self._conn.commit()


__all__ = ["SQLiteEmbeddingsCache"]
