"""Simplified SQLite memory store with FAISS-like API."""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List

from .interface import AgentMemory


class SQLiteFAISSMemory(AgentMemory):
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, session TEXT, content TEXT)"
        )
        self.session = "default"

    async def enter(self, session_id: str) -> None:
        self.session = session_id

    async def add(self, item: Dict) -> str:
        content = json.dumps(item)
        cur = self.conn.execute(
            "INSERT INTO memory(session, content) VALUES (?, ?)",
            (self.session, content),
        )
        self.conn.commit()
        return str(cur.lastrowid)

    async def similar(self, query: str, k: int = 5) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT content FROM memory WHERE session=? ORDER BY id DESC LIMIT ?",
            (self.session, k),
        )
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    async def reset(self) -> None:
        self.conn.execute("DELETE FROM memory WHERE session=?", (self.session,))
        self.conn.commit()
