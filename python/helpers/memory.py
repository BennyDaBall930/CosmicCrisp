from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from simpleeval import simple_eval

from python.helpers.print_style import PrintStyle
from python.runtime.memory import get_memory
from python.runtime.memory.schema import MemoryItem
from python.runtime.memory.store import MemoryStore
from python.helpers import files, knowledge_import
from python.helpers.log import LogItem


class Memory:
    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"
        INSTRUMENTS = "instruments"

    _instances: Dict[str, "Memory"] = {}

    def __init__(self, agent, store: MemoryStore, session: str) -> None:
        self.agent = agent
        self.store = store
        self.session = session or "default"

    @classmethod
    async def get(cls, agent) -> "Memory":
        session = getattr(getattr(agent, "config", None), "memory_subdir", "") or "default"
        instance = cls._instances.get(session)
        if instance is None:
            store = get_memory()
            await store.enter(session)
            instance = cls(agent, store, session)
            cls._instances[session] = instance
            try:
                log_item: Optional[LogItem] = agent.context.log.log(  # type: ignore[attr-defined]
                    type="util",
                    heading=f"Initializing memory for '/{session}'",
                )
            except Exception:
                log_item = None
            knowledge_dirs = getattr(getattr(agent, "config", None), "knowledge_subdirs", None)
            if knowledge_dirs:
                if isinstance(knowledge_dirs, (list, tuple, set)):
                    dirs = list(knowledge_dirs)
                else:
                    dirs = [knowledge_dirs]
                await instance.preload_knowledge(
                    log_item,
                    dirs,  # type: ignore[arg-type]
                    session,
                )
            if log_item:
                log_item.update(result="Memory initialized")
        else:
            await instance.store.enter(session)
            instance.agent = agent
        return instance

    @classmethod
    async def reload(cls, agent) -> "Memory":
        session = getattr(getattr(agent, "config", None), "memory_subdir", "") or "default"
        cls._instances.pop(session, None)
        return await cls.get(agent)

    async def search_similarity_threshold(
        self,
        query: str,
        limit: int,
        threshold: float,
        filter: str = "",
    ) -> List[Document]:
        await self.store.enter(self.session)
        k = max(limit * 3 if limit else 25, 25)
        raw = await self.store.similar_paginated(query=query, k=k, offset=0)
        comparator = self._get_comparator(filter) if filter else None

        results: List[Document] = []
        for item in raw:
            metadata = item.get("meta", {}) or {}
            if comparator and not comparator(metadata):
                continue
            score = float(item.get("score", 0.0) or 0.0)
            if score < threshold:
                continue
            results.append(self._to_document(item))
            if limit and len(results) >= limit:
                break

        if not results and query:
            tokens = [tok for tok in query.lower().strip().split() if tok]
            if tokens:
                fallback_docs = await self.store.all()
                for item in fallback_docs:
                    text = str(item.get("text", ""))
                    text_lower = text.lower()
                    if not all(tok in text_lower for tok in tokens):
                        continue
                    match_ratio = sum(tok in text_lower for tok in tokens) / len(tokens)
                    score_hint = max(match_ratio, threshold) if threshold else match_ratio
                    item.setdefault("meta", {})
                    item["score"] = max(score_hint, 0.1)
                    doc = self._to_document(item)
                    doc.metadata["score"] = doc.metadata.get("score", score_hint)
                    results.append(doc)
                    if limit and len(results) >= limit:
                        break
                if limit:
                    results = results[:limit]

        return results

    async def search_by_metadata(self, filter: str, limit: int = 0) -> List[Document]:
        await self.store.enter(self.session)
        comparator = self._get_comparator(filter)
        items = await self.store.all()
        matches: List[Document] = []
        for item in items:
            metadata = item.get("meta", {}) or {}
            if comparator(metadata):
                matches.append(self._to_document(item))
                if limit and len(matches) >= limit:
                    break
        return matches

    async def insert_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        await self.store.enter(self.session)
        meta = dict(metadata or {})
        kind = meta.pop("kind", "note")
        tags_raw = meta.pop("tags", [])
        tags = list(tags_raw if isinstance(tags_raw, list) else [tags_raw])
        if not meta.get("area"):
            meta["area"] = Memory.Area.MAIN.value
        if "import_timestamp" not in meta or meta.get("import_timestamp") in (None, ""):
            meta["import_timestamp"] = self.get_timestamp()
        item = MemoryItem(
            kind=kind,
            ts=datetime.utcnow(),
            tags=tags,
            text=text,
            meta=meta,
        )
        item.meta.setdefault("session", self.session)
        return await self.store.add(item)

    async def insert_documents(self, docs: Sequence[Document]) -> List[str]:
        ids: List[str] = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            text = getattr(doc, "page_content", "")
            new_id = await self.insert_text(text, dict(metadata))
            ids.append(new_id)
        return ids

    async def delete_documents_by_ids(self, ids: Sequence[str]) -> List[Document]:
        await self.store.enter(self.session)
        existing = await self.store.get_many(list(ids))
        docs = [self._to_document(item) for item in existing]
        if ids:
            await self.store.bulk_delete(list(ids))
        return docs

    async def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        if not ids:
            return []
        await self.store.enter(self.session)
        items = await self.store.get_many(list(ids))
        return [self._to_document(item) for item in items]

    async def get_recent(self, limit: int = 10, offset: int = 0) -> List[Document]:
        await self.store.enter(self.session)
        raw = await self.store.recent_paginated(k=limit, offset=offset)
        return [self._to_document(item) for item in raw]

    async def delete_documents_by_query(self, query: str, threshold: float, filter: str = "") -> List[Document]:
        await self.store.enter(self.session)
        comparator = self._get_comparator(filter) if filter else None
        k = 200
        raw = await self.store.similar_paginated(query=query, k=k, offset=0)
        matches: List[Dict[str, Any]] = []
        ids: List[str] = []
        for item in raw:
            score = float(item.get("score", 0.0) or 0.0)
            meta = item.get("meta", {}) or {}
            if score < threshold:
                continue
            if comparator and not comparator(meta):
                continue
            matches.append(item)
            if item.get("id"):
                ids.append(str(item["id"]))
        if not ids:
            return []
        docs = [self._to_document(m) for m in matches]
        await self.store.bulk_delete(ids)
        return docs

    async def preload_knowledge(
        self,
        log_item: Optional[LogItem],
        kn_dirs: List[str],
        memory_subdir: str,
    ) -> None:
        if not kn_dirs:
            return

        if log_item:
            log_item.update(heading="Preloading knowledge...")

        db_dir = self._abs_db_dir(memory_subdir)
        os.makedirs(db_dir, exist_ok=True)
        index_path = files.get_abs_path(db_dir, "knowledge_import.json")

        index: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as fh:
                    index = json.load(fh)
            except Exception:
                index = {}

        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        for file_path, entry in list(index.items()):
            state = entry.get("state")
            ids = entry.get("ids", []) or []
            documents = entry.get("documents", []) or []

            if state in {"changed", "removed"} and ids:
                await self.delete_documents_by_ids([str(i) for i in ids])

            if state == "changed" and documents:
                try:
                    entry["ids"] = await self.insert_documents(documents)
                except Exception as exc:
                    PrintStyle().error(f"Failed to insert knowledge from {file_path}: {exc}")
                    entry["ids"] = []

        index = {k: v for k, v in index.items() if v.get("state") != "removed"}
        for entry in index.values():
            entry.pop("documents", None)
            entry.pop("state", None)

        try:
            with open(index_path, "w") as fh:
                json.dump(index, fh)
        except Exception as exc:
            PrintStyle().error(f"Failed to persist knowledge index: {exc}")

    def _preload_knowledge_folders(
        self,
        log_item: Optional[LogItem],
        kn_dirs: List[str],
        index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        for kn_dir in kn_dirs:
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    files.get_abs_path("knowledge", kn_dir, area.value),
                    index,
                    {"area": area.value},
                )

        index = knowledge_import.load_knowledge(
            log_item,
            files.get_abs_path("instruments"),
            index,
            {"area": Memory.Area.INSTRUMENTS.value},
            filename_pattern="**/*.md",
        )
        return index

    @staticmethod
    def _abs_db_dir(memory_subdir: str) -> str:
        return files.get_abs_path("memory", memory_subdir)

    @staticmethod
    def format_docs_plain(docs: Sequence[Document]) -> List[str]:
        formatted: List[str] = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            lines = [f"{key}: {value}" for key, value in metadata.items()]
            lines.append(f"Content: {getattr(doc, 'page_content', '')}")
            formatted.append("\n".join(lines))
        return formatted

    @staticmethod
    def get_timestamp() -> str:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    @staticmethod
    def _get_comparator(condition: str):
        def comparator(data: Dict[str, Any]) -> bool:
            try:
                return bool(simple_eval(condition, names=data))
            except Exception as exc:  # pragma: no cover - defensive
                PrintStyle.error(f"Error evaluating condition '{condition}': {exc}")
                return False

        return comparator

    @staticmethod
    def _to_document(item: Dict[str, Any]) -> Document:
        metadata = dict(item.get("meta", {}) or {})
        metadata.setdefault("id", item.get("id"))
        metadata.setdefault("kind", item.get("kind", "note"))
        metadata.setdefault("tags", item.get("tags", []))
        metadata.setdefault("timestamp", item.get("ts"))
        if "score" not in metadata and item.get("score") is not None:
            metadata["score"] = item.get("score")
        if metadata.get("session") is None and item.get("meta"):
            metadata["session"] = item["meta"].get("session")
        text = item.get("text", "")
        return Document(page_content=text, metadata=metadata)


def get_memory_subdir_abs(agent) -> str:
    """Project-relative path to the agent's memory subdirectory."""
    subdir = getattr(getattr(agent, "config", None), "memory_subdir", "") or "default"
    return files.get_abs_path("memory", subdir)


def get_custom_knowledge_subdir_abs(agent) -> str:
    """Absolute path to the agent's custom knowledge subdir."""
    for dir_name in getattr(agent.config, "knowledge_subdirs", []) or []:
        if dir_name != "default":
            return files.get_abs_path("knowledge", dir_name)
    raise Exception("No custom knowledge subdir set")


def reload():
    Memory._instances = {}
