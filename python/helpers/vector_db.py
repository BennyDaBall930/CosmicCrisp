from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document
from simpleeval import simple_eval

from python.helpers.print_style import PrintStyle


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _get_comparator(condition: str):
    def comparator(metadata: Dict[str, Any]) -> bool:
        try:
            return bool(simple_eval(condition, names=metadata))
        except Exception as exc:  # pragma: no cover - defensive
            PrintStyle.error(f"Error evaluating condition '{condition}': {exc}")
            return False

    return comparator


class VectorDB:
    _cached_embeddings: Dict[str, Any] = {}

    @staticmethod
    def _get_embeddings(agent, cache: bool = True):
        model = agent.get_embedding_model()
        if not cache:
            return model
        namespace = getattr(model, "model_name", "default")
        # Cache the embedding model to avoid re-instantiation cost
        if namespace not in VectorDB._cached_embeddings:
            VectorDB._cached_embeddings[namespace] = model
        return VectorDB._cached_embeddings[namespace]

    def __init__(self, agent, cache: bool = True) -> None:
        self.agent = agent
        self.embeddings = self._get_embeddings(agent, cache=cache)
        self._docs: Dict[str, Document] = {}
        self._vectors: Dict[str, List[float]] = {}

    async def search_by_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ) -> List[Document]:
        comparator = _get_comparator(filter) if filter else None
        query_vector = await self._embed_query(query)
        scored: List[tuple[float, Document]] = []
        for doc_id, vector in self._vectors.items():
            score = _cosine_similarity(query_vector, vector)
            if score < threshold:
                continue
            doc = self._docs[doc_id]
            if comparator and not comparator(doc.metadata):
                continue
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    async def search_by_metadata(self, filter: str, limit: int = 0) -> List[Document]:
        comparator = _get_comparator(filter)
        results: List[Document] = []
        for doc in self._docs.values():
            if comparator(doc.metadata):
                results.append(doc)
                if limit and len(results) >= limit:
                    break
        return results

    async def insert_documents(self, docs: Sequence[Document]) -> List[str]:
        texts = [doc.page_content for doc in docs]
        vectors = await self._embed_documents(texts)
        ids: List[str] = []
        for doc, vector in zip(docs, vectors):
            doc_id = str(uuid.uuid4())
            doc.metadata["id"] = doc_id
            self._docs[doc_id] = doc
            self._vectors[doc_id] = vector
            ids.append(doc_id)
        return ids

    async def delete_documents_by_ids(self, ids: Sequence[str]) -> List[Document]:
        removed: List[Document] = []
        for doc_id in ids:
            doc = self._docs.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
            if doc:
                removed.append(doc)
        return removed

    def get_all_docs(self) -> Dict[str, Document]:
        return dict(self._docs)

    async def _embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if hasattr(self.embeddings, "embed_documents"):
            return await asyncio.to_thread(self.embeddings.embed_documents, list(texts))
        # Fallback: use embed_query per text
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(await self._embed_query(text))
        return vectors

    async def _embed_query(self, text: str) -> List[float]:
        if hasattr(self.embeddings, "embed_query"):
            return await asyncio.to_thread(self.embeddings.embed_query, text)
        if hasattr(self.embeddings, "embed_documents"):
            result = await asyncio.to_thread(self.embeddings.embed_documents, [text])
            return result[0] if result else []
        raise RuntimeError("Embedding model does not support embedding queries")
