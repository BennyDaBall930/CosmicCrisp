"""CLI utility to migrate legacy memory exports into the unified store."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from ..config import load_runtime_config
from ..embeddings import get_embeddings
from ..memory import get_memory
from ..memory.schema import MemoryItem


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import legacy memory JSON/NDJSON into the runtime store.")
    parser.add_argument("input", type=Path, help="File or directory containing exported memory data.")
    parser.add_argument(
        "--session",
        help="Override session/tag applied to imported items.",
    )
    parser.add_argument(
        "--kind",
        default="note",
        help="Default kind for imported items when not provided (default: note).",
    )
    return parser


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_dir():
        for candidate in sorted(path.rglob("*.json")):
            yield candidate
        for candidate in sorted(path.rglob("*.ndjson")):
            yield candidate
    else:
        yield path


def _load_file(path: Path) -> Iterable[Dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]
    if isinstance(data, dict):
        records = data.get("items")
        if isinstance(records, list):
            return [obj for obj in records if isinstance(obj, dict)]
    return []


async def _run(input_path: Path, *, session: Optional[str], default_kind: str) -> None:
    config = load_runtime_config()
    embeddings = get_embeddings(config)
    store = get_memory(config, embeddings=embeddings)

    if session and hasattr(store, "enter"):
        await getattr(store, "enter")(session)

    imported = 0
    for file in _iter_files(input_path):
        for record in _load_file(file):
            payload = dict(record)
            payload.setdefault("kind", default_kind)
            tags = payload.get("tags") or []
            if not isinstance(tags, list):
                tags = list(tags)
            meta = payload.get("meta") or {}
            if session:
                if session not in tags:
                    tags.append(session)
                meta = dict(meta)
                meta.setdefault("session", session)
            payload["tags"] = tags
            payload["meta"] = meta
            item = MemoryItem(**payload)
            await store.add(item)
            imported += 1
    print(f"Imported {imported} records into memory store")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args.input, session=args.session, default_kind=args.kind))


if __name__ == "__main__":  # pragma: no cover
    main()
