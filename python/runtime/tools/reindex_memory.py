"""CLI utility to rebuild the local memory vector cache."""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from ..config import load_runtime_config
from ..embeddings import get_embeddings
from ..memory import get_memory

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the SQLite memory vector cache.")
    parser.add_argument(
        "--session",
        help="Reindex a specific session only (requires store support).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging during reindexing.",
    )
    return parser


async def _run(session: Optional[str], verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO)
    config = load_runtime_config()
    embeddings = get_embeddings(config)
    store = get_memory(config, embeddings=embeddings)

    if session and hasattr(store, "enter"):
        await getattr(store, "enter")(session)

    reindex = getattr(store, "reindex", None)
    if not callable(reindex):
        logger.error("Active memory store does not expose a reindex() method")
        return

    logger.info("Rebuilding memory vectors%s", f" for session '{session}'" if session else "")
    await reindex()
    logger.info("Memory reindex completed")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args.session, args.verbose))


if __name__ == "__main__":  # pragma: no cover
    main()
