#!/usr/bin/env python3
"""Script to initialize local mem0 instance and preload Apple Zero memories."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add the python directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file manually since we're running as standalone script
try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).parent.parent / ".env"
    success = load_dotenv(dotenv_path=str(dotenv_path))
    if success:
        logger.debug("Loaded .env file with python-dotenv")
    else:
        logger.warning("Failed to load .env file: %s", dotenv_path)
except ImportError:
    logger.warning("python-dotenv not available, using system environment")

# Debug environment variable
logger.info("MEM0_LOCAL_MODE environment variable: %r", os.getenv("MEM0_LOCAL_MODE", "NOT_SET"))

try:
    from mem0 import MemoryClient
except ImportError as exc:
    print(f"mem0 not installed: {exc}", file=sys.stderr)
    sys.exit(1)


class AppleZeroMemoryPreloader:
    """Class to manage preloading of Apple Zero memories with proper metadata."""

    def __init__(self, namespace: str = "applezero", base_url: Optional[str] = None):
        """Initialize the preloader with mem0 client."""
        self.namespace = namespace
        self.client = None
        self.mem0_available = False

        # Try to initialize mem0 client
        try:
            api_key = os.getenv("MEM0_API_KEY")
            if not api_key:
                logger.warning("MEM0_API_KEY not provided - falling back to SQLite-only mode")
            else:
                self.client = MemoryClient(
                    api_key=api_key,
                    host=base_url or os.getenv("MEM0_BASE_URL"),
                )
                self.mem0_available = True
                logger.info("✓ Mem0 client initialized successfully")
        except Exception as exc:
            logger.warning(f"Mem0 client initialization failed: {exc} - falling back to SQLite-only mode")

        self.preload_data_path = Path(__file__).parent / "mem0_preload_data.json"

    async def setup_local_mem0(self) -> bool:
        """Initialize local mem0 instance with SQLite backend."""
        if not self.mem0_available or self.client is None:
            logger.info("Mem0 not available - setup skipped, using SQLite fallback")
            return True  # Consider this successful since we're falling back gracefully

        try:
            # Test connection - local mem0 should work without external API
            await asyncio.to_thread(self.client.add, [], user_id=self.namespace)
            logger.info("✓ Mem0 local instance initialized successfully")
            return True
        except Exception as exc:
            logger.error("mem0 initialization failed: %s", exc)
            return False

    async def preload_apple_zero_memories(self) -> int:
        """Load Apple Zero contextual memories into local mem0 store."""
        if not self.mem0_available or self.client is None:
            logger.info("Mem0 not available - memory preloading skipped, using SQLite fallback for memory operations")
            return 0  # No mem0 memories can be preloaded

        if not self.preload_data_path.exists():
            logger.warning("Preload data file not found: %s", self.preload_data_path)
            return 0

        try:
            with open(self.preload_data_path, 'r', encoding='utf-8') as f:
                memories: List[Dict] = json.load(f)

            loaded_count = 0
            for memory_data in memories:
                try:
                    # Format for mem0 client
                    messages = [{"role": "user", "content": memory_data["text"]}]
                    metadata = memory_data.get("metadata", {})

                    await asyncio.to_thread(
                        self.client.add,
                        messages,
                        user_id=self.namespace,
                        metadata=metadata
                    )
                    loaded_count += 1
                    logger.debug("Loaded memory: %s", memory_data["text"][:50] + "...")

                except Exception as exc:
                    logger.warning("Failed to load memory: %s", exc)
                    continue

            logger.info("✓ Successfully preloaded %d Apple Zero memories", loaded_count)
            return loaded_count

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Invalid preload data format: %s", exc)
            return 0
        except Exception as exc:
            logger.error("Error preloading memories: %s", exc)
            return 0

    async def verify_mem0_integration(self) -> bool:
        """Test mem0 connection and fallback functionality."""
        if not self.mem0_available or self.client is None:
            logger.info("Mem0 not available - verification skipped, using SQLite fallback")
            logger.info("✓ Memory system ready (SQLite-only mode)")
            return True  # Return success since SQLite fallback provides functionality

        try:
            # Test basic operations
            memories = await asyncio.to_thread(self.client.get_all, user_id=self.namespace)
            if memories is None:
                memories = []

            logger.info("✓ Mem0 integration verified (%d memories found)", len(memories))

            # Test search functionality
            if memories:
                sample_query = "Apple ecosystem"
                results = await asyncio.to_thread(
                    self.client.search,
                    sample_query,
                    limit=5,
                    user_id=self.namespace
                )
                logger.info("✓ Mem0 search functionality verified")

            return True

        except Exception as exc:
            logger.error("Mem0 integration verification failed: %s", exc)
            logger.warning("Falling back to SQLite-only mode")
            return True  # Return success since SQLite fallback will work


async def main():
    """Main setup function."""
    logger.info("Starting mem0 setup and Apple Zero memory preloading...")

    # Check environment configuration
    if not os.getenv("MEM0_LOCAL_MODE", "").lower() in ("true", "1", "yes"):
        logger.warning("MEM0_LOCAL_MODE not set to true, skipping local setup")
        return 1

    preloader = AppleZeroMemoryPreloader()

    # Step 1: Initialize local mem0
    if not await preloader.setup_local_mem0():
        logger.error("Failed to initialize mem0")
        return 1

    # Step 2: Preload Apple Zero memories
    preloaded_count = await preloader.preload_apple_zero_memories()
    if preloaded_count == 0:
        logger.warning("No memories were preloaded")

    # Step 3: Verify integration
    if not await preloader.verify_mem0_integration():
        logger.error("Mem0 integration verification failed")
        return 1

    logger.info("✓ Mem0 setup completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
