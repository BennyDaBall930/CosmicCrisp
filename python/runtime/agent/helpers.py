"""Miscellaneous helper functions for the agent."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict


async def stream_dict(data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Yield dict as a JSON line for streaming."""
    yield json.dumps(data) + "\n"
