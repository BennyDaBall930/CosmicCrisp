"""Unified tool registry for the runtime."""

from __future__ import annotations

from .registry import registry

# Import default tools to ensure they are registered on package import
from . import browser  # noqa: F401  # pylint: disable=unused-import
from . import code  # noqa: F401  # pylint: disable=unused-import
from . import image  # noqa: F401  # pylint: disable=unused-import
from . import search  # noqa: F401  # pylint: disable=unused-import

__all__ = ["registry"]

