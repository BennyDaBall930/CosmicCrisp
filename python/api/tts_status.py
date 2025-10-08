"""TTS status endpoint to check health and availability."""
import logging
import sys
from typing import Any

from python.helpers.api import ApiHandler, Request, Response
from python.helpers import settings

logger = logging.getLogger(__name__)


def _get_tts_settings() -> dict:
    all_settings = settings.get_settings()
    tts_settings = all_settings.get("tts")
    if not isinstance(tts_settings, dict):
        tts_settings = settings.get_default_settings()["tts"]
    return tts_settings


def _check_xtts_health() -> dict:
    """Check XTTS sidecar health by attempting connection."""
    import urllib.request
    import urllib.error

    health_url = "http://127.0.0.1:7055/healthz"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                return {"available": True, "healthy": True, "url": health_url}
            else:
                return {"available": True, "healthy": False, "status": resp.status}
    except urllib.error.HTTPError as e:
        return {"available": True, "healthy": False, "error": str(e), "status": e.code}
    except urllib.error.URLError as e:
        return {"available": False, "healthy": False, "error": str(e)}
    except Exception as e:
        return {"available": False, "healthy": False, "error": str(e)}


def _check_chatterbox_availability() -> dict:
    """Check if Chatterbox is available (Python 3.10 or 3.11 required)."""
    try:
        # Chatterbox requires Python 3.11, fails on 3.12+
        if sys.version_info >= (3, 12):
            return {"available": False, "reason": "Python 3.12+ not supported by Chatterbox"}

        if sys.version_info < (3, 10):
            return {"available": False, "reason": "Python >= 3.10 required for Chatterbox"}

        # Try to import
        import torch  # noqa: F401
        from python.helpers.chatterbox_tts import get_backend # noqa: F401
        return {"available": True, "python_version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    except ImportError as e:
        return {"available": False, "reason": f"Import failed: {e}"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _check_kokoro_availability() -> dict:
    """Check if Kokoro is available and can be initialized."""
    try:
        # Test if kokoro package is available
        import torch  # noqa: F401
        import soundfile  # noqa: F401

        # Try a quick import test without full pipeline initialization
        from kokoro import KPipeline  # noqa: F401
        return {"available": True}
    except ImportError as e:
        return {"available": False, "reason": f"Import failed: {e}"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


class TtsStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        tts_settings = _get_tts_settings()
        active_engine = tts_settings.get("engine", "chatterbox")

        status = {
            "active_engine": active_engine,
            "engines": {}
        }

        # Check each engine's status
        # Browser - always available (client-side)
        status["engines"]["browser"] = {"available": True, "note": "Client-side TTS, requires user interaction"}

        # XTTS - check sidecar health
        status["engines"]["xtts"] = _check_xtts_health()

        # Chatterbox - check Python version and imports
        status["engines"]["chatterbox"] = _check_chatterbox_availability()

        # Kokoro - check imports
        status["engines"]["kokoro"] = _check_kokoro_availability()

        # Note about Piper VC - would require additional checks, but keeping simple for now
        status["engines"]["piper_vc"] = {"available": False, "note": "Complex setup required"}

        return status
