from python.helpers.api import ApiHandler, Request, Response

from typing import Any

from python.helpers.mlx_server import MLXServerManager
from python.helpers.settings import get_settings


class MlxServerStart(ApiHandler):
    async def process(self, input: dict[Any, Any], request: Request) -> dict[Any, Any] | Response:
        try:
            settings = get_settings()
            manager = MLXServerManager.get_instance()
            result = manager.start_server(
                settings_path=None,  # Use current settings
                port=settings.get("mlx_server_port")
            )
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
