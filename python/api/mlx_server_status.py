from python.helpers.api import ApiHandler, Request, Response

from typing import Any

from python.helpers.mlx_server import MLXServerManager


class MlxServerStatus(ApiHandler):
    async def process(self, input: dict[Any, Any], request: Request) -> dict[Any, Any] | Response:
        try:
            manager = MLXServerManager.get_instance()
            status = manager.get_status()
            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "error": str(e)}
