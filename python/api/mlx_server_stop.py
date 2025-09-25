from python.helpers.api import ApiHandler, Request, Response

from typing import Any

from python.helpers.mlx_server import MLXServerManager


class MlxServerStop(ApiHandler):
    async def process(self, input: dict[Any, Any], request: Request) -> dict[Any, Any] | Response:
        try:
            manager = MLXServerManager.get_instance()
            result = manager.stop_server()
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
