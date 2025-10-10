from python.helpers.api import ApiHandler, Request, Response
from python.runtime import get_config, get_memory_store, get_observability, tool_registry
from python.helpers.mlx_server import MLXServerManager


class RuntimeStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        config = get_config()

        registry = tool_registry()
        tools = []
        try:
            for name, entry in registry.items():
                tools.append(
                    {
                        "name": name,
                        "enabled": entry.enabled,
                        "description": entry.description,
                        "tags": list(entry.tags),
                    }
                )
        except Exception:
            tools = []

        memory_stats = {}
        memory_store = get_memory_store()
        stats_fn = getattr(memory_store, "stats", None)
        if stats_fn:
            try:
                memory_stats = await stats_fn()
            except Exception:
                memory_stats = {}

        observability = get_observability()
        observability_info = {
            "helicone_enabled": observability.config.helicone_enabled,
            "metrics_namespace": observability.config.metrics_namespace,
            "json_log_path": str(observability.config.json_log_path)
            if observability.config.json_log_path
            else None,
        }

        mlx_manager = MLXServerManager.get_instance()
        try:
            mlx_state = mlx_manager.get_status()
        except Exception:
            mlx_state = {"status": "unknown"}

        agent_cfg = config.agent

        return {
            "agent": {
                "max_loops": agent_cfg.max_loops,
                "subagent_max_depth": agent_cfg.subagent_max_depth,
                "subagent_timeout": agent_cfg.subagent_timeout,
                "planner_profile": agent_cfg.planner_profile,
                "execution_profile": agent_cfg.execution_profile,
            },
            "memory": memory_stats,
            "observability": observability_info,
            "tools": tools,
            "mlx": mlx_state,
        }
