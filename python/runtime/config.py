"""Runtime configuration loading utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from .agent.prompts import DEFAULT_LIBRARY_PATH
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, List

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    dotenv_path = os.getenv("RUNTIME_DOTENV")
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

DEFAULT_TOKEN_BUDGETS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4.1-mini": 64_000,
    "claude-3-5": 200_000,
    "gemini-2.5-pro": 1_000_000,
    "local-mlx": 8_192,
}


@dataclass(slots=True)
class EmbeddingsConfig:
    provider: str = "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_path: Path = Path("./tmp/embeddings.sqlite")


@dataclass(slots=True)
class MemoryConfig:
    db_path: Path = Path("./tmp/memory.sqlite")
    top_k: int = 6
    summarize_overflow: bool = True


@dataclass(slots=True)
class TokenConfig:
    budgets: Dict[str, int] = field(default_factory=lambda: DEFAULT_TOKEN_BUDGETS.copy())
    summarizer_model: Optional[str] = None


@dataclass(slots=True)
class ToolsConfig:
    enabled: Sequence[str] = field(default_factory=tuple)
    disabled: Sequence[str] = field(default_factory=tuple)
    plugins: Sequence[str] = field(default_factory=tuple)
    auto_modules: Sequence[str] = field(
        default_factory=lambda: (
            "python.runtime.tools.search",
            "python.runtime.tools.browser",
            "python.runtime.tools.code",
            "python.runtime.tools.image",
            "python.runtime.tools.shell",
            "python.runtime.tools.files",
            "python.runtime.tools.data",
            "python.runtime.tools.prompt_tool",
        )
    )
    workspace: Path = Path(".")
    shell_allow_list: Sequence[str] = field(default_factory=lambda: ("ls", "pwd", "cat"))


@dataclass(slots=True)
class PromptsConfig:
    library_path: Path = Path(DEFAULT_LIBRARY_PATH)
    overrides_path: Optional[Path] = None
    extra_safety: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class ObservabilityConfig:
    helicone_enabled: bool = False
    helicone_base_url: Optional[str] = None
    helicone_api_key: Optional[str] = None
    json_log_path: Optional[Path] = Path("./logs/runtime_observability.jsonl")
    metrics_namespace: str = "apple_zero"


@dataclass(slots=True)
class AgentConfig:
    max_loops: int = 20
    subagent_max_depth: int = 3
    subagent_timeout: float = 120.0
    persona: str = "default"
    planner_profile: str = "planner"
    execution_profile: str = "general"


@dataclass(slots=True)
class RouterModelConfig:
    name: str
    provider: str = "openai"
    priority: int = 10
    max_context: int = 0
    cost_per_1k: float = 0.0
    latency: str = "medium"
    capabilities: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    is_local: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RouterConfig:
    default_model: str = "gpt-4o"
    default_strategy: str = "balanced"
    strategy_overrides: Dict[str, str] = field(default_factory=dict)
    models: Sequence[RouterModelConfig] = field(default_factory=tuple)



@dataclass(slots=True)
class RuntimeConfig:
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    router: RouterConfig = field(default_factory=RouterConfig)


def _load_config_file() -> Dict[str, Any]:
    env_path = os.getenv("RUNTIME_CONFIG") or os.getenv("RUNTIME_CONFIG_FILE")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        Path(p)
        for p in (
            "runtime.toml",
            "config/runtime.toml",
            "conf/runtime.toml",
            "python/runtime.toml",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            if tomllib is None:
                raise RuntimeError(
                    "Runtime configuration file specified but tomllib is unavailable."
                )
            with candidate.expanduser().open("rb") as fh:
                return tomllib.load(fh)
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lookup(settings: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = settings
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _normalize_model_key(model: str, existing: MutableMapping[str, int]) -> str:
    candidate = model.replace("__", "/").replace("_", "-")
    for key in existing:
        if key.lower() == candidate.lower():
            return key
    return candidate


def _merge_token_budgets(
    base: MutableMapping[str, int],
    config_section: Any,
    environ: Mapping[str, str],
) -> Dict[str, int]:
    budgets = dict(base)
    if isinstance(config_section, Mapping):
        for model, value in config_section.items():
            key = _normalize_model_key(str(model), budgets)
            parsed = _as_int(value, budgets.get(key, 0))
            if parsed:
                budgets[key] = parsed
    env_json = environ.get("TOKEN_BUDGETS")
    if env_json:
        try:
            overrides = json.loads(env_json)
        except json.JSONDecodeError:
            overrides = {}
        if isinstance(overrides, Mapping):
            for model, value in overrides.items():
                key = _normalize_model_key(str(model), budgets)
                parsed = _as_int(value, budgets.get(key, 0))
                if parsed:
                    budgets[key] = parsed
    prefix = "TOKEN_BUDGET_"
    for env_key, env_value in environ.items():
        if not env_key.startswith(prefix):
            continue
        model_key = env_key[len(prefix) :]
        key = _normalize_model_key(model_key, budgets)
        parsed = _as_int(env_value, budgets.get(key, 0))
        if parsed:
            budgets[key] = parsed
    return budgets


def _parse_list(value: Any, default: Sequence[str]) -> Sequence[str]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(
        item.strip()
        for item in str(value).split(",")
        if item and item.strip()
    )




def _parse_router_models(entries: Any) -> Sequence[RouterModelConfig]:
    models: List[RouterModelConfig] = []
    if not isinstance(entries, Sequence):
        return tuple(models)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        metadata = entry.get("metadata")
        models.append(
            RouterModelConfig(
                name=name,
                provider=str(entry.get("provider", "openai") or "openai").strip() or "openai",
                priority=_as_int(entry.get("priority"), 10),
                max_context=_as_int(entry.get("max_context"), 0),
                cost_per_1k=_as_float(
                    entry.get("cost_per_1k", entry.get("cost", 0.0)),
                    0.0,
                ),
                latency=str(entry.get("latency", "medium") or "medium").strip().lower(),
                capabilities=_parse_list(entry.get("capabilities"), ()),
                tags=_parse_list(entry.get("tags"), ()),
                is_local=_as_bool(entry.get("is_local"), False),
                metadata={
                    str(k): str(v)
                    for k, v in metadata.items()
                }
                if isinstance(metadata, Mapping)
                else {},
            )
        )
    return tuple(models)

@lru_cache(maxsize=1)
def load_runtime_config() -> RuntimeConfig:
    settings = _load_config_file()
    env = os.environ

    embeddings = EmbeddingsConfig(
        provider=str(
            env.get(
                "EMBEDDINGS_PROVIDER",
                _lookup(settings, "embeddings", "provider", default="huggingface"),
            )
        ).strip(),
        model=str(
            env.get(
                "EMBEDDINGS_MODEL",
                _lookup(
                    settings,
                    "embeddings",
                    "model",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                ),
            )
        ).strip(),
        batch_size=_as_int(
            env.get(
                "EMBEDDINGS_BATCH",
                _lookup(settings, "embeddings", "batch", default=32),
            ),
            32,
        ),
        cache_path=Path(
            env.get(
                "EMBEDDINGS_CACHE_PATH",
                _lookup(settings, "embeddings", "cache_path", default="./tmp/embeddings.sqlite"),
            )
        ).expanduser(),
    )

    memory = MemoryConfig(
        db_path=Path(
            env.get(
                "MEMORY_DB_PATH",
                _lookup(settings, "memory", "db_path", default="./tmp/memory.sqlite"),
            )
        ).expanduser(),
        top_k=_as_int(
            env.get(
                "MEMORY_K",
                _lookup(settings, "memory", "top_k", default=6),
            ),
            6,
        ),
        summarize_overflow=_as_bool(
            env.get(
                "MEMORY_SUMMARIZE_OVERFLOW",
                _lookup(settings, "memory", "summarize_overflow", default=True),
            ),
            True,
        ),
    )

    token_budgets = _merge_token_budgets(
        DEFAULT_TOKEN_BUDGETS,
        _lookup(settings, "token_budgets", default={}),
        env,
    )
    tokens = TokenConfig(
        budgets=token_budgets,
        summarizer_model=str(
            env.get(
                "SUMMARIZER_MODEL",
                _lookup(settings, "tokens", "summarizer_model", default="gpt-4.1-mini"),
            )
        ).strip()
        or None,
    )

    prompts = PromptsConfig(
        library_path=Path(
            env.get(
                "PROMPTS_LIBRARY_PATH",
                _lookup(settings, "prompts", "library_path", default=str(DEFAULT_LIBRARY_PATH)),
            )
        ).expanduser(),
        overrides_path=(
            Path(
                env.get(
                    "PROMPTS_OVERRIDES_PATH",
                    _lookup(settings, "prompts", "overrides_path", default=""),
                )
            ).expanduser()
            if env.get("PROMPTS_OVERRIDES_PATH")
            or _lookup(settings, "prompts", "overrides_path")
            else None
        ),
        extra_safety=_parse_list(
            env.get(
                "PROMPTS_EXTRA_SAFETY",
                _lookup(settings, "prompts", "extra_safety", default=()),
            ),
            (),
        ),
    )

    agent = AgentConfig(
        max_loops=_as_int(
            env.get("AGENT_MAX_LOOPS", _lookup(settings, "agent", "max_loops", default=20)),
            20,
        ),
        subagent_max_depth=_as_int(
            env.get(
                "AGENT_SUBAGENT_MAX_DEPTH",
                _lookup(settings, "agent", "subagent_max_depth", default=3),
            ),
            3,
        ),
        subagent_timeout=_as_float(
            env.get(
                "AGENT_SUBAGENT_TIMEOUT",
                _lookup(settings, "agent", "subagent_timeout", default=120.0),
            ),
            120.0,
        ),
        persona=str(
            env.get(
                "AGENT_PERSONA",
                _lookup(settings, "agent", "persona", default="default"),
            )
        ).strip()
        or "default",
        planner_profile=str(
            env.get(
                "AGENT_PLANNER_PROFILE",
                _lookup(settings, "agent", "planner_profile", default="planner"),
            )
        ).strip()
        or "planner",
        execution_profile=str(
            env.get(
                "AGENT_EXECUTION_PROFILE",
                _lookup(settings, "agent", "execution_profile", default="general"),
            )
        ).strip()
        or "general",
    )

    tools = ToolsConfig(
        enabled=_parse_list(env.get("TOOLS_ENABLED"), _lookup(settings, "tools", "enabled", default=())),
        disabled=_parse_list(env.get("TOOLS_DISABLED"), _lookup(settings, "tools", "disabled", default=())),
        plugins=_parse_list(env.get("TOOLS_PLUGINS"), _lookup(settings, "tools", "plugins", default=())),
        auto_modules=_parse_list(env.get("TOOLS_AUTO_MODULES"), _lookup(settings, "tools", "auto_modules", default=(
            "python.runtime.tools.search",
            "python.runtime.tools.browser",
            "python.runtime.tools.code",
            "python.runtime.tools.image",
            "python.runtime.tools.shell",
            "python.runtime.tools.files",
            "python.runtime.tools.data",
            "python.runtime.tools.prompt_tool",
        ))),
        workspace=Path(
            env.get(
                "TOOLS_WORKSPACE",
                _lookup(settings, "tools", "workspace", default="."),
            )
        ).expanduser(),
        shell_allow_list=_parse_list(
            env.get("TOOLS_SHELL_ALLOW", _lookup(settings, "tools", "shell_allow_list", default=("ls", "pwd", "cat"))),
            ("ls", "pwd", "cat"),
        ),
    )

    json_log_setting = env.get(
        "OBSERVABILITY_LOG_PATH",
        _lookup(settings, "observability", "json_log_path", default="./logs/runtime_observability.jsonl"),
    )
    json_log_path: Optional[Path]
    if json_log_setting is None:
        json_log_path = None
    else:
        if isinstance(json_log_setting, Path):
            candidate = str(json_log_setting)
        else:
            candidate = str(json_log_setting).strip()
        if candidate.lower() in {"", "none", "null", "disable", "disabled"}:
            json_log_path = None
        else:
            json_log_path = Path(candidate).expanduser()

    observability = ObservabilityConfig(
        helicone_enabled=_as_bool(
            env.get(
                "HELICONE_ENABLED",
                _lookup(settings, "observability", "helicone_enabled", default=False),
            ),
            False,
        ),
        helicone_base_url=(
            str(
                env.get(
                    "HELICONE_BASE_URL",
                    _lookup(settings, "observability", "helicone_base_url", default=""),
                )
            ).strip()
            or None
        ),
        helicone_api_key=(
            str(
                env.get(
                    "HELICONE_API_KEY",
                    _lookup(settings, "observability", "helicone_api_key", default=""),
                )
            ).strip()
            or None
        ),
        json_log_path=json_log_path,
        metrics_namespace=str(
            env.get(
                "OBSERVABILITY_METRICS_NS",
                _lookup(settings, "observability", "metrics_namespace", default="apple_zero"),
            )
        ).strip()
        or "apple_zero",
    )

    raw_router_models = _lookup(settings, "router", "models", default=())
    env_router_models = env.get("ROUTER_MODELS")
    if env_router_models:
        try:
            parsed_models = json.loads(env_router_models)
        except json.JSONDecodeError:
            parsed_models = ()
        if parsed_models:
            raw_router_models = parsed_models
    router_models = _parse_router_models(raw_router_models)

    strategy_overrides: Dict[str, str] = {}
    raw_strategy_overrides = _lookup(settings, "router", "strategy_overrides", default={})
    if isinstance(raw_strategy_overrides, Mapping):
        strategy_overrides.update({str(k).lower(): str(v).lower() for k, v in raw_strategy_overrides.items()})
    env_strategy_overrides = env.get("ROUTER_STRATEGY_OVERRIDES")
    if env_strategy_overrides:
        try:
            parsed_overrides = json.loads(env_strategy_overrides)
        except json.JSONDecodeError:
            parsed_overrides = None
        if isinstance(parsed_overrides, Mapping):
            strategy_overrides.update({str(k).lower(): str(v).lower() for k, v in parsed_overrides.items()})

    router = RouterConfig(
        default_model=str(
            env.get(
                "ROUTER_DEFAULT_MODEL",
                _lookup(settings, "router", "default_model", default="gpt-4o"),
            )
        ).strip()
        or "gpt-4o",
        default_strategy=str(
            env.get(
                "ROUTER_DEFAULT_STRATEGY",
                _lookup(settings, "router", "default_strategy", default="balanced"),
            )
        ).strip()
        or "balanced",
        strategy_overrides=strategy_overrides,
        models=router_models,
    )

    return RuntimeConfig(
        embeddings=embeddings,
        memory=memory,
        tokens=tokens,
        prompts=prompts,
        observability=observability,
        agent=agent,
        tools=tools,
        router=router,
    )


__all__ = [
    "EmbeddingsConfig",
    "MemoryConfig",
    "RuntimeConfig",
    "TokenConfig",
    "PromptsConfig",
    "ToolsConfig",
    "AgentConfig",
    "RouterModelConfig",
    "RouterConfig",
    "DEFAULT_TOKEN_BUDGETS",
    "load_runtime_config",
]
