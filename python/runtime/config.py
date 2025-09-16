"""Runtime configuration loading utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

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
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    batch_size: int = 32
    cache_path: Path = Path("./tmp/embeddings.sqlite")


@dataclass(slots=True)
class MemoryConfig:
    db_path: Path = Path("./tmp/memory.sqlite")
    top_k: int = 6
    mem0_enabled: bool = False
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
class AgentConfig:
    max_loops: int = 20
    subagent_max_depth: int = 3
    subagent_timeout: float = 120.0
    persona: str = "default"
    planner_profile: str = "planner"
    execution_profile: str = "general"


@dataclass(slots=True)
class RuntimeConfig:
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)


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


@lru_cache(maxsize=1)
def load_runtime_config() -> RuntimeConfig:
    settings = _load_config_file()
    env = os.environ

    embeddings = EmbeddingsConfig(
        provider=str(
            env.get(
                "EMBEDDINGS_PROVIDER",
                _lookup(settings, "embeddings", "provider", default="openai"),
            )
        ).strip(),
        model=str(
            env.get(
                "EMBEDDINGS_MODEL",
                _lookup(settings, "embeddings", "model", default="text-embedding-3-large"),
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
        mem0_enabled=_as_bool(
            env.get(
                "MEM0_ENABLED",
                _lookup(settings, "memory", "mem0_enabled", default=False),
            ),
            False,
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

    return RuntimeConfig(
        embeddings=embeddings,
        memory=memory,
        tokens=tokens,
        agent=agent,
        tools=tools,
    )


__all__ = [
    "EmbeddingsConfig",
    "MemoryConfig",
    "RuntimeConfig",
    "TokenConfig",
    "ToolsConfig",
    "AgentConfig",
    "DEFAULT_TOKEN_BUDGETS",
    "load_runtime_config",
]
