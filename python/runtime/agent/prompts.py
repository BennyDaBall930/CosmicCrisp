"""Prompt library helpers for Apple Zero.

The default prompt sets are stored as YAML files under ``python/runtime/prompts/library``.
They are derived from community prompt collections (e.g. x1xhlol/system-prompts) and
adapted for Apple Zero's macOS-focused workflow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
    raise RuntimeError("PyYAML is required to load prompt templates") from exc

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_LIBRARY_PATH = DEFAULT_ROOT / "library"

PERSONA_MODIFIERS: Dict[str, str] = {
    "default": "Maintain a balanced, professional tone.",
    "concise": "Respond succinctly. Prefer bullet lists and short sentences.",
    "detailed": "Elaborate thoroughly with context, examples, and caveats.",
    "mentor": "Adopt a coaching voice that explains rationale and best practices.",
}

SAFETY_GUARDRAILS = (
    "Operate within legal, ethical, and corporate guidelines. Avoid destructive actions. "
    "Escalate to a human when approval or credentials are required."
)


def _load_yaml_file(path: Path) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a mapping")
    return {str(key).lower(): str(value).strip() for key, value in data.items()}


def load_prompt_library(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    root = path or DEFAULT_LIBRARY_PATH
    library: Dict[str, Dict[str, str]] = {}
    if not root.exists():
        raise FileNotFoundError(f"Prompt library directory not found: {root}")
    for file in root.glob("*.yaml"):
        profile = file.stem.lower()
        library[profile] = _load_yaml_file(file)
    return library


def load_prompt_overrides(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    overrides: Dict[str, Dict[str, str]] = {}
    if path is None:
        return overrides
    root = Path(path)
    if not root.exists():
        return overrides
    for profile_dir in root.iterdir():
        if profile_dir.is_file() and profile_dir.suffix in {".yaml", ".yml"}:
            overrides[profile_dir.stem.lower()] = _load_yaml_file(profile_dir)
            continue
        if not profile_dir.is_dir():
            continue
        profile_key = profile_dir.name.lower()
        overrides.setdefault(profile_key, {})
        for override_file in profile_dir.glob("*.yaml"):
            overrides[profile_key].update(_load_yaml_file(override_file))
        for override_file in profile_dir.glob("*.md"):
            overrides[profile_key][override_file.stem.lower()] = override_file.read_text().strip()
    return overrides


__all__ = [
    "DEFAULT_LIBRARY_PATH",
    "PERSONA_MODIFIERS",
    "SAFETY_GUARDRAILS",
    "load_prompt_library",
    "load_prompt_overrides",
]
