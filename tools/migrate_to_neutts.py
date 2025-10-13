#!/usr/bin/env python3
"""
Migration helper to remove legacy TTS engines and adopt NeuTTS-Air.

This script normalises persisted settings, removes legacy sidecar artefacts,
and ensures the NeuTTS cache layout exists.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

NEW_TTS_SETTINGS = {
    "provider": "neutts",
    "sample_rate": 24_000,
    "stream_default": True,
    "neutts": {
        "backbone_repo": "neuphonic/neutts-air-q4-gguf",
        "codec_repo": "neuphonic/neucodec-onnx-decoder",
        "backbone_device": "mps",
        "codec_device": "cpu",
        "model_cache_dir": "data/tts/neutts/models",
        "quality_default": "q4",
        "default_voice_id": None,
        "stream_chunk_seconds": 0.32,
    },
}


def _load_settings(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except json.JSONDecodeError:
        return None


def _save_settings(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def migrate_settings_file(settings_path: Path) -> bool:
    data = _load_settings(settings_path)
    if data is None:
        return False

    updated = False
    payload = dict(data)
    if payload.get("tts") != NEW_TTS_SETTINGS:
        payload["tts"] = NEW_TTS_SETTINGS
        updated = True
    if updated:
        print(f"[migrate] Updating {settings_path}")
        _save_settings(settings_path, payload)
    return updated


def remove_legacy_tree(root: Path) -> None:
    targets = [
        root / "sidecar",
        root / "run.sh",
        root / "setup.sh",
        root / "logs" / "tts_sidecar.err",
        root / "logs" / "tts_sidecar.out",
    ]
    for target in targets:
        if target.is_dir():
            print(f"[migrate] Removing legacy directory {target}")
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            print(f"[migrate] Removing legacy file {target}")
            target.unlink(missing_ok=True)


def ensure_neutts_dirs(root: Path) -> None:
    base = root / "data" / "tts" / "neutts"
    for sub in ("models", "voices"):
        path = base / sub
        if not path.exists():
            print(f"[migrate] Creating {path}")
            path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    settings_path = repo_root / "tmp" / "settings.json"
    migrate_settings_file(settings_path)
    remove_legacy_tree(repo_root)
    ensure_neutts_dirs(repo_root)
    print("[migrate] NeuTTS-Air migration complete.")


if __name__ == "__main__":
    main()
