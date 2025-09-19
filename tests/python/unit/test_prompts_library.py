from __future__ import annotations

from pathlib import Path

import pytest

from python.runtime.agent import prompts


def test_load_prompt_library_reads_yaml(tmp_path: Path):
    lib = tmp_path / 'library'
    lib.mkdir()
    (lib / 'default.yaml').write_text('greeting: Hello\nFarewell: bye\n')
    library = prompts.load_prompt_library(lib)
    assert library['default']['greeting'] == 'Hello'
    assert library['default']['farewell'] == 'bye'


def test_load_prompt_library_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prompts.load_prompt_library(tmp_path / 'missing')


def test_load_prompt_overrides_accepts_mixed_formats(tmp_path: Path):
    override_root = tmp_path / 'overrides'
    override_root.mkdir()
    (override_root / 'mentor.yaml').write_text('tone: supportive\n')
    subdir = override_root / 'researcher'
    subdir.mkdir()
    (subdir / 'context.yaml').write_text('context: In-depth')
    (subdir / 'notes.md').write_text('Keep citations concise.')

    overrides = prompts.load_prompt_overrides(override_root)
    assert overrides['mentor']['tone'] == 'supportive'
    assert overrides['researcher']['context'] == 'In-depth'
    assert overrides['researcher']['notes'] == 'Keep citations concise.'


def test_persona_modifiers_cover_default_persona():
    assert 'default' in prompts.PERSONA_MODIFIERS
    assert prompts.PERSONA_MODIFIERS['default'].endswith('.')
