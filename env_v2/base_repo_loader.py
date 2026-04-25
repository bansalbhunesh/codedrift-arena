"""Materialize the base repo into a fresh temp directory per episode.

The on-disk template lives at ``env_v2/base_repo``. ``copy_base_repo`` clones
it into ``dest`` so generator mutations stay isolated to one episode.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_TEMPLATE = Path(__file__).resolve().parent / "base_repo"


def template_root() -> Path:
    return _TEMPLATE


def copy_base_repo(dest: Path) -> Path:
    """Copy the template repo into ``dest`` and return its path."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_TEMPLATE, dest)
    return dest


def list_source_files(repo_root: Path) -> list[Path]:
    """All .py files under src/ that the generator may mutate."""
    return sorted((repo_root / "src").rglob("*.py"))


def list_test_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "tests").rglob("test_*.py"))
