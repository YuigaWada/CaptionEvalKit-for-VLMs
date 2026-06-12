from __future__ import annotations

from pathlib import Path

from .context import default_context


def repo_root() -> Path:
    return default_context().project_root


def metrics_root() -> Path:
    return default_context().metrics_root


def envs_root() -> Path:
    return repo_root() / "envs"


def cache_root() -> Path:
    return default_context().cache_root
