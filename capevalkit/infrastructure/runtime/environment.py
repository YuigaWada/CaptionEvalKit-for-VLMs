from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path


def apply_runtime_environment(
    env: MutableMapping[str, str],
    project_root: Path,
    *,
    cache_root: Path | None = None,
) -> None:
    cache = cache_root or project_root / ".cache"
    env.setdefault("UV_LINK_MODE", "hardlink")
    env.setdefault("CAPEVALKIT_HOME", str(cache))
    env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
    env.setdefault("CLIP_DOWNLOAD_ROOT", str(cache / "clip"))
    env.setdefault("TORCH_HOME", str(cache / "torch"))
    env.setdefault("HF_HOME", str(cache / "huggingface"))
    env.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
