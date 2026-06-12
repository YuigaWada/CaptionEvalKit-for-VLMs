from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectContext:
    package_root: Path
    resource_root: Path
    project_root: Path
    cache_root: Path
    source_root: Path | None
    source_mode: bool
    lock_path: Path
    lock_digest: str

    @property
    def metrics_root(self) -> Path:
        return self.project_root / "metrics"

    @property
    def upstreams_root(self) -> Path:
        return self.project_root / "metrics" / "upstreams"

    @property
    def overlays_root(self) -> Path:
        return self.project_root / "overlays"

    @property
    def expected_root(self) -> Path:
        return self.project_root / "benchmarks" / "expected"


def default_context() -> ProjectContext:
    package_root = PACKAGE_ROOT
    source = source_root()
    forced_runtime = _runtime_forced()
    source_mode = source is not None and not forced_runtime
    resource_root = source if source is not None else package_root / "resources"
    lock_path = _lock_path(package_root)
    lock_digest = _lock_digest(lock_path)
    cache_root = capevalkit_home()

    if source_mode:
        project_root = source
    else:
        runtime_root = os.environ.get("CAPEVALKIT_RUNTIME_ROOT")
        project_root = (
            Path(runtime_root).expanduser()
            if runtime_root
            else cache_root / "runtime" / lock_digest[:12]
        )

    return ProjectContext(
        package_root=package_root,
        resource_root=resource_root,
        project_root=project_root,
        cache_root=cache_root,
        source_root=source,
        source_mode=source_mode,
        lock_path=lock_path,
        lock_digest=lock_digest,
    )


def source_root() -> Path | None:
    candidate = PACKAGE_ROOT.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "metrics").is_dir():
        return candidate
    return None


def capevalkit_home() -> Path:
    value = os.environ.get("CAPEVALKIT_HOME")
    if value:
        return Path(value).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "capevalkit"


def _runtime_forced() -> bool:
    mode = os.environ.get("CAPEVALKIT_RUNTIME_MODE", "").strip().lower()
    if mode in {"cache", "runtime", "installed"}:
        return True
    if os.environ.get("CAPEVALKIT_RUNTIME_ROOT"):
        return True
    force = os.environ.get("CAPEVALKIT_FORCE_RUNTIME", "").strip().lower()
    return force in {"1", "true", "yes", "on"}


def _lock_path(package_root: Path) -> Path:
    path = package_root / "resources" / "upstreams.lock.json"
    if path.exists():
        return path
    return path


def _lock_digest(path: Path) -> str:
    if path.exists():
        payload = path.read_bytes()
    else:
        payload = b"{}"
    return hashlib.sha256(payload).hexdigest()
