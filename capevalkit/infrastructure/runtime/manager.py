from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any

from capevalkit.domain.metrics import MetricManifest
from capevalkit.infrastructure.runtime.context import ProjectContext, default_context
from capevalkit.infrastructure.runtime.overlays import ensure_overlays

_SOURCE_SUBMODULE_LOCK = threading.Lock()


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    source: str
    url: str
    rev: str
    path: str
    uv_project: str
    overlays: tuple[str, ...]


class RuntimeManager:
    def __init__(self, context: ProjectContext | None = None) -> None:
        self.context = context or default_context()
        self._lock = self._load_lock()

    def prepare_base(self) -> Path:
        if self.context.source_mode:
            return self.context.project_root

        self.context.project_root.mkdir(parents=True, exist_ok=True)
        self._copy_metric_manifests()
        self._copy_tree("overlays")
        self._copy_expected()
        self._write_state()
        return self.context.project_root

    def ensure_metric(self, manifest: MetricManifest) -> Path:
        upstream = self.upstream_for_path(manifest.repo_dir)
        if upstream is None:
            return self.prepare_base() / manifest.repo_dir
        return self.ensure_upstream(upstream.name)

    def ensure_upstreams(self, upstream_names: list[str] | tuple[str, ...] | set[str]) -> list[Path]:
        return [self.ensure_upstream(name) for name in sorted(set(upstream_names))]

    def upstream_names(self) -> list[str]:
        upstreams = self._lock.get("upstreams", {})
        if not isinstance(upstreams, dict):
            return []
        return sorted(str(name) for name in upstreams)

    def ensure_upstream(self, name: str) -> Path:
        spec = self.upstream(name)
        root = self.prepare_base()
        path = root / spec.path

        if self.context.source_mode:
            if self._source_checkout_is_ready(root, path):
                ensure_overlays(root=root, overlays=spec.overlays)
                return path
            with _SOURCE_SUBMODULE_LOCK:
                if self._source_checkout_is_ready(root, path):
                    ensure_overlays(root=root, overlays=spec.overlays)
                    return path
                self._remove_overlay_only_stub(root, path, spec)
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", spec.path],
                    cwd=root,
                    check=True,
                )
                if self._source_checkout_is_ready(root, path):
                    ensure_overlays(root=root, overlays=spec.overlays)
                    return path
            hint = f"run: git submodule update --init --recursive {spec.path}"
            raise FileNotFoundError(f"incomplete metric repository: {path}; {hint}")

        if self._git_checkout_is_ready(path, spec.rev):
            ensure_overlays(root=root, overlays=spec.overlays)
            return path

        path_is_empty = path.exists() and not any(path.iterdir())
        if path.exists() and not path_is_empty and not (path / ".git").exists():
            self._remove_overlay_only_stub(root, path, spec)
            path_is_empty = path.exists() and not any(path.iterdir())
            if path.exists() and not path_is_empty and not (path / ".git").exists():
                raise RuntimeError(f"runtime upstream path exists but is not a git checkout: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path_is_empty:
            subprocess.run(["git", "clone", spec.url, str(path)], check=True)
        subprocess.run(["git", "fetch", "--tags", "origin"], cwd=path, check=True)
        subprocess.run(["git", "checkout", spec.rev], cwd=path, check=True)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=path, check=True)
        ensure_overlays(root=root, overlays=spec.overlays)
        return path

    def upstream(self, name: str) -> UpstreamSpec:
        upstreams = self._lock.get("upstreams", {})
        if not isinstance(upstreams, dict) or name not in upstreams:
            known = ", ".join(sorted(upstreams)) if isinstance(upstreams, dict) else "(none)"
            raise KeyError(f"unknown upstream {name!r}; known: {known}")
        data = upstreams[name]
        if not isinstance(data, dict):
            raise ValueError(f"invalid upstream lock entry: {name}")
        return UpstreamSpec(
            name=name,
            source=str(data.get("source", "git")),
            url=str(data["url"]),
            rev=str(data["rev"]),
            path=str(data["path"]),
            uv_project=str(data.get("uv_project", data["path"])),
            overlays=tuple(str(item) for item in data.get("overlays", [])),
        )

    def upstream_for_path(self, repo_dir: str) -> UpstreamSpec | None:
        upstreams = self._lock.get("upstreams", {})
        if not isinstance(upstreams, dict):
            return None
        normalized = repo_dir.rstrip("/")
        for name, data in upstreams.items():
            if isinstance(data, dict) and str(data.get("path", "")).rstrip("/") == normalized:
                return self.upstream(str(name))
        return None

    def _load_lock(self) -> dict[str, Any]:
        if not self.context.lock_path.exists():
            return {"schema_version": 1, "upstreams": {}}
        return json.loads(self.context.lock_path.read_text())

    def _copy_metric_manifests(self) -> None:
        source = self.context.resource_root / "metrics"
        target = self.context.project_root / "metrics"
        if not source.exists():
            return
        for manifest in source.glob("*/metric.toml"):
            relative = manifest.relative_to(source)
            destination = target / relative
            self._copy_file(manifest, destination)

    def _copy_expected(self) -> None:
        source = self.context.resource_root / "benchmarks" / "expected"
        target = self.context.project_root / "benchmarks" / "expected"
        if source.exists():
            self._copy_tree_from_to(source, target)

    def _copy_tree(self, relative: str) -> None:
        source = self.context.resource_root / relative
        target = self.context.project_root / relative
        if source.exists():
            self._copy_tree_from_to(source, target)

    def _copy_tree_from_to(self, source: Path, target: Path) -> None:
        for item in source.rglob("*"):
            if item.is_dir():
                continue
            relative = item.relative_to(source)
            self._copy_file(item, target / relative)

    def _copy_file(self, source: Path, target: Path) -> None:
        if target.exists() and target.read_bytes() == source.read_bytes():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _write_state(self) -> None:
        state = {
            "lock_digest": self.context.lock_digest,
            "lock_path": str(self.context.lock_path),
        }
        path = self.context.project_root / "runtime-state.json"
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    def _git_checkout_is_ready(self, path: Path, rev: str) -> bool:
        if not (path / ".git").exists():
            return False
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0 and result.stdout.strip() == rev

    def _source_checkout_is_ready(self, root: Path, path: Path) -> bool:
        if not path.exists() or not any(path.iterdir()):
            return False
        if (path / ".git").exists():
            nested_ready = self._source_submodule_tree_is_ready(root, path)
            return True if nested_ready is None else nested_ready
        nested_ready = self._source_submodule_tree_is_ready(root, path)
        return False if nested_ready is None else nested_ready

    def _source_submodule_tree_is_ready(self, root: Path, path: Path) -> bool | None:
        result = subprocess.run(
            ["git", "submodule", "status", "--recursive", "--", str(path.relative_to(root))],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or not lines:
            return None
        return all(not line.lstrip().startswith("-") for line in lines)

    def _remove_overlay_only_stub(self, root: Path, path: Path, spec: UpstreamSpec) -> None:
        if not path.exists() or not any(path.iterdir()):
            return
        allowed = {
            root / Path(overlay).relative_to("overlays")
            for overlay in spec.overlays
            if Path(overlay).parts[:1] == ("overlays",)
        }
        files = {item for item in path.rglob("*") if item.is_file()}
        if files and files.issubset(allowed):
            shutil.rmtree(path)
