from __future__ import annotations

from pathlib import Path

from .domain.metrics import BenchmarkSpec, MetricManifest
from .infrastructure.manifests import TomlMetricManifestRepository
from .infrastructure.manifests import load_manifest as load_manifest
from .paths import metrics_root


def _default_repository(root: Path | None = None) -> TomlMetricManifestRepository:
    if root is not None:
        return TomlMetricManifestRepository(root)

    from .runtime import RuntimeManager

    manager = RuntimeManager()
    return TomlMetricManifestRepository(
        prepare_runtime=manager.prepare_base,
        root_provider=metrics_root,
    )


def load_manifests(root: Path | None = None) -> dict[str, MetricManifest]:
    return dict(_default_repository(root).list())


def get_manifest(metric_name: str) -> MetricManifest:
    return _default_repository().get(metric_name)


__all__ = [
    "BenchmarkSpec",
    "MetricManifest",
    "get_manifest",
    "load_manifest",
    "load_manifests",
]

