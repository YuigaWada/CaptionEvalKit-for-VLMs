from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from .paths import metrics_root


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    args: tuple[str, ...]
    expected: str | None


@dataclass(frozen=True)
class MetricManifest:
    name: str
    python: str
    module: str
    benchmark_module: str | None
    repo_dir: str
    repo_url: str | None
    uv_project: str
    runner: tuple[str, ...]
    benchmarks: dict[str, BenchmarkSpec]
    smoke_command: tuple[str, ...]
    merge_policy: str
    path: Path


def _as_tuple(value: object, key: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {key} must be a list of strings")
    return tuple(value)


def load_manifest(path: Path) -> MetricManifest | None:
    data = tomllib.loads(path.read_text())
    metric = data.get("metric")
    if not isinstance(metric, dict):
        raise ValueError(f"{path}: missing [metric] table")

    checks = data.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError(f"{path}: [checks] must be a table")

    compatibility = data.get("compatibility", {})
    if not isinstance(compatibility, dict):
        raise ValueError(f"{path}: [compatibility] must be a table")

    repository = data.get("repository", {})
    if not isinstance(repository, dict):
        raise ValueError(f"{path}: [repository] must be a table")

    runner = data.get("runner", {})
    if not isinstance(runner, dict):
        raise ValueError(f"{path}: [runner] must be a table")

    benchmarks_table = data.get("benchmarks", {})
    if not isinstance(benchmarks_table, dict):
        raise ValueError(f"{path}: [benchmarks] must be a table")

    required = ("name", "python", "module")
    missing = [key for key in required if not isinstance(metric.get(key), str)]
    if missing:
        raise ValueError(f"{path}: missing string keys: {', '.join(missing)}")
    if metric.get("enabled") is False:
        return None
    if not isinstance(repository.get("dir"), str):
        raise ValueError(f"{path}: missing repository.dir")

    smoke = checks.get("smoke", [])
    benchmark_specs: dict[str, BenchmarkSpec] = {}
    for name, value in benchmarks_table.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: benchmarks.{name} must be a table")
        benchmark_specs[name] = BenchmarkSpec(
            name=name,
            args=_as_tuple(value.get("args", []), f"benchmarks.{name}.args", path),
            expected=value.get("expected") if isinstance(value.get("expected"), str) else None,
        )

    return MetricManifest(
        name=metric["name"],
        python=metric["python"],
        module=metric["module"],
        benchmark_module=metric.get("benchmark_module"),
        repo_dir=repository["dir"],
        repo_url=repository.get("url") if isinstance(repository.get("url"), str) else None,
        uv_project=repository.get("uv_project", repository["dir"]),
        runner=_as_tuple(runner.get("command", []), "runner.command", path),
        benchmarks=benchmark_specs,
        smoke_command=_as_tuple(smoke, "checks.smoke", path),
        merge_policy=str(compatibility.get("merge_policy", "single-metric-until-proven-compatible")),
        path=path,
    )


def load_manifests(root: Path | None = None) -> dict[str, MetricManifest]:
    if root is None:
        from .runtime import RuntimeManager

        RuntimeManager().prepare_base()
        root = metrics_root()
    manifests: dict[str, MetricManifest] = {}
    for path in sorted(root.glob("*/metric.toml")):
        manifest = load_manifest(path)
        if manifest is None:
            continue
        if manifest.name in manifests:
            raise ValueError(f"duplicate metric manifest: {manifest.name}")
        manifests[manifest.name] = manifest
    return manifests


def get_manifest(metric_name: str) -> MetricManifest:
    manifests = load_manifests()
    try:
        return manifests[metric_name]
    except KeyError as exc:
        known = ", ".join(sorted(manifests)) or "(none)"
        raise KeyError(f"unknown metric: {metric_name}; known metrics: {known}") from exc
