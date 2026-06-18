from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

