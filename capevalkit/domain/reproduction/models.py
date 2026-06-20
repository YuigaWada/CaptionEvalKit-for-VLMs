from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReproduceTask:
    metric: str
    benchmark: str
    expected: Path
    output: Path


@dataclass
class ReproduceResult:
    metric: str
    benchmark: str
    status: str
    output: str
    expected: str
    message: str = ""


@dataclass(frozen=True)
class ReproduceJob:
    tasks: tuple[ReproduceTask, ...]
    runner_metric: str
    benchmark: str
    metric_args: tuple[str, ...] = ()
    use_references: bool = True
    resource: str = "cpu"

