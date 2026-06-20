from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MetricRunRequest:
    metric_name: str
    args: list[str]
    quiet: bool = False
    progress_total: int | None = None
    progress_desc: str | None = None


@dataclass(frozen=True)
class MetricRunResult:
    return_code: int


class MetricRunner(Protocol):
    def run(self, request: MetricRunRequest) -> MetricRunResult:
        ...

