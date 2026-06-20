from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from capevalkit.domain.metrics import MetricManifest


class MetricManifestRepository(Protocol):
    def list(self) -> Mapping[str, MetricManifest]:
        ...

    def get(self, metric_id: str) -> MetricManifest:
        ...

