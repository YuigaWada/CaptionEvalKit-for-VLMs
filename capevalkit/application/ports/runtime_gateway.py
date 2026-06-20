from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class RuntimeInfo:
    source_mode: bool
    project_root: Path
    cache_root: Path
    lock_path: Path
    lock_digest: str


class RuntimeGateway(Protocol):
    def context(self) -> RuntimeInfo:
        ...

    def prepare_base(self) -> Path:
        ...

    def upstream_names(self) -> list[str]:
        ...

    def upstream_for_path(self, path: str) -> str | None:
        ...

    def ensure_upstreams(self, names: Iterable[str]) -> list[Path]:
        ...

