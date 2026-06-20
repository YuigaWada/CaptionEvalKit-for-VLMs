from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    image: str
    caption: str
    references: list[str]
    score: float


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    items: tuple[BenchmarkItem, ...]

    @classmethod
    def from_items(cls, name: str, items: Iterable[BenchmarkItem]) -> BenchmarkDataset:
        return cls(name=name, items=tuple(items))

    def require_non_empty(self) -> None:
        if not self.items:
            raise ValueError(f"{self.name} has no benchmark items")

