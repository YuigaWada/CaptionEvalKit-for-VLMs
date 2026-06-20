from __future__ import annotations

import os
from typing import Iterable, TypeVar


T = TypeVar("T")


def progress_update(count: int = 1) -> None:
    path = os.environ.get("CAPEVALKIT_PROGRESS_FILE")
    if not path or count <= 0:
        return
    try:
        with open(path, "a") as file:
            file.write(f"{count}\n")
    except OSError:
        return


def progress_iter(iterable: Iterable[T]) -> Iterable[T]:
    for item in iterable:
        yield item
        progress_update()
