from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import shutil

from .paths import repo_root


@dataclass(frozen=True)
class OverlayAction:
    source: Path
    target: Path
    changed: bool


def ensure_overlays(
    *,
    root: Path | None = None,
    overlays: Iterable[str] | None = None,
) -> list[OverlayAction]:
    project_root = root or repo_root()
    overlays_root = project_root / "overlays"
    actions: list[OverlayAction] = []
    if not overlays_root.exists():
        return actions

    if overlays is None:
        sources = sorted(path for path in overlays_root.rglob("*") if path.is_file())
    else:
        sources = []
        for overlay in overlays:
            source = project_root / overlay
            if source.is_file():
                sources.append(source)

    for source in sources:
        relative = source.relative_to(overlays_root)
        target = project_root / relative
        changed = not target.exists() or source.read_bytes() != target.read_bytes()
        if changed:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        actions.append(OverlayAction(source=source, target=target, changed=changed))
    return actions
