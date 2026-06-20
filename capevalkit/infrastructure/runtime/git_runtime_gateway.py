from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from capevalkit.application.ports import RuntimeInfo
from capevalkit.infrastructure.runtime.manager import RuntimeManager


class GitRuntimeGateway:
    def __init__(self, manager: RuntimeManager | None = None) -> None:
        self.manager = manager or RuntimeManager()

    def context(self) -> RuntimeInfo:
        context = self.manager.context
        return RuntimeInfo(
            source_mode=context.source_mode,
            project_root=context.project_root,
            cache_root=context.cache_root,
            lock_path=context.lock_path,
            lock_digest=context.lock_digest,
        )

    def prepare_base(self) -> Path:
        return self.manager.prepare_base()

    def upstream_names(self) -> list[str]:
        return self.manager.upstream_names()

    def upstream_for_path(self, path: str) -> str | None:
        upstream = self.manager.upstream_for_path(path)
        return upstream.name if upstream else None

    def ensure_upstreams(self, names: Iterable[str]) -> list[Path]:
        return self.manager.ensure_upstreams(names)
