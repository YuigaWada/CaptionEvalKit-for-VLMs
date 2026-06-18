from __future__ import annotations

import os
from pathlib import Path
import sys

from capevalkit.interfaces import cli
from capevalkit.infrastructure.runtime.context import default_context
from capevalkit.infrastructure.runtime.environment import apply_runtime_environment
from capevalkit.infrastructure.runtime.manager import RuntimeManager


def configure_environment(root: Path | None = None) -> Path:
    if root is None:
        manager = RuntimeManager()
        project_root = manager.prepare_base()
        cache_root = manager.context.cache_root
    else:
        project_root = root
        cache_root = default_context().cache_root
    apply_runtime_environment(os.environ, project_root, cache_root=cache_root)
    return project_root


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    return cli.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
