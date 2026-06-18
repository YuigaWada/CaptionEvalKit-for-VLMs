from __future__ import annotations

from .application.ports import MetricRunRequest
from .domain.metrics import MetricManifest
from .infrastructure.execution.uv_subprocess_metric_runner import (
    UvSubprocessMetricRunner,
    build_uv_command as _build_uv_command,
    call_with_progress as _call_with_progress,
    drain_progress as _drain_progress,
    metric_repo,
    package_import_root as _package_import_root,
    pythonpath as _pythonpath,
    uv_project,
)
from .manifests import get_manifest
from .paths import cache_root, repo_root


def build_uv_command(metric_name: str, args: list[str]) -> list[str]:
    return _build_uv_command(metric_name, args, manifest_provider=get_manifest)


def dispatch(
    metric_name: str,
    args: list[str],
    *,
    quiet: bool = False,
    progress_total: int | None = None,
    progress_desc: str | None = None,
) -> int:
    runner = UvSubprocessMetricRunner(get_manifest)
    result = runner.run(
        MetricRunRequest(
            metric_name=metric_name,
            args=args,
            quiet=quiet,
            progress_total=progress_total,
            progress_desc=progress_desc,
        )
    )
    return result.return_code


def print_command(metric_name: str, args: list[str]) -> None:
    manifest = get_manifest(metric_name)
    command = " ".join(_quote(part) for part in build_uv_command(metric_name, args))
    root = repo_root()
    cache_dir = cache_root() / "uv"
    clip_cache_dir = cache_root() / "clip"
    from .context import default_context

    context = default_context()
    pythonpath = _pythonpath(_package_import_root(context), root, None)
    print(
        f"cd {_quote(str(metric_repo(manifest)))} && "
        f"UV_CACHE_DIR={_quote(str(cache_dir))} UV_LINK_MODE=hardlink "
        f"CLIP_DOWNLOAD_ROOT={_quote(str(clip_cache_dir))} "
        f"PYTHONPATH={_quote(pythonpath)} {command}"
    )


def _quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


def exit_with_dispatch(metric_name: str, args: list[str]) -> None:
    raise SystemExit(dispatch(metric_name, args))


__all__ = [
    "MetricManifest",
    "_call_with_progress",
    "_drain_progress",
    "_package_import_root",
    "_pythonpath",
    "build_uv_command",
    "dispatch",
    "exit_with_dispatch",
    "metric_repo",
    "print_command",
    "uv_project",
]

