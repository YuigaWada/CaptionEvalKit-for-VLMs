from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

from capevalkit.application.ports import MetricRunRequest, MetricRunResult
from capevalkit.context import ProjectContext, default_context
from capevalkit.domain.metrics import MetricManifest
from capevalkit.paths import cache_root, repo_root
from capevalkit.runtime import RuntimeManager
from capevalkit.runtime_env import apply_runtime_environment


ManifestProvider = Callable[[str], MetricManifest]


def metric_repo(manifest: MetricManifest) -> Path:
    path = RuntimeManager().ensure_metric(manifest)
    if not path.exists():
        hint = f" clone {manifest.repo_url}" if manifest.repo_url else ""
        raise FileNotFoundError(f"missing metric repository: {path}{hint}")
    return path


def uv_project(manifest: MetricManifest) -> Path:
    RuntimeManager().ensure_metric(manifest)
    path = repo_root() / manifest.uv_project
    if not path.exists():
        raise FileNotFoundError(f"missing uv project for {manifest.name}: {path}")
    return path


def build_uv_command(metric_name: str, args: list[str], *, manifest_provider: ManifestProvider) -> list[str]:
    manifest = manifest_provider(metric_name)
    return ["uv", "run", "--project", str(uv_project(manifest)), *args]


class UvSubprocessMetricRunner:
    def __init__(self, manifest_provider: ManifestProvider) -> None:
        self.manifest_provider = manifest_provider

    def run(self, request: MetricRunRequest) -> MetricRunResult:
        manifest = self.manifest_provider(request.metric_name)
        command = build_uv_command(
            request.metric_name,
            request.args,
            manifest_provider=self.manifest_provider,
        )
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        context = default_context()
        apply_runtime_environment(env, repo_root(), cache_root=context.cache_root)
        env["PYTHONPATH"] = pythonpath(package_import_root(context), repo_root(), env.get("PYTHONPATH"))
        cwd = metric_repo(manifest)
        if request.quiet and request.progress_total and request.progress_total > 0:
            return MetricRunResult(
                call_with_progress(
                    command,
                    cwd=cwd,
                    env=env,
                    total=request.progress_total,
                    desc=request.progress_desc or request.metric_name,
                )
            )
        stream = subprocess.DEVNULL if request.quiet else None
        return MetricRunResult(subprocess.call(command, cwd=cwd, env=env, stdout=stream, stderr=stream))


def call_with_progress(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    total: int,
    desc: str,
) -> int:
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ModuleNotFoundError:
        stream = subprocess.DEVNULL
        return subprocess.call(command, cwd=cwd, env=env, stdout=stream, stderr=stream)

    fd, progress_name = tempfile.mkstemp(prefix="capevalkit-progress-", text=True)
    os.close(fd)
    progress_path = Path(progress_name)
    env = env.copy()
    env["CAPEVALKIT_PROGRESS_FILE"] = str(progress_path)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with progress_path.open("r") as progress_file:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),
                transient=True,
                disable=not sys.stderr.isatty(),
                redirect_stdout=False,
                redirect_stderr=False,
            ) as progress:
                task_id = progress.add_task(desc, total=total)
                completed = 0
                while process.poll() is None:
                    completed = drain_progress(progress_file, progress, task_id, total, completed)
                    time.sleep(0.1)
                return_code = process.wait()
                completed = drain_progress(progress_file, progress, task_id, total, completed)
                if return_code == 0 and completed < total:
                    progress.update(task_id, advance=total - completed)
        return return_code
    finally:
        try:
            progress_path.unlink()
        except OSError:
            pass


def drain_progress(progress_file, progress, task_id, total: int, completed: int) -> int:
    for line in progress_file:
        try:
            count = int(line.strip())
        except ValueError:
            continue
        remaining = total - completed
        if remaining <= 0:
            continue
        advance = min(count, remaining)
        progress.update(task_id, advance=advance)
        completed += advance
    return completed


def pythonpath(*entries: Path | str | None) -> str:
    values: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        for value in str(entry).split(os.pathsep):
            if value and value not in values:
                values.append(value)
    return os.pathsep.join(values)


def package_import_root(context: ProjectContext | None = None) -> Path:
    context = context or default_context()
    if context.source_mode:
        return context.package_root.parent

    bridge = context.project_root / ".pythonpath"
    target = bridge / "capevalkit"
    package_root = context.package_root
    bridge.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == package_root:
            return bridge
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(package_root, target_is_directory=True)
    except OSError:
        shutil.copytree(package_root, target)
    return bridge

