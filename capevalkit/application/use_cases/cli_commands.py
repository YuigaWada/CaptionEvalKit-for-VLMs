from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capevalkit.application.ports import (
    MetricManifestRepository,
    MetricRunRequest,
    MetricRunner,
    RuntimeGateway,
    RuntimeInfo,
)
from capevalkit.domain.evaluation import ReferenceRequirementPolicy
from capevalkit.domain.metrics import MetricManifest


@dataclass(frozen=True)
class MetricListItem:
    name: str
    repo_dir: str
    uv_project: str
    python: str
    native_benchmarks: tuple[str, ...]


@dataclass(frozen=True)
class ListMetricsResult:
    metrics: tuple[MetricListItem, ...]
    generic_benchmarks: tuple[str, ...]


@dataclass(frozen=True)
class DoctorResult:
    runtime: RuntimeInfo
    git: str
    uv: str


@dataclass(frozen=True)
class SyncRuntimeRequest:
    all_metrics: bool = False
    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyncRuntimeResult:
    base_path: Path | None = None
    upstream_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RunCommandRequest:
    metric: str
    command: list[str]


@dataclass(frozen=True)
class ScoreCaptionsRequest:
    metric: str
    predictions: Path
    output: Path
    references: Path | None = None
    image_dir: Path | None = None
    extra_args: list[str] | None = None


@dataclass(frozen=True)
class BenchmarkRequest:
    metric: str
    benchmark: str | None
    output: Path | None
    output_root: Path
    data_root: str | None = None
    native: bool = False
    no_references: bool = False
    score_key: str | None = None
    limit: int | None = None
    extra_args: list[str] | None = None


@dataclass(frozen=True)
class VerifyResultsRequest:
    results: str
    expected: str
    tolerance: float
    round_decimals: int | None = None


@dataclass(frozen=True)
class SuiteRequest:
    metrics: tuple[str, ...]
    benchmarks: tuple[str, ...]
    output_dir: Path
    expected_root: Path
    data_root: str | None = None
    verify: bool = False
    tolerance: float = 0.2
    round_decimals: int | None = None
    no_references: bool = False
    limit: int | None = None


@dataclass(frozen=True)
class AllReproduceRequest:
    metrics: tuple[str, ...]
    benchmarks: tuple[str, ...]
    data_root: str | None
    output_dir: Path
    expected_root: Path
    tolerance: float
    round_decimals: int | None
    limit: int | None = None
    fail_fast: bool = False
    dry_run: bool = False
    verbose: bool = False
    color: str = "auto"
    report_missing: bool = True
    jobs: int = 1
    gpu_jobs: int = 1
    pair_filter: Callable[[str, str], bool] | None = None
    allow_mismatch: bool = False


@dataclass(frozen=True)
class AllReproduceResult:
    exit_code: int
    results: list[Any]


@dataclass(frozen=True)
class DownloadAssetsRequest:
    assets: tuple[str, ...]
    all_assets: bool = False
    list_only: bool = False
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class DownloadAssetsResult:
    rows: str | None = None
    paths: tuple[Path, ...] = ()


def list_metrics(
    repository: MetricManifestRepository,
    *,
    generic_benchmarks: Iterable[str],
) -> ListMetricsResult:
    manifests = repository.list()
    rows = tuple(
        MetricListItem(
            name=name,
            repo_dir=manifest.repo_dir,
            uv_project=manifest.uv_project,
            python=manifest.python,
            native_benchmarks=tuple(sorted(manifest.benchmarks)),
        )
        for name, manifest in sorted(manifests.items())
    )
    return ListMetricsResult(metrics=rows, generic_benchmarks=tuple(generic_benchmarks))


def doctor(
    runtime: RuntimeGateway,
    *,
    tool_lookup: Callable[[str], str | None],
) -> DoctorResult:
    runtime.prepare_base()
    return DoctorResult(
        runtime=runtime.context(),
        git=tool_lookup("git") or "missing",
        uv=tool_lookup("uv") or "missing",
    )


def sync_runtime(
    request: SyncRuntimeRequest,
    *,
    runtime: RuntimeGateway,
    repository: MetricManifestRepository,
) -> SyncRuntimeResult:
    runtime.prepare_base()
    if request.all_metrics:
        upstreams = runtime.upstream_names()
    elif request.metrics:
        manifests = repository.list()
        upstreams = []
        for metric in request.metrics:
            try:
                manifest = manifests[metric]
            except KeyError as exc:
                known = ", ".join(sorted(manifests))
                raise KeyError(f"unknown metric: {metric}; known metrics: {known}") from exc
            upstream = runtime.upstream_for_path(manifest.repo_dir)
            if upstream:
                upstreams.append(upstream)
    else:
        return SyncRuntimeResult(base_path=runtime.context().project_root)

    return SyncRuntimeResult(upstream_paths=tuple(runtime.ensure_upstreams(upstreams)))


def run_command(request: RunCommandRequest, *, runner: MetricRunner) -> int:
    return runner.run(MetricRunRequest(metric_name=request.metric, args=request.command)).return_code


def score_captions(
    request: ScoreCaptionsRequest,
    *,
    repository: MetricManifestRepository,
    runner: MetricRunner,
) -> int:
    manifest = repository.get(request.metric)
    command = [
        *manifest.runner,
        "--predictions",
        str(request.predictions.resolve()),
        "--output",
        str(request.output.resolve()),
    ]
    if request.references:
        command.extend(["--references", str(request.references.resolve())])
    if request.image_dir:
        command.extend(["--image-dir", str(request.image_dir.resolve())])
    command.extend(request.extra_args or [])
    return runner.run(MetricRunRequest(metric_name=request.metric, args=command)).return_code


def benchmark(
    request: BenchmarkRequest,
    *,
    repository: MetricManifestRepository,
    runner: MetricRunner,
    benchmark_runner: Callable[..., int],
    import_module: Callable[[str], Any],
    reference_policy: ReferenceRequirementPolicy | None = None,
) -> int:
    manifest = repository.get(request.metric)
    extra_args = request.extra_args or []
    reference_policy = reference_policy or ReferenceRequirementPolicy()
    if request.benchmark and not request.native:
        output = request.output or request.output_root / request.metric / f"{request.benchmark}.json"
        return benchmark_runner(
            request.metric,
            request.benchmark,
            str(output),
            data_root=request.data_root,
            metric_args=extra_args,
            use_references=reference_policy.use_references(
                request.metric,
                no_references=request.no_references,
            ),
            score_key=request.score_key,
            limit=request.limit,
        )

    if request.benchmark:
        try:
            spec = manifest.benchmarks[request.benchmark]
        except KeyError as exc:
            known = ", ".join(sorted(manifest.benchmarks)) or "(none)"
            raise KeyError(f"{request.metric} has no benchmark {request.benchmark}; known: {known}") from exc
        output = request.output or request.output_root / request.metric / f"{request.benchmark}.json"
        command = [*manifest.runner, *spec.args, "--output", str(output), *extra_args]
        return runner.run(MetricRunRequest(metric_name=request.metric, args=command)).return_code

    if manifest.benchmark_module:
        module = import_module(manifest.benchmark_module)
        return int(module.main(extra_args) or 0)
    raise ValueError(f"{request.metric} does not declare a benchmark_module or benchmark specs")


def verify_results_use_case(
    request: VerifyResultsRequest,
    *,
    verifier: Callable[..., None],
) -> None:
    verifier(
        request.results,
        request.expected,
        tolerance=request.tolerance,
        round_decimals=request.round_decimals,
    )


def suite(
    request: SuiteRequest,
    *,
    benchmark_runner: Callable[..., int],
    verifier: Callable[..., None],
    reference_policy: ReferenceRequirementPolicy | None = None,
    printer: Callable[[str], None] | None = None,
) -> int:
    reference_policy = reference_policy or ReferenceRequirementPolicy()
    printer = printer or print
    for metric in request.metrics:
        for benchmark_name in request.benchmarks:
            output = request.output_dir / metric / f"{benchmark_name}.json"
            code = benchmark_runner(
                metric,
                benchmark_name,
                str(output),
                data_root=request.data_root,
                use_references=reference_policy.use_references(
                    metric,
                    no_references=request.no_references,
                ),
                limit=request.limit,
            )
            if code != 0:
                return code
            if request.verify:
                expected = request.expected_root / metric / f"{benchmark_name}.json"
                verifier(
                    str(output),
                    str(expected),
                    tolerance=request.tolerance,
                    round_decimals=request.round_decimals,
                )
            printer(f"ok\t{metric}\t{benchmark_name}\t{output}")
    return 0


def all_reproduce(
    request: AllReproduceRequest,
    *,
    reproduce_runner: Callable[..., tuple[int, list[Any]]],
) -> AllReproduceResult:
    code, results = reproduce_runner(
        metrics=list(request.metrics),
        benchmarks=list(request.benchmarks),
        data_root=request.data_root,
        output_dir=request.output_dir,
        expected_root=request.expected_root,
        tolerance=request.tolerance,
        round_decimals=request.round_decimals,
        limit=request.limit,
        fail_fast=request.fail_fast,
        dry_run=request.dry_run,
        verbose=request.verbose,
        color=request.color,
        report_missing=request.report_missing,
        jobs=request.jobs,
        gpu_jobs=request.gpu_jobs,
        pair_filter=request.pair_filter,
        allow_mismatch=request.allow_mismatch,
    )
    return AllReproduceResult(exit_code=code, results=results)


def download_assets(
    request: DownloadAssetsRequest,
    *,
    catalog: Iterable[Any],
    formatter: Callable[[Iterable[Any]], str],
    selector: Callable[..., list[Any]],
    downloader: Callable[..., list[Path]],
) -> DownloadAssetsResult:
    if request.list_only:
        return DownloadAssetsResult(rows=formatter(catalog))
    selected = selector(
        request.assets,
        all_assets=request.all_assets,
    )
    paths = downloader(
        selected,
        force=request.force,
        dry_run=request.dry_run,
    )
    return DownloadAssetsResult(paths=tuple(paths))
