from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
import threading
from typing import Callable, Iterable

from .benchmarks import (
    LONGCAPARENA_BENCHMARKS,
    run_metric_on_benchmark,
    write_benchmark_result,
)
from .domain.evaluation import NO_REFERENCE_METRICS
from .domain.reproduction import (
    EXCLUSIVE_GPU_METRICS,
    FLEUR_METRICS,
    GPU_METRICS,
    PYCOCO_METRICS,
    TOLERANCE_OVERRIDES,
    JobGroupingPolicy,
    ReproduceJob,
    ReproduceResult,
    ReproduceTask,
    ResourceRequirementPolicy,
    TolerancePolicy,
)
from .paths import repo_root
from .verify import NumericComparison, compare_results


DEFAULT_REPRO_METRICS = [
    "bleu",
    "rouge",
    "meteor",
    "cider",
    "spice",
    "clipscore",
    "clipscoreavg",
    "refclipscore",
    "pacscore",
    "refpacscore",
    "polos",
    "fleur",
    "reffleur",
    "vela",
    "expert",
]
DEFAULT_REPRO_BENCHMARKS = ["composite", "flickr8k-ex", "flickr8k-cf", "nebula", "polaris", *LONGCAPARENA_BENCHMARKS]
DEFAULT_REPRO_LONGCAPARENA_METRICS = ("vela",)
STATUS_LABELS = {
    "ok": "OK",
    "smoke": "OK",
    "mismatch": "DIFF",
    "error": "ERR",
    "skip": "SKIP",
    "planned": "PLAN",
}
STATUS_COLORS = {
    "ok": "32",
    "smoke": "32",
    "mismatch": "31",
    "error": "31;1",
    "skip": "33",
    "planned": "36",
}
LOCAL_IMAGE_BENCHMARKS = {"nebula", "polaris"}


def default_reproduce_pair(metric: str, benchmark: str) -> bool:
    if benchmark in LONGCAPARENA_BENCHMARKS:
        return metric in DEFAULT_REPRO_LONGCAPARENA_METRICS
    return True


@dataclass(frozen=True)
class DisplayComparison:
    check: str
    reprod: str
    original: str
    diff: str


def expected_tasks(
    *,
    expected_root: Path,
    output_dir: Path,
    metrics: Iterable[str],
    benchmarks: Iterable[str],
) -> list[ReproduceTask]:
    tasks = []
    seen: set[tuple[str, str]] = set()
    for metric in metrics:
        for benchmark in benchmarks:
            key = (metric, benchmark)
            if key in seen:
                continue
            seen.add(key)
            expected = expected_root / metric / f"{benchmark}.json"
            if expected.exists():
                tasks.append(
                    ReproduceTask(
                        metric=metric,
                        benchmark=benchmark,
                        expected=expected,
                        output=output_dir / metric / f"{benchmark}.json",
                    )
                )
    return tasks


def missing_expected_pairs(
    *,
    expected_root: Path,
    metrics: Iterable[str],
    benchmarks: Iterable[str],
) -> list[tuple[str, str]]:
    missing = []
    seen: set[tuple[str, str]] = set()
    for metric in metrics:
        for benchmark in benchmarks:
            key = (metric, benchmark)
            if key in seen:
                continue
            seen.add(key)
            if not (expected_root / metric / f"{benchmark}.json").exists():
                missing.append((metric, benchmark))
    return missing


def run_all_reproduce(
    *,
    metrics: list[str],
    benchmarks: list[str],
    data_root: str | None,
    output_dir: Path,
    expected_root: Path,
    tolerance: float,
    round_decimals: int | None,
    limit: int | None = None,
    fail_fast: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    color: str = "auto",
    report_missing: bool = True,
    jobs: int = 1,
    gpu_jobs: int = 1,
    pair_filter: Callable[[str, str], bool] | None = None,
    allow_mismatch: bool = False,
) -> tuple[int, list[ReproduceResult]]:
    tasks = expected_tasks(
        expected_root=expected_root,
        output_dir=output_dir,
        metrics=metrics,
        benchmarks=benchmarks,
    )
    if pair_filter is not None:
        tasks = [task for task in tasks if pair_filter(task.metric, task.benchmark)]
    results: list[ReproduceResult] = []

    if report_missing:
        for metric, benchmark in missing_expected_pairs(
            expected_root=expected_root,
            metrics=metrics,
            benchmarks=benchmarks,
        ):
            if pair_filter is not None and not pair_filter(metric, benchmark):
                continue
            results.append(
                ReproduceResult(
                    metric=metric,
                    benchmark=benchmark,
                    status="skip",
                    output=str(output_dir / metric / f"{benchmark}.json"),
                    expected=str(expected_root / metric / f"{benchmark}.json"),
                    message="missing expected file",
                )
            )
    pre_run_skips = [result for result in results if result.status == "skip"]

    use_color = should_color(color)

    if dry_run:
        planned = [
            ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="planned",
                output=str(task.output),
                expected=str(task.expected),
            )
            for task in tasks
        ]
        results.extend(planned)
        print_results_header()
        for index, result in enumerate(planned, 1):
            print_result(result, index=index, total=len(planned), use_color=use_color)
        for result in pre_run_skips:
            print_result(result, index=None, total=len(planned), use_color=use_color)
        return 0, sorted_results(results)

    reproduce_jobs = build_reproduce_jobs(tasks)
    print_results_header()
    print_lock = threading.Lock()

    with ReproduceProgress(total=len(tasks)) as progress:
        def emit_job_start(job: ReproduceJob) -> None:
            with print_lock:
                progress.start(job)

        def emit_result(result: ReproduceResult, *, index: int | None, total: int | None) -> None:
            with print_lock:
                progress.update()
                print_result(
                    result,
                    index=index,
                    total=total,
                    use_color=use_color,
                    printer=progress.print,
                )

        if fail_fast or (jobs <= 1 and gpu_jobs <= 1):
            completed = 0
            for job in reproduce_jobs:
                job_results = run_reproduce_job(
                    job,
                    data_root=data_root,
                    tolerance=tolerance,
                    round_decimals=round_decimals,
                    limit=limit,
                    verbose=verbose,
                    compare=not allow_mismatch,
                    on_start=emit_job_start,
                )
                for result in job_results:
                    completed += 1
                    results.append(result)
                    emit_result(result, index=completed, total=len(tasks))
                if fail_fast and any(result.status != "ok" for result in job_results):
                    break
        else:
            completed = 0
            max_cpu_workers = max(1, jobs)
            max_gpu_workers = max(1, gpu_jobs)
            cpu_executor = ThreadPoolExecutor(max_workers=max_cpu_workers)
            gpu_executor = ThreadPoolExecutor(max_workers=max_gpu_workers)
            futures = {}
            try:
                normal_jobs = [job for job in reproduce_jobs if job.resource != "exclusive-gpu"]
                exclusive_gpu_jobs = [job for job in reproduce_jobs if job.resource == "exclusive-gpu"]
                for job in normal_jobs:
                    executor = gpu_executor if job.resource == "gpu" else cpu_executor
                    future = executor.submit(
                        run_reproduce_job,
                        job,
                        data_root=data_root,
                        tolerance=tolerance,
                        round_decimals=round_decimals,
                        limit=limit,
                        verbose=verbose,
                        compare=not allow_mismatch,
                        on_start=emit_job_start,
                    )
                    futures[future] = job
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        for result in future.result():
                            completed += 1
                            results.append(result)
                            emit_result(result, index=completed, total=len(tasks))
                for job in exclusive_gpu_jobs:
                    for result in run_reproduce_job(
                        job,
                        data_root=data_root,
                        tolerance=tolerance,
                        round_decimals=round_decimals,
                        limit=limit,
                        verbose=verbose,
                        compare=not allow_mismatch,
                        on_start=emit_job_start,
                    ):
                        completed += 1
                        results.append(result)
                        emit_result(result, index=completed, total=len(tasks))
            finally:
                cpu_executor.shutdown(wait=True, cancel_futures=True)
                gpu_executor.shutdown(wait=True, cancel_futures=True)

        if not fail_fast:
            for result in pre_run_skips:
                with print_lock:
                    print_result(
                        result,
                        index=None,
                        total=len(tasks),
                        use_color=use_color,
                        printer=progress.print,
                    )

    results = sorted_results(results)
    success_statuses = {"ok", "smoke", "skip"}
    if allow_mismatch:
        success_statuses = {"ok", "smoke"}
    return (0 if all(result.status in success_statuses for result in results) else 1), results


def build_reproduce_jobs(tasks: list[ReproduceTask]) -> list[ReproduceJob]:
    return JobGroupingPolicy().build_jobs(tasks)


def grouped_job_for_task(
    task: ReproduceTask,
    task_by_pair: dict[tuple[str, str], ReproduceTask],
    assigned: set[tuple[str, str]],
) -> ReproduceJob:
    return JobGroupingPolicy().grouped_job_for_task(task, task_by_pair, assigned)


def _resource_for_metric(metric: str) -> str:
    return ResourceRequirementPolicy().resource_for_metric(metric)


def _available_group(
    metrics: Iterable[str],
    benchmark: str,
    task_by_pair: dict[tuple[str, str], ReproduceTask],
    assigned: set[tuple[str, str]],
) -> list[ReproduceTask]:
    return JobGroupingPolicy._available_group(metrics, benchmark, task_by_pair, assigned)


def run_reproduce_job(
    job: ReproduceJob,
    *,
    data_root: str | None,
    tolerance: float,
    round_decimals: int | None,
    limit: int | None,
    verbose: bool,
    compare: bool = True,
    on_start: Callable[[ReproduceJob], None] | None = None,
) -> list[ReproduceResult]:
    if on_start is not None:
        on_start(job)
    missing_prerequisite = missing_job_prerequisite(job, data_root=data_root)
    if missing_prerequisite:
        return [
            ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="skip",
                output=str(task.output),
                expected=str(task.expected),
                message=missing_prerequisite,
            )
            for task in job.tasks
        ]

    try:
        code, items, metric_output = run_metric_on_benchmark(
            job.runner_metric,
            job.benchmark,
            data_root=data_root,
            metric_args=list(job.metric_args),
            use_references=job.use_references,
            limit=limit,
            quiet=not verbose,
            show_progress=False,
        )
        if code != 0:
            return [
                ReproduceResult(
                    metric=task.metric,
                    benchmark=task.benchmark,
                    status="error",
                    output=str(task.output),
                    expected=str(task.expected),
                    message=f"benchmark exited with code {code}",
                )
                for task in job.tasks
            ]
    except Exception as exc:
        return [
            ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="error",
                output=str(task.output),
                expected=str(task.expected),
                message=f"{type(exc).__name__}: {exc}",
            )
            for task in job.tasks
        ]

    results = []
    for task in job.tasks:
        try:
            write_benchmark_result(
                task.metric,
                task.benchmark,
                task.output,
                items=items,
                metric_output=metric_output,
            )
            if not compare:
                result = ReproduceResult(
                    metric=task.metric,
                    benchmark=task.benchmark,
                    status="smoke",
                    output=str(task.output),
                    expected=str(task.expected),
                    message="smoke run completed",
                )
                results.append(result)
                continue
            comparisons = compare_results(
                str(task.output),
                str(task.expected),
                tolerance=task_tolerance(task.metric, task.benchmark, tolerance),
                round_decimals=round_decimals,
            )
            mismatches = [comparison for comparison in comparisons if not comparison.ok]
            if mismatches:
                raise AssertionError(format_comparisons(mismatches))
        except AssertionError as exc:
            result = ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="mismatch",
                output=str(task.output),
                expected=str(task.expected),
                message=str(exc),
            )
        except Exception as exc:
            result = ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="error",
                output=str(task.output),
                expected=str(task.expected),
                message=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = ReproduceResult(
                metric=task.metric,
                benchmark=task.benchmark,
                status="ok",
                output=str(task.output),
                expected=str(task.expected),
                message=format_comparisons(comparisons),
            )
        results.append(result)
    return results


def missing_job_prerequisite(job: ReproduceJob, *, data_root: str | None = None) -> str | None:
    if (
        _explicit_non_repo_data_root(data_root)
        and job.benchmark in LOCAL_IMAGE_BENCHMARKS
        and any(task.metric in GPU_METRICS for task in job.tasks)
    ):
        image_dir = _benchmark_image_dir(job.benchmark, data_root)
        if not image_dir.exists():
            return f"missing {job.benchmark} images: {image_dir}"
    if any(task.metric in FLEUR_METRICS for task in job.tasks):
        numpy_version = locked_project_package_version(
            repo_root() / "metrics" / "upstreams" / "fleur" / "uv.lock",
            "numpy",
        )
        if numpy_version and major_version(numpy_version) >= 2:
            return f"incompatible FLEUR NumPy runtime: numpy {numpy_version}; run uv lock/sync with numpy<2"
    return None


def _benchmark_image_dir(benchmark: str, data_root: str | None) -> Path:
    base = Path(data_root).expanduser().absolute() if data_root else repo_root() / "data"
    return base / benchmark / "images"


def _explicit_non_repo_data_root(data_root: str | None) -> bool:
    if data_root is None:
        return False
    return Path(data_root).expanduser().absolute() != (repo_root() / "data").absolute()


def locked_project_package_version(lock_path: Path, package: str) -> str | None:
    if not lock_path.exists():
        return None
    pattern = re.compile(
        rf'\[\[package\]\]\s+name = "{re.escape(package)}"\s+version = "([^"]+)"',
        re.MULTILINE,
    )
    match = pattern.search(lock_path.read_text())
    return match.group(1) if match else None


def major_version(version: str) -> int:
    match = re.match(r"(\d+)", version)
    return int(match.group(1)) if match else 0


def sorted_results(results: list[ReproduceResult]) -> list[ReproduceResult]:
    order = {name: index for index, name in enumerate(DEFAULT_REPRO_METRICS)}
    bench_order = {name: index for index, name in enumerate(DEFAULT_REPRO_BENCHMARKS)}
    return sorted(
        results,
        key=lambda result: (
            order.get(result.metric, len(order)),
            result.metric,
            bench_order.get(result.benchmark, len(bench_order)),
            result.benchmark,
            result.status,
        ),
    )


def task_tolerance(metric: str, benchmark: str, default: float) -> float:
    return TolerancePolicy().tolerance(metric, benchmark, default)


def print_result(
    result: ReproduceResult,
    *,
    index: int | None = None,
    total: int | None = None,
    use_color: bool = False,
    printer: Callable[[str], None] | None = None,
) -> None:
    raw_label = STATUS_LABELS.get(result.status, result.status.upper())
    label = f"{raw_label:<6}"
    if use_color:
        label = colorize(label, STATUS_COLORS.get(result.status, "0"))
    progress = format_progress(index, total)
    target = f"{result.metric}/{result.benchmark}"
    rows = display_comparisons(result.message)
    if rows:
        print_result_row(
            progress=progress,
            label=label,
            target=target,
            comparisons=rows,
            printer=printer,
        )
        return
    print_result_row(
        progress=progress,
        label=label,
        target=target,
        comparisons=[],
        note=compact_message(result.message),
        printer=printer,
    )


def format_job_target(job: ReproduceJob) -> str:
    metrics = [task.metric for task in job.tasks]
    if len(metrics) == 1:
        metric = metrics[0]
    else:
        metric = f"{metrics[0]}+{len(metrics) - 1}"
    return f"{metric}/{job.benchmark}"


class ReproduceProgress:
    def __init__(self, *, total: int) -> None:
        self.total = total
        self.progress = None
        self.task_id = None

    def __enter__(self) -> ReproduceProgress:
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
            return self
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(),
            transient=True,
            disable=self.total <= 0 or not sys.stderr.isatty(),
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.progress.start()
        self.task_id = self.progress.add_task("all_reproduce", total=self.total)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.progress is not None:
            self.progress.stop()

    def start(self, job: ReproduceJob) -> None:
        if self.progress is None or self.task_id is None:
            return
        self.progress.update(self.task_id, description=format_job_target(job))
        self.progress.refresh()

    def update(self) -> None:
        if self.progress is not None and self.task_id is not None:
            self.progress.update(self.task_id, advance=1)
            self.progress.refresh()

    def print(self, line: str) -> None:
        if self.progress is not None:
            self.progress.console.print(line, highlight=False, markup=False, soft_wrap=True)
            return
        print(line, flush=True)


def print_results_header() -> None:
    line = (
        f"{'ITEM':<7} {'STATUS':<6} {'TARGET':<32} "
        f"{'τb_REPROD':>10} {'τb_ORIG':>8} "
        f"{'τc_REPROD':>10} {'τc_ORIG':>8} "
        f"{'τb_DIFF':>8} {'τc_DIFF':>8}  NOTE"
    )
    print(line.rstrip(), flush=True)


def print_result_row(
    *,
    progress: str,
    label: str,
    target: str,
    comparisons: Iterable[DisplayComparison],
    note: str = "",
    printer: Callable[[str], None] | None = None,
) -> None:
    columns = format_display_columns(comparisons)
    line = f"{progress:<7} {label:<6} {target:<32} {columns}"
    if note:
        line = f"{line}  {note}"
    if printer is None:
        print(line.rstrip(), flush=True)
    else:
        printer(line.rstrip())


def format_display_columns(comparisons: Iterable[DisplayComparison]) -> str:
    by_check = {comparison.check: comparison for comparison in comparisons}
    tau_b = by_check.get("tau-b", empty_display_comparison("tau-b"))
    tau_c = by_check.get("tau-c", empty_display_comparison("tau-c"))
    return (
        f"{tau_b.reprod:>10} {tau_b.original:>8} "
        f"{tau_c.reprod:>10} {tau_c.original:>8} "
        f"{tau_b.diff:>8} {tau_c.diff:>8}"
    )


def empty_display_comparison(check: str) -> DisplayComparison:
    return DisplayComparison(check=check, reprod="-", original="-", diff="-")


def format_progress(index: int | None, total: int | None) -> str:
    if index is None or total is None:
        return "--"
    width = len(str(total))
    return f"{index:0{width}d}/{total}"


def should_color(color: str) -> bool:
    if color == "always":
        return True
    if color == "never":
        return False
    return sys.stdout.isatty()


def colorize(value: str, code: str) -> str:
    return f"\033[{code}m{value}\033[0m"


def format_comparisons(comparisons: Iterable[NumericComparison]) -> str:
    parts = []
    for comparison in comparisons:
        key = compact_key(comparison.key)
        parts.append(format_comparison_part(key, comparison.actual_display, comparison.expected_display))
    return "; ".join(parts)


def compact_key(key: str) -> str:
    if key.endswith("kendall_tau_b"):
        return "tau-b"
    if key.endswith("kendall_tau_c"):
        return "tau-c"
    return key.rsplit(".", 1)[-1]


def compact_message(message: str) -> str:
    if not message:
        return ""
    mismatch_parts = []
    if message.startswith("missing PACScore OpenCLIP checkpoint:"):
        return "missing PACScore OpenCLIP checkpoint"
    if message.startswith("missing PACScore CLIP checkpoint:"):
        return "missing PACScore CLIP checkpoint"
    if message.startswith("missing PAC-S++ checkpoint:"):
        return "missing PAC-S++ checkpoint"
    if message.startswith("missing nebula images:"):
        return "missing nebula images"
    if message.startswith("missing polaris images:"):
        return "missing polaris images"
    if message.startswith("incompatible FLEUR NumPy runtime:"):
        return "incompatible FLEUR NumPy runtime"
    pattern = re.compile(
        r"correlations\.kendall_tau_([bc]): actual=[^ ]+ rounded=([^ ]+) "
        r"expected=[^ ]+ rounded_expected=([^ ;]+)"
    )
    for key, actual, expected in pattern.findall(message):
        mismatch_parts.append(format_comparison_part(f"tau-{key}", actual, expected))
    if mismatch_parts:
        return "; ".join(mismatch_parts)

    missing = re.search(r"missing ([^;]+); checked:", message)
    if missing:
        return f"missing {missing.group(1)}"

    one_line = " ".join(message.split())
    return one_line if len(one_line) <= 140 else f"{one_line[:137]}..."


def format_comparison_part(key: str, actual: str, expected: str) -> str:
    delta = display_delta(actual, expected)
    if delta:
        return f"{key} reprod={actual} original={expected} diff={delta}"
    return f"{key} reprod={actual} original={expected}"


def display_comparisons(message: str) -> list[DisplayComparison]:
    rows = []
    pattern = re.compile(
        r"(?P<check>[^ ;]+) reprod=(?P<reprod>[^ ;]+) "
        r"original=(?P<original>[^ ;]+)(?: diff=(?P<diff>[^ ;]+))?"
    )
    for match in pattern.finditer(compact_message(message)):
        rows.append(
            DisplayComparison(
                check=match.group("check"),
                reprod=match.group("reprod"),
                original=match.group("original"),
                diff=match.group("diff") or "-",
            )
        )
    return rows


def display_delta(actual: str, expected: str) -> str:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except ValueError:
        return ""
    if not (math.isfinite(actual_value) and math.isfinite(expected_value)):
        return ""
    decimals = max(decimal_places(actual), decimal_places(expected))
    return f"{actual_value - expected_value:+.{decimals}f}"


def decimal_places(value: str) -> int:
    _, _, fraction = value.partition(".")
    return len(fraction) if fraction else 0


def write_summary(path: Path, results: list[ReproduceResult]) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    payload = {
        "counts": counts,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def print_summary(path: Path, results: list[ReproduceResult], *, use_color: bool = False) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    parts = []
    for status in ["ok", "smoke", "mismatch", "error", "skip", "planned"]:
        count = counts.get(status, 0)
        if not count:
            continue
        label = STATUS_LABELS[status]
        if use_color:
            label = colorize(label, STATUS_COLORS[status])
        parts.append(f"{label}={count}")
    print(f"Summary {' '.join(parts)}  {path}", flush=True)


def default_expected_root() -> Path:
    return repo_root() / "benchmarks" / "expected"
