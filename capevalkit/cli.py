from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil

from .benchmarks import LONGCAPARENA_BENCHMARKS, benchmark_metric
from .dispatcher import dispatch, print_command
from .manifests import get_manifest, load_manifests
from .paths import repo_root
from .runtime import RuntimeManager
from .reproduce import (
    DEFAULT_REPRO_BENCHMARKS,
    DEFAULT_REPRO_METRICS,
    NO_REFERENCE_METRICS,
    default_expected_root,
    default_reproduce_pair,
    print_result,
    print_summary,
    run_all_reproduce,
    should_color,
    write_summary,
)
from .verify import verify_results


def _split_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _split_csv(value: str) -> list[str]:
    items: list[str] = []
    for item in (part.strip() for part in value.split(",") if part.strip()):
        if item == "longcaparena":
            items.extend(LONGCAPARENA_BENCHMARKS)
        else:
            items.append(item)
    return items


def list_metrics(_: argparse.Namespace) -> int:
    manifests = load_manifests()
    for name in sorted(manifests):
        manifest = manifests[name]
        native = ",".join(sorted(manifest.benchmarks)) or "-"
        print(
            f"{name}\trepo={manifest.repo_dir}\tuv_project={manifest.uv_project}"
            f"\tpython={manifest.python}\tnative={native}"
        )
    longcaparena = ",".join(LONGCAPARENA_BENCHMARKS)
    print(f"generic-benchmarks\tcomposite,flickr8k-ex,flickr8k-cf,nebula,polaris,{longcaparena}")
    return 0


def env_command(args: argparse.Namespace) -> int:
    print_command(args.metric, _split_remainder(args.command))
    return 0


def run(args: argparse.Namespace) -> int:
    return dispatch(args.metric, _split_remainder(args.command))


def doctor(_: argparse.Namespace) -> int:
    manager = RuntimeManager()
    context = manager.context
    manager.prepare_base()
    print(f"mode\t{'source' if context.source_mode else 'runtime-cache'}")
    print(f"project_root\t{context.project_root}")
    print(f"cache_root\t{context.cache_root}")
    print(f"lock\t{context.lock_path}")
    print(f"lock_digest\t{context.lock_digest[:12]}")
    print(f"git\t{shutil.which('git') or 'missing'}")
    print(f"uv\t{shutil.which('uv') or 'missing'}")
    return 0


def sync(args: argparse.Namespace) -> int:
    manager = RuntimeManager()
    manager.prepare_base()
    if args.all:
        upstreams = manager.upstream_names()
    elif args.metrics:
        manifests = load_manifests()
        upstreams = []
        for metric in _split_csv(args.metrics):
            try:
                manifest = manifests[metric]
            except KeyError as exc:
                known = ", ".join(sorted(manifests))
                raise SystemExit(f"unknown metric: {metric}; known metrics: {known}") from exc
            upstream = manager.upstream_for_path(manifest.repo_dir)
            if upstream:
                upstreams.append(upstream.name)
    else:
        print(f"OK\tbase\t{manager.context.project_root}")
        return 0

    for path in manager.ensure_upstreams(upstreams):
        print(f"OK\tupstream\t{path}")
    return 0


def score(args: argparse.Namespace) -> int:
    manifest = get_manifest(args.metric)
    command = [
        *manifest.runner,
        "--predictions",
        str(Path(args.predictions).resolve()),
        "--output",
        str(Path(args.output).resolve()),
    ]
    if args.references:
        command.extend(["--references", str(Path(args.references).resolve())])
    if args.image_dir:
        command.extend(["--image-dir", str(Path(args.image_dir).resolve())])
    command.extend(_split_remainder(args.args))
    return dispatch(args.metric, command)


def benchmark(args: argparse.Namespace) -> int:
    manifest = get_manifest(args.metric)
    extra_args = _split_remainder(args.args)
    if args.benchmark and not args.native:
        output_root = Path(os.environ.get("IC_EVAL_OUTPUT_DIR", repo_root() / "outputs"))
        output = args.output or str(output_root / args.metric / f"{args.benchmark}.json")
        return benchmark_metric(
            args.metric,
            args.benchmark,
            output,
            data_root=args.data_root,
            metric_args=extra_args,
            use_references=_use_references(args.metric, no_references=args.no_references),
            score_key=args.score_key,
            limit=args.limit,
        )

    if args.benchmark:
        try:
            spec = manifest.benchmarks[args.benchmark]
        except KeyError as exc:
            known = ", ".join(sorted(manifest.benchmarks)) or "(none)"
            raise SystemExit(f"{args.metric} has no benchmark {args.benchmark}; known: {known}") from exc
        output_root = Path(os.environ.get("IC_EVAL_OUTPUT_DIR", repo_root() / "outputs"))
        output = args.output or str(output_root / args.metric / f"{args.benchmark}.json")
        command = [*manifest.runner, *spec.args, "--output", output, *extra_args]
        return dispatch(args.metric, command)

    if manifest.benchmark_module:
        module = importlib.import_module(manifest.benchmark_module)
        return int(module.main(extra_args) or 0)
    raise SystemExit(f"{args.metric} does not declare a benchmark_module or benchmark specs")


def verify(args: argparse.Namespace) -> int:
    verify_results(args.results, args.expected, tolerance=args.tolerance, round_decimals=args.round_decimals)
    print("ok")
    return 0


def download_assets_command(args: argparse.Namespace) -> int:
    from .downloads import (
        DOWNLOADABLE_ASSETS,
        download_assets,
        format_asset_rows,
        select_assets,
    )

    if args.list:
        print(format_asset_rows(DOWNLOADABLE_ASSETS))
        return 0
    try:
        selected = select_assets(
            args.assets,
            all_assets=args.all,
        )
        download_assets(
            selected,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def suite(args: argparse.Namespace) -> int:
    metrics = _split_csv(args.metrics)
    benchmarks = _split_csv(args.benchmarks)
    output_dir = Path(args.output_dir)
    for metric in metrics:
        for benchmark_name in benchmarks:
            output = output_dir / metric / f"{benchmark_name}.json"
            code = benchmark_metric(
                metric,
            benchmark_name,
            str(output),
            data_root=args.data_root,
            use_references=_use_references(metric, no_references=args.no_references),
            limit=args.limit,
        )
            if code != 0:
                return code
            if args.verify:
                expected = Path(args.expected_root) / metric / f"{benchmark_name}.json"
                verify_results(
                    str(output),
                    str(expected),
                    tolerance=args.tolerance,
                    round_decimals=args.round_decimals,
                )
            print(f"ok\t{metric}\t{benchmark_name}\t{output}")
    return 0


def all_reproduce(args: argparse.Namespace) -> int:
    metrics = _split_csv(args.metrics) if args.metrics else list(DEFAULT_REPRO_METRICS)
    benchmarks = _split_csv(args.benchmarks) if args.benchmarks else list(DEFAULT_REPRO_BENCHMARKS)
    pair_filter = default_reproduce_pair if args.metrics is None and args.benchmarks is None else None
    output_dir = Path(args.output_dir)
    expected_root = Path(args.expected_root)
    report_missing = args.show_missing or (args.metrics is not None and args.benchmarks is not None)
    limit = 1 if args.smoke and args.limit is None else args.limit
    code, results = run_all_reproduce(
        metrics=metrics,
        benchmarks=benchmarks,
        data_root=args.data_root,
        output_dir=output_dir,
        expected_root=expected_root,
        tolerance=args.tolerance,
        round_decimals=args.round_decimals,
        limit=limit,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        verbose=args.verbose,
        color=args.color,
        report_missing=report_missing,
        jobs=args.jobs,
        gpu_jobs=args.gpu_jobs,
        pair_filter=pair_filter,
        allow_mismatch=args.smoke,
    )
    summary = Path(args.summary) if args.summary else output_dir / "summary.json"
    write_summary(summary, results)
    print_summary(summary, results, use_color=should_color(args.color))
    return code


def _use_references(metric: str, *, no_references: bool) -> bool:
    return not no_references and metric not in NO_REFERENCE_METRICS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capevalkit")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    list_parser = subparsers.add_parser("list-metrics")
    list_parser.set_defaults(func=list_metrics)

    doctor_parser = subparsers.add_parser("doctor", help="show runtime and tool availability")
    doctor_parser.set_defaults(func=doctor)

    sync_parser = subparsers.add_parser("sync", help="materialize runtime upstream repositories")
    sync_parser.add_argument("--metrics", help="comma-separated metrics whose upstreams should be synced")
    sync_parser.add_argument("--all", action="store_true", help="sync every locked upstream")
    sync_parser.set_defaults(func=sync)

    env_parser = subparsers.add_parser("env-command")
    env_parser.add_argument("--metric", required=True)
    env_parser.add_argument("command", nargs=argparse.REMAINDER)
    env_parser.set_defaults(func=env_command)

    repo_parser = subparsers.add_parser("repo-command")
    repo_parser.add_argument("--metric", required=True)
    repo_parser.add_argument("command", nargs=argparse.REMAINDER)
    repo_parser.set_defaults(func=env_command)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--metric", required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=run)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--metric", required=True)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--references")
    score_parser.add_argument("--image-dir")
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("args", nargs=argparse.REMAINDER)
    score_parser.set_defaults(func=score)

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--metric", required=True)
    bench_parser.add_argument("--benchmark")
    bench_parser.add_argument("--data-root")
    bench_parser.add_argument("--native", action="store_true")
    bench_parser.add_argument("--no-references", action="store_true")
    bench_parser.add_argument("--score-key")
    bench_parser.add_argument("--limit", type=int)
    bench_parser.add_argument("--output")
    bench_parser.add_argument("args", nargs=argparse.REMAINDER)
    bench_parser.set_defaults(func=benchmark)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--results", required=True)
    verify_parser.add_argument("--expected", required=True)
    verify_parser.add_argument("--tolerance", type=float, default=1e-4)
    verify_parser.add_argument("--round-decimals", type=int)
    verify_parser.set_defaults(func=verify)

    download_parser = subparsers.add_parser(
        "download-assets",
        help="download scriptable checkpoints and model assets",
    )
    download_parser.add_argument("assets", nargs="*", help="asset names; omitted means default downloadable assets")
    download_parser.add_argument("--all", action="store_true", help="select every downloadable asset")
    download_parser.add_argument("--list", action="store_true", help="list known downloadable assets")
    download_parser.add_argument("--force", action="store_true", help="overwrite existing downloaded files")
    download_parser.add_argument("--dry-run", action="store_true", help="print selected assets without downloading")
    download_parser.set_defaults(func=download_assets_command)

    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument("--metrics", default="bleu,rouge,meteor,cider,spice,clipscore,pacscore,polos")
    suite_parser.add_argument("--benchmarks", default="composite,flickr8k-ex,flickr8k-cf,nebula,polaris")
    suite_parser.add_argument("--data-root")
    suite_parser.add_argument("--output-dir", default=str(repo_root() / "outputs"))
    suite_parser.add_argument("--expected-root", default=str(repo_root() / "benchmarks" / "expected"))
    suite_parser.add_argument("--verify", action="store_true")
    suite_parser.add_argument("--tolerance", type=float, default=0.2)
    suite_parser.add_argument("--round-decimals", type=int)
    suite_parser.add_argument("--no-references", action="store_true")
    suite_parser.add_argument("--limit", type=int)
    suite_parser.set_defaults(func=suite)

    repro_parser = subparsers.add_parser(
        "all_reproduce",
        aliases=["all-reproduce"],
        help="run every expected metric x benchmark pair and verify reproducibility",
    )
    repro_parser.add_argument(
        "--metrics",
        help=f"default: {','.join(DEFAULT_REPRO_METRICS)}",
    )
    repro_parser.add_argument(
        "--benchmarks",
        help=f"default: {','.join(DEFAULT_REPRO_BENCHMARKS)}; LongCap-Arena default pairs are VELA only",
    )
    repro_parser.add_argument("--data-root")
    repro_parser.add_argument("--output-dir", default=str(repo_root() / "outputs" / "all-reproduce"))
    repro_parser.add_argument("--expected-root", default=str(default_expected_root()))
    repro_parser.add_argument("--summary")
    repro_parser.add_argument("--tolerance", type=float, default=0.2)
    repro_parser.add_argument("--round-decimals", type=int, default=1)
    repro_parser.add_argument("--limit", type=int)
    repro_parser.add_argument("--smoke", action="store_true", help="run one sample per pair and fail only on ERR/SKIP")
    repro_parser.add_argument("--fail-fast", action="store_true")
    repro_parser.add_argument("--jobs", type=int, default=1, help="parallel CPU reproduce jobs")
    repro_parser.add_argument("--gpu-jobs", type=int, default=1, help="parallel GPU reproduce jobs")
    repro_parser.add_argument("--dry-run", action="store_true")
    repro_parser.add_argument("--verbose", action="store_true", help="show metric subprocess logs")
    repro_parser.add_argument("--show-missing", action="store_true", help="show selected pairs without expected files")
    repro_parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    repro_parser.set_defaults(func=all_reproduce)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
