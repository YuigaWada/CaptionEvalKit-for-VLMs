from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.machinery
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import capevalkit.api as api
import capevalkit.benchmarks as benchmarks
import capevalkit.launcher as launcher
from capevalkit.context import ProjectContext, default_context
from capevalkit.benchmarks import DEFAULT_SCORE_KEYS, _kendall, load_benchmark
from capevalkit.cli import _split_csv, _use_references
from capevalkit.compat import zip_strict
from capevalkit.dispatcher import _call_with_progress, _package_import_root, _pythonpath, build_uv_command
from capevalkit.manifests import get_manifest, load_manifests
from capevalkit.metrics.pycocoevalcap_metrics import _per_item_score
from capevalkit.overlays import ensure_overlays
from capevalkit.progress import progress_iter, progress_update
from capevalkit.runtime import RuntimeManager
from capevalkit.runtime_env import apply_runtime_environment
from capevalkit.reproduce import (
    DEFAULT_REPRO_BENCHMARKS,
    DEFAULT_REPRO_METRICS,
    ReproduceJob,
    ReproduceResult,
    ReproduceTask,
    build_reproduce_jobs,
    compact_message,
    default_reproduce_pair,
    expected_tasks,
    format_comparisons,
    missing_job_prerequisite,
    missing_expected_pairs,
    print_result,
    print_results_header,
    run_all_reproduce,
)
from capevalkit.verify import NumericComparison
from capevalkit.verify import verify_results


class ArchitectureTest(unittest.TestCase):
    def test_polos_manifest_uses_repo_uv_project(self) -> None:
        manifest = get_manifest("polos")
        self.assertEqual(manifest.repo_dir, "metrics/upstreams/polos")
        self.assertEqual(manifest.uv_project, "metrics/upstreams/polos")
        self.assertIn("flickr8k-cf", manifest.benchmarks)
        self.assertIn("flickr8k-ex", manifest.benchmarks)
        self.assertIn("polaris", manifest.benchmarks)

    def test_uv_command_points_at_metric_repo_project(self) -> None:
        command = build_uv_command("polos", ["python", "--version"])
        self.assertEqual(command[:3], ["uv", "run", "--project"])
        self.assertTrue(command[3].endswith("/metrics/upstreams/polos"))

    def test_dispatcher_pythonpath_includes_package_parent_and_runtime_root(self) -> None:
        value = _pythonpath("/site-packages", "/runtime", f"/runtime{os.pathsep}/existing")
        self.assertEqual(value.split(os.pathsep), ["/site-packages", "/runtime", "/existing"])

    def test_dispatcher_package_bridge_exposes_only_capevalkit_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_root = root / "site-packages" / "capevalkit"
            package_root.mkdir(parents=True)
            (package_root / "__init__.py").write_text("")
            context = ProjectContext(
                package_root=package_root,
                resource_root=root / "resources",
                project_root=root / "runtime",
                cache_root=root / "cache",
                source_root=None,
                source_mode=False,
                lock_path=root / "lock.json",
                lock_digest="digest",
            )

            bridge = _package_import_root(context)

            self.assertEqual(bridge, root / "runtime" / ".pythonpath")
            self.assertTrue((bridge / "capevalkit").exists())
            self.assertFalse((bridge / "numpy").exists())

    def test_progress_iter_writes_event_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_file = Path(tmp) / "progress.txt"
            with patch.dict(os.environ, {"CAPEVALKIT_PROGRESS_FILE": str(progress_file)}):
                self.assertEqual(list(progress_iter(["a", "b"])), ["a", "b"])
                progress_update(3)

            self.assertEqual(progress_file.read_text().splitlines(), ["1", "1", "3"])

    def test_dispatcher_call_with_progress_consumes_child_events(self) -> None:
        script = (
            "from capevalkit.progress import progress_update\n"
            "progress_update(1)\n"
            "progress_update(2)\n"
        )
        env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                code = _call_with_progress(
                    [sys.executable, "-c", script],
                    cwd=Path(tmp),
                    env=env,
                    total=3,
                    desc="metric/bench",
                )

        self.assertEqual(code, 0)

    def test_runtime_environment_points_model_caches_at_cache_root(self) -> None:
        env: dict[str, str] = {}
        apply_runtime_environment(env, Path("/repo"), cache_root=Path("/cache/capevalkit"))

        self.assertEqual(env["UV_LINK_MODE"], "hardlink")
        self.assertEqual(env["CAPEVALKIT_HOME"], "/cache/capevalkit")
        self.assertEqual(env["UV_CACHE_DIR"], "/cache/capevalkit/uv")
        self.assertEqual(env["CLIP_DOWNLOAD_ROOT"], "/cache/capevalkit/clip")
        self.assertEqual(env["TORCH_HOME"], "/cache/capevalkit/torch")
        self.assertEqual(env["HF_HOME"], "/cache/capevalkit/huggingface")
        self.assertEqual(env["XDG_CACHE_HOME"], "/cache/capevalkit/xdg")

    def test_default_context_can_use_runtime_cache_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(
                os.environ,
                {"CAPEVALKIT_HOME": str(home), "CAPEVALKIT_RUNTIME_MODE": "cache"},
                clear=False,
            ):
                context = default_context()

        self.assertFalse(context.source_mode)
        self.assertTrue(str(context.project_root).startswith(str(home / "runtime")))
        self.assertEqual(context.cache_root, home)

    def test_runtime_manager_materializes_base_resources_without_upstreams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            runtime = root / "runtime"
            lock = root / "lock.json"
            (resources / "metrics" / "m").mkdir(parents=True)
            (resources / "metrics" / "m" / "metric.toml").write_text("[metric]\nname = 'm'\n")
            (resources / "metrics" / "upstreams" / "ignored").mkdir(parents=True)
            (resources / "overlays" / "metrics" / "upstreams" / "u").mkdir(parents=True)
            (resources / "overlays" / "metrics" / "upstreams" / "u" / "uv.toml").write_text("")
            (resources / "benchmarks" / "expected" / "m").mkdir(parents=True)
            (resources / "benchmarks" / "expected" / "m" / "b.json").write_text("{}")
            lock.write_text(json.dumps({"schema_version": 1, "upstreams": {}}))
            context = ProjectContext(
                package_root=root,
                resource_root=resources,
                project_root=runtime,
                cache_root=root / "cache",
                source_root=None,
                source_mode=False,
                lock_path=lock,
                lock_digest="digest",
            )

            RuntimeManager(context).prepare_base()

            self.assertTrue((runtime / "metrics" / "m" / "metric.toml").exists())
            self.assertFalse((runtime / "metrics" / "upstreams" / "ignored").exists())
            self.assertTrue((runtime / "overlays" / "metrics" / "upstreams" / "u" / "uv.toml").exists())
            self.assertTrue((runtime / "benchmarks" / "expected" / "m" / "b.json").exists())

    def test_runtime_manager_clones_locked_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            runtime = root / "runtime"
            lock = root / "lock.json"
            resources.mkdir()
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "upstreams": {
                            "demo": {
                                "source": "git",
                                "url": "https://example.test/demo.git",
                                "rev": "abc123",
                                "path": "metrics/upstreams/demo",
                                "uv_project": "metrics/upstreams/demo",
                                "overlays": [],
                            }
                        },
                    }
                )
            )
            context = ProjectContext(
                package_root=root,
                resource_root=resources,
                project_root=runtime,
                cache_root=root / "cache",
                source_root=None,
                source_mode=False,
                lock_path=lock,
                lock_digest="digest",
            )
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if command[:2] == ["git", "clone"]:
                    checkout = Path(command[-1])
                    (checkout / ".git").mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0, stdout="")

            with patch("capevalkit.runtime.subprocess.run", side_effect=fake_run):
                RuntimeManager(context).ensure_upstream("demo")

            self.assertEqual(commands[0], ["git", "clone", "https://example.test/demo.git", str(runtime / "metrics" / "upstreams" / "demo")])
            self.assertIn(["git", "checkout", "abc123"], commands)

    def test_runtime_manager_initializes_empty_source_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            checkout = root / "checkout"
            upstream = checkout / "metrics" / "upstreams" / "demo"
            lock = root / "lock.json"
            upstream.mkdir(parents=True)
            resources.mkdir()
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "upstreams": {
                            "demo": {
                                "source": "git",
                                "url": "https://example.test/demo.git",
                                "rev": "abc123",
                                "path": "metrics/upstreams/demo",
                                "uv_project": "metrics/upstreams/demo",
                                "overlays": [],
                            }
                        },
                    }
                )
            )
            context = ProjectContext(
                package_root=root,
                resource_root=resources,
                project_root=checkout,
                cache_root=root / "cache",
                source_root=checkout,
                source_mode=True,
                lock_path=lock,
                lock_digest="digest",
            )
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if command[:3] == ["git", "submodule", "update"]:
                    (upstream / ".git").write_text("gitdir: ../../../.git/modules/demo\n")
                return subprocess.CompletedProcess(command, 0, stdout="")

            with patch("capevalkit.runtime.subprocess.run", side_effect=fake_run):
                result = RuntimeManager(context).ensure_upstream("demo")

            self.assertEqual(result, upstream)
            self.assertIn(
                ["git", "submodule", "update", "--init", "--recursive", "metrics/upstreams/demo"],
                commands,
            )

    def test_launcher_preserves_invocation_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                with patch("capevalkit.launcher.configure_environment", return_value=Path(tmp)):
                    with patch("capevalkit.launcher.cli.main", return_value=0):
                        with contextlib.chdir(tmp):
                            launcher.main([])
                            self.assertEqual(Path.cwd(), Path(tmp))
            finally:
                if Path.cwd() != cwd:
                    raise AssertionError("launcher test leaked cwd")

    def test_launcher_does_not_apply_overlays_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("capevalkit.launcher.configure_environment", return_value=Path(tmp)):
                with patch("capevalkit.overlays.ensure_overlays") as overlays:
                    with patch("capevalkit.launcher.cli.main", return_value=0):
                        launcher.main([])

        overlays.assert_not_called()

    def test_runtime_manager_removes_overlay_only_stub_before_submodule_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            checkout = root / "checkout"
            upstream = checkout / "metrics" / "upstreams" / "demo"
            overlay_file = upstream / "pyproject.toml"
            overlay_source = checkout / "overlays" / "metrics" / "upstreams" / "demo" / "pyproject.toml"
            lock = root / "lock.json"
            overlay_file.parent.mkdir(parents=True)
            overlay_file.write_text("[project]\nname='demo'\n")
            overlay_source.parent.mkdir(parents=True)
            overlay_source.write_text("[project]\nname='demo'\n")
            resources.mkdir()
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "upstreams": {
                            "demo": {
                                "source": "git",
                                "url": "https://example.test/demo.git",
                                "rev": "abc123",
                                "path": "metrics/upstreams/demo",
                                "uv_project": "metrics/upstreams/demo",
                                "overlays": ["overlays/metrics/upstreams/demo/pyproject.toml"],
                            }
                        },
                    }
                )
            )
            context = ProjectContext(
                package_root=root,
                resource_root=resources,
                project_root=checkout,
                cache_root=root / "cache",
                source_root=checkout,
                source_mode=True,
                lock_path=lock,
                lock_digest="digest",
            )

            def fake_run(command, **kwargs):
                if command[:3] == ["git", "submodule", "update"]:
                    upstream.mkdir(parents=True, exist_ok=True)
                    (upstream / ".git").write_text("gitdir: ../../../.git/modules/demo\n")
                    (upstream / "demo.py").write_text("")
                return subprocess.CompletedProcess(command, 0, stdout="")

            with patch("capevalkit.runtime.subprocess.run", side_effect=fake_run):
                result = RuntimeManager(context).ensure_upstream("demo")

            self.assertEqual(result, upstream)
            self.assertTrue((upstream / ".git").exists())
            self.assertTrue((upstream / "pyproject.toml").exists())

    def test_zip_strict_is_python_39_compatible(self) -> None:
        self.assertEqual(list(zip_strict([1, 2], ["a", "b"])), [(1, "a"), (2, "b")])
        with self.assertRaises(ValueError):
            list(zip_strict([1], ["a", "b"]))

    def test_cli_reference_default_respects_metric_variant(self) -> None:
        self.assertFalse(_use_references("fleur", no_references=False))
        self.assertTrue(_use_references("reffleur", no_references=False))
        self.assertTrue(_use_references("vela", no_references=False))
        self.assertFalse(_use_references("vela", no_references=True))

    def test_evaluate_metric_calls_python_metric_with_samples(self) -> None:
        items = [
            benchmarks.BenchmarkItem("a", "a.jpg", "candidate a", ["ref a"], 0.1),
            benchmarks.BenchmarkItem("b", "b.jpg", "candidate b", ["ref b"], 0.9),
        ]
        seen_predictions = []

        def metric(samples: list[api.CaptionSample]) -> dict[str, float]:
            seen_predictions.extend(sample.prediction for sample in samples)
            return {sample.id: float(index) for index, sample in enumerate(samples)}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metric.json"
            with patch("capevalkit.api.load_benchmark", return_value=items):
                result = api.evaluate_metric(
                    benchmark="bench",
                    metric=metric,
                    metric_name="MyMetric",
                    predictions={"a": "override a", "b": "override b"},
                    output=output,
                )
            wrote_output = output.exists()
            written = json.loads(output.read_text())

        self.assertEqual(seen_predictions, ["override a", "override b"])
        self.assertEqual(result["metric"], "MyMetric")
        self.assertEqual(result["benchmark"], "bench")
        self.assertEqual(result["score_name"], "MyMetric")
        self.assertEqual(result["num_samples"], 2)
        item = result["raw_metric_output"]["MyMetric"]["per_item"]["a"]
        self.assertEqual(item["score"], 0.0)
        self.assertEqual(item["ground_truth_score"], 0.1)
        self.assertEqual(item["caption"], "override a")
        self.assertEqual(item["image"], "a.jpg")
        self.assertEqual(item["references"], ["ref a"])
        self.assertEqual(written, result)
        self.assertTrue(wrote_output)

    def test_evaluate_caption_model_records_batches_and_runs_metrics(self) -> None:
        reference_flags: dict[str, bool] = {}

        def fake_dispatch(metric_name: str, command: list[str], *, quiet: bool = False) -> int:
            reference_flags[metric_name] = "--references" in command
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        metric_name: {
                            "score": 1.0,
                            "per_item": {"a": 1.0, "b": 1.0},
                        }
                    }
                )
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch("capevalkit.api.dispatch", side_effect=fake_dispatch):
                results = api.evaluate_caption_model(
                    images=["a.jpg", "b.jpg"],
                    ids=["a", "b"],
                    references=[["ref a"], ["ref b"]],
                    metrics=["cider", "clipscore"],
                    predict=lambda batch: [f"caption {item_id}" for item_id in batch.ids],
                    output_dir=output_dir,
                    batch_size=1,
                )

            rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]

        self.assertEqual([row["caption"] for row in rows], ["caption a", "caption b"])
        self.assertTrue(reference_flags["cider"])
        self.assertFalse(reference_flags["clipscore"])
        self.assertIn("cider", results)
        self.assertIn("clipscore", results)

    def test_evaluate_captions_accepts_image_caption_pairs(self) -> None:
        seen_predictions = []

        def fake_dispatch(metric_name: str, command: list[str], *, quiet: bool = False) -> int:
            predictions = Path(command[command.index("--predictions") + 1])
            seen_predictions.extend(json.loads(line)["caption"] for line in predictions.read_text().splitlines())
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        metric_name: {
                            "score": 0.5,
                            "per_item": {"a": 0.5, "b": 0.5},
                        }
                    }
                )
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            with patch("capevalkit.api.dispatch", side_effect=fake_dispatch):
                results = api.evaluate_captions(
                    pairs=[
                        {"id": "a", "image": "a.jpg", "caption": "caption a", "references": ["ref a"]},
                        {"id": "b", "image": "b.jpg", "caption": "caption b", "references": ["ref b"]},
                    ],
                    metrics=["cider"],
                    output_dir=tmp,
                )

        self.assertEqual(seen_predictions, ["caption a", "caption b"])
        self.assertIn("cider", results)

    def test_ensure_overlays_copies_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "overlays" / "metric" / "uv.toml"
            target = root / "metric" / "uv.toml"
            source.parent.mkdir(parents=True)
            source.write_text('link-mode = "hardlink"\n')

            first = ensure_overlays(root=root)
            self.assertEqual(target.read_text(), source.read_text())
            self.assertTrue(first[0].changed)

            second = ensure_overlays(root=root)
            self.assertFalse(second[0].changed)

    def test_ensure_overlays_can_apply_only_selected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "overlays" / "metrics" / "upstreams" / "selected" / "uv.toml"
            skipped = root / "overlays" / "metrics" / "upstreams" / "skipped" / "uv.toml"
            selected.parent.mkdir(parents=True)
            skipped.parent.mkdir(parents=True)
            selected.write_text('link-mode = "hardlink"\n')
            skipped.write_text('link-mode = "hardlink"\n')

            actions = ensure_overlays(
                root=root,
                overlays=["overlays/metrics/upstreams/selected/uv.toml"],
            )

            self.assertEqual(len(actions), 1)
            self.assertTrue((root / "metrics" / "upstreams" / "selected" / "uv.toml").exists())
            self.assertFalse((root / "metrics" / "upstreams" / "skipped" / "uv.toml").exists())

    def test_python310_upstream_overlays_include_tomli(self) -> None:
        upstreams = ["pycocoevalcap", "polos", "fleur", "vela"]
        for upstream in upstreams:
            path = Path("overlays") / "metrics" / "upstreams" / upstream / "pyproject.toml"
            dependencies = tomllib.loads(path.read_text())["project"]["dependencies"]
            self.assertTrue(
                any(dep.startswith("tomli>=2") for dep in dependencies),
                f"{path} must include tomli for Python < 3.11 metric runners",
            )

    def test_classic_metrics_share_pycocoevalcap_repo(self) -> None:
        manifests = load_manifests()
        for metric in ["bleu", "rouge", "meteor", "cider", "spice"]:
            self.assertIn(metric, manifests)
            self.assertEqual(manifests[metric].repo_dir, "metrics/upstreams/pycocoevalcap")

    def test_requested_heavy_metrics_are_registered(self) -> None:
        manifests = load_manifests()
        expected_repos = {
            "clipscore": "metrics/upstreams/clipscore",
            "clipscore-vitl": "metrics/upstreams/clipscore",
            "clipscoreavg": "metrics/upstreams/clipscore",
            "refclipscore": "metrics/upstreams/clipscore",
            "refclipscore-vitl": "metrics/upstreams/clipscore",
            "pacscore": "metrics/upstreams/pacscore",
            "pacscore-vitl": "metrics/upstreams/pacscore",
            "pacscoreavg": "metrics/upstreams/pacscore",
            "refpacscore": "metrics/upstreams/pacscore",
            "refpacscore-vitl": "metrics/upstreams/pacscore",
            "pacscorepp": "metrics/upstreams/pacscore",
            "pacscoreppavg": "metrics/upstreams/pacscore",
            "refpacscorepp": "metrics/upstreams/pacscore",
            "polos": "metrics/upstreams/polos",
            "fleur": "metrics/upstreams/fleur",
            "reffleur": "metrics/upstreams/fleur",
            "vela": "metrics/upstreams/vela",
        }
        for metric, repo in expected_repos.items():
            self.assertEqual(manifests[metric].repo_dir, repo)

    def test_verify_compares_nested_numeric_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.json"
            expected = root / "expected.json"
            payload = {"flickr": {"Polos": {"Kendall": 72.3456}}}
            results.write_text(json.dumps(payload))
            expected.write_text(json.dumps(payload))
            verify_results(str(results), str(expected), tolerance=1e-4)

    def test_verify_can_compare_rounded_paper_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.json"
            expected = root / "expected.json"
            results.write_text(json.dumps({"correlations": {"kendall_tau_c": 30.6408}}))
            expected.write_text(json.dumps({"correlations": {"kendall_tau_c": 30.6}}))
            verify_results(str(results), str(expected), round_decimals=1)

    def test_verify_accepts_rounded_values_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.json"
            expected = root / "expected.json"
            results.write_text(json.dumps({"correlations": {"kendall_tau_c": 46.24}}))
            expected.write_text(json.dumps({"correlations": {"kendall_tau_c": 46.3}}))
            verify_results(str(results), str(expected), tolerance=0.15, round_decimals=1)

    def test_loads_composite_benchmark_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "composite"
            (root / "images").mkdir(parents=True)
            (root / "en_test_composite_da2.csv").write_text(
                "imgid,mt,refs,score\n"
                "a.jpg,a caption,\"['one ref', 'two ref']\",3.5\n"
            )
            items = load_benchmark("composite", str(tmp))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].caption, "a caption")
            self.assertEqual(items[0].references, ["one ref", "two ref"])

    def test_loads_composite_from_hf_for_repo_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = benchmarks.BenchmarkItem(
                id="row0",
                image="image.jpg",
                caption="caption",
                references=["ref"],
                score=1.0,
            )
            with patch.object(benchmarks, "repo_root", return_value=root):
                with patch.object(benchmarks, "_load_hf_composite", return_value=[expected]) as loader:
                    items = load_benchmark("composite", str(root / "data"))
        self.assertEqual(items, [expected])
        loader.assert_called_once_with(limit=None)

    def test_loads_flickr_from_hf_for_repo_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = benchmarks.BenchmarkItem(
                id="row0",
                image="image.jpg",
                caption="caption",
                references=["ref"],
                score=1.0,
            )
            with patch.object(benchmarks, "repo_root", return_value=root):
                with patch.object(benchmarks, "_load_hf_flickr8k", return_value=[expected]) as loader:
                    items = load_benchmark("flickr8k-cf", str(root / "data"))
        self.assertEqual(items, [expected])
        loader.assert_called_once_with("cf", limit=None)

    def test_loads_legacy_ex_alias_from_hf_for_repo_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = benchmarks.BenchmarkItem(
                id="row0",
                image="image.jpg",
                caption="caption",
                references=["ref"],
                score=1.0,
            )
            with patch.object(benchmarks, "repo_root", return_value=root):
                with patch.object(benchmarks, "_load_hf_flickr8k", return_value=[expected]) as loader:
                    items = load_benchmark("ex", str(root / "data"))
        self.assertEqual(items, [expected])
        loader.assert_called_once_with("expert", limit=None)

    def test_load_benchmark_forwards_limit_to_hf_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = benchmarks.BenchmarkItem(
                id="row0",
                image="image.jpg",
                caption="caption",
                references=["ref"],
                score=1.0,
            )
            with patch.object(benchmarks, "repo_root", return_value=root):
                with patch.object(benchmarks, "_load_hf_composite", return_value=[expected]) as loader:
                    items = load_benchmark("composite", str(root / "data"), limit=1)
        self.assertEqual(items, [expected])
        loader.assert_called_once_with(limit=1)

    def test_hf_embedded_row_to_item_writes_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            item = benchmarks._hf_embedded_row_to_item(
                {
                    "id": "sample",
                    "imgid": "a.jpg",
                    "img": {"bytes": b"image-bytes", "path": None},
                    "refs": [" one ref "],
                    "cand": " a caption ",
                    "human_score": 0.5,
                },
                image_dir=image_dir,
                row_index=0,
            )

            self.assertEqual(item.id, "sample")
            self.assertEqual(item.image, str(image_dir / "a.jpg"))
            self.assertEqual(item.caption, "a caption")
            self.assertEqual(item.references, ["one ref"])
            self.assertEqual(item.score, 0.5)
            self.assertEqual((image_dir / "a.jpg").read_bytes(), b"image-bytes")

    def test_loads_flickr_cf_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flickr8k"
            (root / "images").mkdir(parents=True)
            (root / "crowdflower_flickr8k.json").write_text(
                json.dumps(
                    {
                        "img1": {
                            "image_path": "images/img1.jpg",
                            "ground_truth": ["ref"],
                            "human_judgement": [{"caption": "cand", "rating": 4.0}],
                        }
                    }
                )
            )
            items = load_benchmark("flickr8k-cf", str(tmp))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].score, 4.0)

    def test_loads_nebula_benchmark_csv_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nebula"
            root.mkdir(parents=True)
            (root / "nebula_test.csv").write_text(
                "imgid,mt,refs,score\n"
                "hash_id,a caption,\"['one ref', 'two ref']\",0.75\n"
            )
            items = load_benchmark("nebula", str(tmp))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image, str(root / "images" / "hash_id"))

    def test_nebula_score_correction_uses_spica_average_by_image_and_caption(self) -> None:
        item = benchmarks.BenchmarkItem(
            id="sample",
            image="/tmp/images/hash_id.jpg",
            caption="a   caption",
            references=["ref"],
            score=0.0,
        )
        with patch.object(
            benchmarks,
            "_load_spica_score_lookup",
            return_value={"hash_id\ta caption": 0.75},
        ):
            corrected = benchmarks._apply_nebula_score_corrections([item])
        self.assertEqual(corrected[0].score, 0.75)

    def test_hf_nebula_row_writes_embedded_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = benchmarks._hf_nebula_row_to_item(
                {
                    "file_name": "hash_id.jpg",
                    "image": {"bytes": b"image-bytes"},
                    "refs": ["one ref", "two ref"],
                    "mt": "a caption",
                    "human_score": 0.75,
                },
                "test",
                0,
                Path(tmp),
            )

            self.assertEqual(item.image, str(Path(tmp) / "hash_id.jpg"))
            self.assertEqual((Path(tmp) / "hash_id.jpg").read_bytes(), b"image-bytes")
            self.assertEqual(item.references, ["one ref", "two ref"])
            self.assertEqual(item.score, 0.75)

    def test_hf_image_cache_rejects_stale_local_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "nebula-test.jsonl"
            stale_image = root / "data" / "nebula" / "images" / "hash_id"
            stale_image.parent.mkdir(parents=True)
            stale_image.write_bytes(b"image-bytes")
            cache_path.write_text(
                json.dumps(
                    benchmarks.BenchmarkItem(
                        id="sample",
                        image=str(stale_image),
                        caption="a caption",
                        references=["ref"],
                        score=1.0,
                    ).__dict__
                )
                + "\n"
            )

            self.assertFalse(benchmarks._hf_image_cache_is_current(cache_path, root / "nebula-images"))

    def test_hf_image_cache_accepts_cached_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "nebula-test.jsonl"
            cached_image = root / "nebula-images" / "test" / "hash_id"
            cached_image.parent.mkdir(parents=True)
            cached_image.write_bytes(b"image-bytes")
            cache_path.write_text(
                json.dumps(
                    benchmarks.BenchmarkItem(
                        id="sample",
                        image=str(cached_image),
                        caption="a caption",
                        references=["ref"],
                        score=1.0,
                    ).__dict__
                )
                + "\n"
            )

            self.assertTrue(benchmarks._hf_image_cache_is_current(cache_path, root / "nebula-images"))

    def test_hf_cache_temp_paths_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "nebula-test.jsonl"
            self.assertNotEqual(
                benchmarks._cache_tmp_path(cache_path),
                benchmarks._cache_tmp_path(cache_path),
            )

    def test_hf_image_column_cache_writes_parquet_images(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "0000.parquet"
            table = pa.table(
                {
                    "file_name": pa.array(["hash_id.jpg"]),
                    "image": pa.array(
                        [{"bytes": b"image-bytes", "path": None}],
                        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                    ),
                    "refs": pa.array([["one ref"]], type=pa.list_(pa.string())),
                    "mt": pa.array(["a caption"]),
                    "human_score": pa.array([0.75]),
                }
            )
            pq.write_table(table, parquet_path)

            class FakeFs:
                def open(self, _: str, mode: str):
                    return parquet_path.open(mode)

            cache_path = root / "nebula-test.jsonl"
            image_dir = root / "nebula-images"
            with patch("capevalkit.benchmarks._hf_parquet_paths", return_value=["default/test/0000.parquet"]):
                with patch("fsspec.filesystem", return_value=FakeFs()):
                    benchmarks._write_hf_image_column_cache(
                        repo_id="org/nebula",
                        splits=("test",),
                        cache_path=cache_path,
                        image_dir=image_dir,
                        columns=["file_name", "image", "refs", "mt", "human_score"],
                        image_column="image",
                        row_to_item=benchmarks._hf_nebula_row_to_item,
                    )

            items = benchmarks._read_cached_items(cache_path)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image, str(image_dir / "test" / "hash_id.jpg"))
            self.assertEqual((image_dir / "test" / "hash_id.jpg").read_bytes(), b"image-bytes")

    def test_hf_image_column_cache_respects_limit_before_writing_images(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "0000.parquet"
            table = pa.table(
                {
                    "file_name": pa.array(["first.jpg", "second.jpg"]),
                    "image": pa.array(
                        [
                            {"bytes": b"first-image", "path": None},
                            {"bytes": b"second-image", "path": None},
                        ],
                        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                    ),
                    "refs": pa.array([["one ref"], ["two ref"]], type=pa.list_(pa.string())),
                    "mt": pa.array(["first caption", "second caption"]),
                    "human_score": pa.array([0.75, 0.25]),
                }
            )
            pq.write_table(table, parquet_path)

            class FakeFs:
                def open(self, _: str, mode: str):
                    return parquet_path.open(mode)

            cache_path = root / "nebula-test-limit1.jsonl"
            image_dir = root / "nebula-images-limit1"
            with patch("capevalkit.benchmarks._hf_parquet_paths", return_value=["default/test/0000.parquet"]):
                with patch("fsspec.filesystem", return_value=FakeFs()):
                    benchmarks._write_hf_image_column_cache(
                        repo_id="org/nebula",
                        splits=("test",),
                        cache_path=cache_path,
                        image_dir=image_dir,
                        columns=["file_name", "image", "refs", "mt", "human_score"],
                        image_column="image",
                        row_to_item=benchmarks._hf_nebula_row_to_item,
                        limit=1,
                    )

            items = benchmarks._read_cached_items(cache_path)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image, str(image_dir / "test" / "first.jpg"))
            self.assertEqual((image_dir / "test" / "first.jpg").read_bytes(), b"first-image")
            self.assertFalse((image_dir / "test" / "second.jpg").exists())

    def test_hf_image_column_cache_retries_incomplete_parquet_read(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "0000.parquet"
            table = pa.table(
                {
                    "file_name": pa.array(["hash_id.jpg"]),
                    "image": pa.array(
                        [{"bytes": b"image-bytes", "path": None}],
                        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                    ),
                    "refs": pa.array([["one ref"]], type=pa.list_(pa.string())),
                    "mt": pa.array(["a caption"]),
                    "human_score": pa.array([0.75]),
                }
            )
            pq.write_table(table, parquet_path)

            class FlakyFs:
                attempts = 0

                def open(self, _: str, mode: str):
                    self.attempts += 1
                    if self.attempts == 1:
                        raise OSError("not enough data to satisfy content length header")
                    return parquet_path.open(mode)

            fake_fs = FlakyFs()
            cache_path = root / "nebula-test.jsonl"
            image_dir = root / "nebula-images"
            with patch("capevalkit.benchmarks._hf_parquet_paths", return_value=["default/test/0000.parquet"]):
                with patch("fsspec.filesystem", return_value=fake_fs):
                    with patch("capevalkit.benchmarks._hf_read_retries", return_value=2):
                        with patch("capevalkit.benchmarks._hf_retry_delay_seconds", return_value=0.0):
                            benchmarks._write_hf_image_column_cache(
                                repo_id="org/nebula",
                                splits=("test",),
                                cache_path=cache_path,
                                image_dir=image_dir,
                                columns=["file_name", "image", "refs", "mt", "human_score"],
                                image_column="image",
                                row_to_item=benchmarks._hf_nebula_row_to_item,
                            )

            items = benchmarks._read_cached_items(cache_path)
            self.assertEqual(fake_fs.attempts, 2)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image, str(image_dir / "test" / "hash_id.jpg"))

    def test_hf_cache_creation_is_serialized_and_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fake_writer(**kwargs) -> None:
                calls.append(time.monotonic())
                time.sleep(0.05)
                item = benchmarks.BenchmarkItem(
                    id="sample",
                    image="image.jpg",
                    caption="caption",
                    references=["ref"],
                    score=1.0,
                )
                kwargs["cache_path"].write_text(json.dumps(item.__dict__) + "\n")

            with patch.object(benchmarks, "HF_BENCHMARK_CACHE", root):
                with patch.object(benchmarks, "_write_hf_cache", side_effect=fake_writer):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [
                            executor.submit(
                                benchmarks._load_hf_cached_benchmark,
                                cache_name="sample",
                                repo_id="org/sample",
                                splits=("test",),
                                columns=["caption"],
                                row_to_item=lambda row, split, index: row,
                            )
                            for _ in range(2)
                        ]
                        results = [future.result() for future in futures]

            self.assertEqual(len(calls), 1)
            self.assertEqual([len(result) for result in results], [1, 1])

    def test_clipscore_loader_adds_pycocoevalcap_upstream_to_import_path(self) -> None:
        from capevalkit.metrics import clipscore_metric

        old_path = list(sys.path)

        class Loader:
            def create_module(self, spec):
                return None

            def exec_module(self, module) -> None:
                module.loaded = True

        spec = importlib.machinery.ModuleSpec("official_clipscore", Loader())
        pycoco_path = str(Path("metrics/upstreams").resolve())
        try:
            with patch("importlib.util.spec_from_file_location", return_value=spec):
                module = clipscore_metric._load_official_module()
            self.assertTrue(module.loaded)
            self.assertIn(pycoco_path, sys.path)
        finally:
            sys.path[:] = old_path

    def test_loads_polaris_benchmark_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "polaris"
            root.mkdir(parents=True)
            (root / "polaris_test.csv").write_text(
                "imgid,mt,refs,score\n"
                "hash_id,a caption,\"['one ref', 'two ref']\",0.5\n"
            )
            items = load_benchmark("polaris", str(tmp))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].score, 0.5)

    def test_hf_polaris_row_writes_embedded_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = benchmarks._hf_polaris_row_to_item(
                {
                    "img": {"bytes": b"image-bytes", "path": "row_image.jpg"},
                    "refs": ["one ref", "two ref"],
                    "cand": "a caption",
                    "human_score": 0.5,
                },
                "test",
                3,
                Path(tmp),
            )

            self.assertEqual(item.image, str(Path(tmp) / "row_image.jpg"))
            self.assertEqual((Path(tmp) / "row_image.jpg").read_bytes(), b"image-bytes")
            self.assertEqual(item.caption, "a caption")
            self.assertEqual(item.score, 0.5)

    def test_loads_longcaparena_local_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "longcaparena"
            (root / "test").mkdir(parents=True)
            (root / "images").mkdir()
            (root / "test" / "desc_dci_val.csv").write_text(
                "imgid,cand,score,refs\n"
                "sa_1.jpg,a long caption,0.8,\"['one long ref']\"\n"
            )
            items = load_benchmark("longcaparena-testa-desc", str(tmp))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].caption, "a long caption")
            self.assertEqual(items[0].references, ["one long ref"])
            self.assertEqual(items[0].image, str(root / "images" / "sa_1.jpg"))

    def test_bleu_benchmark_defaults_to_bleu_4(self) -> None:
        self.assertEqual(DEFAULT_SCORE_KEYS["bleu"], "BLEU-4")

    def test_longcaparena_cli_alias_expands_to_six_benchmarks(self) -> None:
        self.assertEqual(
            _split_csv("longcaparena"),
            [
                "longcaparena-testa-desc",
                "longcaparena-testa-rel",
                "longcaparena-testa-flu",
                "longcaparena-testb-desc",
                "longcaparena-testb-rel",
                "longcaparena-testb-flu",
            ],
        )

    def test_kendall_matches_bruteforce_with_ties(self) -> None:
        values = [1.0, 2.0, 2.0, 4.0, 3.0, 3.0]
        targets = [1.0, 3.0, 2.0, 2.0, 3.0, 4.0]
        expected = self._kendall_bruteforce(values, targets)
        actual = _kendall(values, targets)
        self.assertAlmostEqual(actual["kendall_tau_b"], expected["kendall_tau_b"])
        self.assertAlmostEqual(actual["kendall_tau_c"], expected["kendall_tau_c"])

    def test_benchmark_result_enriches_root_per_item_scores(self) -> None:
        items = [
            benchmarks.BenchmarkItem("a", "a.jpg", "candidate a", ["ref a"], 0.1),
            benchmarks.BenchmarkItem("b", "b.jpg", "candidate b", ["ref b"], 0.9),
        ]
        result = benchmarks.benchmark_result(
            "refclipscore",
            "bench",
            items=items,
            metric_output={
                "CLIPScore": 1.0,
                "RefCLIPScore": 2.0,
                "per_item": {
                    "a": {"CLIPScore": 0.3, "RefCLIPScore": 0.8},
                    "b": {"CLIPScore": 0.4, "RefCLIPScore": 0.2},
                },
            },
        )

        item = result["raw_metric_output"]["per_item"]["a"]
        self.assertEqual(result["score_name"], "RefCLIPScore")
        self.assertEqual(item["score"], 0.8)
        self.assertEqual(item["scores"], {"CLIPScore": 0.3, "RefCLIPScore": 0.8})
        self.assertEqual(item["ground_truth_score"], 0.1)
        self.assertEqual(item["caption"], "candidate a")

    @staticmethod
    def _kendall_bruteforce(values: list[float], targets: list[float]) -> dict[str, float]:
        from math import sqrt

        concordant = discordant = tie_x = tie_y = 0
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                dx = (values[i] > values[j]) - (values[i] < values[j])
                dy = (targets[i] > targets[j]) - (targets[i] < targets[j])
                if dx == 0:
                    tie_x += 1
                if dy == 0:
                    tie_y += 1
                if dx and dy:
                    if dx == dy:
                        concordant += 1
                    else:
                        discordant += 1
        pairs = n * (n - 1) / 2
        numerator = concordant - discordant
        distinct = min(len(set(values)), len(set(targets)))
        return {
            "kendall_tau_b": 100 * numerator / sqrt((pairs - tie_x) * (pairs - tie_y)),
            "kendall_tau_c": 100 * numerator / (n * n * (distinct - 1) / (2 * distinct)),
        }

    def test_spice_per_item_score_extracts_all_f(self) -> None:
        self.assertEqual(_per_item_score({"All": {"f": "0.25"}}), 0.25)

    def test_table_1_expected_values_exist(self) -> None:
        root = Path("benchmarks/expected")
        metrics = ["bleu", "rouge", "meteor", "cider", "spice", "clipscore", "pacscore", "polos"]
        benchmarks = ["composite", "flickr8k-ex", "flickr8k-cf", "nebula"]
        for metric in metrics:
            for benchmark in benchmarks:
                path = root / metric / f"{benchmark}.json"
                self.assertTrue(path.exists(), str(path))
                payload = json.loads(path.read_text())
                self.assertIn("kendall_tau_b", payload["correlations"])
                self.assertIn("kendall_tau_c", payload["correlations"])

    def test_completed_metric_polaris_expected_values_exist(self) -> None:
        for metric in ["bleu", "rouge", "meteor", "cider", "spice", "clipscore", "pacscore", "polos"]:
            path = Path("benchmarks/expected") / metric / "polaris.json"
            payload = json.loads(path.read_text())
            self.assertIn("kendall_tau_c", payload["correlations"])

    def test_vela_longcaparena_expected_values_exist(self) -> None:
        for benchmark in _split_csv("longcaparena"):
            path = Path("benchmarks/expected") / "vela" / f"{benchmark}.json"
            payload = json.loads(path.read_text())
            self.assertIn("kendall_tau_c", payload["correlations"])

    def test_all_reproduce_discovers_expected_tasks(self) -> None:
        root = Path("benchmarks/expected")
        tasks = expected_tasks(
            expected_root=root,
            output_dir=Path("outputs/all-reproduce"),
            metrics=["clipscore", "bleu"],
            benchmarks=["composite", "polaris"],
        )
        self.assertEqual(
            [(task.metric, task.benchmark) for task in tasks],
            [
                ("clipscore", "composite"),
                ("clipscore", "polaris"),
                ("bleu", "composite"),
                ("bleu", "polaris"),
            ],
        )

    def test_all_reproduce_normalizes_legacy_ex_alias(self) -> None:
        tasks = expected_tasks(
            expected_root=Path("benchmarks/expected"),
            output_dir=Path("outputs/all-reproduce"),
            metrics=["pacscore"],
            benchmarks=["ex"],
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].benchmark, "flickr8k-ex")
        self.assertEqual(tasks[0].expected, Path("benchmarks/expected/pacscore/flickr8k-ex.json"))
        self.assertEqual(tasks[0].output, Path("outputs/all-reproduce/pacscore/flickr8k-ex.json"))

    def test_all_reproduce_groups_pycocoevalcap_metrics_by_benchmark(self) -> None:
        tasks = expected_tasks(
            expected_root=Path("benchmarks/expected"),
            output_dir=Path("outputs/all-reproduce"),
            metrics=["bleu", "rouge", "clipscore"],
            benchmarks=["composite"],
        )
        jobs = build_reproduce_jobs(tasks)

        self.assertEqual(
            [(task.metric, task.benchmark) for task in jobs[0].tasks],
            [("bleu", "composite"), ("rouge", "composite")],
        )
        self.assertEqual(jobs[0].runner_metric, "bleu")
        self.assertEqual(jobs[0].metric_args, ("--metrics", "bleu,rouge"))
        self.assertEqual(jobs[0].resource, "cpu")

    def test_all_reproduce_groups_clipscore_with_reference_variant(self) -> None:
        tasks = expected_tasks(
            expected_root=Path("benchmarks/expected"),
            output_dir=Path("outputs/all-reproduce"),
            metrics=["clipscore", "refclipscore"],
            benchmarks=["composite"],
        )
        jobs = build_reproduce_jobs(tasks)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            [(task.metric, task.benchmark) for task in jobs[0].tasks],
            [("clipscore", "composite"), ("refclipscore", "composite")],
        )
        self.assertEqual(jobs[0].runner_metric, "refclipscore")
        self.assertTrue(jobs[0].use_references)
        self.assertEqual(jobs[0].resource, "gpu")

    def test_all_reproduce_groups_pacscore_with_reference_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["pacscorepp", "refpacscorepp"]:
                expected = root / "expected" / metric / "composite.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            tasks = expected_tasks(
                expected_root=root / "expected",
                output_dir=root / "outputs",
                metrics=["pacscorepp", "refpacscorepp"],
                benchmarks=["composite"],
            )
        jobs = build_reproduce_jobs(tasks)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            [(task.metric, task.benchmark) for task in jobs[0].tasks],
            [("pacscorepp", "composite"), ("refpacscorepp", "composite")],
        )
        self.assertEqual(jobs[0].runner_metric, "refpacscorepp")
        self.assertTrue(jobs[0].use_references)
        self.assertEqual(jobs[0].resource, "gpu")

    def test_all_reproduce_marks_large_model_metrics_as_exclusive_gpu(self) -> None:
        tasks = expected_tasks(
            expected_root=Path("benchmarks/expected"),
            output_dir=Path("outputs/all-reproduce"),
            metrics=["fleur", "reffleur", "vela"],
            benchmarks=["composite", "longcaparena-testa-desc"],
        )
        jobs = build_reproduce_jobs(tasks)
        resources = {(job.runner_metric, job.benchmark): job.resource for job in jobs}

        self.assertEqual(resources[("fleur", "composite")], "exclusive-gpu")
        self.assertEqual(resources[("reffleur", "composite")], "exclusive-gpu")
        self.assertEqual(resources[("vela", "longcaparena-testa-desc")], "exclusive-gpu")

    def test_all_reproduce_runs_grouped_classic_metrics_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["bleu", "rouge"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch(
                        "capevalkit.reproduce.run_metric_on_benchmark",
                        return_value=(0, [], {}),
                    )
                )
                write_result = stack.enter_context(
                    patch("capevalkit.reproduce.write_benchmark_result")
                )
                stack.enter_context(
                    patch("capevalkit.reproduce.compare_results", return_value=[])
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["bleu", "rouge"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                    jobs=2,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        run_metric.assert_called_once()
        self.assertEqual(run_metric.call_args.args[:2], ("bleu", "bench"))
        self.assertEqual(
            run_metric.call_args.kwargs["metric_args"],
            ["--metrics", "bleu,rouge"],
        )
        self.assertFalse(run_metric.call_args.kwargs["show_progress"])
        self.assertEqual(write_result.call_count, 2)

    def test_all_reproduce_uses_rich_for_current_grouped_job(self) -> None:
        progress_events = []

        class FakeProgress:
            def __init__(self, *, total: int) -> None:
                progress_events.append(("total", total))

            def __enter__(self) -> FakeProgress:
                progress_events.append(("enter", None))
                return self

            def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
                progress_events.append(("exit", None))

            def start(self, job: ReproduceJob) -> None:
                progress_events.append(("start", f"{job.tasks[0].metric}+{len(job.tasks) - 1}/{job.benchmark}"))

            def update(self) -> None:
                progress_events.append(("update", None))

            def print(self, line: str) -> None:
                progress_events.append(("print", line))
                print(line)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["bleu", "rouge"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "capevalkit.reproduce.run_metric_on_benchmark",
                        return_value=(0, [], {}),
                    )
                )
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(patch("capevalkit.reproduce.ReproduceProgress", FakeProgress))
                stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                run_all_reproduce(
                    metrics=["bleu", "rouge"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        output = stdout.getvalue()
        self.assertNotIn("RUN", output)
        self.assertIn(("total", 2), progress_events)
        self.assertIn(("start", "bleu+1/bench"), progress_events)
        self.assertTrue(any(event[0] == "print" and "bleu/bench" in event[1] for event in progress_events))
        self.assertEqual(progress_events.count(("update", None)), 2)

    def test_all_reproduce_runs_grouped_reference_metric_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["clipscore", "refclipscore"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch(
                        "capevalkit.reproduce.run_metric_on_benchmark",
                        return_value=(0, [], {}),
                    )
                )
                write_result = stack.enter_context(
                    patch("capevalkit.reproduce.write_benchmark_result")
                )
                stack.enter_context(
                    patch("capevalkit.reproduce.compare_results", return_value=[])
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["clipscore", "refclipscore"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                    gpu_jobs=1,
                    jobs=2,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        run_metric.assert_called_once()
        self.assertEqual(run_metric.call_args.args[:2], ("refclipscore", "bench"))
        self.assertTrue(run_metric.call_args.kwargs["use_references"])
        self.assertEqual(write_result.call_count, 2)

    def test_all_reproduce_runs_exclusive_gpu_jobs_after_regular_gpu_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["clipscore", "vela"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            call_order = []
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch(
                        "capevalkit.reproduce.run_metric_on_benchmark",
                        side_effect=lambda metric, benchmark, **_: call_order.append(metric) or (0, [], {}),
                    )
                )
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["clipscore", "vela"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                    gpu_jobs=4,
                    jobs=4,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        self.assertEqual(call_order, ["clipscore", "vela"])
        self.assertEqual(run_metric.call_count, 2)

    def test_all_reproduce_runs_pacscore_vitl_without_checkpoint_preskip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["pacscore-vitl", "refpacscore-vitl"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark", return_value=(0, [], {}))
                )
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["pacscore-vitl", "refpacscore-vitl"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        run_metric.assert_called_once()

    def test_all_reproduce_runs_pacscore_without_checkpoint_preskip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["pacscore", "refpacscore"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark", return_value=(0, [], {}))
                )
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["pacscore", "refpacscore"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        run_metric.assert_called_once()

    def test_all_reproduce_runs_pacscorepp_without_checkpoint_preskip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for metric in ["pacscorepp", "refpacscorepp"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark", return_value=(0, [], {}))
                )
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["pacscorepp", "refpacscorepp"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok", "ok"])
        run_metric.assert_called_once()

    def test_all_reproduce_skips_nebula_image_metrics_when_images_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "expected" / "clipscore" / "nebula.json"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark")
                )
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["clipscore"],
                    benchmarks=["nebula"],
                    data_root=str(root / "external-data"),
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["skip"])
        self.assertIn("missing nebula images", results[0].message)
        run_metric.assert_not_called()

    def test_all_reproduce_uses_hf_images_for_repo_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = ReproduceJob(
                tasks=(
                    ReproduceTask(
                        metric="clipscore",
                        benchmark="nebula",
                        expected=root / "expected.json",
                        output=root / "output.json",
                    ),
                ),
                runner_metric="clipscore",
                benchmark="nebula",
                resource="gpu",
            )
            with patch("capevalkit.reproduce.repo_root", return_value=root):
                missing = missing_job_prerequisite(job, data_root=str(root / "data"))

        self.assertIsNone(missing)

    def test_all_reproduce_skips_fleur_when_numpy2_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "metrics" / "upstreams" / "fleur" / "uv.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text('[[package]]\nname = "numpy"\nversion = "2.2.6"\n')
            for metric in ["fleur", "reffleur"]:
                expected = root / "expected" / metric / "bench.json"
                expected.parent.mkdir(parents=True)
                expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark")
                )
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["fleur", "reffleur"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["skip", "skip"])
        self.assertIn("incompatible FLEUR NumPy runtime", results[0].message)
        run_metric.assert_not_called()

    def test_all_reproduce_runs_fleur_when_numpy1_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "metrics" / "upstreams" / "fleur" / "uv.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text('[[package]]\nname = "numpy"\nversion = "1.26.4"\n')
            expected = root / "expected" / "fleur" / "bench.json"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}")
            with contextlib.ExitStack() as stack:
                run_metric = stack.enter_context(
                    patch(
                        "capevalkit.reproduce.run_metric_on_benchmark",
                        return_value=(0, [], {}),
                    )
                )
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[]))
                stack.enter_context(patch("capevalkit.reproduce.repo_root", return_value=root))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["fleur"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["ok"])
        run_metric.assert_called_once()

    def test_default_all_reproduce_expected_files_match_default_pairs(self) -> None:
        root = Path("benchmarks/expected")
        tasks = expected_tasks(
            expected_root=root,
            output_dir=Path("outputs/all-reproduce"),
            metrics=DEFAULT_REPRO_METRICS,
            benchmarks=DEFAULT_REPRO_BENCHMARKS,
        )
        pairs = {(task.metric, task.benchmark) for task in tasks}
        non_default_pairs = sorted(pair for pair in pairs if not default_reproduce_pair(*pair))
        actual_pairs = {(path.parent.name, path.stem) for path in root.glob("*/*.json")}
        unexpected_actual_pairs = sorted(actual_pairs - pairs)

        self.assertIn(("vela", "longcaparena-testa-desc"), pairs)
        self.assertIn(("vela", "longcaparena-testb-flu"), pairs)
        self.assertIn(("bleu", "composite"), pairs)
        self.assertNotIn(("bleu", "longcaparena-testa-desc"), pairs)
        self.assertNotIn(("pacscorepp", "composite"), pairs)
        self.assertNotIn(("pacscoreppavg", "longcaparena-testb-rel"), pairs)
        self.assertEqual(non_default_pairs, [])
        self.assertEqual(unexpected_actual_pairs, [])

    def test_all_reproduce_reports_missing_expected_pairs(self) -> None:
        missing = missing_expected_pairs(
            expected_root=Path("benchmarks/expected"),
            metrics=["clipscore"],
            benchmarks=["missing-benchmark"],
        )
        self.assertEqual(missing, [("clipscore", "missing-benchmark")])

    def test_all_reproduce_can_hide_missing_expected_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "expected" / "metric" / "bench.json"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                code, results = run_all_reproduce(
                    metrics=["metric"],
                    benchmarks=["bench", "missing"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    dry_run=True,
                    color="never",
                    report_missing=False,
                )
        self.assertEqual(code, 0)
        self.assertEqual(
            [(result.metric, result.benchmark, result.status) for result in results],
            [("metric", "bench", "planned")],
        )

    def test_all_reproduce_smoke_skips_value_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "expected" / "metric" / "bench.json"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}")
            comparison = NumericComparison(
                key="correlations.kendall_tau_c",
                actual=0.0,
                expected=1.0,
                actual_display="0.0",
                expected_display="1.0",
                ok=False,
            )
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch("capevalkit.reproduce.run_metric_on_benchmark", return_value=(0, [], {}))
                )
                stack.enter_context(patch("capevalkit.reproduce.write_benchmark_result"))
                stack.enter_context(patch("capevalkit.reproduce.compare_results", return_value=[comparison]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code, results = run_all_reproduce(
                    metrics=["metric"],
                    benchmarks=["bench"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=False,
                    allow_mismatch=True,
                )

        self.assertEqual(code, 0)
        self.assertEqual([result.status for result in results], ["smoke"])
        self.assertEqual([result.message for result in results], ["smoke run completed"])

    def test_all_reproduce_smoke_fails_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code, results = run_all_reproduce(
                    metrics=["metric"],
                    benchmarks=["missing"],
                    data_root=None,
                    output_dir=root / "outputs",
                    expected_root=root / "expected",
                    tolerance=0.2,
                    round_decimals=1,
                    color="never",
                    report_missing=True,
                    allow_mismatch=True,
                )

        self.assertEqual(code, 1)
        self.assertEqual([result.status for result in results], ["skip"])

    def test_all_reproduce_compacts_mismatch_messages(self) -> None:
        message = (
            "correlations.kendall_tau_b: actual=30.132 rounded=30.1 "
            "expected=30.0 rounded_expected=30.0; "
            "correlations.kendall_tau_c: actual=32.518 rounded=32.5 "
            "expected=32.4 rounded_expected=32.4"
        )
        self.assertEqual(
            compact_message(message),
            "tau-b reprod=30.1 original=30.0 diff=+0.1; "
            "tau-c reprod=32.5 original=32.4 diff=+0.1",
        )

    def test_all_reproduce_formats_ok_comparisons_as_actual_over_paper(self) -> None:
        message = format_comparisons(
            [
                NumericComparison(
                    key="correlations.kendall_tau_b",
                    actual=28.36,
                    expected=28.3,
                    actual_display="28.4",
                    expected_display="28.3",
                    ok=True,
                ),
                NumericComparison(
                    key="correlations.kendall_tau_c",
                    actual=45.9,
                    expected=46.5,
                    actual_display="45.9",
                    expected_display="46.5",
                    ok=False,
                ),
            ]
        )
        self.assertEqual(
            message,
            "tau-b reprod=28.4 original=28.3 diff=+0.1; "
            "tau-c reprod=45.9 original=46.5 diff=-0.6",
        )

    def test_all_reproduce_prints_tau_b_and_tau_c_on_one_row(self) -> None:
        result = ReproduceResult(
            metric="bleu",
            benchmark="flickr8k-cf",
            status="ok",
            output="outputs/bleu/flickr8k-cf.json",
            expected="benchmarks/expected/bleu/flickr8k-cf.json",
            message=(
                "tau-b reprod=16.9 original=16.9 diff=+0.0; "
                "tau-c reprod=8.7 original=8.7 diff=+0.0"
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            print_results_header()
            print_result(result, index=1, total=9, use_color=False)

        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("τb_REPROD", lines[0])
        self.assertIn("τc_REPROD", lines[0])
        self.assertTrue(lines[0].index("τb_DIFF") > lines[0].index("τc_ORIG"))
        headers = ["τb_REPROD", "τb_ORIG", "τc_REPROD", "τc_ORIG", "τb_DIFF", "τc_DIFF", "NOTE"]
        starts = [lines[0].index(header) for header in headers]
        gaps = [
            next_start - (start + len(header))
            for start, header, next_start in zip(starts, headers, starts[1:])
        ]
        self.assertEqual(gaps, [2, 2, 2, 2, 2, 2])
        self.assertEqual(lines[1].count("bleu/flickr8k-cf"), 1)
        self.assertNotIn("tau-b", lines[1])
        self.assertNotIn("tau-c", lines[1])
        self.assertIn("16.9", lines[1])
        self.assertIn("8.7", lines[1])
        self.assertTrue(lines[1].rfind("+0.0") > lines[1].index("8.7"))

    def test_all_reproduce_compacts_missing_data_messages(self) -> None:
        message = "FileNotFoundError: missing polaris/polaris_test.csv; checked: a, b"
        self.assertEqual(compact_message(message), "missing polaris/polaris_test.csv")

    def test_all_reproduce_compacts_missing_checkpoint_messages(self) -> None:
        message = (
            "missing PACScore OpenCLIP checkpoint: "
            "/repo/metrics/upstreams/pacscore/checkpoints/openClip_ViT-L-14.pth"
        )
        self.assertEqual(compact_message(message), "missing PACScore OpenCLIP checkpoint")

    def test_all_reproduce_compacts_fleur_numpy_messages(self) -> None:
        message = "incompatible FLEUR NumPy runtime: numpy 2.2.6; run uv lock/sync with numpy<2"
        self.assertEqual(compact_message(message), "incompatible FLEUR NumPy runtime")


if __name__ == "__main__":
    unittest.main()
