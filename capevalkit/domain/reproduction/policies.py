from __future__ import annotations

from collections.abc import Iterable

from capevalkit.domain.evaluation import NO_REFERENCE_METRICS, ReferenceRequirementPolicy

from .models import ReproduceJob, ReproduceTask


TOLERANCE_OVERRIDES = {
    ("polos", "composite"): 0.55,
    ("expert", "composite"): 2.0,
    ("expert", "flickr8k-ex"): 2.0,
    ("expert", "flickr8k-cf"): 2.0,
    ("expert", "nebula"): 2.0,
    ("expert", "polaris"): 2.0,
    ("vela", "longcaparena-testa-desc"): 3.0,
    ("vela", "longcaparena-testa-rel"): 3.0,
    ("vela", "longcaparena-testa-flu"): 3.0,
    ("vela", "longcaparena-testb-desc"): 3.0,
    ("vela", "longcaparena-testb-rel"): 3.0,
    ("vela", "longcaparena-testb-flu"): 3.0,
}
PYCOCO_METRICS = ("bleu", "rouge", "meteor", "cider", "spice")
GPU_METRICS = {
    "clipscore",
    "clipscore-vitl",
    "clipscoreavg",
    "refclipscore",
    "refclipscore-vitl",
    "pacscore",
    "pacscore-vitl",
    "pacscoreavg",
    "refpacscore",
    "refpacscore-vitl",
    "pacscorepp",
    "pacscoreppavg",
    "refpacscorepp",
    "polos",
    "fleur",
    "reffleur",
    "vela",
    "expert",
}
FLEUR_METRICS = {"fleur", "reffleur"}
EXCLUSIVE_GPU_METRICS = {*FLEUR_METRICS, "vela", "expert"}
REFERENCE_PAIRS = (
    ("clipscore", "refclipscore"),
    ("clipscore-vitl", "refclipscore-vitl"),
    ("pacscore", "refpacscore"),
    ("pacscore-vitl", "refpacscore-vitl"),
    ("pacscorepp", "refpacscorepp"),
)


class ResourceRequirementPolicy:
    def __init__(
        self,
        *,
        gpu_metrics: set[str] | None = None,
        exclusive_gpu_metrics: set[str] | None = None,
    ) -> None:
        self.gpu_metrics = gpu_metrics or GPU_METRICS
        self.exclusive_gpu_metrics = exclusive_gpu_metrics or EXCLUSIVE_GPU_METRICS

    def resource_for_metric(self, metric: str) -> str:
        if metric in self.exclusive_gpu_metrics:
            return "exclusive-gpu"
        if metric in self.gpu_metrics:
            return "gpu"
        return "cpu"


class TolerancePolicy:
    def __init__(self, overrides: dict[tuple[str, str], float] | None = None) -> None:
        self.overrides = overrides or TOLERANCE_OVERRIDES

    def tolerance(self, metric: str, benchmark: str, default: float) -> float:
        return max(default, self.overrides.get((metric, benchmark), default))


class JobGroupingPolicy:
    def __init__(
        self,
        *,
        resource_policy: ResourceRequirementPolicy | None = None,
        reference_policy: ReferenceRequirementPolicy | None = None,
        pycoco_metrics: tuple[str, ...] = PYCOCO_METRICS,
        reference_pairs: tuple[tuple[str, str], ...] = REFERENCE_PAIRS,
    ) -> None:
        self.resource_policy = resource_policy or ResourceRequirementPolicy()
        self.reference_policy = reference_policy or ReferenceRequirementPolicy(NO_REFERENCE_METRICS)
        self.pycoco_metrics = pycoco_metrics
        self.reference_pairs = reference_pairs

    def build_jobs(self, tasks: list[ReproduceTask]) -> list[ReproduceJob]:
        task_by_pair = {(task.metric, task.benchmark): task for task in tasks}
        assigned: set[tuple[str, str]] = set()
        jobs: list[ReproduceJob] = []
        for task in tasks:
            key = (task.metric, task.benchmark)
            if key in assigned:
                continue
            job = self.grouped_job_for_task(task, task_by_pair, assigned)
            jobs.append(job)
            assigned.update((job_task.metric, job_task.benchmark) for job_task in job.tasks)
        return jobs

    def grouped_job_for_task(
        self,
        task: ReproduceTask,
        task_by_pair: dict[tuple[str, str], ReproduceTask],
        assigned: set[tuple[str, str]],
    ) -> ReproduceJob:
        benchmark = task.benchmark
        metric = task.metric
        if metric in self.pycoco_metrics:
            group = self._available_group(self.pycoco_metrics, benchmark, task_by_pair, assigned)
            metrics = ",".join(task.metric for task in group)
            return ReproduceJob(
                tasks=tuple(group),
                runner_metric=group[0].metric,
                benchmark=benchmark,
                metric_args=("--metrics", metrics),
                use_references=True,
                resource="cpu",
            )

        for no_ref, with_ref in self.reference_pairs:
            if metric in {no_ref, with_ref}:
                group = self._available_group((no_ref, with_ref), benchmark, task_by_pair, assigned)
                has_ref = any(task.metric == with_ref for task in group)
                runner_metric = with_ref if has_ref else no_ref
                return ReproduceJob(
                    tasks=tuple(group),
                    runner_metric=runner_metric,
                    benchmark=benchmark,
                    use_references=has_ref,
                    resource=self.resource_policy.resource_for_metric(runner_metric),
                )

        return ReproduceJob(
            tasks=(task,),
            runner_metric=metric,
            benchmark=benchmark,
            use_references=self.reference_policy.use_references(metric),
            resource=self.resource_policy.resource_for_metric(metric),
        )

    @staticmethod
    def _available_group(
        metrics: Iterable[str],
        benchmark: str,
        task_by_pair: dict[tuple[str, str], ReproduceTask],
        assigned: set[tuple[str, str]],
    ) -> list[ReproduceTask]:
        group = []
        for metric in metrics:
            key = (metric, benchmark)
            if key in assigned:
                continue
            task = task_by_pair.get(key)
            if task:
                group.append(task)
        return group

