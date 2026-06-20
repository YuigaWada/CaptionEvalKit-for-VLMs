from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Union

from capevalkit.infrastructure.benchmarks.legacy import (
    BenchmarkItem,
    benchmark_metric,
    benchmark_result,
    load_benchmark,
)
from capevalkit.domain.evaluation import ReferenceRequirementPolicy
from capevalkit.infrastructure.execution.dispatcher import dispatch
from capevalkit.infrastructure.manifests.catalog import get_manifest, load_manifests
from capevalkit.infrastructure.runtime.paths import repo_root
from capevalkit.shared.compat import zip_strict


@dataclass(frozen=True)
class CaptionSample:
    id: str
    image: str
    references: list[str]
    prediction: str | None = None
    human_score: float | None = None


@dataclass(frozen=True)
class CaptionBatch:
    ids: list[str]
    images: list[str]
    references: list[list[str]]
    samples: list[CaptionSample]


@dataclass(frozen=True)
class MetricOutput:
    name: str
    per_item: Mapping[str, float]
    score: float | None = None


MetricCallable = Callable[[list[CaptionSample]], Union[Mapping[str, float], MetricOutput, dict[str, Any]]]


def score(
    metric: str,
    predictions: str,
    output: str,
    *,
    references: str | None = None,
    image_dir: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    manifest = get_manifest(metric)
    command = [
        *manifest.runner,
        "--predictions",
        str(Path(predictions).resolve()),
        "--output",
        str(Path(output).resolve()),
    ]
    if references:
        command.extend(["--references", str(Path(references).resolve())])
    if image_dir:
        command.extend(["--image-dir", str(Path(image_dir).resolve())])
    command.extend(extra_args or [])
    return dispatch(metric, command)


def benchmark(
    metric: str,
    benchmark_name: str,
    output: str,
    *,
    data_root: str | None = None,
    extra_args: list[str] | None = None,
    use_references: bool = True,
    score_key: str | None = None,
    limit: int | None = None,
) -> int:
    return benchmark_metric(
        metric,
        benchmark_name,
        output,
        data_root=data_root,
        metric_args=extra_args,
        use_references=use_references,
        score_key=score_key,
        limit=limit,
    )


def load_samples(
    benchmark_name: str,
    *,
    data_root: str | None = None,
    predictions: str | Path | Mapping[str, str] | None = None,
    limit: int | None = None,
) -> list[CaptionSample]:
    items = load_benchmark(benchmark_name, data_root, limit=limit)
    prediction_map = _load_prediction_map(predictions)
    return [_sample_from_item(item, prediction_map) for item in items]


def evaluate_metric(
    *,
    benchmark: str,
    metric: MetricCallable,
    metric_name: str = "metric",
    data_root: str | None = None,
    predictions: str | Path | Mapping[str, str] | None = None,
    output: str | Path | None = None,
    score_key: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    items = load_benchmark(benchmark, data_root, limit=limit)
    if not items:
        raise ValueError(f"{benchmark} has no benchmark items")
    prediction_map = _load_prediction_map(predictions)
    samples = [_sample_from_item(item, prediction_map) for item in items]
    metric_output = _normalize_metric_output(metric(samples), metric_name)
    evaluated_items = [
        BenchmarkItem(
            id=item.id,
            image=item.image,
            caption=sample.prediction or "",
            references=item.references,
            score=item.score,
        )
        for item, sample in zip_strict(items, samples)
    ]
    result = benchmark_result(
        metric_name,
        benchmark,
        items=evaluated_items,
        metric_output=metric_output,
        score_key=score_key,
    )
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


class CaptionEvalRun:
    def __init__(
        self,
        *,
        images: Sequence[str | Path],
        metrics: Sequence[str],
        ids: Sequence[str] | None = None,
        references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None = None,
        output_dir: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else repo_root() / "outputs" / "caption-model"
        )
        self.samples = _samples_from_images(images, ids=ids, references=references, limit=limit)
        if not self.samples:
            raise ValueError("no images were provided")
        self._captions: dict[str, str] = {}

    def __enter__(self) -> CaptionEvalRun:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def iter_batches(self, batch_size: int = 1) -> Iterable[CaptionBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.samples), batch_size):
            samples = self.samples[start:start + batch_size]
            yield CaptionBatch(
                ids=[sample.id for sample in samples],
                images=[sample.image for sample in samples],
                references=[sample.references for sample in samples],
                samples=samples,
            )

    def record(
        self,
        ids: Sequence[str] | Mapping[str, str],
        captions: Sequence[str] | None = None,
    ) -> None:
        if isinstance(ids, Mapping):
            if captions is not None:
                raise ValueError("captions must be omitted when ids is a mapping")
            items = ids.items()
        else:
            if captions is None:
                raise ValueError("captions are required when ids is a sequence")
            items = zip_strict(ids, captions)
        known_ids = {sample.id for sample in self.samples}
        for item_id, caption in items:
            item_id = str(item_id)
            if item_id not in known_ids:
                raise KeyError(f"unknown sample id: {item_id}")
            self._captions[item_id] = str(caption)

    def evaluate(
        self,
        *,
        extra_args_by_metric: Mapping[str, Sequence[str]] | None = None,
        quiet: bool = False,
    ) -> dict[str, Any]:
        missing = [sample.id for sample in self.samples if sample.id not in self._captions]
        if missing:
            raise ValueError(f"missing captions for sample ids: {missing[:5]}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.jsonl"
        references_path = self.output_dir / "references.jsonl"
        _write_jsonl(
            predictions_path,
            [
                {
                    "id": sample.id,
                    "caption": self._captions[sample.id],
                    "image": sample.image,
                }
                for sample in self.samples
            ],
        )
        _write_jsonl(
            references_path,
            [{"id": sample.id, "references": sample.references} for sample in self.samples],
        )

        results: dict[str, Any] = {}
        for metric in self.metrics:
            output_path = self.output_dir / f"{metric}.json"
            command = _metric_score_command(
                metric,
                predictions=predictions_path,
                references=references_path,
                output=output_path,
                extra_args=list((extra_args_by_metric or {}).get(metric, ())),
            )
            code = dispatch(metric, command, quiet=quiet)
            if code != 0:
                raise RuntimeError(f"{metric} exited with code {code}")
            results[metric] = json.loads(output_path.read_text())
        return results


def evaluate_caption_model(
    *,
    images: Sequence[str | Path],
    metrics: Sequence[str],
    predict: Callable[[CaptionBatch], Sequence[str] | Mapping[str, str]],
    ids: Sequence[str] | None = None,
    references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    extra_args_by_metric: Mapping[str, Sequence[str]] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    with CaptionEvalRun(
        images=images,
        metrics=metrics,
        ids=ids,
        references=references,
        output_dir=output_dir,
        limit=limit,
    ) as run:
        for batch in run.iter_batches(batch_size=batch_size):
            captions = predict(batch)
            if isinstance(captions, Mapping):
                run.record(captions)
            else:
                run.record(batch.ids, captions)
        return run.evaluate(extra_args_by_metric=extra_args_by_metric, quiet=quiet)


def evaluate_captions(
    *,
    metrics: Sequence[str],
    pairs: Sequence[Mapping[str, Any]] | None = None,
    images: Sequence[str | Path] | None = None,
    captions: Sequence[str] | Mapping[str, str] | None = None,
    ids: Sequence[str] | None = None,
    references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    extra_args_by_metric: Mapping[str, Sequence[str]] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    images, ids, captions, references = _caption_inputs(
        pairs=pairs,
        images=images,
        captions=captions,
        ids=ids,
        references=references,
    )
    with CaptionEvalRun(
        images=images,
        metrics=metrics,
        ids=ids,
        references=references,
        output_dir=output_dir,
        limit=limit,
    ) as run:
        run.record({sample.id: captions[sample.id] for sample in run.samples})
        return run.evaluate(extra_args_by_metric=extra_args_by_metric, quiet=quiet)


def _sample_from_item(item: BenchmarkItem, predictions: Mapping[str, str] | None) -> CaptionSample:
    return CaptionSample(
        id=item.id,
        image=item.image,
        references=item.references,
        prediction=predictions[item.id] if predictions is not None else item.caption,
        human_score=item.score,
    )


def _samples_from_images(
    images: Sequence[str | Path],
    *,
    ids: Sequence[str] | None,
    references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None,
    limit: int | None,
) -> list[CaptionSample]:
    image_values = [str(Path(image).resolve()) for image in images]
    if limit is not None:
        image_values = image_values[:limit]
    if ids is None:
        id_values = [str(index) for index in range(len(image_values))]
    else:
        id_values = [str(item_id) for item_id in ids]
        if limit is not None:
            id_values = id_values[:limit]
    if len(id_values) != len(image_values):
        raise ValueError("ids and images must have the same length")
    refs_by_id = _references_by_id(id_values, references, limit=limit)
    return [
        CaptionSample(id=item_id, image=image, references=refs_by_id[item_id])
        for item_id, image in zip_strict(id_values, image_values)
    ]


def _caption_inputs(
    *,
    pairs: Sequence[Mapping[str, Any]] | None,
    images: Sequence[str | Path] | None,
    captions: Sequence[str] | Mapping[str, str] | None,
    ids: Sequence[str] | None,
    references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None,
) -> tuple[list[str], list[str], dict[str, str], Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None]:
    if pairs is not None:
        if images is not None or captions is not None or ids is not None or references is not None:
            raise ValueError("pairs cannot be combined with images, captions, ids, or references")
        image_values: list[str] = []
        id_values: list[str] = []
        caption_values: dict[str, str] = {}
        reference_values: dict[str, str | Sequence[str]] = {}
        for index, pair in enumerate(pairs):
            item_id = str(pair.get("id", index))
            caption = pair.get("caption", pair.get("prediction"))
            if not isinstance(caption, str):
                raise ValueError("each pair must contain caption or prediction")
            image = pair.get("image", pair.get("image_path"))
            if image is None:
                raise ValueError("each pair must contain image or image_path")
            image_values.append(str(image))
            id_values.append(item_id)
            caption_values[item_id] = caption
            if "references" in pair:
                reference_values[item_id] = pair["references"]
            elif "captions" in pair:
                reference_values[item_id] = pair["captions"]
        return image_values, id_values, caption_values, reference_values

    if images is None or captions is None:
        raise ValueError("either pairs or both images and captions are required")
    image_values = [str(image) for image in images]
    id_values = [str(item_id) for item_id in ids] if ids is not None else [str(index) for index in range(len(image_values))]
    if len(id_values) != len(image_values):
        raise ValueError("ids and images must have the same length")
    if isinstance(captions, Mapping):
        caption_values = {str(item_id): str(caption) for item_id, caption in captions.items()}
    else:
        caption_list = [str(caption) for caption in captions]
        if len(caption_list) != len(image_values):
            raise ValueError("captions and images must have the same length")
        caption_values = dict(zip_strict(id_values, caption_list))
    return image_values, id_values, caption_values, references


def _references_by_id(
    ids: Sequence[str],
    references: Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]] | None,
    *,
    limit: int | None,
) -> dict[str, list[str]]:
    if references is None:
        return {item_id: [] for item_id in ids}
    if isinstance(references, Mapping):
        return {item_id: _reference_list(references.get(item_id, [])) for item_id in ids}
    reference_values = list(references)
    if limit is not None:
        reference_values = reference_values[:limit]
    if len(reference_values) != len(ids):
        raise ValueError("references and images must have the same length")
    return {
        item_id: _reference_list(reference)
        for item_id, reference in zip_strict(ids, reference_values)
    }


def _reference_list(value: str | Sequence[str] | Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("references must be strings or sequences of strings")


def _load_prediction_map(predictions: str | Path | Mapping[str, str] | None) -> dict[str, str] | None:
    if predictions is None:
        return None
    if isinstance(predictions, Mapping):
        return {str(item_id): str(caption) for item_id, caption in predictions.items()}
    rows = {}
    for line in Path(predictions).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        caption = row.get("caption", row.get("prediction"))
        if not isinstance(caption, str):
            raise ValueError("prediction rows must contain caption or prediction")
        rows[str(row["id"])] = caption
    return rows


def _normalize_metric_output(result: Mapping[str, float] | MetricOutput | dict[str, Any], metric_name: str) -> dict[str, Any]:
    if isinstance(result, MetricOutput):
        per_item = {str(item_id): float(score) for item_id, score in result.per_item.items()}
        return {
            result.name: {
                "score": float(result.score) if result.score is not None else _mean(per_item.values()),
                "per_item": per_item,
            }
        }
    if not isinstance(result, Mapping):
        raise TypeError("metric callable must return a mapping or MetricOutput")
    if _looks_like_metric_output(result):
        return dict(result)
    per_item = {str(item_id): float(score) for item_id, score in result.items()}
    return {metric_name: {"score": _mean(per_item.values()), "per_item": per_item}}


def _looks_like_metric_output(result: Mapping[str, Any]) -> bool:
    if isinstance(result.get("per_item"), Mapping):
        return True
    return any(isinstance(value, Mapping) and isinstance(value.get("per_item"), Mapping) for value in result.values())


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _metric_score_command(
    metric: str,
    *,
    predictions: Path,
    references: Path,
    output: Path,
    extra_args: list[str],
) -> list[str]:
    manifest = get_manifest(metric)
    command = [*manifest.runner, "--predictions", str(predictions), "--output", str(output), *extra_args]
    if ReferenceRequirementPolicy().use_references(metric):
        command[command.index("--output"):command.index("--output")] = ["--references", str(references)]
    return command


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


__all__ = [
    "CaptionBatch",
    "CaptionEvalRun",
    "CaptionSample",
    "MetricOutput",
    "benchmark",
    "evaluate_caption_model",
    "evaluate_captions",
    "evaluate_metric",
    "get_manifest",
    "load_manifests",
    "load_samples",
    "score",
]
