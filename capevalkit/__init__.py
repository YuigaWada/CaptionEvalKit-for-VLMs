"""Image captioning metric evaluation kit."""

from capevalkit.api import (
    CaptionBatch,
    CaptionEvalRun,
    CaptionSample,
    MetricOutput,
    benchmark,
    evaluate_caption_model,
    evaluate_captions,
    evaluate_metric,
    get_manifest,
    load_manifests,
    load_samples,
    score,
)

__version__ = "0.1.3"

__all__ = [
    "__version__",
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
