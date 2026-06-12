"""Image captioning metric evaluation kit."""

from .api import (
    CaptionBatch,
    CaptionEvalRun,
    CaptionSample,
    MetricOutput,
    benchmark,
    evaluate_caption_model,
    evaluate_captions,
    evaluate_metric,
    load_samples,
    score,
)

__version__ = "0.1.0"

__all__ = [
    "CaptionBatch",
    "CaptionEvalRun",
    "CaptionSample",
    "MetricOutput",
    "benchmark",
    "evaluate_caption_model",
    "evaluate_captions",
    "evaluate_metric",
    "load_samples",
    "score",
    "__version__",
]
