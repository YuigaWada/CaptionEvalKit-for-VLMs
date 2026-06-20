from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_SCORE_KEYS = {
    "bleu": "BLEU-4",
    "rouge": "ROUGE-L",
    "meteor": "METEOR",
    "cider": "CIDEr",
    "spice": "SPICE",
    "jaspice": "JaSPICE",
    "clipscore": "CLIPScore",
    "clipscore-vitl": "CLIPScore",
    "clipscoreavg": "CLIPScoreavg",
    "refclipscore": "RefCLIPScore",
    "refclipscore-vitl": "RefCLIPScore",
    "pacscore": "PAC-S",
    "pacscore-vitl": "PAC-S",
    "pacscoreavg": "PAC-Savg",
    "refpacscore": "RefPAC-S",
    "refpacscore-vitl": "RefPAC-S",
    "pacscorepp": "PAC-S++",
    "pacscoreppavg": "PAC-S++avg",
    "refpacscorepp": "RefPAC-S++",
    "polos": "Polos",
    "vela": "VELA",
    "fleur": "FLEUR",
    "reffleur": "RefFLEUR",
    "expert": "EXPERT",
}

NO_REFERENCE_METRICS = {
    "clipscore",
    "clipscore-vitl",
    "clipscoreavg",
    "pacscore",
    "pacscore-vitl",
    "pacscoreavg",
    "pacscorepp",
    "pacscoreppavg",
    "fleur",
    "expert",
}

ITEM_METADATA_KEYS = {"id", "score", "scores", "ground_truth_score", "caption", "image", "references"}


class ReferenceRequirementPolicy:
    def __init__(self, no_reference_metrics: set[str] | None = None) -> None:
        self.no_reference_metrics = no_reference_metrics or NO_REFERENCE_METRICS

    def use_references(self, metric: str, *, no_references: bool = False) -> bool:
        return not no_references and metric not in self.no_reference_metrics


class ScoreKeyPolicy:
    def __init__(self, defaults: Mapping[str, str] | None = None, longcaparena_benchmarks: set[str] | None = None) -> None:
        self.defaults = defaults or DEFAULT_SCORE_KEYS
        self.longcaparena_benchmarks = longcaparena_benchmarks or set()

    def score_key(self, metric_name: str, benchmark_name: str, override: str | None = None) -> str | None:
        if override:
            return override
        score_key = self.defaults.get(metric_name)
        if metric_name == "vela" and benchmark_name in self.longcaparena_benchmarks:
            return score_key or "VELA"
        return score_key


class BenchmarkModePolicy:
    def __init__(self, longcaparena_modes: Mapping[str, str] | None = None) -> None:
        self.longcaparena_modes = longcaparena_modes or {}

    def metric_args(self, metric_name: str, benchmark_name: str, metric_args: list[str] | None) -> list[str]:
        args = list(metric_args or [])
        if metric_name != "vela":
            return args
        mode = self.longcaparena_modes.get(benchmark_name)
        if mode and "--mode" not in args:
            return ["--mode", mode, *args]
        return args


class MetricOutputNormalizationPolicy:
    def score_values(self, metric_output: dict[str, Any], score_key: str | None = None) -> tuple[str, dict[str, float]]:
        for key, value in metric_output.items():
            if score_key and key != score_key:
                continue
            if isinstance(value, dict) and isinstance(value.get("per_item"), dict):
                return key, {str(item_id): self.extract_item_score(score, key) for item_id, score in value["per_item"].items()}
        if isinstance(metric_output.get("per_item"), dict):
            per_item = metric_output["per_item"]
            first = next(iter(per_item.values()), None)
            if isinstance(first, dict):
                key = score_key or self.default_item_score_key(first)
                return key, {str(item_id): self.extract_item_score(scores, key) for item_id, scores in per_item.items()}
            key = score_key or "score"
            return key, {str(item_id): self.extract_item_score(score, key) for item_id, score in per_item.items()}
        raise ValueError("metric output does not contain per_item scores")

    def default_item_score_key(self, value: Any) -> str:
        if not isinstance(value, dict):
            return "score"
        scores = value.get("scores")
        if isinstance(scores, dict) and scores:
            return str(next(iter(scores)))
        if "score" in value:
            return "score"
        return str(next(iter(value)))

    def extract_item_score(self, value: Any, score_key: str | None) -> float:
        if not isinstance(value, dict):
            return float(value)
        if score_key and score_key in value:
            return float(value[score_key])
        scores = value.get("scores")
        if isinstance(scores, dict):
            if score_key and score_key in scores:
                return float(scores[score_key])
            if "score" in scores:
                return float(scores["score"])
            for score in scores.values():
                return float(score)
        if "score" in value:
            return float(value["score"])
        for key, score in value.items():
            if key in ITEM_METADATA_KEYS:
                continue
            try:
                return float(score)
            except (TypeError, ValueError):
                continue
        raise ValueError("per_item entry does not contain a numeric metric score")

    def item_score_map(self, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        scores = value.get("scores")
        if isinstance(scores, dict):
            return {str(key): float(score) for key, score in scores.items()}
        output = {}
        for key, score in value.items():
            if key in ITEM_METADATA_KEYS:
                continue
            try:
                output[str(key)] = float(score)
            except (TypeError, ValueError):
                continue
        return output

