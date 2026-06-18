from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capevalkit.shared.compat import zip_strict
from capevalkit.infrastructure.execution.progress import progress_iter


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _records_to_coco_inputs(
    predictions_path: str,
    references_path: str,
) -> tuple[list[str], dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    predictions = {str(row["id"]): row for row in _read_jsonl(predictions_path)}
    references = {str(row["id"]): row for row in _read_jsonl(references_path)}
    missing = sorted(set(references) - set(predictions))
    if missing:
        raise ValueError(f"missing predictions for ids: {', '.join(missing[:10])}")

    gts: dict[str, list[dict[str, str]]] = {}
    res: dict[str, list[dict[str, str]]] = {}
    item_ids = sorted(references)
    for item_id in progress_iter(item_ids):
        ref_row = references[item_id]
        pred_row = predictions[item_id]
        refs = ref_row.get("references", ref_row.get("captions"))
        if not isinstance(refs, list):
            raise ValueError(f"references row {item_id} must contain references or captions list")
        caption = pred_row.get("caption", pred_row.get("prediction"))
        if not isinstance(caption, str):
            raise ValueError(f"prediction row {item_id} must contain caption or prediction")
        gts[item_id] = [{"caption": str(ref)} for ref in refs]
        res[item_id] = [{"caption": caption}]
    return item_ids, gts, res


def compute_pycocoevalcap(
    predictions_path: str,
    references_path: str,
    metrics: list[str],
) -> dict[str, Any]:
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ModuleNotFoundError:
        from tokenizer.ptbtokenizer import PTBTokenizer

    item_ids, gts, res = _records_to_coco_inputs(predictions_path, references_path)
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    output: dict[str, Any] = {}
    for metric in metrics:
        scorer, names = _build_scorer(metric)
        score, per_item = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for name, score_value, item_values in zip_strict(names, score, per_item):
                values = list(map(float, item_values))
                output[name] = {
                    "score": float(score_value),
                    "per_item": dict(zip_strict(item_ids, values)),
                }
        else:
            values = [_per_item_score(item) for item in per_item]
            output[names[0]] = {
                "score": float(score),
                "per_item": dict(zip_strict(item_ids, values)),
            }
    return output


def _per_item_score(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["All"]["f"])
    return float(value)


def _build_scorer(metric: str):
    try:
        if metric == "bleu":
            from pycocoevalcap.bleu.bleu import Bleu

            return Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        if metric == "meteor":
            from pycocoevalcap.meteor.meteor import Meteor

            return Meteor(), ["METEOR"]
        if metric == "rouge":
            from pycocoevalcap.rouge.rouge import Rouge

            return Rouge(), ["ROUGE-L"]
        if metric == "cider":
            from pycocoevalcap.cider.cider import Cider

            return Cider(), ["CIDEr"]
        if metric == "spice":
            from pycocoevalcap.spice.spice import Spice

            return Spice(), ["SPICE"]
    except ModuleNotFoundError:
        if metric == "bleu":
            from bleu.bleu import Bleu

            return Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        if metric == "meteor":
            from meteor.meteor import Meteor

            return Meteor(), ["METEOR"]
        if metric == "rouge":
            from rouge.rouge import Rouge

            return Rouge(), ["ROUGE-L"]
        if metric == "cider":
            from cider.cider import Cider

            return Cider(), ["CIDEr"]
        if metric == "spice":
            from spice.spice import Spice

            return Spice(), ["SPICE"]
    raise ValueError(f"unknown pycocoevalcap metric: {metric}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--metrics", default="bleu,meteor,rouge,cider,spice")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    metrics = [name.strip().lower() for name in args.metrics.split(",") if name.strip()]
    result = compute_pycocoevalcap(args.predictions, args.references, metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
