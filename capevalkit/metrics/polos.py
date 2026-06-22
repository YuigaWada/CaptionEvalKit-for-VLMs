from __future__ import annotations

from collections.abc import Sequence
import argparse
import json
from pathlib import Path
from typing import Any

from capevalkit.shared.compat import zip_strict
from capevalkit.infrastructure.runtime.paths import repo_root
from capevalkit.infrastructure.execution.progress import progress_status, progress_update


def load_model(model_path: str | None = None, model_name: str = "polos") -> Any:
    from polos.models import download_model, load_checkpoint

    if model_path:
        progress_status(f"Loading Polos checkpoint: {model_path}")
        return load_checkpoint(model_path)
    cache_dir = repo_root() / ".model-cache" / "polos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_status(f"Loading Polos model {model_name}: first use may download to {cache_dir}")
    return load_checkpoint(download_model(model_name, saving_directory=f"{cache_dir}/"))


def score(
    data: Sequence[dict[str, Any]],
    model_path: str | None = None,
    *,
    cuda: bool = True,
    batch_size: int = 8,
) -> list[float]:
    model = load_model(model_path)
    rows = list(data)
    scores: list[float] = []
    step = max(1, batch_size)
    for start in range(0, len(rows), step):
        batch = rows[start:start + step]
        _, batch_scores = model.predict(batch, batch_size=batch_size, cuda=cuda)
        scores.extend(float(score_) for score_ in batch_scores)
        progress_update(len(batch))
    return scores


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    }


def _image(row: dict[str, Any]) -> str:
    value = row.get("image", row.get("image_path"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain image or image_path for Polos")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    predictions = _load_rows(args.predictions)
    references = _load_rows(args.references)
    item_ids = sorted(predictions)
    data = []
    for item_id in item_ids:
        pred = predictions[item_id]
        ref = references[item_id]
        data.append(
            {
                "img": _image(pred),
                "mt": pred.get("caption", pred.get("prediction")),
                "refs": ref.get("references", ref.get("captions", [])),
            }
        )
    scores = score(data, args.model, cuda=not args.cpu, batch_size=args.batch_size)
    result = {
        "Polos": {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "per_item": dict(zip_strict(item_ids, scores)),
        }
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
