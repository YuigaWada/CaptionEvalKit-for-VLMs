from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from capevalkit.compat import zip_strict
from capevalkit.progress import progress_update


VALID_MODES = ("desc", "rel", "flu")


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    }


def _caption(row: dict[str, Any]) -> str:
    value = row.get("caption", row.get("prediction"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain a string caption or prediction")
    return value


def _references(row: dict[str, Any]) -> list[str]:
    value = row.get("references", row.get("captions", []))
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("reference rows must contain references or captions")
    return [str(item) for item in value]


def _image(row: dict[str, Any]) -> str:
    value = row.get("image", row.get("image_path"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain image or image_path for VELA")
    return value


class _VelaRows(Dataset):
    def __init__(
        self,
        *,
        item_ids: list[str],
        predictions: dict[str, dict[str, Any]],
        references: dict[str, dict[str, Any]],
        mode: str,
    ) -> None:
        self.item_ids = item_ids
        self.predictions = predictions
        self.references = references
        self.mode = mode

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_id = self.item_ids[index]
        pred = self.predictions[item_id]
        image_path = _image(pred)
        return {
            "imgid": Path(image_path).name,
            "img": Image.open(image_path).convert("RGB"),
            "refs": _references(self.references[item_id]),
            "cand": _caption(pred),
            "mode": self.mode,
            "score": 0.0,
        }


def compute(
    predictions_path: str,
    references_path: str,
    *,
    mode: str,
    cfg_path: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from dataloader.vela_collate_fn import VELACollateFn
    from vela import load_pretrained_model

    predictions = _load_rows(predictions_path)
    references = _load_rows(references_path)
    item_ids = sorted(predictions)
    if set(item_ids) != set(references):
        missing = sorted(set(item_ids).symmetric_difference(references))
        raise ValueError(f"prediction/reference id mismatch: {missing[:5]}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_model(cfg_path=cfg_path, device=device)
    dataset = _VelaRows(item_ids=item_ids, predictions=predictions, references=references, mode=mode)
    collate_fn = VELACollateFn(model.cfg, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    scores: list[float] = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = model.move_batch_to_device(batch, device)
            outputs = torch.clamp(model(**batch).float(), min=0.0, max=1.0)
            batch_scores = [float(score) for score in outputs.detach().cpu().tolist()]
            scores.extend(batch_scores)
            progress_update(len(batch_scores))

    return {
        "VELA": {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "mode": mode,
            "per_item": dict(zip_strict(item_ids, scores)),
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--cfg-path", default="configs/config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    result = compute(
        args.predictions,
        args.references,
        mode=args.mode,
        cfg_path=args.cfg_path,
        device="cpu" if args.cpu else args.device,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
