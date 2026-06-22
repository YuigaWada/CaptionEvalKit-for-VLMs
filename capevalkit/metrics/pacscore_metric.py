from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.request import urlopen

from capevalkit.shared.compat import zip_strict
from capevalkit.infrastructure.runtime.paths import repo_root
from capevalkit.infrastructure.execution.progress import copy_with_download_progress, progress_status, progress_update


def _pacscore_root() -> Path:
    return repo_root() / "metrics" / "upstreams" / "pacscore"


_DEFAULT_CHECKPOINTS = {
    ("pacs", "ViT-B/32"): "checkpoints/clip_ViT-B-32.pth",
    ("pacs", "open_clip_ViT-L/14"): "checkpoints/openClip_ViT-L-14.pth",
    ("pacs++", "ViT-B/32"): "checkpoints/PAC++_clip_ViT-B-32.pth",
    ("pacs++", "ViT-L/14"): "checkpoints/PAC++_clip_ViT-L-14.pth",
}
_CHECKPOINT_URLS = {
    ("pacs", "ViT-B/32"): "https://drive.usercontent.google.com/download?id=1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae&export=download&confirm=t",
    ("pacs", "open_clip_ViT-L/14"): "https://drive.usercontent.google.com/download?id=1G1DAGQf5fW2U3u7K3Dn-eCC6koMDyvsU&export=download&confirm=t",
    ("pacs++", "ViT-B/32"): "https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-B-32.pth",
    ("pacs++", "ViT-L/14"): "https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth",
}
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    }


def _caption(row: dict[str, Any]) -> str:
    value = row.get("caption", row.get("prediction"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain caption or prediction")
    return value


def _split_sentences(caption: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", caption) if part.strip()]
    return sentences or [caption]


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
        raise ValueError("prediction rows must contain image or image_path for PACScore")
    return value


def _checkpoint_path(variant: str, clip_model: str, checkpoint: str | None) -> Path:
    value = checkpoint or _DEFAULT_CHECKPOINTS[(variant, clip_model)]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _pacscore_root() / path
    if not path.exists():
        url = _CHECKPOINT_URLS.get((variant, clip_model))
        if url:
            with _checkpoint_download_lock(path):
                if not path.exists():
                    _download_checkpoint(url, path)
        else:
            raise FileNotFoundError(
                f"missing PACScore checkpoint for {clip_model}: {path}; "
                "no automatic checkpoint URL is registered"
            )
    return path


@contextmanager
def _checkpoint_download_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _download_checkpoint(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=120) as response, tmp_path.open("wb") as handle:
        copy_with_download_progress(response, handle, label=f"PACScore checkpoint {path.name}")
    tmp_path.replace(path)
    progress_status(f"Cached PACScore checkpoint: {path}")


def _ensure_clip_lora_tokenizer() -> None:
    target = _pacscore_root() / "models" / "clip_lora" / "bpe_simple_vocab_16e6.txt.gz"
    if target.exists():
        return
    source = _pacscore_root() / "models" / "clip" / "bpe_simple_vocab_16e6.txt.gz"
    if not source.exists():
        source = _pacscore_root() / "data" / "tokenizer" / "bpe_simple_vocab_16e6.txt.gz"
    if not source.exists():
        raise FileNotFoundError(f"missing CLIP tokenizer vocabulary for PAC-S++: {target}")
    shutil.copyfile(source, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32", choices=["ViT-B/32", "open_clip_ViT-L/14", "ViT-L/14"])
    parser.add_argument("--variant", default="pacs", choices=["pacs", "pacs++"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compute-refpac", action="store_true")
    parser.add_argument("--sentence-average", action="store_true")
    args = parser.parse_args(argv)
    if args.sentence_average and args.compute_refpac:
        raise SystemExit("--sentence-average is only defined for PAC-S/PAC-S++ without RefPAC")

    import torch
    import numpy as np
    from evaluation import PACScore, RefPACScore

    predictions = _load_rows(args.predictions)
    references = _load_rows(args.references) if args.references else {}
    item_ids = sorted(predictions)
    images = [_image(predictions[item_id]) for item_id in item_ids]
    candidates = [_caption(predictions[item_id]) for item_id in item_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.variant == "pacs++":
        _ensure_clip_lora_tokenizer()
        from models.clip_lora import clip_lora

        model, preprocess = clip_lora.load(args.clip_model, device=device, lora=4)
    elif args.clip_model.startswith("open_clip"):
        from models import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    else:
        from models.clip import clip

        model, preprocess = clip.load(args.clip_model, device=device)
    model = model.to(device).float()
    checkpoint = torch.load(_checkpoint_path(args.variant, args.clip_model, args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    pac_key = "PAC-S++" if args.variant == "pacs++" else "PAC-S"
    ref_key = "RefPAC-S++" if args.variant == "pacs++" else "RefPAC-S"
    if args.sentence_average:
        flattened_images = []
        flattened_candidates = []
        flattened_item_ids = []
        for item_id, image, candidate in zip_strict(item_ids, images, candidates):
            for sentence in _split_sentences(candidate):
                flattened_item_ids.append(item_id)
                flattened_images.append(image)
                flattened_candidates.append(sentence)
        _, sentence_scores, _, _ = PACScore(model, preprocess, flattened_images, flattened_candidates, device, w=2.0)
        by_item: dict[str, list[float]] = {item_id: [] for item_id in item_ids}
        for item_id, score in zip_strict(flattened_item_ids, sentence_scores):
            by_item[item_id].append(float(score))
        per_item_scores = {item_id: float(np.mean(scores)) for item_id, scores in by_item.items()}
        avg_key = f"{pac_key}avg"
        result = {
            avg_key: {
                "score": float(np.mean(list(per_item_scores.values()))),
                "per_item": per_item_scores,
            }
        }
        progress_update(len(item_ids))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
        return 0

    _, pac_scores, candidate_feats, len_candidates = PACScore(model, preprocess, images, candidates, device, w=2.0)
    result: dict[str, Any] = {
        pac_key: {
            "score": float(np.mean(pac_scores)),
            "per_item": {item_id: float(score) for item_id, score in zip_strict(item_ids, pac_scores)},
        }
    }
    if args.compute_refpac and references:
        refs = [_references(references[item_id]) for item_id in item_ids]
        _, text_text = RefPACScore(model, refs, candidate_feats, device, torch.tensor(len_candidates))
        ref_scores = 2 * pac_scores * text_text / (pac_scores + text_text)
        result[ref_key] = {
            "score": float(np.mean(ref_scores)),
            "per_item": {item_id: float(score) for item_id, score in zip_strict(item_ids, ref_scores)},
        }
    progress_update(len(item_ids))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
