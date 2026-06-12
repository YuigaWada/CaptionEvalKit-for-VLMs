from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from capevalkit.compat import zip_strict
from capevalkit.paths import repo_root
from capevalkit.progress import progress_update


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["id"])] = row
    return rows


def _caption(row: dict[str, Any]) -> str:
    value = row.get("caption", row.get("prediction"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain a string caption or prediction")
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


def _image_path(row: dict[str, Any], image_dir: str | None) -> str:
    value = row.get("image", row.get("image_path"))
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return str(path)
        if image_dir and (Path(image_dir) / value).exists():
            return str(Path(image_dir) / value)
        return value
    if not image_dir:
        raise ValueError("image path missing; provide prediction.image or --image-dir")
    image_root = Path(image_dir)
    item_id = str(row["id"])
    for suffix in ("", ".jpg", ".jpeg", ".png", ".tiff"):
        candidate = image_root / f"{item_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"could not resolve image for id={item_id} under {image_dir}")


def _load_official_module():
    repo = repo_root() / "metrics" / "upstreams" / "clipscore"
    pycoco_repo = repo_root() / "metrics" / "upstreams" / "pycocoevalcap"
    path = repo / "clipscore.py"
    for import_path in (pycoco_repo.parent, repo):
        sys.path.insert(0, str(import_path))
    spec = importlib.util.spec_from_file_location("official_clipscore", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_clipscore_features(official: Any, features: Any) -> Any:
    if official.version.parse(official.np.__version__) < official.version.parse("1.21"):
        return official.sklearn.preprocessing.normalize(features, axis=1)
    return features / official.np.sqrt(official.np.sum(features**2, axis=1, keepdims=True))


def _clip_score_from_features(official: Any, image_features: Any, candidate_features: Any) -> Any:
    image_features = _normalize_clipscore_features(official, image_features)
    candidate_features = _normalize_clipscore_features(official, candidate_features)
    per = 2.5 * official.np.clip(official.np.sum(image_features * candidate_features, axis=1), 0, None)
    return official.np.mean(per), per


def _extract_image_features(official: Any, images: list[str], model: Any, device: str) -> Any:
    unique_images = list(dict.fromkeys(images))
    unique_features = official.extract_all_images(unique_images, model, device)
    features_by_image = dict(zip_strict(unique_images, unique_features))
    return official.np.vstack([features_by_image[image] for image in images])


def _extract_caption_features(official: Any, captions: list[str], model: Any, device: str) -> Any:
    unique_captions = list(dict.fromkeys(captions))
    unique_features = official.extract_all_captions(unique_captions, model, device)
    features_by_caption = dict(zip_strict(unique_captions, unique_features))
    return official.np.vstack([features_by_caption[caption] for caption in captions])


def _ref_clip_score_from_features(official: Any, references: list[list[str]], candidate_features: Any, model: Any, device: str) -> Any:
    candidate_features = _normalize_clipscore_features(official, candidate_features)
    flattened_refs = []
    flattened_ref_indexes = []
    for index, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_ref_indexes.extend([index for _ in refs])

    flattened_refs = _extract_caption_features(official, flattened_refs, model, device)
    flattened_refs = _normalize_clipscore_features(official, flattened_refs)

    refs_by_candidate: dict[int, list[Any]] = {}
    for ref_features, candidate_index in zip_strict(flattened_refs, flattened_ref_indexes):
        refs_by_candidate.setdefault(candidate_index, []).append(ref_features)

    per = []
    for candidate_index, candidate in enumerate(candidate_features):
        refs = official.np.vstack(refs_by_candidate[candidate_index])
        per.append(official.np.max(candidate.dot(refs.transpose())))
    return official.np.mean(per), per


def compute(
    predictions_path: str,
    references_path: str | None,
    *,
    image_dir: str | None,
    clip_model: str = "ViT-B/32",
    sentence_average: bool = False,
) -> dict[str, Any]:
    official = _load_official_module()
    worker_count = int(os.environ.get("CLIPSCORE_NUM_WORKERS", "0"))
    extract_images = official.extract_all_images
    extract_captions = official.extract_all_captions

    def extract_images_with_workers(images, model, device, batch_size=64, num_workers=8):
        return extract_images(images, model, device, batch_size=batch_size, num_workers=worker_count)

    def extract_captions_with_workers(captions, model, device, batch_size=256, num_workers=8):
        return extract_captions(captions, model, device, batch_size=batch_size, num_workers=worker_count)

    official.extract_all_images = extract_images_with_workers
    official.extract_all_captions = extract_captions_with_workers
    import torch
    import numpy as np

    predictions = _load_rows(predictions_path)
    references = _load_rows(references_path) if references_path else {}

    item_ids = sorted(predictions)
    images = [_image_path(predictions[item_id], image_dir) for item_id in item_ids]
    candidates = [_caption(predictions[item_id]) for item_id in item_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_root = os.environ.get("CLIP_DOWNLOAD_ROOT", str(Path.cwd() / ".cache" / "clip"))
    model, _ = official.clip.load(clip_model, device=device, jit=False, download_root=download_root)
    model.eval()

    if sentence_average:
        flattened_images = []
        flattened_captions = []
        flattened_item_ids = []
        for item_id, image, candidate in zip_strict(item_ids, images, candidates):
            for sentence in _split_sentences(candidate):
                flattened_item_ids.append(item_id)
                flattened_images.append(image)
                flattened_captions.append(sentence)

        image_features = _extract_image_features(official, flattened_images, model, device)
        candidate_features = _extract_caption_features(official, flattened_captions, model, device)
        _, per_sentence = _clip_score_from_features(official, image_features, candidate_features)
        by_item: dict[str, list[float]] = {item_id: [] for item_id in item_ids}
        for item_id, score in zip_strict(flattened_item_ids, per_sentence):
            by_item[item_id].append(float(score))
        per_item_scores = {item_id: float(np.mean(scores)) for item_id, scores in by_item.items()}
        progress_update(len(item_ids))
        return {
            "CLIPScoreavg": {
                "score": float(np.mean(list(per_item_scores.values()))),
                "per_item": per_item_scores,
            }
        }

    image_features = _extract_image_features(official, images, model, device)
    candidate_features = _extract_caption_features(official, candidates, model, device)
    _, per_image_text = _clip_score_from_features(official, image_features, candidate_features)
    per_item: dict[str, dict[str, float]] = {
        item_id: {"CLIPScore": float(score)}
        for item_id, score in zip_strict(item_ids, per_image_text)
    }
    output: dict[str, Any] = {
        "CLIPScore": float(np.mean(per_image_text)),
        "per_item": per_item,
    }

    if references:
        refs = [_references(references[item_id]) for item_id in item_ids]
        _, per_text_text = _ref_clip_score_from_features(official, refs, candidate_features, model, device)
        ref_scores = 2 * per_image_text * per_text_text / (per_image_text + per_text_text)
        for item_id, score in zip_strict(item_ids, ref_scores):
            per_item[item_id]["RefCLIPScore"] = float(score)
        output["RefCLIPScore"] = float(np.mean(ref_scores))
    progress_update(len(item_ids))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references")
    parser.add_argument("--image-dir")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--sentence-average", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = compute(
        args.predictions,
        args.references,
        image_dir=args.image_dir,
        clip_model=args.clip_model,
        sentence_average=args.sentence_average,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
