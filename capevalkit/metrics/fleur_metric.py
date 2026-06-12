from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image

from capevalkit.compat import zip_strict
from capevalkit.paths import repo_root
from capevalkit.progress import progress_iter


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
        raise ValueError("prediction rows must contain image or image_path for FLEUR")
    return value


def _load_llava():
    llava_root = repo_root() / "metrics" / "upstreams" / "fleur" / "LLaVA"
    if llava_root.exists():
        sys.path.insert(0, str(llava_root))
    try:
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FLEUR requires the LLaVA package. Initialize metrics/upstreams/fleur/LLaVA "
            "and sync the FLEUR uv project."
        ) from exc
    return SimpleNamespace(
        torch=torch,
        DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
        SeparatorStyle=SeparatorStyle,
        conv_templates=conv_templates,
        KeywordsStoppingCriteria=KeywordsStoppingCriteria,
        get_model_name_from_path=get_model_name_from_path,
        process_images=process_images,
        tokenizer_image_token=tokenizer_image_token,
        load_pretrained_model=load_pretrained_model,
    )


def _prompt(caption: str, references: list[str] | None) -> str:
    criteria = (
        "Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given "
        "Grading Criteria. (Print Real Number Score ONLY)\n\n"
        "Grading Criteria:\n\n"
        "0.0: The caption does not describe the image at all.\n"
        "1.0: The caption accurately and clearly describes the image.\n\n"
    )
    if references is None:
        return f"{criteria}Caption: {caption}\n\nScore(Choose a rating from 0.0 to 1.0):"
    return (
        f"{criteria}Reference Captions: {references}\n\n"
        f"Candidate Caption: {caption}\n\nScore(Choose a rating from 0.0 to 1.0):"
    )


def _parse_score(text: str) -> float:
    match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
    if match is None:
        raise ValueError(f"could not parse FLEUR score from model output: {text!r}")
    value = float(match.group(0))
    if not 0 <= value <= 1:
        raise ValueError(f"FLEUR score outside [0, 1]: {value}")
    return value


def _select_token_index(generated_tokens: Any, token_id: int, *, prefer_second_zero: bool = False) -> int:
    matches = (generated_tokens == token_id).nonzero().flatten()
    if len(matches) == 0:
        raise ValueError(f"generated score token id {token_id} not found")
    if prefer_second_zero and len(matches) > 1:
        return int(matches[1])
    return int(matches[0])


def _smooth_score(llava: Any, tokenizer: Any, output: Any, input_len: int, raw_score: float) -> float:
    rate2token = {digit: tokenizer.encode(str(digit))[-1] for digit in range(10)}
    generated_tokens = output.sequences[0, input_len:]
    raw = str(raw_score)

    if raw_score < 1.0:
        first_digit = int(raw[raw.index(".") + 1])
        first_index = _select_token_index(
            generated_tokens,
            rate2token[first_digit],
            prefer_second_zero=first_digit == 0,
        )
        probs = llava.torch.nn.functional.softmax(output.scores[first_index], dim=-1)[0]
        score = sum(float(probs[token]) * digit * 0.1 for digit, token in rate2token.items())

        if len(raw) > raw.index(".") + 2:
            second_digit = int(raw[raw.index(".") + 2])
            second_index = _select_token_index(generated_tokens, rate2token[second_digit], prefer_second_zero=True)
            probs2 = llava.torch.nn.functional.softmax(output.scores[second_index], dim=-1)[0]
            score += sum(float(probs2[token]) * digit * 0.01 for digit, token in rate2token.items())
        return float(score)

    one_index = _select_token_index(generated_tokens, rate2token[1])
    probs = llava.torch.nn.functional.softmax(output.scores[one_index], dim=-1)[0]
    return float(0.9 * probs[rate2token[0]] + probs[rate2token[1]])


def compute(
    predictions_path: str,
    references_path: str | None,
    *,
    model_path: str,
    conv_mode: str,
    max_new_tokens: int,
    load_4bit: bool,
    load_8bit: bool,
) -> dict[str, Any]:
    if load_4bit and load_8bit:
        raise ValueError("FLEUR can load either 4bit or 8bit, not both")

    llava = _load_llava()
    predictions = _load_rows(predictions_path)
    references = _load_rows(references_path) if references_path else {}
    item_ids = sorted(predictions)

    model_name = llava.get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = llava.load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
    )
    model.eval()
    device = next(model.parameters()).device

    scores = []
    for item_id in progress_iter(item_ids):
        pred = predictions[item_id]
        refs = _references(references[item_id]) if references else None
        query = _prompt(_caption(pred), refs)
        conv = llava.conv_templates[conv_mode].copy()

        if model.config.mm_use_im_start_end:
            query = llava.DEFAULT_IM_START_TOKEN + llava.DEFAULT_IMAGE_TOKEN + llava.DEFAULT_IM_END_TOKEN + "\n" + query
        else:
            query = llava.DEFAULT_IMAGE_TOKEN + "\n" + query
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)

        image = Image.open(_image(pred)).convert("RGB")
        image_tensor = llava.process_images([image], image_processor, SimpleNamespace(image_aspect_ratio=None))
        if isinstance(image_tensor, list):
            image_tensor = [tensor.to(device, dtype=llava.torch.float16) for tensor in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=llava.torch.float16)

        input_ids = llava.tokenizer_image_token(
            conv.get_prompt(),
            tokenizer,
            llava.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)
        stop_str = conv.sep if conv.sep_style != llava.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = llava.KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with llava.torch.inference_mode():
            output = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = tokenizer.decode(output.sequences[0, input_ids.shape[1]:]).strip()
        raw_score = _parse_score(text)
        scores.append(_smooth_score(llava, tokenizer, output, input_ids.shape[1], raw_score))

    key = "RefFLEUR" if references else "FLEUR"
    return {
        key: {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "per_item": dict(zip_strict(item_ids, scores)),
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    parser.set_defaults(load_4bit=True)
    args = parser.parse_args(argv)

    result = compute(
        args.predictions,
        args.references,
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
