from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image

from capevalkit.shared.compat import zip_strict
from capevalkit.infrastructure.runtime.paths import repo_root
from capevalkit.infrastructure.execution.progress import progress_iter, progress_status


DEFAULT_MODEL_PATH = "hjkim811/EXPERT-llava-13b-lora"
DEFAULT_MODEL_BASE = "liuhaotian/llava-v1.5-13b"


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    }


def _caption(row: dict[str, Any]) -> str:
    value = row.get("caption", row.get("prediction"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain caption or prediction")
    return " ".join(value.split())


def _image(row: dict[str, Any]) -> str:
    value = row.get("image", row.get("image_path"))
    if not isinstance(value, str):
        raise ValueError("prediction rows must contain image or image_path for EXPERT")
    return value


def _load_llava():
    llava_root = repo_root() / "metrics" / "upstreams" / "expert" / "LLaVA"
    if llava_root.exists():
        sys.path.insert(0, str(llava_root))
    try:
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EXPERT requires the LLaVA package. Initialize metrics/upstreams/expert/LLaVA "
            "and sync the EXPERT uv project."
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
        disable_torch_init=disable_torch_init,
    )


def _prompt(caption: str) -> str:
    return (
        "Evaluate the caption and assign a score on a scale of 0.0 to 1.0.\n\n"
        f"Caption: {caption}\n\n"
        "Score (0.0~1.0):"
    )


def _parse_score(text: str) -> float:
    matches = re.findall(r"\d+(?:\.\d+)?|\.\d+", text)
    if not matches:
        raise ValueError(f"could not parse EXPERT score from model output: {text!r}")
    for match in matches:
        value = float(match)
        if 0 <= value <= 1:
            return value
        if 1 < value <= 100:
            return value / 100.0
    raise ValueError(f"EXPERT score outside [0, 1] or [0, 100]: {text!r}")


def _select_token_index(generated_tokens: Any, token_id: int, *, prefer_second_zero: bool = False) -> int:
    matches = (generated_tokens == token_id).nonzero().flatten()
    if len(matches) == 0:
        raise ValueError(f"generated score token id {token_id} not found")
    if prefer_second_zero and len(matches) > 1:
        return int(matches[1])
    return int(matches[0])


def _score_text(raw_score: float) -> str:
    text = str(raw_score)
    if "." not in text:
        return f"{raw_score:.1f}"
    return text


def _generated_tokens(output: Any) -> Any:
    return output.sequences[0]


def _score_tokens(output: Any) -> Any:
    return output.sequences[0, 1:]


def _smooth_score(llava: Any, tokenizer: Any, output: Any, input_len: int, raw_score: float) -> float:
    rate2token = {digit: tokenizer.encode(str(digit))[-1] for digit in range(10)}
    generated_tokens = _score_tokens(output)
    raw = _score_text(raw_score)

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
    *,
    model_path: str,
    model_base: str,
    conv_mode: str,
    max_new_tokens: int,
    load_4bit: bool,
    load_8bit: bool,
    image_aspect_ratio: str,
) -> dict[str, Any]:
    if load_4bit and load_8bit:
        raise ValueError("EXPERT can load either 4bit or 8bit, not both")

    llava = _load_llava()
    predictions = _load_rows(predictions_path)
    item_ids = sorted(predictions)

    llava.disable_torch_init()
    model_name = llava.get_model_name_from_path(model_path)
    progress_status(
        f"Loading EXPERT/LLaVA model {model_path}: first use may download Hugging Face assets"
    )
    tokenizer, model, image_processor, _ = llava.load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
    )
    model.eval()
    device = next(model.parameters()).device

    scores: list[float] = []
    score_cache: dict[tuple[str, str], float] = {}
    for item_id in progress_iter(item_ids):
        pred = predictions[item_id]
        caption = _caption(pred)
        image_path = _image(pred)
        cache_key = (image_path, caption)
        cached_score = score_cache.get(cache_key)
        if cached_score is not None:
            scores.append(cached_score)
            continue

        query = _prompt(caption)
        conv = llava.conv_templates[conv_mode].copy()

        if model.config.mm_use_im_start_end:
            query = llava.DEFAULT_IM_START_TOKEN + llava.DEFAULT_IMAGE_TOKEN + llava.DEFAULT_IM_END_TOKEN + "\n" + query
        else:
            query = llava.DEFAULT_IMAGE_TOKEN + "\n" + query
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_tensor = llava.process_images(
            [image],
            image_processor,
            SimpleNamespace(image_aspect_ratio=image_aspect_ratio),
        )
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
        generation_model = getattr(getattr(model, "base_model", None), "model", model)
        with llava.torch.inference_mode():
            output = generation_model.generate(
                inputs=input_ids,
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
        text = tokenizer.decode(_generated_tokens(output), skip_special_tokens=True).strip()
        try:
            raw_score = _parse_score(text)
        except ValueError as exc:
            print(f"EXPERT warning: {exc}; using 0.0", file=sys.stderr)
            score = 0.0
            scores.append(score)
            score_cache[cache_key] = score
            continue
        try:
            score = _smooth_score(llava, tokenizer, output, input_ids.shape[1], raw_score)
        except ValueError as exc:
            print(f"EXPERT warning: {exc}; using raw score {raw_score}", file=sys.stderr)
            score = raw_score
        scores.append(score)
        score_cache[cache_key] = score

    return {
        "EXPERT": {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "per_item": dict(zip_strict(item_ids, scores)),
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load-4bit", dest="quantization", action="store_const", const="4bit")
    quantization.add_argument("--load-8bit", dest="quantization", action="store_const", const="8bit")
    quantization.add_argument("--no-load-4bit", dest="quantization", action="store_const", const="none")
    quantization.add_argument("--no-load-8bit", dest="quantization", action="store_const", const="none")
    parser.add_argument("--image-aspect-ratio", default="pad")
    parser.set_defaults(quantization="8bit")
    args = parser.parse_args(argv)

    result = compute(
        args.predictions,
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        load_4bit=args.quantization == "4bit",
        load_8bit=args.quantization == "8bit",
        image_aspect_ratio=args.image_aspect_ratio,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
