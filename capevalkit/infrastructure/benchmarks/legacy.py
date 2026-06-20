from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import ast
import csv
from functools import lru_cache
import json
from math import isnan
import os
from pathlib import Path
import time
import tempfile
import threading
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import urlopen

from capevalkit.domain.benchmarks import BenchmarkItem
from capevalkit.domain.evaluation import (
    DEFAULT_SCORE_KEYS,
    BenchmarkModePolicy,
    MetricOutputNormalizationPolicy,
    ScoreKeyPolicy,
)
from capevalkit.domain.evaluation.correlations import kendall_correlations
from capevalkit.infrastructure.execution.dispatcher import dispatch
from capevalkit.infrastructure.manifests.catalog import get_manifest
from capevalkit.infrastructure.runtime.paths import repo_root

HF_BENCHMARK_CACHE = repo_root() / ".hf-cache" / "benchmarks"
DEFAULT_HF_COMPOSITE_REPO = "yuwd/Composite"
DEFAULT_HF_FLICKR8K_REPO = "yuwd/Flickr8k-HumanEval"
HF_NEBULA_REPO = "Ka2ukiMatsuda/Nebula"
HF_POLARIS_REPO = "yuwd/Polaris"
HF_SPICA_REPO = "hiranohachiman/Spica"
HF_LONGCAP_ARENA_REPO = "Ka2ukiMatsuda/LongCap-Arena"
SPICA_SPLIT_FILES = ("spica_train.csv", "spica_val.csv", "spica_test.csv")
_HF_CACHE_THREAD_LOCK = threading.Lock()
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]
LONGCAPARENA_BENCHMARKS = {
    "longcaparena-testa-desc": ("dci_val.desc", "desc", "desc_dci_val.csv"),
    "longcaparena-testa-rel": ("dci_val.rel", "rel", "rel_dci_val.csv"),
    "longcaparena-testa-flu": ("dci_val.flu", "flu", "flu_dci_val.csv"),
    "longcaparena-testb-desc": ("dci_test.desc", "desc", "desc_dci_test.csv"),
    "longcaparena-testb-rel": ("dci_test.rel", "rel", "rel_dci_test.csv"),
    "longcaparena-testb-flu": ("dci_test.flu", "flu", "flu_dci_test.csv"),
}


def _literal_refs(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _image_path(image_dir: Path, image_name: str) -> str:
    path = image_dir / image_name
    if path.exists():
        return str(path)
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_dir / f"{image_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(path)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def _load_csv_benchmark(csv_path: Path, image_dir: Path) -> list[BenchmarkItem]:
    items = []
    with csv_path.open(newline="") as file:
        for index, row in enumerate(csv.DictReader(file)):
            item_id = str(row.get("id") or row.get("imgid") or index)
            image_name = str(row.get("image") or row.get("image_path") or row.get("imgid"))
            caption = row.get("mt") if row.get("mt") is not None else row.get("cand")
            if caption is None:
                raise ValueError(f"{csv_path}: missing mt/cand column")
            items.append(
                BenchmarkItem(
                    id=f"{item_id}_{index}",
                    image=_image_path(image_dir, image_name),
                    caption=_normalize_text(caption),
                    references=[_normalize_text(ref) for ref in _literal_refs(row.get("refs", "[]"))],
                    score=float(row["score"]),
                )
            )
    return items


def _explicit_non_repo_data_root(data_root: str | None) -> bool:
    if data_root is None:
        return False
    return Path(data_root).expanduser().absolute() != (repo_root() / "data").absolute()


def _hf_splits(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    env_name = f"IC_EVAL_{name.upper()}_SPLITS"
    value = os.environ.get(env_name)
    if not value:
        return default
    splits = tuple(split.strip() for split in value.split(",") if split.strip())
    if not splits:
        raise ValueError(f"{env_name} must contain at least one split")
    return splits


def _hf_cache_path(cache_name: str, splits: tuple[str, ...]) -> Path:
    split_key = "_".join(splits)
    return HF_BENCHMARK_CACHE / f"{cache_name}-{split_key}.jsonl"


def _limited_path(path: Path, limit: int | None) -> Path:
    if limit is None:
        return path
    return path.with_name(f"{path.stem}-limit{limit}{path.suffix}")


def _limited_dir(path: Path, limit: int | None) -> Path:
    if limit is None:
        return path
    return path.with_name(f"{path.name}-limit{limit}")


def _limit_items(items: list[BenchmarkItem], limit: int | None) -> list[BenchmarkItem]:
    if limit is None:
        return items
    return items[:limit]


def _hf_repo(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


def _hf_read_retries() -> int:
    value = os.environ.get("IC_EVAL_HF_READ_RETRIES", "3")
    try:
        return max(1, int(value))
    except ValueError:
        return 3


def _hf_retry_delay_seconds() -> float:
    value = os.environ.get("IC_EVAL_HF_RETRY_DELAY", "2")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 2.0


def _retry_hf_read(description: str, read: Callable[[], Any]) -> Any:
    attempts = _hf_read_retries()
    delay = _hf_retry_delay_seconds()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return read()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            if delay:
                time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"failed to read {description} after {attempts} attempts") from last_error


def _cached_hf_file(repo_id: str, filename: str, cache_name: str) -> Path:
    cache_path = HF_BENCHMARK_CACHE / cache_name
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = _cache_tmp_path(cache_path)
                with urlopen(url, timeout=120) as response:
                    tmp_path.write_bytes(response.read())
                tmp_path.replace(cache_path)
    return cache_path


def _cache_tmp_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")


@contextmanager
def _hf_cache_write_lock():
    HF_BENCHMARK_CACHE.mkdir(parents=True, exist_ok=True)
    lock_path = HF_BENCHMARK_CACHE / ".write.lock"
    with _HF_CACHE_THREAD_LOCK:
        with lock_path.open("w") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)


def _hf_parquet_paths(repo_id: str, splits: tuple[str, ...]) -> list[str]:
    api = (
        "https://huggingface.co/api/datasets/"
        f"{quote(repo_id, safe='/')}/tree/refs%2Fconvert%2Fparquet/default?recursive=true"
    )
    with urlopen(api, timeout=60) as response:
        entries = json.load(response)
    split_prefixes = {f"default/{split}/" for split in splits}
    paths = [
        entry["path"]
        for entry in entries
        if entry.get("type") == "file"
        and str(entry.get("path", "")).endswith(".parquet")
        and any(str(entry["path"]).startswith(prefix) for prefix in split_prefixes)
    ]
    if not paths:
        raise FileNotFoundError(f"no Hugging Face parquet shards found for {repo_id} splits={splits}")
    return sorted(paths)


def _write_hf_cache(
    *,
    repo_id: str,
    cache_path: Path,
    splits: tuple[str, ...],
    columns: list[str],
    row_to_item: Callable[[dict[str, Any], str, int], BenchmarkItem],
    limit: int | None = None,
) -> None:
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face benchmark loading requires fsspec and pyarrow; run `uv sync`"
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _cache_tmp_path(cache_path)
    fs = fsspec.filesystem("https")
    split_offsets: dict[str, int] = {}
    written = 0
    with tmp_path.open("w") as file:
        for parquet_path in _hf_parquet_paths(repo_id, splits):
            if limit is not None and written >= limit:
                break
            split = Path(parquet_path).parent.name
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/refs%2Fconvert%2Fparquet/{parquet_path}"
            def read_table():
                with fs.open(url, "rb") as parquet_file:
                    return pq.read_table(parquet_file, columns=columns)

            table = _retry_hf_read(f"{repo_id}/{parquet_path}", read_table)
            offset = split_offsets.get(split, 0)
            for row_index, row in enumerate(table.to_pylist()):
                if limit is not None and written >= limit:
                    break
                item = row_to_item(row, split, offset + row_index)
                file.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                written += 1
            split_offsets[split] = offset + table.num_rows
    tmp_path.replace(cache_path)


def _read_cached_items(path: Path) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _hf_image_cache_is_current(cache_path: Path, image_dir: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        image_root = image_dir.resolve()
        for line in cache_path.read_text().splitlines():
            if not line.strip():
                continue
            item = BenchmarkItem(**json.loads(line))
            image_path = Path(item.image)
            image_path.resolve().relative_to(image_root)
            return image_path.exists()
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False
    return False


def _write_hf_embedded_image_cache(
    *,
    repo_id: str,
    config_name: str,
    split: str,
    cache_path: Path,
    image_dir: Path,
    limit: int | None = None,
) -> None:
    try:
        from datasets import Image, load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Hugging Face dataset loading requires datasets; run `uv sync`") from exc

    dataset_split = f"{split}[:{limit}]" if limit is not None else split
    dataset = load_dataset(repo_id, config_name, split=dataset_split)
    dataset = dataset.cast_column("img", Image(decode=False))
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _cache_tmp_path(cache_path)
    with tmp_path.open("w") as file:
        for row_index, row in enumerate(dataset):
            if limit is not None and row_index >= limit:
                break
            item = _hf_embedded_row_to_item(row, image_dir=image_dir, row_index=row_index)
            file.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
    tmp_path.replace(cache_path)


def _write_hf_image_column_cache(
    *,
    repo_id: str,
    splits: tuple[str, ...],
    cache_path: Path,
    image_dir: Path,
    columns: list[str],
    image_column: str,
    row_to_item: Callable[[dict[str, Any], str, int, Path], BenchmarkItem],
    limit: int | None = None,
) -> None:
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face image benchmark loading requires fsspec and pyarrow; run `uv sync`"
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _cache_tmp_path(cache_path)
    fs = fsspec.filesystem("https")
    split_offsets: dict[str, int] = {}
    written = 0
    with tmp_path.open("w") as file:
        for parquet_path in _hf_parquet_paths(repo_id, splits):
            if limit is not None and written >= limit:
                break
            split = Path(parquet_path).parent.name
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/refs%2Fconvert%2Fparquet/{parquet_path}"
            split_image_dir = image_dir / split
            offset = split_offsets.get(split, 0)
            def read_rows() -> tuple[list[str], int]:
                rows = []
                row_offset = offset
                remaining = None if limit is None else max(0, limit - written)
                with fs.open(url, "rb") as parquet_file:
                    reader = pq.ParquetFile(parquet_file)
                    for batch in reader.iter_batches(columns=columns, batch_size=256):
                        for row in batch.to_pylist():
                            if remaining is not None and len(rows) >= remaining:
                                return rows, row_offset
                            if row.get(image_column) is None:
                                raise ValueError(f"{repo_id}/{split}: missing {image_column} image column")
                            item = row_to_item(row, split, row_offset, split_image_dir)
                            rows.append(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                            row_offset += 1
                return rows, row_offset

            rows, offset = _retry_hf_read(f"{repo_id}/{parquet_path}", read_rows)
            file.writelines(rows)
            written += len(rows)
            split_offsets[split] = offset
    tmp_path.replace(cache_path)


def _hf_embedded_row_to_item(row: dict[str, Any], *, image_dir: Path, row_index: int) -> BenchmarkItem:
    image_name = str(row.get("imgid") or row.get("file_name") or row.get("id") or row_index)
    image_path = _write_hf_embedded_image(row.get("img"), image_dir, image_name)
    return BenchmarkItem(
        id=str(row.get("id") or f"{Path(image_name).stem}_{row_index}"),
        image=str(image_path),
        caption=_normalize_text(row.get("cand", row.get("mt", row.get("caption", "")))),
        references=[_normalize_text(ref) for ref in _literal_refs(row.get("refs", row.get("references", [])))],
        score=float(row.get("human_score", row.get("score"))),
    )


def _write_hf_embedded_image(image_value: Any, image_dir: Path, image_name: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / Path(image_name).name
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if image_bytes is not None:
            if not target.exists():
                target.write_bytes(image_bytes)
            return target
        image_path = image_value.get("path")
        if image_path:
            source = Path(str(image_path))
            if source.exists():
                if not target.exists():
                    target.write_bytes(source.read_bytes())
                return target
    if target.exists():
        return target
    raise FileNotFoundError(f"HF row for {image_name} does not contain image bytes and {target} is missing")


def _load_hf_embedded_image_benchmark(
    *,
    cache_name: str,
    repo_id: str,
    config_name: str,
    split: str = "test",
    limit: int | None = None,
) -> list[BenchmarkItem]:
    cache_path = _limited_path(HF_BENCHMARK_CACHE / f"{cache_name}-{config_name}-{split}.jsonl", limit)
    image_dir = _limited_dir(HF_BENCHMARK_CACHE / f"{cache_name}-images", limit) / config_name / split
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
                _write_hf_embedded_image_cache(
                    repo_id=repo_id,
                    config_name=config_name,
                    split=split,
                    cache_path=cache_path,
                    image_dir=image_dir,
                    limit=limit,
                )
    items = _read_cached_items(cache_path)
    if not items:
        raise ValueError(f"{cache_name}/{config_name}/{split} Hugging Face cache has no benchmark items: {cache_path}")
    return items


def _load_hf_cached_benchmark(
    *,
    cache_name: str,
    repo_id: str,
    splits: tuple[str, ...],
    columns: list[str],
    row_to_item: Callable[[dict[str, Any], str, int], BenchmarkItem],
    limit: int | None = None,
) -> list[BenchmarkItem]:
    cache_path = _limited_path(_hf_cache_path(cache_name, splits), limit)
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
                _write_hf_cache(
                    repo_id=repo_id,
                    cache_path=cache_path,
                    splits=splits,
                    columns=columns,
                    row_to_item=row_to_item,
                    limit=limit,
                )
    items = _read_cached_items(cache_path)
    if not items:
        raise ValueError(f"{cache_name} Hugging Face cache has no benchmark items: {cache_path}")
    return items


def _hf_image_name(image_value: Any, fallback: int) -> str:
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        if image_path:
            return Path(str(image_path)).name
    return str(fallback)


def _hf_nebula_row_to_item(row: dict[str, Any], split: str, row_index: int, image_dir: Path) -> BenchmarkItem:
    image_name = str(row.get("file_name") or row.get("imgid") or _hf_image_name(row.get("image"), row_index))
    image_path = _write_hf_embedded_image(row.get("image"), image_dir, image_name)
    return BenchmarkItem(
        id=f"{split}_{row_index}_{Path(image_name).stem}",
        image=str(image_path),
        caption=_normalize_text(row["mt"]),
        references=[_normalize_text(ref) for ref in _literal_refs(row["refs"])],
        score=float(row.get("human_score", row.get("score"))),
    )


def _hf_polaris_row_to_item(row: dict[str, Any], split: str, row_index: int, image_dir: Path) -> BenchmarkItem:
    image_name = str(row.get("imgid") or row.get("file_name") or row.get("id") or _hf_image_name(row.get("img"), row_index))
    image_path = _write_hf_embedded_image(row.get("img"), image_dir, image_name)
    caption = row.get("cand") if row.get("cand") is not None else row.get("mt")
    if caption is None:
        raise ValueError("Polaris Hugging Face row is missing cand/mt")
    return BenchmarkItem(
        id=f"{Path(image_name).stem}_{row_index}",
        image=str(image_path),
        caption=_normalize_text(caption),
        references=[_normalize_text(ref) for ref in _literal_refs(row["refs"])],
        score=float(row.get("human_score", row.get("score"))),
    )


def _item_score_key(image: str, caption: str) -> str:
    return f"{Path(image).stem}\t{_normalize_text(caption)}"


@lru_cache(maxsize=1)
def _load_spica_score_lookup() -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for filename in SPICA_SPLIT_FILES:
        path = _cached_hf_file(HF_SPICA_REPO, filename, filename)
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                key = _item_score_key(str(row["imgid"]), str(row["mt"]))
                totals[key] = totals.get(key, 0.0) + float(row["score"])
                counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def _apply_nebula_score_corrections(items: list[BenchmarkItem]) -> list[BenchmarkItem]:
    if os.environ.get("IC_EVAL_DISABLE_NEBULA_SCORE_CORRECTIONS"):
        return items

    lookup = _load_spica_score_lookup()
    corrected = []
    for item in items:
        score = lookup.get(_item_score_key(item.image, item.caption), item.score)
        corrected.append(
            BenchmarkItem(
                id=item.id,
                image=item.image,
                caption=item.caption,
                references=item.references,
                score=score,
            )
        )
    return corrected


def _load_hf_nebula(data_root: str | None = None, limit: int | None = None) -> list[BenchmarkItem]:
    splits = _hf_splits("nebula", ("test",))
    cache_path = _limited_path(_hf_cache_path("nebula", splits), limit)
    image_dir = _limited_dir(HF_BENCHMARK_CACHE / "nebula-images", limit)
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not _hf_image_cache_is_current(cache_path, image_dir):
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not _hf_image_cache_is_current(cache_path, image_dir):
                _write_hf_image_column_cache(
                    repo_id=HF_NEBULA_REPO,
                    splits=splits,
                    cache_path=cache_path,
                    image_dir=image_dir,
                    columns=["file_name", "image", "refs", "mt", "human_score"],
                    image_column="image",
                    row_to_item=_hf_nebula_row_to_item,
                    limit=limit,
                )
    items = _read_cached_items(cache_path)
    if splits == ("test",):
        return _apply_nebula_score_corrections(items)
    return items


def _load_hf_polaris(data_root: str | None = None, limit: int | None = None) -> list[BenchmarkItem]:
    splits = _hf_splits("polaris", ("test",))
    if splits != ("test",):
        raise ValueError("Polaris Hugging Face loader currently supports only IC_EVAL_POLARIS_SPLITS=test")
    cache_path = _limited_path(_hf_cache_path("polaris", splits), limit)
    image_dir = _limited_dir(HF_BENCHMARK_CACHE / "polaris-images", limit)
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not _hf_image_cache_is_current(cache_path, image_dir):
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not _hf_image_cache_is_current(cache_path, image_dir):
                _write_hf_image_column_cache(
                    repo_id=HF_POLARIS_REPO,
                    splits=splits,
                    cache_path=cache_path,
                    image_dir=image_dir,
                    columns=["refs", "cand", "img", "human_score"],
                    image_column="img",
                    row_to_item=_hf_polaris_row_to_item,
                    limit=limit,
                )
    return _read_cached_items(cache_path)

def _load_hf_composite(limit: int | None = None) -> list[BenchmarkItem]:
    return _load_hf_embedded_image_benchmark(
        cache_name="composite",
        repo_id=_hf_repo("IC_EVAL_HF_COMPOSITE_REPO", DEFAULT_HF_COMPOSITE_REPO),
        config_name="default",
        limit=limit,
    )


def _load_hf_flickr8k(variant: str, limit: int | None = None) -> list[BenchmarkItem]:
    config_name = "crowdflower" if variant == "cf" else "expert"
    return _load_hf_embedded_image_benchmark(
        cache_name="flickr8k",
        repo_id=_hf_repo("IC_EVAL_HF_FLICKR8K_REPO", DEFAULT_HF_FLICKR8K_REPO),
        config_name=config_name,
        limit=limit,
    )


def longcaparena_mode(benchmark_name: str) -> str | None:
    spec = LONGCAPARENA_BENCHMARKS.get(benchmark_name)
    return spec[1] if spec else None


def _write_hf_longcaparena_cache(cache_path: Path, split: str, limit: int | None = None) -> None:
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LongCap-Arena loading requires fsspec and pyarrow; run `uv sync`"
        ) from exc

    paths = _hf_parquet_paths(HF_LONGCAP_ARENA_REPO, (split,))
    image_dir = _limited_dir(HF_BENCHMARK_CACHE / "longcaparena-images", limit) / split
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _cache_tmp_path(cache_path)
    fs = fsspec.filesystem("https")
    with tmp_path.open("w") as file:
        row_offset = 0
        written = 0
        for parquet_path in paths:
            if limit is not None and written >= limit:
                break
            url = (
                "https://huggingface.co/datasets/"
                f"{HF_LONGCAP_ARENA_REPO}/resolve/refs%2Fconvert%2Fparquet/{parquet_path}"
            )
            with fs.open(url, "rb") as parquet_file:
                reader = pq.ParquetFile(parquet_file)
                for batch in reader.iter_batches(
                    columns=["file_name", "image", "refs", "cand", "score"],
                    batch_size=256,
                ):
                    for row in batch.to_pylist():
                        if limit is not None and written >= limit:
                            break
                        file_name = str(row["file_name"])
                        image_path = image_dir / file_name
                        image_payload = row.get("image")
                        image_bytes = image_payload.get("bytes") if isinstance(image_payload, dict) else None
                        if image_bytes is not None and not image_path.exists():
                            image_path.write_bytes(image_bytes)
                        item = BenchmarkItem(
                            id=f"{split}_{row_offset}_{Path(file_name).stem}",
                            image=str(image_path),
                            caption=_normalize_text(row["cand"]),
                            references=[_normalize_text(ref) for ref in row["refs"]],
                            score=float(row["score"]),
                        )
                        file.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                        row_offset += 1
                        written += 1
                    if limit is not None and written >= limit:
                        break
    tmp_path.replace(cache_path)


def _load_hf_longcaparena(benchmark_name: str, limit: int | None = None) -> list[BenchmarkItem]:
    split, _, _ = LONGCAPARENA_BENCHMARKS[benchmark_name]
    cache_path = _limited_path(HF_BENCHMARK_CACHE / f"{benchmark_name}.jsonl", limit)
    if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
        with _hf_cache_write_lock():
            if os.environ.get("IC_EVAL_REFRESH_HF_CACHE") or not cache_path.exists():
                _write_hf_longcaparena_cache(cache_path, split, limit=limit)
    items = _read_cached_items(cache_path)
    if not items:
        raise ValueError(f"{benchmark_name} Hugging Face cache has no benchmark items: {cache_path}")
    return items


def _load_local_longcaparena(benchmark_name: str, data_root: str | None) -> list[BenchmarkItem]:
    _, _, csv_name = LONGCAPARENA_BENCHMARKS[benchmark_name]
    checked = []
    for root in _data_roots(data_root):
        for base in (
            root / "longcaparena",
            root / "LongCap-Arena",
            root / "metrics" / "upstreams" / "vela" / "data",
            root,
        ):
            for csv_path in (base / "test" / csv_name, base / csv_name):
                checked.append(str(csv_path))
                if csv_path.exists():
                    for image_dir in (
                        base / "images",
                        base.parent / "images",
                        root / "longcaparena" / "images",
                        root / "images",
                    ):
                        if image_dir.exists():
                            return _load_csv_benchmark(csv_path, image_dir)
                    return _load_csv_benchmark(csv_path, base / "images")
    raise FileNotFoundError(f"missing LongCap-Arena {csv_name}; checked: {', '.join(checked)}")


def _load_flickr(data_root: Path, variant: str) -> list[BenchmarkItem]:
    json_name = "crowdflower_flickr8k.json" if variant == "cf" else "flickr8k.json"
    data = json.loads((data_root / json_name).read_text())
    items = []
    for image_id, record in data.items():
        image_name = Path(record["image_path"]).name
        refs = [_normalize_text(ref) for ref in record["ground_truth"]]
        for index, judgement in enumerate(record["human_judgement"]):
            score = float(judgement["rating"])
            if isnan(score):
                continue
            items.append(
                BenchmarkItem(
                    id=f"{Path(image_name).stem}_{index}",
                    image=str(data_root / "images" / image_name),
                    caption=_normalize_text(judgement["caption"]),
                    references=refs,
                    score=score,
                )
            )
    return items


def default_data_root() -> Path:
    return _data_roots()[0]


def _data_roots(data_root: str | None = None) -> list[Path]:
    roots = []
    if data_root:
        roots.append(Path(data_root).expanduser().absolute())
    if os.environ.get("IC_EVAL_DATA_ROOT"):
        roots.append(Path(os.environ["IC_EVAL_DATA_ROOT"]).expanduser().absolute())
    roots.extend(
        [
            repo_root() / "data",
            repo_root(),
        ]
    )
    unique = []
    explicit = Path(data_root) if data_root else None
    for root in roots:
        if root not in unique and (root.exists() or root == explicit):
            unique.append(root)
    return unique


def _data_dir(data_root: str | None, subdir: str, filename: str) -> Path:
    checked = []
    for root in _data_roots(data_root):
        for candidate in (root, root / subdir):
            checked.append(str(candidate / filename))
            if (candidate / filename).exists():
                return candidate
    raise FileNotFoundError(f"missing {subdir}/{filename}; checked: {', '.join(checked)}")


def load_benchmark(name: str, data_root: str | None = None, limit: int | None = None) -> list[BenchmarkItem]:
    if name == "composite":
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_composite(limit=limit)
        try:
            data = _data_dir(data_root, "composite", "en_test_composite_da2.csv")
            filename = "en_test_composite_da2.csv"
        except FileNotFoundError:
            data = _data_dir(data_root, "composite", "en_test_composite_da.csv")
            filename = "en_test_composite_da.csv"
        return _limit_items(_load_csv_benchmark(data / filename, data / "images"), limit)
    if name == "polaris":
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_polaris(data_root, limit=limit)
        data = _data_dir(data_root, "polaris", "polaris_test.csv")
        return _limit_items(_load_csv_benchmark(data / "polaris_test.csv", data / "images"), limit)
    if name == "nebula":
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_nebula(data_root, limit=limit)
        data = _data_dir(data_root, "nebula", "nebula_test.csv")
        return _limit_items(_load_csv_benchmark(data / "nebula_test.csv", data / "images"), limit)
    if name == "flickr8k-cf":
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_flickr8k("cf", limit=limit)
        data = _data_dir(data_root, "flickr8k", "crowdflower_flickr8k.json")
        return _limit_items(_load_flickr(data, "cf"), limit)
    if name == "flickr8k-ex":
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_flickr8k("expert", limit=limit)
        data = _data_dir(data_root, "flickr8k", "flickr8k.json")
        return _limit_items(_load_flickr(data, "expert"), limit)
    if name in LONGCAPARENA_BENCHMARKS:
        if not _explicit_non_repo_data_root(data_root):
            return _load_hf_longcaparena(name, limit=limit)
        return _limit_items(_load_local_longcaparena(name, data_root), limit)
    raise ValueError(f"unknown benchmark: {name}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _score_values(metric_output: dict[str, Any], score_key: str | None = None) -> tuple[str, dict[str, float]]:
    return MetricOutputNormalizationPolicy().score_values(metric_output, score_key)


def _default_item_score_key(value: Any) -> str:
    return MetricOutputNormalizationPolicy().default_item_score_key(value)


def _extract_item_score(value: Any, score_key: str | None) -> float:
    return MetricOutputNormalizationPolicy().extract_item_score(value, score_key)


def _enrich_metric_output(
    metric_output: dict[str, Any],
    *,
    items: list[BenchmarkItem],
    score_key: str | None,
) -> dict[str, Any]:
    output = deepcopy(metric_output)
    items_by_id = {item.id: item for item in items}
    for key, value in output.items():
        if isinstance(value, dict) and isinstance(value.get("per_item"), dict):
            value["per_item"] = _enrich_per_item_scores(value["per_item"], items_by_id, key)
    if isinstance(output.get("per_item"), dict):
        output["per_item"] = _enrich_per_item_scores(
            output["per_item"],
            items_by_id,
            score_key or _default_item_score_key(next(iter(output["per_item"].values()), {})),
        )
    return output


def _enrich_per_item_scores(
    per_item: dict[str, Any],
    items_by_id: dict[str, BenchmarkItem],
    score_key: str | None,
) -> dict[str, dict[str, Any]]:
    enriched = {}
    for item_id, value in per_item.items():
        item_id = str(item_id)
        item = items_by_id.get(item_id)
        if item is None:
            continue
        payload: dict[str, Any] = {
            "score": _extract_item_score(value, score_key),
            "ground_truth_score": item.score,
            "caption": item.caption,
            "image": item.image,
            "references": list(item.references),
        }
        scores = _item_score_map(value)
        if scores:
            payload["scores"] = scores
        enriched[item_id] = payload
    return enriched


def _item_score_map(value: Any) -> dict[str, float]:
    return MetricOutputNormalizationPolicy().item_score_map(value)


def _kendall(values: list[float], targets: list[float]) -> dict[str, float]:
    return kendall_correlations(values, targets)


def benchmark_metric(
    metric_name: str,
    benchmark_name: str,
    output: str,
    *,
    data_root: str | None = None,
    metric_args: list[str] | None = None,
    use_references: bool = True,
    score_key: str | None = None,
    limit: int | None = None,
    quiet: bool = False,
) -> int:
    code, items, metric_output = run_metric_on_benchmark(
        metric_name,
        benchmark_name,
        data_root=data_root,
        metric_args=metric_args,
        use_references=use_references,
        limit=limit,
        quiet=quiet,
    )
    if code != 0:
        return code
    write_benchmark_result(
        metric_name,
        benchmark_name,
        output,
        items=items,
        metric_output=metric_output,
        score_key=score_key,
    )
    return 0


def run_metric_on_benchmark(
    metric_name: str,
    benchmark_name: str,
    *,
    data_root: str | None = None,
    metric_args: list[str] | None = None,
    use_references: bool = True,
    limit: int | None = None,
    quiet: bool = False,
    show_progress: bool = True,
) -> tuple[int, list[BenchmarkItem], dict[str, Any]]:
    manifest = get_manifest(metric_name)
    items = load_benchmark(benchmark_name, data_root, limit=limit)
    if not items:
        raise ValueError(f"{benchmark_name} has no benchmark items")
    mode_policy = BenchmarkModePolicy(
        {name: mode for name, (_, mode, _) in LONGCAPARENA_BENCHMARKS.items()}
    )
    metric_args = mode_policy.metric_args(metric_name, benchmark_name, metric_args)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        predictions = tmp_path / "predictions.jsonl"
        references = tmp_path / "references.jsonl"
        raw_scores = tmp_path / "scores.json"
        _write_jsonl(
            predictions,
            [{"id": item.id, "caption": item.caption, "image": item.image} for item in items],
        )
        _write_jsonl(
            references,
            [{"id": item.id, "references": item.references} for item in items],
        )
        command = [
            *manifest.runner,
            "--predictions",
            str(predictions),
            "--output",
            str(raw_scores),
            *metric_args,
        ]
        if use_references:
            command[command.index("--output"):command.index("--output")] = ["--references", str(references)]
        code = dispatch(
            metric_name,
            command,
            quiet=quiet,
            progress_total=len(items) if show_progress else None,
            progress_desc=f"{metric_name}/{benchmark_name}" if show_progress else None,
        )
        if code != 0:
            return code, items, {}
        metric_output = json.loads(raw_scores.read_text())
    return 0, items, metric_output


def write_benchmark_result(
    metric_name: str,
    benchmark_name: str,
    output: str | Path,
    *,
    items: list[BenchmarkItem],
    metric_output: dict[str, Any],
    score_key: str | None = None,
) -> None:
    result = benchmark_result(
        metric_name,
        benchmark_name,
        items=items,
        metric_output=metric_output,
        score_key=score_key,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def benchmark_result(
    metric_name: str,
    benchmark_name: str,
    *,
    items: list[BenchmarkItem],
    metric_output: dict[str, Any],
    score_key: str | None = None,
) -> dict[str, Any]:
    score_key = ScoreKeyPolicy(
        DEFAULT_SCORE_KEYS,
        longcaparena_benchmarks=set(LONGCAPARENA_BENCHMARKS),
    ).score_key(metric_name, benchmark_name, score_key)
    score_name, per_item = _score_values(metric_output, score_key)
    ordered_scores = [per_item[item.id] for item in items]
    human_scores = [item.score for item in items]
    return {
        "metric": metric_name,
        "benchmark": benchmark_name,
        "score_name": score_name,
        "num_samples": len(items),
        "correlations": _kendall(ordered_scores, human_scores),
        "raw_metric_output": _enrich_metric_output(metric_output, items=items, score_key=score_name),
    }
