from __future__ import annotations

from contextlib import contextmanager
import os
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, BinaryIO, Iterable, TypeVar


T = TypeVar("T")
_STATUS_REPORTER: Callable[[str], None] | None = None
_STATUS_REPORTER_LOCK = threading.RLock()
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def progress_update(count: int = 1) -> None:
    path = os.environ.get("CAPEVALKIT_PROGRESS_FILE")
    if not path or count <= 0:
        return
    try:
        with open(path, "a") as file:
            file.write(f"{count}\n")
    except OSError:
        return


@contextmanager
def progress_status_reporter(reporter: Callable[[str], None]) -> Iterator[None]:
    global _STATUS_REPORTER
    with _STATUS_REPORTER_LOCK:
        previous = _STATUS_REPORTER
        _STATUS_REPORTER = reporter
    try:
        yield
    finally:
        with _STATUS_REPORTER_LOCK:
            _STATUS_REPORTER = previous


def progress_status(message: str) -> None:
    message = " ".join(str(message).split())
    if not message:
        return
    with _STATUS_REPORTER_LOCK:
        reporter = _STATUS_REPORTER
    if reporter is not None:
        try:
            reporter(message)
            return
        except Exception:
            pass
    path = os.environ.get("CAPEVALKIT_PROGRESS_FILE")
    if path:
        try:
            with open(path, "a") as file:
                file.write(f"STATUS\t{message}\n")
            return
        except OSError:
            pass
    print(f"{_YELLOW}{format_status_line(message)}{_RESET}", file=sys.stderr, flush=True)


def format_status_line(message: str) -> str:
    return f"  {status_emoji(message)} {message}"


def status_emoji(message: str) -> str:
    normalized = message.strip().lower()
    if normalized.startswith("cached ") or normalized.startswith("downloaded "):
        return "✅"
    if normalized.startswith("using cached "):
        return "♻️"
    if normalized.startswith("preparing "):
        return "🛠️"
    if normalized.startswith("downloading "):
        return "⬇️"
    if normalized.startswith("syncing "):
        return "🔄"
    if normalized.startswith("loading "):
        return "📦"
    if normalized.startswith("extracting "):
        return "📂"
    return "✨"


def progress_iter(iterable: Iterable[T]) -> Iterable[T]:
    for item in iterable:
        yield item
        progress_update()


def copy_with_download_progress(
    response: Any,
    output: BinaryIO,
    *,
    label: str,
    chunk_size: int = 1024 * 1024,
) -> int:
    total = _content_length(response)
    downloaded = 0
    _download_status("Downloading", label, downloaded, total)
    next_fraction = 0.1
    next_bytes = 64 * 1024 * 1024
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        output.write(chunk)
        downloaded += len(chunk)
        if total:
            fraction = downloaded / total
            if fraction >= next_fraction:
                _download_status("Downloading", label, downloaded, total)
                next_fraction += 0.1
        elif downloaded >= next_bytes:
            _download_status("Downloading", label, downloaded, total)
            next_bytes += 64 * 1024 * 1024
    _download_status("Downloaded", label, downloaded, total)
    return downloaded


def _download_status(action: str, label: str, downloaded: int, total: int | None) -> None:
    if total:
        percent = 100.0 * downloaded / total if total else 0.0
        progress_status(
            f"{action} {label}: {_format_bytes(downloaded)} / {_format_bytes(total)} ({percent:.0f}%)"
        )
        return
    progress_status(f"{action} {label}: {_format_bytes(downloaded)}")


def _content_length(response: Any) -> int | None:
    value = None
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        value = headers.get("content-length") or headers.get("Content-Length")
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
