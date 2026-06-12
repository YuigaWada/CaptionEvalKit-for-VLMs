from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

from capevalkit.compat import zip_strict

DEFAULT_SERVER_URL = "http://localhost:2115"
DEFAULT_DOCKER_IMAGE = "capevalkit-jaspice:latest"
DEFAULT_CONTAINER_NAME = "capevalkit-jaspice-server"


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _records_to_jaspice_inputs(
    predictions_path: str,
    references_path: str,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    predictions = {str(row["id"]): row for row in _read_jsonl(predictions_path)}
    references = {str(row["id"]): row for row in _read_jsonl(references_path)}
    missing = sorted(set(references) - set(predictions))
    if missing:
        raise ValueError(f"missing predictions for ids: {', '.join(missing[:10])}")

    item_ids = sorted(references)
    jaspice_references: dict[str, list[str]] = {}
    candidates: dict[str, list[str]] = {}
    for item_id in item_ids:
        ref_row = references[item_id]
        pred_row = predictions[item_id]
        refs = ref_row.get("references", ref_row.get("captions"))
        if not isinstance(refs, list):
            raise ValueError(f"references row {item_id} must contain references or captions list")
        caption = pred_row.get("caption", pred_row.get("prediction"))
        if not isinstance(caption, str):
            raise ValueError(f"prediction row {item_id} must contain caption or prediction")
        jaspice_references[item_id] = [str(ref) for ref in refs]
        candidates[item_id] = [caption]
    return item_ids, jaspice_references, candidates


def compute_jaspice(
    predictions_path: str,
    references_path: str,
    *,
    batch_size: int = 16,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 60.0,
    auto_docker: bool = True,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    container_name: str = DEFAULT_CONTAINER_NAME,
) -> dict[str, Any]:
    item_ids, references, candidates = _records_to_jaspice_inputs(predictions_path, references_path)
    if auto_docker:
        _ensure_jaspice_server(
            server_url,
            docker_image=docker_image,
            container_name=container_name,
        )
    values = _compute_via_server(
        item_ids,
        references,
        candidates,
        batch_size=batch_size,
        server_url=server_url,
        timeout=timeout,
    )
    return {
        "JaSPICE": {
            "score": sum(values) / len(values) if values else 0.0,
            "per_item": dict(zip_strict(item_ids, values)),
        }
    }


def _compute_via_server(
    item_ids: list[str],
    references: dict[str, list[str]],
    candidates: dict[str, list[str]],
    *,
    batch_size: int,
    server_url: str,
    timeout: float,
) -> list[float]:
    values: list[float] = []
    step = max(1, batch_size)
    for start in range(0, len(item_ids), step):
        batch_ids = item_ids[start:start + step]
        batch_references = [
            [reference.replace(" ", "") for reference in references[item_id]]
            for item_id in batch_ids
        ]
        batch_candidates = [
            candidates[item_id][0].replace(" ", "")
            for item_id in batch_ids
        ]
        values.extend(
            _post_server(
                server_url,
                {
                    "references": batch_references,
                    "candidates": batch_candidates,
                    "batch_size": step,
                },
                timeout=timeout,
            )
        )
    return values


def _post_server(server_url: str, payload: dict[str, Any], *, timeout: float) -> list[float]:
    import urllib.error
    import urllib.request

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach JaSPICE server at {server_url}; "
            "install Docker or start a JaSPICE server manually"
        ) from exc

    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, list):
        raise ValueError(f"JaSPICE server returned non-list response: {decoded!r}")
    return [float(value) for value in decoded]


def _ensure_jaspice_server(
    server_url: str,
    *,
    docker_image: str,
    container_name: str,
) -> None:
    if _server_ready(server_url, timeout=1.0):
        return
    parsed = urlparse(server_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return

    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            f"JaSPICE server is not running at {server_url}, and Docker was not found"
        )

    host_port = parsed.port or 2115
    if not _docker_image_exists(docker, docker_image):
        _build_docker_image(docker, docker_image)

    if _docker_container_exists(docker, container_name):
        subprocess.run([docker, "start", container_name], check=True)
    else:
        subprocess.run(
            [
                docker,
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{host_port}:2115",
                docker_image,
            ],
            check=True,
        )

    wait_seconds = float(os.environ.get("CAPEVALKIT_JASPICE_DOCKER_TIMEOUT", "120"))
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _server_ready(server_url, timeout=2.0):
            return
        time.sleep(1.0)
    raise RuntimeError(f"JaSPICE Docker container started, but {server_url} did not become ready")


def _server_ready(server_url: str, *, timeout: float) -> bool:
    try:
        _post_server(
            server_url,
            {"references": [], "candidates": [], "batch_size": 1},
            timeout=timeout,
        )
    except Exception:
        return False
    return True


def _docker_image_exists(docker: str, image: str) -> bool:
    result = subprocess.run(
        [docker, "image", "inspect", image],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _docker_container_exists(docker: str, container_name: str) -> bool:
    result = subprocess.run(
        [docker, "container", "inspect", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _build_docker_image(docker: str, docker_image: str) -> None:
    source = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="capevalkit-jaspice-docker-") as tmp:
        context = Path(tmp) / "jaspice"
        shutil.copytree(
            source,
            context,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                "pyproject.toml",
                "uv.toml",
                "parsed.db",
                "wnjpn.db",
                "wnjpn.db.gz",
            ),
        )
        _patch_jaspice_dockerfile(context / "Dockerfile")
        subprocess.run([docker, "build", "-t", docker_image, str(context)], check=True)


def _patch_jaspice_dockerfile(path: Path) -> None:
    text = path.read_text()
    text = text.replace("ENV DEBIAN_FRONTEND noninteractive", "ENV DEBIAN_FRONTEND=noninteractive")
    text = text.replace("cd /tmp/knp-4.20 / &&\\", "cd /tmp/knp-4.20 &&\\")
    if "archive.debian.org/debian" not in text:
        marker = "RUN apt-get update --fix-missing &&\\\n"
        archive_setup = (
            "RUN sed -i \\\n"
            "    -e 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' \\\n"
            "    -e 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' \\\n"
            "    /etc/apt/sources.list &&\\\n"
            "    printf 'Acquire::Check-Valid-Until \"false\";\\n' > /etc/apt/apt.conf.d/99archive &&\\\n"
        )
        text = text.replace(marker, archive_setup + "    apt-get update --fix-missing &&\\\n", 1)
    path.write_text(text)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CAPEVALKIT_JASPICE_BATCH_SIZE", "16")))
    parser.add_argument("--server-url", default=os.environ.get("CAPEVALKIT_JASPICE_SERVER_URL", DEFAULT_SERVER_URL))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("CAPEVALKIT_JASPICE_TIMEOUT", "60")))
    parser.add_argument(
        "--no-docker",
        dest="auto_docker",
        action="store_false",
        default=_env_flag("CAPEVALKIT_JASPICE_AUTO_DOCKER", True),
        help="do not build/start the local JaSPICE Docker server automatically",
    )
    parser.add_argument("--docker-image", default=os.environ.get("CAPEVALKIT_JASPICE_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE))
    parser.add_argument("--container-name", default=os.environ.get("CAPEVALKIT_JASPICE_CONTAINER", DEFAULT_CONTAINER_NAME))
    args = parser.parse_args(argv)

    result = compute_jaspice(
        args.predictions,
        args.references,
        batch_size=args.batch_size,
        server_url=args.server_url,
        timeout=args.timeout,
        auto_docker=args.auto_docker,
        docker_image=args.docker_image,
        container_name=args.container_name,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
