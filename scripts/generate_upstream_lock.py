from __future__ import annotations

import argparse
import configparser
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate capevalkit upstream lock from git submodules.")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--output",
        default="capevalkit/resources/upstreams.lock.json",
        help="lock file to write",
    )
    parser.add_argument("--version", default="0.1.1", help="capevalkit version to record")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / args.output
    submodules = _read_gitmodules(root / ".gitmodules")
    manifest_urls = _manifest_urls(root)
    revs = _submodule_revs(root)
    overlays = _overlay_paths(root)

    lock = {
        "capevalkit_version": args.version,
        "schema_version": 1,
        "upstreams": {},
    }
    for name, data in sorted(submodules.items()):
        path = data["path"]
        lock["upstreams"][name] = {
            "source": "git",
            "url": manifest_urls.get(path, data["url"]),
            "rev": revs.get(path, ""),
            "path": path,
            "uv_project": path,
            "overlays": overlays.get(path, []),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return 0


def _read_gitmodules(path: Path) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(path)
    result: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        name = section.split('"', 2)[1]
        result[name] = {
            "path": parser.get(section, "path"),
            "url": parser.get(section, "url"),
        }
    return result


def _submodule_revs(root: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    revs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            revs[parts[1]] = parts[0].lstrip("-+U")
    return revs


def _manifest_urls(root: Path) -> dict[str, str]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    result: dict[str, str] = {}
    for manifest in sorted((root / "metrics").glob("*/metric.toml")):
        data = tomllib.loads(manifest.read_text())
        repository = data.get("repository", {})
        if isinstance(repository, dict) and isinstance(repository.get("dir"), str):
            url = repository.get("url")
            if isinstance(url, str):
                result[repository["dir"]] = url
    return result


def _overlay_paths(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    overlays_root = root / "overlays"
    if not overlays_root.exists():
        return result
    for path in sorted(overlays_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(overlays_root).as_posix()
        parts = Path(relative).parts
        if len(parts) >= 3 and parts[0] == "metrics" and parts[1] == "upstreams":
            upstream_path = "/".join(parts[:3])
            result.setdefault(upstream_path, []).append(f"overlays/{relative}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
