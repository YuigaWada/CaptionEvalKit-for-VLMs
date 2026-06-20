from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import shutil
from pathlib import Path
import sys

from capevalkit.infrastructure.runtime.paths import repo_root


@contextmanager
def _polos_validate_paths():
    root = repo_root()
    upstream = root / "metrics" / "upstreams" / "polos"
    paths = [upstream, upstream / "validate"]
    old_path = list(sys.path)
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old_path


def load_validate_cvpr():
    with _polos_validate_paths():
        return importlib.import_module("validate.validate_cvpr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polos-validate")
    parser.add_argument("--model", default=None)
    parser.add_argument("--hparams", default=None)
    parser.add_argument("--polos", action="store_true", default=True)
    parser.add_argument("--coef", action="store_true")
    parser.add_argument("--flickr", action="store_true")
    parser.add_argument("--pascal", action="store_true")
    parser.add_argument("--foil", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output
    args.output = None
    module = load_validate_cvpr()
    module.main(args)
    if output:
        source = Path("zeroshot_test_results.json")
        if not source.exists():
            raise FileNotFoundError(f"Polos validate did not create {source}")
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
