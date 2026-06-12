#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CAPEVALKIT_SMOKE_TMP:-}" ]]; then
  WORKDIR="$CAPEVALKIT_SMOKE_TMP"
  CUSTOM_WORKDIR=1
else
  WORKDIR="$(mktemp -d /tmp/capevalkit-dist-smoke.XXXXXX)"
  CUSTOM_WORKDIR=0
fi
VENV="$WORKDIR/venv"
CACHE="$WORKDIR/cache"
KEEP=0
VERBOSE=0
PYTHON_BIN="${PYTHON:-python3}"
ALL_REPRODUCE_JOBS="${CAPEVALKIT_SMOKE_JOBS:-1}"
ALL_REPRODUCE_GPU_JOBS="${CAPEVALKIT_SMOKE_GPU_JOBS:-1}"

usage() {
  cat <<'USAGE'
Usage: scripts/smoke_dist.sh [--keep] [--verbose]

Build the local package, install the wheel into a clean venv outside the
source tree, and run capevalkit all_reproduce from that installed wheel.

Options:
  --keep      keep the temporary test directory
  --verbose   show metric subprocess logs from all_reproduce
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep)
      KEEP=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cleanup() {
  if [[ "$KEEP" -eq 0 && "$CUSTOM_WORKDIR" -eq 0 ]]; then
    rm -rf "$WORKDIR"
  else
    echo "kept smoke dir: $WORKDIR"
  fi
}
trap cleanup EXIT

install_wheel_into_clean_venv() {
  if "$PYTHON_BIN" -c 'import ensurepip' >/dev/null 2>&1; then
    if "$PYTHON_BIN" -m venv "$VENV"; then
      "$VENV/bin/python" -m pip install --no-cache-dir "$WHEEL"
      return
    fi
    echo "python venv failed; falling back to uv venv" >&2
    rm -rf "$VENV"
  else
    echo "python ensurepip is unavailable; using uv venv" >&2
  fi

  uv venv --python "$PYTHON_BIN" "$VENV"
  uv pip install --python "$VENV/bin/python" "$WHEEL"
}

echo "==> source root: $ROOT"
echo "==> smoke dir:   $WORKDIR"

mkdir -p "$WORKDIR"
if [[ "$CUSTOM_WORKDIR" -eq 1 && -n "$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "CAPEVALKIT_SMOKE_TMP must be empty to avoid reusing caches: $WORKDIR" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Set PYTHON=/path/to/python or install python3." >&2
  exit 127
fi

unset PYTHONHOME
unset PYTHONPATH

export CAPEVALKIT_HOME="$CACHE"
export CAPEVALKIT_RUNTIME_MODE=cache
export CAPEVALKIT_RUNTIME_ROOT="$WORKDIR/runtime"
export CLIP_DOWNLOAD_ROOT="$WORKDIR/clip-cache"
export HF_DATASETS_CACHE="$WORKDIR/hf-cache/datasets"
export HF_HOME="$WORKDIR/hf-cache/home"
export HUGGINGFACE_HUB_CACHE="$WORKDIR/hf-cache/hub"
export PIP_CACHE_DIR="$WORKDIR/pip-cache"
export TORCH_HOME="$WORKDIR/torch-cache"
export UV_CACHE_DIR="$WORKDIR/uv-cache"
export UV_LINK_MODE="hardlink"
export XDG_CACHE_HOME="$WORKDIR/xdg-cache"

mkdir -p \
  "$CAPEVALKIT_HOME" \
  "$CAPEVALKIT_RUNTIME_ROOT" \
  "$CLIP_DOWNLOAD_ROOT" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME/torch/kernels"

cd "$ROOT"
rm -rf dist
uv build --wheel

WHEEL="$(find "$ROOT/dist" -maxdepth 1 -name '*.whl' | sort | tail -n 1)"
if [[ -z "$WHEEL" ]]; then
  echo "wheel was not built" >&2
  exit 1
fi

install_wheel_into_clean_venv

cd "$WORKDIR"

echo "==> doctor"
"$VENV/bin/capevalkit" doctor

echo "==> list metrics"
"$VENV/bin/capevalkit" list-metrics >"$WORKDIR/list-metrics.txt"
head -n 5 "$WORKDIR/list-metrics.txt"

PROJECT_ROOT="$("$VENV/bin/capevalkit" doctor | awk -F '\t' '$1 == "project_root" {print $2}')"
if [[ "$PROJECT_ROOT" == "$ROOT"* ]]; then
  echo "installed wheel is using the source tree as project_root: $PROJECT_ROOT" >&2
  exit 1
fi

echo "==> verify wheel resources"
"$VENV/bin/python" - "$ROOT" "$WHEEL" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

root = Path(sys.argv[1])
wheel = Path(sys.argv[2])
pyproject = tomllib.loads((root / "pyproject.toml").read_text())
force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())

missing: list[str] = []
errors: list[str] = []

if any(name.startswith("capevalkit/resources/metrics/upstreams/") for name in names):
    errors.append("wheel unexpectedly contains upstream repository payloads")

if "capevalkit/resources/upstreams.lock.json" not in names:
    missing.append("capevalkit/resources/upstreams.lock.json")

for source, destination in sorted(force_include.items()):
    source_path = root / source
    if not source_path.exists():
        missing.append(f"{source} (source path does not exist)")
        continue
    if source_path.is_file():
        expected = destination
        if expected not in names:
            missing.append(expected)
        continue
    for file_path in sorted(path for path in source_path.rglob("*") if path.is_file()):
        rel = file_path.relative_to(source_path).as_posix()
        expected = f"{destination}/{rel}"
        if expected not in names:
            missing.append(expected)

if errors or missing:
    for message in errors:
        print(message, file=sys.stderr)
    if missing:
        print("wheel is missing required resource files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "==> all_reproduce smoke"
ALL_REPRODUCE_ARGS=(
  all_reproduce
  --smoke \
  --jobs "$ALL_REPRODUCE_JOBS" \
  --gpu-jobs "$ALL_REPRODUCE_GPU_JOBS" \
  --output-dir "$WORKDIR/outputs/all-reproduce" \
  --summary "$WORKDIR/outputs/all-reproduce/summary.json" \
  --color never
)
if [[ "$VERBOSE" -eq 1 ]]; then
  ALL_REPRODUCE_ARGS+=(--verbose)
fi
"$VENV/bin/capevalkit" "${ALL_REPRODUCE_ARGS[@]}"

echo "dist smoke passed"
