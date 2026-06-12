from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from .paths import repo_root
from .runtime import RuntimeManager
from .runtime_env import apply_runtime_environment

SourceType = Literal["url", "hf-file", "hf-snapshot"]


@dataclass(frozen=True)
class Asset:
    name: str
    description: str
    source_type: SourceType
    default: bool = True
    url: str | None = None
    hf_repo: str | None = None
    hf_filename: str | None = None
    destination: str | None = None
    ready_path: str | None = None
    extract_to: str | None = None
    license_gated: bool = False
    license_note: str = ""
    distribution_note: str = ""


DOWNLOADABLE_ASSETS: tuple[Asset, ...] = (
    Asset(
        name="pacscore-pacs-vitb",
        description="PAC-S CLIP ViT-B/32 checkpoint used by pacscore/refpacscore.",
        source_type="url",
        url="https://drive.usercontent.google.com/download?id=1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae&export=download&confirm=t",
        destination="metrics/upstreams/pacscore/checkpoints/clip_ViT-B-32.pth",
        ready_path="metrics/upstreams/pacscore/checkpoints/clip_ViT-B-32.pth",
        license_note="Public upstream Google Drive checkpoint; checkpoint-specific license is not declared upstream.",
        distribution_note="Download on demand. Do not bundle in PyPI wheels or sdists.",
    ),
    Asset(
        name="pacscore-pacs-openclip-vitl",
        description="PAC-S OpenCLIP ViT-L/14 checkpoint used by pacscore-vitl/refpacscore-vitl/pacscoreavg.",
        source_type="url",
        default=False,
        url="https://drive.usercontent.google.com/download?id=1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae&export=download&confirm=t",
        destination="metrics/upstreams/pacscore/checkpoints/openClip_ViT-L-14.pth",
        ready_path="metrics/upstreams/pacscore/checkpoints/openClip_ViT-L-14.pth",
        license_note="Public upstream Google Drive checkpoint; checkpoint-specific license is not declared upstream.",
        distribution_note="Download on demand. Do not bundle in PyPI wheels or sdists.",
    ),
    Asset(
        name="pacscore-pacspp-vitb",
        description="PAC-S++ CLIP ViT-B/32 checkpoint.",
        source_type="url",
        default=False,
        url="https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-B-32.pth",
        destination="metrics/upstreams/pacscore/checkpoints/PAC++_clip_ViT-B-32.pth",
        ready_path="metrics/upstreams/pacscore/checkpoints/PAC++_clip_ViT-B-32.pth",
        license_note="Public upstream checkpoint URL; checkpoint-specific license is not declared upstream.",
        distribution_note="Download on demand. Do not bundle in PyPI wheels or sdists.",
    ),
    Asset(
        name="pacscore-pacspp-vitl",
        description="PAC-S++ CLIP ViT-L/14 checkpoint.",
        source_type="url",
        default=False,
        url="https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth",
        destination="metrics/upstreams/pacscore/checkpoints/PAC++_clip_ViT-L-14.pth",
        ready_path="metrics/upstreams/pacscore/checkpoints/PAC++_clip_ViT-L-14.pth",
        license_note="Public upstream checkpoint URL; checkpoint-specific license is not declared upstream.",
        distribution_note="Download on demand. Do not bundle in PyPI wheels or sdists.",
    ),
    Asset(
        name="polos-reprod",
        description="Polos reproduction checkpoint archive.",
        source_type="url",
        url="https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip",
        destination=".model-cache/polos/reprod.zip",
        ready_path=".model-cache/polos/reprod/reprod.ckpt",
        extract_to=".model-cache/polos",
        license_note="Polos code is BSD-3-Clause-Clear; the checkpoint is provided by the upstream project URL.",
        distribution_note="Download on demand and extract into the repo-local model cache.",
    ),
)


def asset_catalog() -> dict[str, Asset]:
    return {asset.name: asset for asset in DOWNLOADABLE_ASSETS}


def select_assets(
    names: Sequence[str],
    *,
    all_assets: bool = False,
    include_license_gated: bool = False,
) -> list[Asset]:
    catalog = asset_catalog()
    if names:
        selected: list[Asset] = []
        for name in names:
            try:
                selected.append(catalog[name])
            except KeyError as exc:
                known = ", ".join(sorted(catalog))
                raise KeyError(f"unknown downloadable asset {name!r}; known: {known}") from exc
        return selected

    if all_assets:
        return [
            asset
            for asset in DOWNLOADABLE_ASSETS
            if include_license_gated or not asset.license_gated
        ]
    return [
        asset
        for asset in DOWNLOADABLE_ASSETS
        if asset.default and (include_license_gated or not asset.license_gated)
    ]


def download_assets(
    assets: Iterable[Asset],
    *,
    root: Path | None = None,
    force: bool = False,
    accept_licenses: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    assets = list(assets)
    project_root = root or repo_root()
    if root is None and not dry_run:
        RuntimeManager().ensure_upstreams(upstreams_for_assets(assets))
    paths: list[Path] = []
    for asset in assets:
        if dry_run:
            target = _asset_target(asset, project_root)
            print(f"DRY {asset.name}\t{target or asset.hf_repo or asset.url}")
            continue
        path = download_asset(
            asset,
            root=project_root,
            force=force,
            accept_licenses=accept_licenses,
        )
        if path is not None:
            paths.append(path)
            print(f"OK  {asset.name}\t{path}")
        else:
            print(f"OK  {asset.name}")
    return paths


def download_asset(
    asset: Asset,
    *,
    root: Path | None = None,
    force: bool = False,
    accept_licenses: bool = False,
    hf_hub_download: Callable[..., str] | None = None,
    snapshot_download: Callable[..., str] | None = None,
    open_url: Callable[..., object] | None = None,
) -> Path | None:
    project_root = root or repo_root()
    if asset.license_gated and not accept_licenses:
        raise PermissionError(f"{asset.name} requires license acknowledgement")

    ready = _resolve_optional(project_root, asset.ready_path)
    if ready and ready.exists() and not force:
        return ready

    if asset.source_type == "url":
        return _download_url_asset(asset, project_root, force=force, open_url=open_url)
    if asset.source_type == "hf-file":
        return _download_hf_file_asset(
            asset,
            project_root,
            force=force,
            hf_hub_download=hf_hub_download,
        )
    if asset.source_type == "hf-snapshot":
        return _download_hf_snapshot_asset(
            asset,
            snapshot_download=snapshot_download,
        )
    raise ValueError(f"unsupported asset source_type: {asset.source_type}")


def format_asset_rows(assets: Iterable[Asset]) -> str:
    lines = []
    for asset in assets:
        flags = []
        if asset.default:
            flags.append("default")
        if asset.license_gated:
            flags.append("license-gated")
        flag_text = ",".join(flags) or "-"
        source = asset.url or (
            f"hf:{asset.hf_repo}/{asset.hf_filename}" if asset.hf_filename else f"hf:{asset.hf_repo}"
        )
        target = asset.destination or asset.ready_path or "(HF cache)"
        lines.append(f"{asset.name}\t{flag_text}\t{target}\t{source}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capevalkit download-assets")
    parser.add_argument("assets", nargs="*", help="asset names; omitted means default downloadable assets")
    parser.add_argument("--all", action="store_true", help="select every downloadable asset")
    parser.add_argument("--list", action="store_true", help="list known downloadable assets")
    parser.add_argument("--force", action="store_true", help="overwrite existing downloaded files")
    parser.add_argument("--dry-run", action="store_true", help="print selected assets without downloading")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(format_asset_rows(DOWNLOADABLE_ASSETS))
        return 0

    manager = RuntimeManager()
    project_root = manager.prepare_base()
    apply_runtime_environment(os.environ, project_root, cache_root=manager.context.cache_root)
    try:
        selected = select_assets(
            args.assets,
            all_assets=args.all,
        )
        download_assets(
            selected,
            root=project_root,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def upstreams_for_assets(assets: Iterable[Asset]) -> list[str]:
    upstreams: set[str] = set()
    for asset in assets:
        for value in (asset.destination, asset.ready_path, asset.extract_to):
            if not value:
                continue
            parts = Path(value).parts
            if len(parts) >= 3 and parts[0] == "metrics" and parts[1] == "upstreams":
                upstreams.add(parts[2])
    return sorted(upstreams)


def _download_url_asset(
    asset: Asset,
    root: Path,
    *,
    force: bool,
    open_url: Callable[..., object] | None,
) -> Path:
    if not asset.url or not asset.destination:
        raise ValueError(f"{asset.name} must declare url and destination")

    destination = root / asset.destination
    if destination.exists() and not force:
        if asset.extract_to:
            _extract_zip(destination, root / asset.extract_to)
            return _resolve_optional(root, asset.ready_path) or destination
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp")
    opener = open_url or urlopen
    request = Request(asset.url, headers={"User-Agent": "Mozilla/5.0"})
    with opener(request) as response, tmp.open("wb") as output:  # type: ignore[attr-defined]
        shutil.copyfileobj(response, output)
    tmp.replace(destination)

    if asset.extract_to:
        _extract_zip(destination, root / asset.extract_to)
        return _resolve_optional(root, asset.ready_path) or destination
    return destination


def _download_hf_file_asset(
    asset: Asset,
    root: Path,
    *,
    force: bool,
    hf_hub_download: Callable[..., str] | None,
) -> Path:
    if not asset.hf_repo or not asset.hf_filename:
        raise ValueError(f"{asset.name} must declare hf_repo and hf_filename")

    downloader = hf_hub_download or _import_hf_hub_download()
    cached_path = Path(downloader(repo_id=asset.hf_repo, filename=asset.hf_filename))
    destination = _resolve_optional(root, asset.destination)
    if destination:
        if destination.exists() and not force:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, destination)
        return destination
    return cached_path


def _download_hf_snapshot_asset(
    asset: Asset,
    *,
    snapshot_download: Callable[..., str] | None,
) -> Path:
    if not asset.hf_repo:
        raise ValueError(f"{asset.name} must declare hf_repo")
    downloader = snapshot_download or _import_snapshot_download()
    return Path(downloader(repo_id=asset.hf_repo))


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not _is_relative_to(target, destination_root):
                raise RuntimeError(f"unsafe zip member outside destination: {member.filename}")
        zf.extractall(destination)


def _asset_target(asset: Asset, root: Path) -> str | None:
    target = _resolve_optional(root, asset.ready_path) or _resolve_optional(root, asset.destination)
    return str(target) if target else None


def _resolve_optional(root: Path, path: str | None) -> Path | None:
    return root / path if path else None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _import_hf_hub_download() -> Callable[..., str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Hugging Face assets") from exc
    return hf_hub_download


def _import_snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Hugging Face assets") from exc
    return snapshot_download


if __name__ == "__main__":
    raise SystemExit(main())
