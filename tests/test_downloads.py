from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from capevalkit.cli import build_parser
from capevalkit.downloads import Asset, asset_catalog, download_asset, select_assets


class DownloadAssetsTest(unittest.TestCase):
    def test_cli_exposes_capevalkit_download_assets_command(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.prog, "capevalkit")
        subparsers_action = next(action for action in parser._actions if action.dest == "command_name")
        self.assertIn("download-assets", subparsers_action.choices)

    def test_catalog_includes_default_pacs_vitb_checkpoint(self) -> None:
        catalog = asset_catalog()

        self.assertIn("pacscore-pacs-vitb", catalog)
        self.assertIn("pacscore-pacs-openclip-vitl", catalog)
        self.assertFalse(catalog["pacscore-pacs-openclip-vitl"].default)

    def test_direct_url_asset_downloads_to_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"checkpoint")
            asset = Asset(
                name="sample",
                description="sample asset",
                source_type="url",
                url=source.as_uri(),
                destination="models/model.bin",
                ready_path="models/model.bin",
            )

            result = download_asset(asset, root=root)

            self.assertEqual(result, root / "models" / "model.bin")
            self.assertEqual(result.read_bytes(), b"checkpoint")

    def test_zip_asset_extracts_to_ready_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reprod-source.zip"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("reprod/reprod.ckpt", "weights")
            asset = Asset(
                name="polos-test",
                description="polos test archive",
                source_type="url",
                url=source.as_uri(),
                destination=".model-cache/polos/reprod.zip",
                ready_path=".model-cache/polos/reprod/reprod.ckpt",
                extract_to=".model-cache/polos",
            )

            result = download_asset(asset, root=root)

            self.assertEqual(result, root / ".model-cache" / "polos" / "reprod" / "reprod.ckpt")
            self.assertEqual(result.read_text(), "weights")

    def test_hf_file_asset_can_copy_to_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "hf-cache" / "model.safetensors"
            cached.parent.mkdir()
            cached.write_bytes(b"hf")
            asset = Asset(
                name="hf-test",
                description="hf file",
                source_type="hf-file",
                hf_repo="org/model",
                hf_filename="model.safetensors",
                destination="runtime/model.safetensors",
                ready_path="runtime/model.safetensors",
            )

            def fake_hf_hub_download(**_: str) -> str:
                return str(cached)

            result = download_asset(asset, root=root, hf_hub_download=fake_hf_hub_download)

            self.assertEqual(result, root / "runtime" / "model.safetensors")
            self.assertEqual(result.read_bytes(), b"hf")

    def test_default_selection_excludes_hf_runtime_assets(self) -> None:
        selected = select_assets([], all_assets=False, include_license_gated=False)
        names = {asset.name for asset in selected}

        self.assertIn("pacscore-pacs-vitb", names)
        self.assertIn("polos-reprod", names)
        self.assertNotIn("pacscore-pacs-openclip-vitl", names)
        self.assertNotIn("pacscore-pacspp-vitb", names)
        self.assertNotIn("pacscore-pacspp-vitl", names)
        self.assertNotIn("vela-longclip-l", names)
        self.assertNotIn("vela-regressor", names)
        self.assertNotIn("vela-ranker", names)
        self.assertNotIn("qwen2.5-3b-instruct", names)
        self.assertNotIn("fleur-llava-v1.5-13b", names)

    def test_catalog_excludes_hf_assets_loaded_by_metric_runtime(self) -> None:
        catalog = asset_catalog()

        self.assertNotIn("vela-longclip-l", catalog)
        self.assertNotIn("vela-regressor", catalog)
        self.assertNotIn("vela-ranker", catalog)
        self.assertNotIn("qwen2.5-3b-instruct", catalog)
        self.assertNotIn("fleur-llava-v1.5-13b", catalog)

    def test_license_gated_asset_requires_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Asset(
                name="gated",
                description="gated snapshot",
                source_type="hf-snapshot",
                hf_repo="org/gated",
                license_gated=True,
            )

            with self.assertRaises(PermissionError):
                download_asset(asset, root=Path(tmp), snapshot_download=lambda **_: tmp)


if __name__ == "__main__":
    unittest.main()
