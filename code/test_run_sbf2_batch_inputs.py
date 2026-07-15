#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import run_sbf2_batch as batch


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


class BatchInputContractTests(unittest.TestCase):
    def test_generic_manifest_matches_legacy_go3055_products(self):
        generic = batch.read_targets_from_csv(
            CODE_DIR / "targets_go3055_manifest.csv", PROJECT_ROOT / "data"
        )
        legacy = batch.read_targets_from_csv(
            CODE_DIR / "article_galaxies_jwst_f150w_selected.csv",
            PROJECT_ROOT / "data",
        )

        self.assertEqual(len(generic), 14)
        self.assertEqual(len(legacy), 14)
        for current, old in zip(generic, legacy):
            self.assertEqual(current["name"], old["name"])
            self.assertEqual(current["signal_product"], old["signal_product"])
            self.assertEqual(current["color_product"], old["color_product"])
            self.assertEqual(current["signal_filter"], "F150W")
            self.assertEqual(current["color_filter"], "F090W")

    def test_known_target_sizes_do_not_replace_alternate_manifest_products(self):
        for galaxy in ("NGC 1380", "NGC 1404"):
            merged = batch.merge_known_targets(
                [
                    {
                        "name": galaxy,
                        "signal_product": "alternate-signal_i2d.fits",
                        "color_product": "alternate-color_i2d.fits",
                        "signal_size": 111,
                        "color_size": 222,
                    }
                ]
            )[0]
            self.assertEqual(merged["signal_size"], 111)
            self.assertEqual(merged["color_size"], 222)

    def test_legacy_worker_flags_are_aliases(self):
        args = batch.parse_args(
            [
                "--worker",
                "--galaxy",
                "test",
                "--f150w",
                "signal.fits",
                "--f090w",
                "color.fits",
            ]
        )
        self.assertEqual(args.signal, "signal.fits")
        self.assertEqual(args.color, "color.fits")

    def test_generic_worker_flags_carry_filter_names(self):
        args = batch.parse_args(
            [
                "--worker",
                "--galaxy",
                "test",
                "--signal",
                "signal.fits",
                "--color",
                "color.fits",
                "--signal-filter",
                "F200W",
                "--color-filter",
                "F115W",
            ]
        )
        self.assertEqual(args.signal_filter, "F200W")
        self.assertEqual(args.color_filter, "F115W")

    def test_unmigrated_filter_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sbf-2.ipynb"):
            batch.validate_notebook_filter_pair(
                CODE_DIR / "sbf-2.ipynb", "F150W", "F115W"
            )

    def test_current_filter_pair_is_accepted(self):
        batch.validate_notebook_filter_pair(
            CODE_DIR / "sbf-2.ipynb", "f150w", "f090w"
        )

    def test_sbf3_accepts_manifest_filter_pair(self):
        batch.validate_notebook_filter_pair(
            CODE_DIR / "sbf-3.ipynb", "F150W", "F115W"
        )

    def test_sbf3_requires_isolated_roots(self):
        with self.assertRaisesRegex(ValueError, "products-root"):
            batch.validate_run_layout(
                CODE_DIR / "sbf-3.ipynb", CODE_DIR / "batch", None
            )
        with self.assertRaisesRegex(ValueError, "separate --batch-root"):
            batch.validate_run_layout(
                CODE_DIR / "sbf-3.ipynb",
                batch.DEFAULT_BATCH_ROOT,
                CODE_DIR / "products",
            )

    def test_renamed_sbf2_copy_keeps_filter_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            renamed = Path(tmp) / "innocent-copy.ipynb"
            renamed.write_bytes((CODE_DIR / "sbf-2.ipynb").read_bytes())
            self.assertEqual(
                batch.notebook_family(renamed), batch.SBF2_NOTEBOOK_FAMILY
            )
            with self.assertRaisesRegex(ValueError, "sbf-2 notebook"):
                batch.validate_notebook_filter_pair(
                    renamed, "F150W", "F115W"
                )

    def test_renamed_sbf3_copy_keeps_layout_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            renamed = Path(tmp) / "innocent-copy.ipynb"
            renamed.write_bytes((CODE_DIR / "sbf-3.ipynb").read_bytes())
            self.assertEqual(
                batch.notebook_family(renamed), batch.SBF3_NOTEBOOK_FAMILY
            )
            with self.assertRaisesRegex(ValueError, "products-root"):
                batch.validate_run_layout(renamed, Path(tmp) / "batch", None)

    def test_mast_product_uri_becomes_download_url(self):
        url = batch.product_uri_download_url(
            "mast:JWST/product/example_i2d.fits", "example_i2d.fits"
        )
        self.assertIn("mast%3AJWST%2Fproduct%2Fexample_i2d.fits", url)

    def test_preseeded_namespace_and_output_directory(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from pathlib import Path\n",
                        "TARGET_GALAXY = globals().get('TARGET_GALAXY', 'wrong')\n",
                        "signal_path = Path(globals().get('signal_path', 'wrong.fits'))\n",
                        "color_path = Path(globals().get('color_path', 'wrong2.fits'))\n",
                        "out_dir = Path(globals().get('out_dir', signal_path.parent))\n",
                        "stem = signal_path.stem\n",
                        "recommended_sbf = {'mbar_weighted': 1.23}\n",
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "synthetic.ipynb"
            template.write_text(json.dumps(notebook))
            output_dir = root / "products" / "test"
            result = batch.execute_template_for_target(
                template,
                "injected galaxy",
                root / "signal.fits",
                root / "color.fits",
                root / "batch",
                signal_filter="F150W",
                color_filter="F115W",
                output_dir=output_dir,
            )
            self.assertEqual(result["galaxy"], "injected galaxy")
            self.assertEqual(result["stem"], "signal")
            self.assertEqual(Path(result["out_dir"]), output_dir.resolve())

    def test_result_reuse_requires_matching_identity(self):
        target = {
            "name": "test",
            "signal_filter": "F150W",
            "color_filter": "F115W",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "sbf-3.ipynb"
            template.write_text("version one")
            signal = root / "signal.fits"
            color = root / "color.fits"
            identity = batch.expected_run_identity(
                template, target, signal, color, products_root=root / "products"
            )
            result_path = batch.result_json_path(root, "test", identity=identity)
            result_path.write_text(json.dumps({"status": "ok", **identity}))
            self.assertIsNotNone(
                batch.final_result_for(target, root, identity=identity)
            )

            template.write_text("version two")
            changed_identity = batch.expected_run_identity(
                template, target, signal, color, products_root=root / "products"
            )
            self.assertIsNone(
                batch.final_result_for(target, root, identity=changed_identity)
            )

    def test_replacing_input_at_same_path_invalidates_run_identity(self):
        target = {
            "name": "test",
            "signal_filter": "F150W",
            "color_filter": "F115W",
        }
        notebook = {
            "metadata": {"sbf_pipeline": {"family": "sbf3"}},
            "cells": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "renamed.ipynb"
            template.write_text(json.dumps(notebook))
            signal = root / "signal.fits"
            color = root / "color.fits"
            signal.write_bytes(b"old")
            color.write_bytes(b"color")

            identity = batch.expected_run_identity(
                template, target, signal, color, products_root=root / "products"
            )
            result_path = batch.result_json_path(root, "test", identity=identity)
            result_path.write_text(json.dumps({"status": "ok", **identity}))

            replacement = root / "replacement.fits"
            replacement.write_bytes(b"replacement with a different size")
            replacement.replace(signal)
            changed_identity = batch.expected_run_identity(
                template, target, signal, color, products_root=root / "products"
            )

            self.assertEqual(identity["signal_path"], changed_identity["signal_path"])
            self.assertNotEqual(
                identity["signal_fingerprint"],
                changed_identity["signal_fingerprint"],
            )
            self.assertNotEqual(
                identity["input_pair_key"], changed_identity["input_pair_key"]
            )
            self.assertNotEqual(identity["out_dir"], changed_identity["out_dir"])
            self.assertIsNone(
                batch.final_result_for(target, root, identity=changed_identity)
            )
            for key in ("resolved_path", "size", "mtime_ns"):
                self.assertIn(key, changed_identity["signal_fingerprint"])

    def test_sbf3_paths_do_not_collide_between_input_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = batch.target_output_dir(
                "same galaxy",
                root / "signal.fits",
                root / "color-a.fits",
                products_root=root / "products",
                signal_filter="F150W",
                color_filter="F090W",
            )
            second = batch.target_output_dir(
                "same galaxy",
                root / "signal.fits",
                root / "color-b.fits",
                products_root=root / "products",
                signal_filter="F150W",
                color_filter="F115W",
            )
            self.assertNotEqual(first, second)

    def test_default_template_remains_frozen_sbf2(self):
        args = batch.parse_args([])
        self.assertEqual(Path(args.template).name, "sbf-2.ipynb")


class Sbf3NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((CODE_DIR / "sbf-3.ipynb").read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_all_code_cells_compile(self):
        for number, cell in enumerate(self.notebook["cells"], start=1):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"cell-{number}", "exec")

    def test_notebook_has_no_stale_outputs(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_generic_input_and_color_contract(self):
        self.assertIn('globals().get("signal_path"', self.source)
        self.assertIn('globals().get("color_path"', self.source)
        self.assertIn('"color_index"', self.source)
        self.assertIn("color_photometry_image", self.source)
        self.assertIn("ndimage.map_coordinates", self.source)
        self.assertIn('output_header["SBFVER"]', self.source)
        for legacy_name in [
            "img_f150",
            "img_f090",
            "hdr150",
            "hdr090",
            "valid150",
            "valid090",
            "color_F090W_F150W",
        ]:
            self.assertNotIn(legacy_name, self.source)


if __name__ == "__main__":
    unittest.main()
