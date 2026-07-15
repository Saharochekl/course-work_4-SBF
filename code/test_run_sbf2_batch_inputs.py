#!/usr/bin/env python3
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
        with self.assertRaisesRegex(ValueError, "numerical notebook"):
            batch.validate_notebook_filter_pair("F150W", "F115W")

    def test_current_filter_pair_is_accepted(self):
        batch.validate_notebook_filter_pair("f150w", "f090w")

    def test_mast_product_uri_becomes_download_url(self):
        url = batch.product_uri_download_url(
            "mast:JWST/product/example_i2d.fits", "example_i2d.fits"
        )
        self.assertIn("mast%3AJWST%2Fproduct%2Fexample_i2d.fits", url)


if __name__ == "__main__":
    unittest.main()
