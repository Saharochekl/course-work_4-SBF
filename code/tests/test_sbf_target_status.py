#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from sbf.sbf_target_status import (
    REQUIRED_SBF2_FITS_KEYS,
    PRIMARY_QUANTITY,
    ensure_target_rows,
    measurement_method,
    read_target_status,
    reusable_result_from_status,
    science_status_fields,
    target_status_key,
    update_target_status,
    validate_reusable_result,
    write_target_status,
)


class TargetStatusCsvTests(unittest.TestCase):
    target = {
        "program": "GO-3055",
        "obsid": "o053_t053",
        "name": "NGC 1380",
        "signal_filter": "f150w",
        "color_filter": "f090w",
    }

    def test_status_roundtrip_is_textual_and_sha_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_status.csv"
            rows = ensure_target_rows({}, [self.target])
            update_target_status(
                rows,
                self.target,
                "done",
                method="sbf2",
                quantity=PRIMARY_QUANTITY,
                result_value=28.3,
                result_unit="AB mag",
                selected_region="circular_inner_lit",
                selection_method="single_annulus_qc_selection_v1",
                result_json=Path(directory) / "result.json",
                qc="pass",
            )
            write_target_status(path, rows)
            restored = read_target_status(path)
            row = restored[target_status_key(self.target)]
            self.assertEqual(row["program"], "3055")
            self.assertEqual(row["status"], "done")
            self.assertEqual(row["method"], "sbf2")
            self.assertEqual(row["result_value"], "28.3")
            self.assertEqual(row["result_unit"], "AB mag")
            self.assertTrue(row["result_json"].endswith("result.json"))
            self.assertNotIn("sha", path.read_text(encoding="utf-8").lower())

    def test_reusable_result_does_not_compare_template_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = root / "products"
            products.mkdir()
            result = self._synthetic_sbf2_result(products)
            result["template_sha256"] = "old-template"
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            self.assertIsNotNone(validate_reusable_result(result_path, self.target))
            rows = ensure_target_rows({}, [self.target])
            update_target_status(
                rows, self.target, "done", result_json=result_path
            )
            self.assertIsNotNone(reusable_result_from_status(rows, self.target))

            fields = science_status_fields(result)
            self.assertEqual(fields["result_value"], "28.3")
            self.assertEqual(fields["result_unit"], "AB mag")
            result["recommended_measurement_method"] = (
                "azimuthal_power_spectrum_psf_fit"
            )
            self.assertEqual(
                measurement_method(result),
                "azimuthal_power_spectrum_psf_fit",
            )

    def test_missing_product_invalidates_done_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = root / "products"
            products.mkdir()
            result = self._synthetic_sbf2_result(products)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            Path(result["science_residual_fits"]).unlink()
            self.assertIsNone(validate_reusable_result(result_path, self.target))

    def test_wrong_pipeline_or_filter_invalidates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._synthetic_sbf2_result(root)
            result_path = root / "result.json"
            for change in ({"template_family": "sbf3"}, {"signal_filter": "F090W"}):
                with self.subTest(change=change):
                    result_path.write_text(json.dumps({**result, **change}), encoding="utf-8")
                    self.assertIsNone(validate_reusable_result(result_path, self.target))

    def test_empty_table_invalidates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._synthetic_sbf2_result(root)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            Path(result["annulus_summary_csv"]).write_text("", encoding="utf-8")
            self.assertIsNone(validate_reusable_result(result_path, self.target))

    def _synthetic_sbf2_result(self, products: Path) -> dict:
        result = {
            "galaxy": "NGC 1380",
            "status": "ok",
            "template_family": "sbf2",
            "signal_filter": "F150W",
            "color_filter": "F090W",
            "recommended_mbar_weighted": 28.3,
            "recommended_kmin": 0.04,
            "recommended_kmax": 0.25,
        }
        for key in REQUIRED_SBF2_FITS_KEYS:
            path = products / f"{key}.fits"
            fits.PrimaryHDU(np.zeros((3, 3), dtype=np.float32)).writeto(path)
            result[key] = str(path)
        df_sbf = products / "df_sbf.csv"
        summary = products / "annulus_summary.csv"
        self._write_measurements(df_sbf)
        summary.write_text(
            "kmin,kmax,mbar_inner,mbar_outer,mbar_weighted\n"
            "0.04,0.25,28.30,28.35,28.32\n",
            encoding="utf-8",
        )
        result["df_sbf_csv"] = str(df_sbf)
        result["annulus_summary_csv"] = str(summary)
        return result

    @staticmethod
    def _write_measurements(path: Path) -> None:
        fields = [
            "region",
            "kmin",
            "kmax",
            "measurement_ok",
            "mbar_spec",
            "P_fluc",
            "n_use",
            "usable_fraction",
            "Pr_over_P0",
            "corr",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for region, value in (
                ("circular_inner_lit", 28.30),
                ("circular_outer_lit", 28.35),
            ):
                writer.writerow(
                    {
                        "region": region,
                        "kmin": 0.04,
                        "kmax": 0.25,
                        "measurement_ok": True,
                        "mbar_spec": value,
                        "P_fluc": 1.0,
                        "n_use": 10000,
                        "usable_fraction": 0.9,
                        "Pr_over_P0": 0.02,
                        "corr": 0.99,
                    }
                )


if __name__ == "__main__":
    unittest.main()
