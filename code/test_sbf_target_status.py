#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import run_sbf_batch as batch
from sbf_target_status import (
    LEGACY_SBF2_FITS_KEYS,
    PRIMARY_QUANTITY,
    annulus_qc,
    ensure_target_rows,
    find_legacy_reusable_result,
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
        "program": "GO-7763",
        "obsid": "o053_t053",
        "name": "IC 3501",
        "signal_filter": "f150w",
        "color_filter": "f115w",
    }

    def test_status_roundtrip_is_textual_and_sha_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_status.csv"
            rows = ensure_target_rows({}, [self.target])
            update_target_status(
                rows,
                self.target,
                "done",
                method="sbf3",
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
            self.assertEqual(row["program"], "7763")
            self.assertEqual(row["status"], "done")
            self.assertEqual(row["method"], "sbf3")
            self.assertEqual(row["result_value"], "28.3")
            self.assertEqual(row["result_unit"], "AB mag")
            self.assertTrue(row["result_json"].endswith("result.json"))
            self.assertNotIn("sha", path.read_text(encoding="utf-8").lower())

    def test_reusable_result_does_not_compare_template_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = root / "products"
            products.mkdir()
            result = self._synthetic_sbf3_result(products)
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
            result = self._synthetic_sbf3_result(products)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            Path(result["working_residual_fits"]).unlink()
            self.assertIsNone(validate_reusable_result(result_path, self.target))

    def _synthetic_sbf3_result(self, products: Path) -> dict:
        result = {
            "galaxy": "IC 3501",
            "status": "ok",
            "template_family": "sbf3",
            "signal_filter": "F150W",
            "color_filter": "F115W",
            "recommended_mbar_weighted": 28.3,
            "recommended_kmin": 0.04,
            "recommended_kmax": 0.25,
        }
        keys = (
            "clean_model_fits",
            "clean_isophotes_fits",
            "full_residual_fits",
            "working_residual_fits",
            "working_annuli_residual_fits",
        )
        for key in keys:
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


class LegacyGo3055AdoptionTests(unittest.TestCase):
    def test_all_fourteen_existing_go3055_results_are_reusable(self):
        targets = batch.read_targets_from_csv(
            batch.SCRIPT_DIR / "targets_go3055_manifest.csv",
            batch.PROJECT_ROOT / "data",
        )
        adopted = [
            find_legacy_reusable_result(
                target, batch.SCRIPT_DIR / "sbf2_batch_outputs"
            )
            for target in targets
        ]
        self.assertEqual(len(targets), 14)
        self.assertEqual(sum(result is not None for result in adopted), 14)
        for target, result in zip(targets, adopted):
            self.assertEqual(result["galaxy"], target["name"])
            self.assertTrue(all(result.get(key) for key in LEGACY_SBF2_FITS_KEYS))
            self.assertRegex(annulus_qc(result), r"^(pass|warn:)")

    def test_parent_adopts_all_fourteen_without_starting_a_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = batch.parse_args(
                [
                    "--template",
                    str(batch.SCRIPT_DIR / "sbf-3.ipynb"),
                    "--target-csv",
                    str(batch.SCRIPT_DIR / "targets_go3055_manifest.csv"),
                    "--programs",
                    "3055",
                    "--data-root",
                    str(batch.PROJECT_ROOT / "data"),
                    "--batch-root",
                    str(root / "batch"),
                    "--products-root",
                    str(root / "products"),
                    "--campaign-root",
                    str(root / "campaign"),
                    "--no-download",
                    "--prefetch-targets",
                    "0",
                    "--no-cleanup-inputs",
                    "--allow-bulk-targets",
                    "--wall-time-hours",
                    "0",
                    "--soft-stop-minutes",
                    "0",
                ]
            )
            with patch.object(
                batch,
                "launch_process_group",
                side_effect=AssertionError("legacy results must suppress workers"),
            ), patch.object(
                batch,
                "start_download_manager",
                side_effect=AssertionError("legacy results must suppress downloads"),
            ):
                self.assertEqual(batch.run_parent(args), 0)

            rows = read_target_status(root / "campaign" / "target_status.csv")
            self.assertEqual(len(rows), 14)
            self.assertEqual(
                {row["status"] for row in rows.values()},
                {"done"},
            )
            self.assertEqual(
                {row["method"] for row in rows.values()},
                {"sbf2_legacy"},
            )
            self.assertEqual(
                {row["quantity"] for row in rows.values()},
                {"apparent_sbf_magnitude"},
            )
            self.assertEqual(
                {row["result_unit"] for row in rows.values()},
                {"AB mag"},
            )
            self.assertTrue(
                all(row["result_value"] for row in rows.values())
            )
            self.assertEqual(
                {row["selected_region"] for row in rows.values()},
                {"circular_inner_lit+circular_outer_lit"},
            )
            events = [
                json.loads(line)
                for line in (root / "campaign" / "campaign_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            reused = [
                event
                for event in events
                if event.get("event_type") == "RESULT_REUSED"
            ]
            self.assertEqual(len(reused), 14)
            self.assertEqual(
                {event["payload"]["reuse_kind"] for event in reused},
                {"legacy-go3055"},
            )


if __name__ == "__main__":
    unittest.main()
