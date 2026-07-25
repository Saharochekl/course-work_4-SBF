#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import run_sbf_2_batch as batch


class DedicatedGo3055RunnerTests(unittest.TestCase):
    def test_defaults_are_offline_isolated_and_go3055_only(self):
        args = batch.parse_args([])
        self.assertEqual(args.programs, ["3055"])
        self.assertTrue(args.no_download)
        self.assertTrue(args.no_cleanup_inputs)
        self.assertEqual(args.prefetch_targets, 0)
        self.assertEqual(args.min_available_ram_gb, 0.0)
        self.assertEqual(args.emergency_available_ram_gb, 0.0)
        self.assertEqual(args.max_worker_rss_gb, 0.0)
        self.assertEqual(args.estimated_worker_output_gb, 6.0)
        self.assertEqual(args.min_processing_free_gb, 40.0)
        self.assertEqual(Path(args.batch_root), batch.DEFAULT_BATCH_ROOT)
        self.assertEqual(Path(args.products_root), batch.DEFAULT_PRODUCTS_ROOT)
        self.assertEqual(Path(args.campaign_root), batch.DEFAULT_CAMPAIGN_ROOT)

    def test_force_reprocess_always_starts_a_new_sqlite_run(self):
        args = batch.parse_args(["--force-reprocess"])
        self.assertTrue(args.force_reprocess)
        self.assertTrue(args.new_run)

    def test_scope_rejects_another_program_or_filter_pair(self):
        args = batch.parse_args([])
        target = {
            "name": "NGC 1404",
            "program": "3055",
            "signal_filter": "F150W",
            "color_filter": "F090W",
        }
        batch.validate_go3055_scope(args, [target], batch.DEFAULT_TEMPLATE)

        wrong_program = dict(target, program="7763")
        with self.assertRaisesRegex(ValueError, "not GO-3055"):
            batch.validate_go3055_scope(
                args, [wrong_program], batch.DEFAULT_TEMPLATE
            )

        wrong_pair = dict(target, color_filter="F115W")
        with self.assertRaisesRegex(ValueError, "filter pair"):
            batch.validate_go3055_scope(
                args, [wrong_pair], batch.DEFAULT_TEMPLATE
            )

    def test_sbf2_execution_records_five_fits_and_two_valid_tables(self):
        notebook = {
            "metadata": {"sbf_pipeline": {"family": "sbf2"}},
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from astropy.io import fits\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "names = [\n",
                        " 'sbf_model_full',\n",
                        " 'sbf_resid_catalog_mask_clip_3p5sigma',\n",
                        " 'sbf_resid_full_science_raw',\n",
                        " 'sbf_resid_science_circular_inner_lit_usable',\n",
                        " 'sbf_resid_science_circular_outer_lit_usable',\n",
                        "]\n",
                        "for name in names:\n",
                        " fits.writeto(out_dir / f'{stem}_{name}.fits', np.zeros((3, 3)), overwrite=True)\n",
                        "rows=[]\n",
                        "for region, mbar in [('circular_inner_lit', 28.10), ('circular_outer_lit', 28.14)]:\n",
                        " for kmin in (0.01, 0.03, 0.04):\n",
                        "  rows.append({'region':region,'kmin':kmin,'kmax':0.25,'measurement_ok':True,'mbar_spec':mbar,'mbar_fit_sigma':0.03,'P_fluc':1.0,'P1':0.1,'P0_fit_sigma':0.05,'n_use':10000,'usable_fraction':0.9,'Pr_over_P0':0.01,'corr':0.9,'psf_n_used':5,'psf_scatter_mag':0.01})\n",
                        "df_sbf=pd.DataFrame(rows)\n",
                        "df_annulus_summary=pd.DataFrame([{'kmin':0.04,'kmax':0.25,'mbar_inner':28.10,'sigma_inner':0.03,'mbar_outer':28.14,'sigma_outer':0.03,'mbar_weighted':28.12,'sigma_adopted':0.03}])\n",
                        "recommended_sbf={'kmin':0.04,'kmax':0.25,'mbar_weighted':28.12,'sigma_adopted':0.03}\n",
                        "inner_cov=(10000, 10000, 9000, 1.0, 0.9)\n",
                        "outer_cov=(10000, 10000, 9000, 1.0, 0.9)\n",
                        "psf_library=[object() for _ in range(5)]\n",
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "synthetic_sbf2.ipynb"
            template.write_text(json.dumps(notebook), encoding="utf-8")
            result = batch.execute_template_for_target(
                template,
                "NGC 1404",
                root / "signal.fits",
                root / "color.fits",
                root / "batch",
                output_dir=root / "products",
            )

        self.assertEqual(result["artifact_count"], 7)
        self.assertEqual(result["fits_artifact_count"], 5)
        self.assertEqual(result["table_artifact_count"], 2)
        self.assertTrue(result["artifacts_verified"])
        self.assertEqual(result["qc_status"], "pass")
        self.assertTrue(
            result["science_residual_fits"].endswith(
                "_sbf_resid_catalog_mask_clip_3p5sigma.fits"
            )
        )
        self.assertEqual(
            {
                item["name"]
                for item in result["artifact_manifest"]
                if item.get("csv_valid")
            },
            {"df_sbf_csv", "annulus_summary_csv"},
        )
        self.assertNotIn(
            "sha256",
            {key for item in result["artifact_manifest"] for key in item},
        )

    def test_single_target_summary_update_preserves_other_galaxies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch.write_summary(
                [
                    {
                        "galaxy": "NGC 1380",
                        "program": "3055",
                        "signal_filter": "F150W",
                        "color_filter": "F090W",
                        "status": "ok",
                        "recommended_mbar_weighted": 28.0,
                    }
                ],
                root,
            )
            batch.write_summary(
                [
                    {
                        "galaxy": "NGC 1404",
                        "program": "3055",
                        "signal_filter": "F150W",
                        "color_filter": "F090W",
                        "status": "failed",
                        "error": "synthetic failure",
                    }
                ],
                root,
            )
            batch.write_summary(
                [
                    {
                        "galaxy": "NGC 1404",
                        "program": "3055",
                        "signal_filter": "F150W",
                        "color_filter": "F090W",
                        "status": "ok",
                        "recommended_mbar_weighted": 28.2,
                    }
                ],
                root,
            )
            stored = json.loads(
                (root / "sbf2_batch_results.json").read_text(encoding="utf-8")
            )
            batch.write_go3055_qc(stored, root)
            qc_text = (root / batch.GO3055_QC_FILENAME).read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            {result["galaxy"] for result in stored},
            {"NGC 1380", "NGC 1404"},
        )
        ngc1404 = [
            result for result in stored if result["galaxy"] == "NGC 1404"
        ]
        self.assertEqual(len(ngc1404), 1)
        self.assertEqual(ngc1404[0]["status"], "ok")
        self.assertNotIn("synthetic failure", qc_text)
        self.assertIn("inner_P_fluc,inner_P1", qc_text)
        self.assertIn("outer_P_fluc,outer_P1", qc_text)


if __name__ == "__main__":
    unittest.main()
