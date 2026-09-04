import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from run_sbf_f090w import (
    AUXILIARY_FILTER,
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    F090W_QC_METRIC_KEYS,
    SBF2_REQUIRED_FITS_KEYS,
    SBF2_REQUIRED_TABLE_KEYS,
    SIGNAL_FILTER,
    _selected_isophote_qc,
    campaign_paths,
    read_targets,
    select_targets,
    serialized_config,
    set_log_context,
    source_result_path,
    source_result_valid,
    sync_status,
    timestamped_print,
    worker_command,
)
from sbf090_pipeline_support import (
    F090W_INNER_MASK_GUARD_METHOD,
    F090W_INNER_MASK_GUARD_TARGETS,
    F090W_ISOPHOTE_METHOD,
    F090W_MASK_METHOD,
    F090W_PSF_COUNT,
    F090W_PSF_SIZE,
    F090W_SOURCE_SCHEMA,
    build_f090w_template,
    isophote_bootstrap_rank,
    is_large_external_contaminant,
    isophote_sequence_diagnostics,
    isophote_sequence_qc,
    load_f150w_reference_center,
    load_f090w_psf_cache,
    write_dataframe_atomic,
    write_psf_cache_atomic,
)
from sbf2_normalized_winsor_core import ExperimentConfig


SCRIPT_DIR = Path(__file__).resolve().parent


class F090WPipelineTests(unittest.TestCase):
    @staticmethod
    def _qc(details):
        return isophote_sequence_qc(
            details,
            required_sma_px=520.0,
            min_isophotes=20,
            max_median_center_shift_px=50.0,
            max_stop2_fraction=0.30,
            max_consecutive_stop2=8,
            max_frozen_stop_fraction=0.25,
            max_consecutive_frozen_stop=20,
            max_singular_stop_count=0,
            max_center_shift_px=15.0,
            max_center_step_px=10.0,
            max_eps_step=0.03,
            max_pa_step_rad=None,
            max_shape_step=0.05,
            max_intensity_rise_fraction=0.05,
        )

    def test_isophote_qc_accepts_a_smooth_converged_sequence(self):
        items = [
            SimpleNamespace(
                sma=float(sma),
                x0=100.0 + 0.01 * index,
                y0=100.0 - 0.01 * index,
                eps=0.40 + 0.001 * np.sin(index),
                pa=0.75 + 0.001 * np.cos(index),
                intens=100.0 * np.exp(-index / 25.0),
                stop_code=0,
            )
            for index, sma in enumerate(np.arange(260.0, 541.0, 10.0))
        ]
        details = isophote_sequence_diagnostics(
            items, 100.0, 100.0, sma_min=260.0, sma_max=540.0
        )
        passed, reason = self._qc(details)

        self.assertTrue(passed, reason)
        self.assertEqual(details["quality_n_isophotes"], len(items))
        self.assertEqual(details["max_consecutive_stop2"], 0)
        self.assertEqual(details["stop2_fraction"], 0.0)
        self.assertLess(details["max_center_step_px"], 0.1)
        self.assertLess(details["max_eps_step"], 0.01)
        self.assertLess(details["max_pa_step_rad"], 0.01)
        self.assertLess(details["max_shape_step"], 0.01)
        self.assertLessEqual(details["max_intensity_rise_fraction"], 0.0)

    def test_isophote_qc_rejects_the_ngc4697_failure_shape_for_every_target(self):
        items = []
        for index, sma in enumerate(np.arange(260.0, 651.0, 10.0)):
            items.append(SimpleNamespace(
                sma=float(sma),
                x0=100.0,
                y0=100.0,
                eps=0.20,
                pa=1.105,
                intens=100.0 - index,
                stop_code=2 if index < 37 else 0,
            ))
        # Reproduce the observed branch switch: centre, ellipticity, position
        # angle and intensity all jump between neighbouring isophotes.
        items[-2].x0 = 108.1
        items[-2].eps = 0.409
        items[-2].pa = 0.765
        items[-2].intens = 1.234 * items[-3].intens

        details = isophote_sequence_diagnostics(
            items, 100.0, 100.0, sma_min=260.0, sma_max=650.0
        )
        passed, reason = self._qc(details)

        self.assertFalse(passed)
        self.assertTrue(reason)
        self.assertGreater(details["stop2_fraction"], 0.30)
        self.assertGreater(details["max_consecutive_stop2"], 8)
        self.assertGreater(details["max_eps_step"], 0.03)
        self.assertGreater(details["max_pa_step_rad"], 0.15)
        self.assertGreater(details["max_shape_step"], 0.05)
        self.assertGreater(details["max_intensity_rise_fraction"], 0.05)

    def test_isophote_position_angle_uses_ellipse_periodicity(self):
        items = [
            SimpleNamespace(
                sma=260.0 + 10.0 * index,
                x0=100.0,
                y0=100.0,
                eps=0.4,
                pa=pa,
                intens=100.0 - index,
                stop_code=0,
            )
            for index, pa in enumerate((
                np.pi / 2.0 - 0.01,
                -np.pi / 2.0 + 0.01,
                -np.pi / 2.0 + 0.02,
            ))
        ]
        details = isophote_sequence_diagnostics(
            items, 100.0, 100.0, sma_min=260.0, sma_max=280.0
        )
        self.assertAlmostEqual(details["max_pa_step_rad"], 0.02, places=6)

    def test_isophote_qc_does_not_reject_position_angle_of_round_isophotes(self):
        items = [
            SimpleNamespace(
                sma=260.0 + 10.0 * index,
                x0=100.0,
                y0=100.0,
                eps=0.001,
                pa=0.0 if index % 2 == 0 else np.pi / 2.0,
                intens=100.0 * np.exp(-index / 25.0),
                stop_code=0,
            )
            for index in range(29)
        ]
        details = isophote_sequence_diagnostics(
            items, 100.0, 100.0, sma_min=260.0, sma_max=540.0
        )
        passed, reason = self._qc(details)

        self.assertGreater(details["max_pa_step_rad"], 1.5)
        self.assertLess(details["max_shape_step"], 0.01)
        self.assertTrue(passed, reason)

    def test_isophote_qc_rejects_a_frozen_stop_code_four_sequence(self):
        items = [
            SimpleNamespace(
                sma=260.0 + 10.0 * index,
                x0=100.0,
                y0=100.0,
                eps=0.03,
                pa=2.23,
                intens=100.0 * np.exp(-index / 25.0),
                stop_code=4,
            )
            for index in range(29)
        ]
        details = isophote_sequence_diagnostics(
            items, 100.0, 100.0, sma_min=260.0, sma_max=540.0
        )
        passed, reason = self._qc(details)

        self.assertFalse(passed)
        self.assertEqual(details["frozen_stop_fraction"], 1.0)
        self.assertEqual(details["max_consecutive_frozen_stop"], len(items))
        self.assertIn("stop-code-1/4/5", reason)

    def test_large_contaminant_rule_masks_only_bright_external_components(self):
        common = {
            "compact_max_area": 5000,
            "core_guard_radius_pixels": 250.0,
            "min_peak_snr": 100.0,
        }
        self.assertTrue(is_large_external_contaminant(
            area_pixels=8382, min_radius_pixels=430.0, peak_snr=1123.0,
            **common,
        ))
        self.assertFalse(is_large_external_contaminant(
            area_pixels=8430, min_radius_pixels=0.0, peak_snr=35.0,
            **common,
        ))
        self.assertFalse(is_large_external_contaminant(
            area_pixels=6406, min_radius_pixels=0.0, peak_snr=250.0,
            **common,
        ))

    def test_bootstrap_ranking_prefers_convergence_and_start_near_fifty(self):
        smooth = {
            "quality_n_isophotes": 18,
            "max_sma_px": 190.0,
            "stop2_fraction": 0.0,
            "max_consecutive_stop2": 0,
            "frozen_stop_count": 0,
            "frozen_stop_fraction": 0.0,
            "max_consecutive_frozen_stop": 0,
            "singular_stop_count": 0,
            "max_center_shift_px": 2.0,
            "max_center_step_px": 0.2,
            "max_eps_step": 0.005,
            "max_pa_step_rad": 0.01,
            "max_shape_step": 0.01,
            "max_intensity_rise_fraction": 0.0,
        }
        poor = dict(smooth)
        poor.update({
            "stop2_fraction": 0.80,
            "max_consecutive_stop2": 15,
            "max_eps_step": 0.20,
        })
        smooth_40 = dict(smooth, start_sma_px=40.0)
        smooth_50 = dict(smooth, start_sma_px=50.0)
        poor_100 = dict(poor, start_sma_px=100.0)

        self.assertLess(
            isophote_bootstrap_rank(smooth_40, 0),
            isophote_bootstrap_rank(poor_100, 1),
        )
        self.assertLess(
            isophote_bootstrap_rank(smooth_50, 1),
            isophote_bootstrap_rank(smooth_40, 0),
        )

    def test_durable_selected_attempt_is_rechecked_before_source_reuse(self):
        good = {
            "phase": "full",
            "selected": True,
            "status": "working",
            "dataset": "real_only",
            "start_sma_px": 50.0,
            "step_px": 10.0,
            "fix_center": False,
            "seed_source": "F090W Sersic initial geometry",
            "seed_eps": 0.4,
            "seed_pa_rad": 0.75,
            "required_sma_px": 520.0,
            "n_isophotes": 60,
            "max_sma_px": 600.0,
            "median_center_shift_px": 1.0,
            "max_center_shift_px": 2.0,
            "quality_n_isophotes": 27,
            "quality_sma_min_px": 260.0,
            "quality_sma_max_px": 520.0,
            "stop2_count": 0,
            "stop2_fraction": 0.0,
            "max_consecutive_stop2": 0,
            "frozen_stop_count": 0,
            "frozen_stop_fraction": 0.0,
            "max_consecutive_frozen_stop": 0,
            "singular_stop_count": 0,
            "max_center_step_px": 0.2,
            "max_eps_step": 0.005,
            "max_pa_step_rad": 0.01,
            "max_shape_step": 0.01,
            "max_intensity_rise_fraction": 0.0,
        }
        self.assertEqual(set(F090W_QC_METRIC_KEYS), set(good) & set(F090W_QC_METRIC_KEYS))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.csv"
            pd.DataFrame([good]).to_csv(path, index=False)
            accepted = _selected_isophote_qc(path)
            self.assertTrue(accepted["passed"], accepted["reason"])
            self.assertEqual(accepted["method"], F090W_ISOPHOTE_METHOD)
            self.assertEqual(accepted["start_sma_px"], 50.0)
            self.assertEqual(accepted["metrics"]["max_consecutive_stop2"], 0)

            broken = dict(good)
            broken.update({
                "stop2_count": 25,
                "stop2_fraction": 0.93,
                "max_consecutive_stop2": 25,
                "max_eps_step": 0.209,
                "max_pa_step_rad": 0.340,
                "max_shape_step": 0.283,
                "max_intensity_rise_fraction": 0.234,
            })
            pd.DataFrame([broken]).to_csv(path, index=False)
            rejected = _selected_isophote_qc(path)
            self.assertFalse(rejected["passed"])
            self.assertIn("stop-code-2", rejected["reason"])
            self.assertIn("complex-shape step", rejected["reason"])
            self.assertIn("outward intensity rise", rejected["reason"])

    def test_completed_config_uses_json_compatible_kmin_list(self):
        config = ExperimentConfig(kmins=(0.01, 0.03, 0.04))
        payload = serialized_config(config)
        self.assertEqual(payload["kmins"], [0.01, 0.03, 0.04])
        self.assertEqual(payload, json.loads(json.dumps(payload)))

    def test_manifest_has_fourteen_swapped_filter_targets(self):
        targets = read_targets(DEFAULT_MANIFEST, DEFAULT_DATA_ROOT)
        self.assertEqual(len(targets), 14)
        self.assertEqual(len({target["name"] for target in targets}), 14)
        for target in targets:
            self.assertEqual(target["signal_filter"], SIGNAL_FILTER)
            self.assertEqual(target["color_filter"], AUXILIARY_FILTER)
            self.assertIn("f090w", target["signal_product"].lower())
            self.assertIn("f150w", target["color_product"].lower())

    def test_generated_notebook_is_f090w_and_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sbf-f090w.ipynb"
            base_notebook = SCRIPT_DIR / "sbf-2.ipynb"
            base_before = base_notebook.read_bytes()
            build_f090w_template(base_notebook, destination)
            self.assertEqual(base_notebook.read_bytes(), base_before)
            notebook = json.loads(destination.read_text(encoding="utf-8"))
            metadata = notebook["metadata"]["sbf_f090w_generated"]
            self.assertEqual(metadata["source_schema"], F090W_SOURCE_SCHEMA)
            self.assertEqual(metadata["isophotes"], F090W_ISOPHOTE_METHOD)
            self.assertEqual(metadata["mask"], F090W_MASK_METHOD)
            self.assertEqual(
                metadata["isophote_inner_mask_guard"],
                F090W_INNER_MASK_GUARD_METHOD,
            )
            self.assertEqual(
                metadata["isophote_inner_mask_guard_targets"], ["NGC 4636"]
            )
            all_source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
            )
            self.assertNotIn("## Цвета в тех же annuli", all_source)
            self.assertNotIn("_orig_print = builtins.print", all_source)
            self.assertNotIn("time.strftime('%H:%M:%S')", all_source)
            self.assertIn("A_F090W_SBF", all_source)
            self.assertNotIn("A_F150W_SBF", all_source)
            self.assertIn("load_f090w_psf_cache", all_source)
            self.assertIn('header["SCIMJD"]', all_source)
            self.assertIn('header["DETPOS"]', all_source)
            for marker in (
                "accepted-F150W-model/WCS",
                "CENTER_LOCAL_RADIUS_PX",
                "ISO_START_SMA_CANDIDATES = (40.0, 50.0, 60.0, 70.0, 100.0)",
                "_sbf_center.csv",
                "_sbf_isophote_attempts.csv",
                "_sbf_isophotes.csv",
                "EXTERNAL_CONTAMINANT_CORE_GUARD_PX = 250.0",
                "EXTERNAL_CONTAMINANT_MIN_PEAK_SNR = 100.0",
                "is_large_external_contaminant",
                "_sbf_external_contaminants.csv",
                "ISO_QC_MAX_CENTER_OFFSET_PX = 15.0",
                "ISO_QC_MAX_STOP2_FRACTION = 0.3",
                "ISO_QC_MAX_CONSECUTIVE_STOP2 = 8",
                "ISO_QC_MAX_FROZEN_STOP_FRACTION = 0.25",
                "ISO_QC_MAX_CONSECUTIVE_FROZEN_STOP = 20",
                "ISO_QC_MAX_SINGULAR_STOP_COUNT = 0",
                "ISO_QC_MAX_CENTER_STEP_PX = 10.0",
                "ISO_QC_MAX_EPS_STEP = 0.03",
                "ISO_QC_MAX_SHAPE_STEP = 0.05",
                "ISO_QC_MAX_INTENSITY_RISE_FRACTION = 0.05",
                "F090W Sersic initial geometry",
                "isophote_sequence_diagnostics",
                "isophote_sequence_qc",
                "isophote_bootstrap_rank",
                "candidates.sort(key=lambda item: item[0])",
                "science_sma_min",
                "external_contaminant_mask",
                "isophote_mask_c",
                "f090_inner_mask_guard_enabled",
                "science mask unchanged",
                "qc_reason",
                "[ISO-QC]",
            ):
                self.assertIn(marker, all_source)
            self.assertNotIn(
                '!= "NGC1399" or phase != "full"', all_source
            )
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell.get("cell_type") == "code":
                    compile(
                        "".join(cell.get("source", [])),
                        f"generated-cell-{index}",
                        "exec",
                    )

    def test_every_console_line_has_context_prefix(self):
        output = io.StringIO()
        set_log_context("NGC 1380", 3, 14)
        try:
            with patch("run_sbf_f090w.time.strftime", return_value="13:50:10"):
                timestamped_print(
                    "[2026-09-03 13:49:59] first line\nsecond line",
                    file=output,
                )
        finally:
            set_log_context("campaign", 0, 14)
        self.assertEqual(
            output.getvalue(),
            "[13:50:10, NGC 1380, 3/14] first line\n"
            "[13:50:10, NGC 1380, 3/14] second line\n",
        )

    def test_worker_receives_manifest_queue_position(self):
        args = SimpleNamespace(
            manifest=Path("manifest.csv"),
            data_root=Path("data"),
            run_root=Path("runs"),
            base_notebook=Path("sbf-2.ipynb"),
            stpsf_data_dir=Path("stpsf-data"),
            wss_opd_dir=Path("wss-opd"),
            e_realizations=64,
            fft_workers=-1,
            force_source=False,
            force_spectra=False,
            rebuild_psf=False,
            rebuild_input_cache=False,
            rebuild_expectation_cache=False,
        )
        command = worker_command(
            args, "NGC 1380", Path("generated.ipynb"), 3, 14
        )
        self.assertEqual(
            command[command.index("--queue-index") + 1], "3"
        )
        self.assertEqual(
            command[command.index("--queue-total") + 1], "14"
        )

    def test_f150w_reference_center_is_transferred_by_wcs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "accepted_f150_model.fits"
            model = fits.PrimaryHDU()
            model.header["SBFXCEN"] = 34.0
            model.header["SBFYCEN"] = 67.0
            model.writeto(model_path)

            f150_wcs = WCS(naxis=2)
            f150_wcs.wcs.crpix = [100.0, 200.0]
            f150_wcs.wcs.cdelt = [-1.0e-5, 1.0e-5]
            f150_wcs.wcs.crval = [150.0, 2.0]
            f150_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            auxiliary_path = root / "f150.fits"
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    np.zeros((4, 4), dtype=np.float32),
                    header=f150_wcs.to_header(),
                    name="SCI",
                ),
            ]).writeto(auxiliary_path)

            signal_wcs = WCS(naxis=2)
            signal_wcs.wcs.crpix = [112.0, 183.0]
            signal_wcs.wcs.cdelt = f150_wcs.wcs.cdelt
            signal_wcs.wcs.crval = f150_wcs.wcs.crval
            signal_wcs.wcs.ctype = f150_wcs.wcs.ctype

            result_path = (
                root / "runs" / "sbf2_go3055" / "batch"
                / "NGC_9999_result.json"
            )
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps({"model_full_fits": str(model_path)}),
                encoding="utf-8",
            )

            reference = load_f150w_reference_center(
                "NGC 9999", auxiliary_path, signal_wcs, project_root=root,
            )
            self.assertEqual(reference["source"], "accepted-F150W-model/WCS")
            self.assertAlmostEqual(reference["f150_x_pixel"], 34.0)
            self.assertAlmostEqual(reference["f150_y_pixel"], 67.0)
            self.assertAlmostEqual(reference["x_pixel"], 46.0, places=5)
            self.assertAlmostEqual(reference["y_pixel"], 50.0, places=5)

    def test_diagnostic_csv_write_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "diagnostics" / "isophotes.csv"
            frame = pd.DataFrame({"sma_px": [40.0, 50.0], "eps": [0.1, 0.2]})
            write_dataframe_atomic(frame, destination)
            restored = pd.read_csv(destination)
            pd.testing.assert_frame_equal(restored, frame)

    def test_source_schema_keeps_legacy_results_and_gates_new_diagnostics(self):
        self.assertGreaterEqual(F090W_SOURCE_SCHEMA, 3)
        with tempfile.TemporaryDirectory() as directory:
            paths = campaign_paths(Path(directory) / "campaign")
            paths["source_batch"].mkdir(parents=True)
            result = {
                "status": "ok",
                "signal_filter": SIGNAL_FILTER,
                "color_filter": AUXILIARY_FILTER,
                "signal_path": str(Path(directory) / "signal.fits"),
                "color_path": str(Path(directory) / "color.fits"),
                "signal_fingerprint": {"stable": True},
                "color_fingerprint": {"stable": True},
                "output_dir": str(Path(directory) / "products"),
                "stem": "science",
            }
            for key in SBF2_REQUIRED_FITS_KEYS:
                result[key] = str(Path(directory) / f"{key}.fits")
            for key in SBF2_REQUIRED_TABLE_KEYS:
                result[key] = str(Path(directory) / f"{key}.csv")

            marker = source_result_path(paths, "NGC 9999")
            common_patches = (
                patch(
                    "run_sbf_f090w.input_fingerprint",
                    return_value={"stable": True},
                ),
                patch(
                    "run_sbf_f090w.fits_is_readable",
                    side_effect=lambda value: (
                        (True, "") if str(value) else (False, "missing path")
                    ),
                ),
                patch(
                    "run_sbf_f090w.load_f090w_psf_cache",
                    return_value={"psf": np.ones((1, 1))},
                ),
                patch(
                    "run_sbf_f090w.pd.read_csv",
                    return_value=pd.DataFrame({"row": [1]}),
                ),
                patch(
                    "run_sbf_f090w._selected_isophote_qc",
                    return_value={
                        "passed": True,
                        "reason": "",
                        "method": F090W_ISOPHOTE_METHOD,
                    },
                ),
            )
            with (
                common_patches[0], common_patches[1], common_patches[2],
                common_patches[3], common_patches[4],
            ):
                marker.write_text(json.dumps(result), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertTrue(valid, message)

                for repair_target in (
                    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 4374",
                    "NGC 4406", "NGC 4472", "NGC 4486", "NGC 4552",
                    "NGC 4621", "NGC 4636", "NGC 4649", "NGC 4697",
                    "NGC 1549", "NGC 3379",
                ):
                    repaired_marker = source_result_path(paths, repair_target)
                    legacy = dict(result, f090_source_schema=F090W_SOURCE_SCHEMA - 1)
                    repaired_marker.write_text(
                        json.dumps(legacy), encoding="utf-8"
                    )
                    valid, _, message = source_result_valid(
                        paths, repair_target
                    )
                    self.assertFalse(valid)
                    self.assertIn("legacy f090w source", message.lower())

                diagnostics = {
                    "center": str(Path(directory) / "center.csv"),
                    "isophote_attempts": str(Path(directory) / "attempts.csv"),
                    "isophotes": str(Path(directory) / "isophotes.csv"),
                    "external_contaminants": str(
                        Path(directory) / "external_contaminants.csv"
                    ),
                }
                current = dict(result)
                current.update({
                    "f090_source_schema": F090W_SOURCE_SCHEMA,
                    "f090_isophote_method": F090W_ISOPHOTE_METHOD,
                    "f090_mask_method": F090W_MASK_METHOD,
                    "f090_diagnostic_tables": diagnostics,
                    "f090_isophote_qc_passed": True,
                    "f090_isophote_qc": {
                        "passed": True,
                        "reason": "",
                        "method": F090W_ISOPHOTE_METHOD,
                    },
                    "f090_external_contaminant_mask_fits": str(
                        Path(directory) / "external_contaminant_mask.fits"
                    ),
                })
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertTrue(valid, message)

                guarded = dict(current)
                guarded["f090_isophote_inner_mask_guard"] = {
                    "enabled": True,
                    "method": F090W_INNER_MASK_GUARD_METHOD,
                    "radius_definition": "inner_sbf_boundary",
                    "affects_sbf_measurement_mask": False,
                }
                guarded_marker = source_result_path(paths, "NGC 4636")
                guarded_marker.write_text(json.dumps(guarded), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 4636")
                self.assertTrue(valid, message)

                guarded.pop("f090_isophote_inner_mask_guard")
                guarded_marker.write_text(json.dumps(guarded), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 4636")
                self.assertFalse(valid)
                self.assertIn("inner isophote-mask guard", message)

                current["f090_isophote_method"] = "obsolete-method"
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertFalse(valid)
                self.assertIn("different isophote method", message)
                current["f090_isophote_method"] = F090W_ISOPHOTE_METHOD

                current["f090_diagnostic_tables"] = {
                    "center": diagnostics["center"],
                    "isophotes": diagnostics["isophotes"],
                }
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertFalse(valid)
                self.assertIn("diagnostics", message)

                current["f090_diagnostic_tables"] = diagnostics
                current["f090_isophote_qc_passed"] = False
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertFalse(valid)
                self.assertIn("QC did not pass", message)

                current["f090_isophote_qc_passed"] = True
                current["f090_isophote_qc"]["passed"] = False
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertFalse(valid)
                self.assertIn("QC did not pass", message)

                current["f090_isophote_qc"]["passed"] = True
                current.pop("f090_external_contaminant_mask_fits")
                marker.write_text(json.dumps(current), encoding="utf-8")
                valid, _, message = source_result_valid(paths, "NGC 9999")
                self.assertFalse(valid)
                self.assertIn("contaminant mask", message)

    def test_targeted_resume_preserves_ten_completed_targets(self):
        targets = read_targets(DEFAULT_MANIFEST, DEFAULT_DATA_ROOT)
        problem = {"NGC 1399", "NGC 4697", "NGC 4621", "NGC 1549"}
        selected = select_targets(targets, list(problem))
        self.assertEqual({target["name"] for target in selected}, problem)
        self.assertEqual(len(selected), 4)

        with tempfile.TemporaryDirectory() as directory:
            paths = campaign_paths(Path(directory) / "campaign")
            paths["root"].mkdir(parents=True)
            config = ExperimentConfig()

            def final_valid(_paths, galaxy, _config):
                return galaxy not in problem, {}, "final result is absent"

            def source_valid(_paths, galaxy):
                if galaxy == "NGC 4697":
                    return True, {}, "ok"
                return galaxy not in problem, {}, "source result is absent"

            with patch("run_sbf_f090w.final_result_valid", side_effect=final_valid), patch(
                "run_sbf_f090w.source_result_valid", side_effect=source_valid
            ):
                status = sync_status(targets, paths, config).set_index("galaxy")

            completed = status.drop(index=list(problem))
            self.assertTrue(completed["status"].eq("ok").all())
            self.assertTrue(completed["stage"].eq("complete").all())
            self.assertEqual(status.loc["NGC 4697", "status"], "pending")
            self.assertEqual(
                status.loc["NGC 4697", "stage"], "spectral measurement"
            )
            for galaxy in problem - {"NGC 4697"}:
                self.assertEqual(status.loc[galaxy, "status"], "pending")
                self.assertEqual(status.loc[galaxy, "stage"], "model/mask/PSF")

    def test_new_isophote_schema_schedules_the_same_rebuild_for_all_targets(self):
        targets = read_targets(DEFAULT_MANIFEST, DEFAULT_DATA_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            paths = campaign_paths(Path(directory) / "campaign")
            paths["root"].mkdir(parents=True)
            config = ExperimentConfig()
            with patch(
                "run_sbf_f090w.final_result_valid",
                return_value=(False, {}, "source stage is not reusable"),
            ), patch(
                "run_sbf_f090w.source_result_valid",
                return_value=(False, {}, "legacy F090W source"),
            ):
                status = sync_status(targets, paths, config)

        self.assertEqual(len(status), 14)
        self.assertTrue(status["status"].eq("pending").all())
        self.assertTrue(status["stage"].eq("model/mask/PSF").all())

    def test_psf_cache_is_atomic_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            science_path = root / "science.fits"
            primary = fits.PrimaryHDU()
            primary.header["FILTER"] = "F090W"
            primary.header["APERNAME"] = "NRCA1_FULL"
            primary.header["DETECTOR"] = "MULTIPLE"
            primary.header["MJD-AVG"] = 60000.0
            science = fits.ImageHDU(
                np.ones((3, 3), dtype=np.float32), name="SCI"
            )
            science.header["PIXAR_SR"] = 2.29e-14
            fits.HDUList([primary, science]).writeto(science_path)

            cache_primary = fits.PrimaryHDU()
            cache_primary.header["PSFMETH"] = "unit-test"
            cache_primary.header["FILTER"] = "F090W"
            cache_primary.header["APERNAME"] = "NRCA1_FULL"
            cache_primary.header["SCIFILE"] = science_path.name
            cache_primary.header["SCIDET"] = "MULTIPLE"
            cache_primary.header["SCIMJD"] = 60000.0
            cache_primary.header["OPDCORR"] = "TEST"
            cache_primary.header["OPDDT"] = 1.0
            scale = float(np.sqrt(2.29e-14 / 2.350443e-11))
            cache_primary.header["SCI_PXS"] = scale
            cache_primary.header["PSF_PXS"] = scale
            hdus = [cache_primary]
            for index in range(F090W_PSF_COUNT):
                array = np.zeros((F090W_PSF_SIZE, F090W_PSF_SIZE), np.float32)
                array[F090W_PSF_SIZE // 2, F090W_PSF_SIZE // 2] = 1.0
                hdu = fits.ImageHDU(array, name=f"PSF{index:02d}")
                hdu.header["PSFID"] = f"test-{index}"
                hdu.header["PSFKIND"] = "model"
                hdu.header["OPDPATH"] = "test_opd.fits"
                hdu.header["PSFEXT"] = "DET_DIST"
                hdu.header["DETPOS"] = f"({index}, {index})"
                hdus.append(hdu)
            cache_path = root / "psf.fits"
            write_psf_cache_atomic(
                cache_path,
                fits.HDUList(hdus),
                expected_filter="F090W",
                expected_size=F090W_PSF_SIZE,
            )
            loaded = load_f090w_psf_cache(
                cache_path,
                science_path,
                root,
                "science",
                expected_filter="F090W",
                expected_size=F090W_PSF_SIZE,
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["psf_library"]), F090W_PSF_COUNT)
            self.assertEqual(loaded["psf"].shape, (129, 129))
            self.assertAlmostEqual(float(loaded["psf"].sum()), 1.0)
            self.assertEqual(
                loaded["psf_library"][0]["opd_path"], "test_opd.fits"
            )
            self.assertEqual(
                loaded["psf_library"][0]["selected_extension"], "DET_DIST"
            )

            fits.setval(science_path, "MJD-AVG", value=60001.0, ext=0)
            stale = load_f090w_psf_cache(
                cache_path,
                science_path,
                root,
                "science",
                expected_filter="F090W",
                expected_size=F090W_PSF_SIZE,
            )
            self.assertIsNone(stale)


if __name__ == "__main__":
    unittest.main()
