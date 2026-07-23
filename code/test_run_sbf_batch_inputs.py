#!/usr/bin/env python3
import csv
import json
import os
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import run_sbf_batch as batch


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


class BatchInputContractTests(unittest.TestCase):
    def test_sbf3_standalone_defaults_and_five_fits_contract(self):
        notebook = json.loads((CODE_DIR / "sbf-3.ipynb").read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("NGC 1404", code)
        self.assertIn("jw03055-o003_t003_nircam_clear-f356w_i2d.fits", code)
        self.assertIn("jw03055-o003_t003_nircam_clear-f277w_i2d.fits", code)
        self.assertIn("PROJECT_ROOT = PYTHON_EXECUTABLE.parents[2]", code)
        self.assertIn("DATA_ROOT = PROJECT_ROOT / \"data\"", code)
        self.assertIn(
            "out_dir = resolve_project_path(globals().get(\"out_dir\", signal_path.parent))",
            code,
        )
        for fits_name in (
            "01_модель_чистая",
            "02_изофоты_чистые",
            "03_остатки_общие",
            "04_остатки_общие_рабочие",
            "05_остатки_общие_рабочие_два_кольца",
        ):
            self.assertIn(fits_name, code)
        self.assertNotIn("save_stage_fits", code)
        self.assertNotIn("save_stage_image", code)

    def test_sbf3_runner_tracks_exactly_five_calibration_fits(self):
        paths = batch.result_paths(Path("products"), "signal", pipeline_label="sbf3")
        fits_paths = {key: path for key, path in paths.items() if path.suffix == ".fits"}
        self.assertEqual(
            set(fits_paths),
            {
                "clean_model_fits",
                "clean_isophotes_fits",
                "full_residual_fits",
                "working_residual_fits",
                "working_annuli_residual_fits",
            },
        )
        self.assertEqual(
            [path.name for path in fits_paths.values()],
            [
                "signal_01_модель_чистая.fits",
                "signal_02_изофоты_чистые.fits",
                "signal_03_остатки_общие.fits",
                "signal_04_остатки_общие_рабочие.fits",
                "signal_05_остатки_общие_рабочие_два_кольца.fits",
            ],
        )

    def test_sbf2_runner_paths_remain_unchanged(self):
        paths = batch.result_paths(Path("products"), "signal", pipeline_label="sbf2")
        self.assertEqual(
            paths["science_residual_fits"].name,
            "signal_sbf_resid_full_science.fits",
        )
        self.assertIn("inner_usable_residual_fits", paths)
        self.assertIn("outer_usable_residual_fits", paths)

    def test_sbf3_result_json_records_five_fits_existence(self):
        fits_keys = {
            "clean_model_fits",
            "clean_isophotes_fits",
            "full_residual_fits",
            "working_residual_fits",
            "working_annuli_residual_fits",
        }
        notebook = {
            "metadata": {"sbf_pipeline": {"family": "sbf3"}},
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from pathlib import Path\n",
                        "import numpy as np\n",
                        "from astropy.io import fits\n",
                        "stem = Path(signal_path).stem\n",
                        "names = [\n",
                        "    '01_модель_чистая',\n",
                        "    '02_изофоты_чистые',\n",
                        "    '03_остатки_общие',\n",
                        "    '04_остатки_общие_рабочие',\n",
                        "    '05_остатки_общие_рабочие_два_кольца',\n",
                        "]\n",
                        "for name in names:\n",
                        "    fits.writeto(out_dir / f'{stem}_{name}.fits', np.zeros((2, 2)), overwrite=True)\n",
                        "recommended_sbf = {'mbar_weighted': 1.23}\n",
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "renamed.ipynb"
            template.write_text(json.dumps(notebook))
            result = batch.execute_template_for_target(
                template,
                "test galaxy",
                root / "signal.fits",
                root / "color.fits",
                root / "batch",
                output_dir=root / "products",
            )
            saved_result = json.loads(
                next((root / "batch").glob("*_result.json")).read_text()
            )

        for key in fits_keys:
            self.assertTrue(result[f"{key}_exists"])
            self.assertTrue(saved_result[f"{key}_exists"])
        self.assertEqual(
            {key.removesuffix("_exists") for key in saved_result if key.endswith("_fits_exists")},
            fits_keys,
        )

    def test_sbf3_residual_links_use_two_working_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "03.fits"
            working = root / "04.fits"
            annuli = root / "05.fits"
            for path in (full, working, annuli):
                path.write_bytes(b"fits")
            batch.link_residuals(
                [
                    {
                        "status": "ok",
                        "galaxy": "NGC 1404",
                        "template_family": batch.SBF3_NOTEBOOK_FAMILY,
                        "full_residual_fits": str(full),
                        "working_residual_fits": str(working),
                        "working_annuli_residual_fits": str(annuli),
                    }
                ],
                root / "batch",
            )
            links = sorted((root / "batch" / "residuals").iterdir())
            self.assertEqual(
                [path.name for path in links],
                [
                    "NGC_1404_working_annuli_residual_fits.fits",
                    "NGC_1404_working_residual_fits.fits",
                ],
            )
            self.assertEqual(
                {path.resolve() for path in links},
                {working.resolve(), annuli.resolve()},
            )

    def test_additional_manifest_snapshot_and_runner_boundary(self):
        manifest = CODE_DIR / "targets_additional_manifest.csv"
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 125)
        enabled = [row for row in rows if row["download_enabled"] == "true"]
        self.assertEqual(len(enabled), 114)
        self.assertEqual(
            Counter(row["program"] for row in enabled),
            Counter({"7763": 74, "5989": 38, "1176": 1, "6565": 1}),
        )
        self.assertEqual(
            sum(
                int(row["signal_content_length_bytes"])
                + int(row["color_content_length_bytes"])
                for row in enabled
            ),
            246_572_758_080,
        )

        actionable = batch.read_targets_from_csv(manifest, PROJECT_ROOT / "data")
        self.assertEqual(len(actionable), 114)
        self.assertNotIn("NGC 4926", {target["name"] for target in actionable})
        self.assertNotIn("Cen A", {target["name"] for target in actionable})
        m104 = next(target for target in actionable if target["name"] == "M104")
        self.assertEqual(
            (m104["signal_filter"], m104["color_filter"]),
            ("F200W", "F090W"),
        )

    def test_live_consumer_loads_both_programs_and_their_own_filter_pairs(self):
        targets = batch.load_manifest_targets(
            CODE_DIR / "targets_go3055_manifest.csv",
            PROJECT_ROOT / "data",
            [CODE_DIR / "targets_additional_manifest.csv"],
        )
        selected = batch.select_targets(
            targets,
            galaxies=None,
            programs=["GO-3055", "07763"],
            allow_bulk_targets=True,
        )
        self.assertEqual(len(selected), 88)
        self.assertEqual(
            Counter(target["program"] for target in selected),
            Counter({"3055": 14, "7763": 74}),
        )
        self.assertEqual(
            Counter(
                (target["signal_filter"], target["color_filter"])
                for target in selected
            ),
            Counter({("F150W", "F090W"): 14, ("F150W", "F115W"): 74}),
        )
        self.assertEqual(selected[0]["name"], "NGC 1380")

    def test_live_consumer_cli_keeps_the_second_manifest_and_program_filter(self):
        args = batch.parse_args(
            [
                "--target-csv",
                "first.csv",
                "--extra-target-csv",
                "second.csv",
                "--programs",
                "3055",
                "7763",
                "--no-download",
                "--prefetch-targets",
                "0",
            ]
        )
        self.assertEqual(args.target_csv, "first.csv")
        self.assertEqual(args.extra_target_csv, ["second.csv"])
        self.assertEqual(args.programs, ["3055", "7763"])
        self.assertTrue(args.no_download)
        self.assertEqual(args.prefetch_targets, 0)

    def test_project_relative_paths_are_independent_of_shell_working_directory(self):
        args = batch.parse_args(
            [
                "--template",
                "code/sbf-3.ipynb",
                "--target-csv",
                "code/targets_go3055_manifest.csv",
                "--extra-target-csv",
                "code/targets_additional_manifest.csv",
                "--data-root",
                "data",
                "--batch-root",
                "runs/test/batch",
            ]
        )
        previous = Path.cwd()
        try:
            os.chdir(CODE_DIR)
            batch.normalize_cli_paths(args)
        finally:
            os.chdir(previous)

        self.assertEqual(Path(args.template), CODE_DIR / "sbf-3.ipynb")
        self.assertEqual(
            Path(args.target_csv), CODE_DIR / "targets_go3055_manifest.csv"
        )
        self.assertEqual(
            [Path(path) for path in args.extra_target_csv],
            [CODE_DIR / "targets_additional_manifest.csv"],
        )
        self.assertEqual(Path(args.data_root), PROJECT_ROOT / "data")
        self.assertEqual(
            Path(args.batch_root), PROJECT_ROOT / "runs" / "test" / "batch"
        )

    def test_program_filter_is_applied_before_bulk_guard(self):
        targets = [
            {"name": f"A-{index}", "program": "1"} for index in range(20)
        ] + [{"name": "B", "program": "2"}]
        self.assertEqual(
            batch.select_targets(targets, None, programs=["2"]),
            [{"name": "B", "program": "2"}],
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            batch.select_targets(targets, None, programs=["999"])

    def test_download_manager_receives_only_parent_selected_targets(self):
        args = Namespace(
            no_download=False,
            target_csv="targets.csv",
            data_root="data",
            batch_root="batch",
            download_retry_seconds=120,
            min_free_gb=30.0,
            no_cleanup_inputs=True,
        )
        targets = [{"name": "NGC 1380"}, {"name": "M104"}]
        with patch.object(batch.subprocess, "Popen") as popen:
            batch.start_download_manager(args, targets, [])
        command = popen.call_args.args[0]
        selected = command[command.index("--galaxies") + 1 :]
        self.assertEqual(selected, ["NGC 1380", "M104"])

    def test_download_manager_is_terminated_at_parent_boundary(self):
        process = Mock()
        process.poll.return_value = None
        batch.stop_download_manager(process, grace_seconds=1)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=1)

    def test_target_selection_is_shared_with_download_worker(self):
        targets = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        self.assertEqual(
            batch.select_targets(targets, ["B"]),
            [{"name": "B"}],
        )
        self.assertEqual(batch.select_targets(targets, None), targets)
        self.assertEqual(batch.select_targets(targets, []), [])

    def test_large_manifest_requires_explicit_selection(self):
        targets = [{"name": f"target-{index}"} for index in range(15)]
        with self.assertRaisesRegex(RuntimeError, "implicit bulk selection"):
            batch.select_targets(targets, None)
        self.assertEqual(
            batch.select_targets(targets, None, allow_bulk_targets=True),
            targets,
        )

    def test_disk_guard_accounts_for_next_transfer(self):
        with patch.object(
            batch,
            "log_resources",
            return_value=({"free_gb": 31.0, "total_gb": 100.0}, {}),
        ):
            self.assertFalse(
                batch.ensure_disk_space_for_downloads(
                    ".",
                    [],
                    min_free_gb=30.0,
                    cleanup_enabled=False,
                    required_bytes=2 * 1024**3,
                )
            )
            self.assertTrue(
                batch.ensure_disk_space_for_downloads(
                    ".",
                    [],
                    min_free_gb=30.0,
                    cleanup_enabled=False,
                    required_bytes=512 * 1024**2,
                )
            )

    def test_live_consumer_disk_budget_subtracts_existing_partial_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = batch.normalize_target(
                {
                    "name": "PARTIAL",
                    "signal_product": "signal.fits",
                    "color_product": "color.fits",
                    "signal_size": 1000,
                    "color_size": 2000,
                }
            )
            target_dir = root / "PARTIAL"
            target_dir.mkdir()
            (target_dir / "signal.fits.part").write_bytes(b"x" * 400)
            (target_dir / "color.fits.restart.part").write_bytes(b"x" * 750)

            growth = batch.remaining_input_growth([target], root)

        self.assertEqual(growth["remaining_bytes"], 1850)
        self.assertEqual(growth["unknown_product_count"], 0)

    def test_extended_manifest_skips_disabled_rows(self):
        columns = [
            "program,target,obsid,signal_filter,color_filter,signal_product,"
            "color_product,signal_product_uri,color_product_uri,download_enabled,"
            "availability_status,science_role,priority,public_release_date\n"
        ]
        columns.append(
            "5989,READY,o001,F150W,F356W,ready-f150w_i2d.fits,"
            "ready-f356w_i2d.fits,,,true,public,calibration,1,\n"
        )
        columns.append(
            "8277,EMBARGOED,o001,F150W,F356W,held-f150w_i2d.fits,"
            "held-f356w_i2d.fits,,,false,proprietary,hold,3,2027-06-01\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "extended.csv"
            manifest.write_text("".join(columns))
            targets = batch.read_targets_from_csv(manifest, Path(tmp) / "data")

        self.assertEqual([target["name"] for target in targets], ["READY"])
        self.assertEqual(targets[0]["availability_status"], "public")
        self.assertEqual(targets[0]["science_role"], "calibration")

    def test_extended_manifest_rejects_bad_enabled_flag(self):
        self.assertFalse(batch.manifest_row_enabled({"download_enabled": "no"}))
        with self.assertRaisesRegex(ValueError, "invalid download_enabled"):
            batch.manifest_row_enabled(
                {"target": "bad", "download_enabled": "eventually"}
            )

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
        self.assertEqual(len(batch.select_targets(generic, None)), 14)
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

    def test_result_reuse_ignores_template_sha_for_same_inputs(self):
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
            self.assertIsNotNone(
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

    def test_notebook_has_no_stale_science_outputs(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                if cell.get("id") == "d34c9dac":
                    # The bootstrap cell intentionally records the resolved
                    # interpreter/project roots for an interactive user.
                    continue
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
