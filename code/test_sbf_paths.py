"""Small portability tests; no science notebook, download or galaxy fit is run."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sbf_paths import (
    PROJECT_ROOT, default_stpsf_data_dir, load_project_json,
    portable_path, project_path, resolve_product_paths,
)


class PortablePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "renamed checkout"
        self.root.mkdir()

    def test_root_relative_is_independent_of_working_directory(self):
        with patch("pathlib.Path.cwd", return_value=self.root / "not-code"):
            self.assertEqual(project_path("data/test.fits", self.root), self.root / "data/test.fits")
            self.assertEqual(project_path("runs/new/output", self.root), self.root / "runs/new/output")

    def test_legacy_absolute_is_rebased_even_if_old_file_exists(self):
        old = self.root.parent / "old" / "course_work-SBF" / "data/test.fits"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"old tree must not be read")
        self.assertEqual(project_path(old, self.root), self.root / "data/test.fits")
        with self.assertRaisesRegex(FileNotFoundError, "Required input is absent"):
            project_path(old, self.root, must_exist=True)

    def test_current_absolute_is_unchanged(self):
        current = self.root / "runs/example.csv"
        self.assertEqual(project_path(current, self.root), current)

    def test_unrelated_absolute_remains_external(self):
        external = self.root.parent / "stpsf-data"
        self.assertEqual(project_path(external, self.root), external)
        self.assertEqual(portable_path(external, self.root), str(external))

    def test_missing_and_empty_paths_fail_clearly(self):
        for value in (None, "", "  "):
            with self.assertRaises(ValueError):
                project_path(value, self.root)
        with self.assertRaises(FileNotFoundError):
            project_path("data/missing.fits", self.root, must_exist=True)

    def test_ambiguous_legacy_path_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            project_path("/old/course_work-SBF/runs/course_work-SBF/data/file", self.root)

    def test_portable_serialization_keeps_repository_location(self):
        path = self.root / "runs/NGC 3379/result.json"
        self.assertEqual(portable_path(path, self.root), "runs/NGC 3379/result.json")

    def test_product_json_resolution_is_in_memory_only(self):
        payload = {
            "table_paths": {"fit": "/old/course_work-SBF/runs/fit.csv"},
            "normalized_fits": [{"path": "runs/normal.fits"}],
            "url": "https://example.org/course_work-SBF/data/file",
            "source_key": "abcdef0123456789",
            "description": "normalization in code is unchanged",
            "kind": "code",
        }
        path = self.root / "result.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = path.read_bytes()
        resolved = load_project_json(path, self.root)
        self.assertEqual(resolved["table_paths"]["fit"], str(self.root / "runs/fit.csv"))
        self.assertEqual(resolved["normalized_fits"][0]["path"], str(self.root / "runs/normal.fits"))
        for key in ("url", "source_key", "description", "kind"):
            self.assertEqual(resolved[key], payload[key])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(resolve_product_paths(payload, self.root), resolved)

    def test_external_stpsf_configuration_is_respected(self):
        external = self.root.parent / "reference-data"
        with patch.dict(os.environ, {"STPSF_PATH": str(external)}):
            self.assertEqual(default_stpsf_data_dir(), external)

    def test_project_default_comes_from_module_not_cwd(self):
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(project_path("code"), PROJECT_ROOT / "code")

    def test_spectral_reader_resolves_legacy_source_products(self):
        from sbf2_normalized_winsor_core import inspect_source

        run_dir = self.root / "runs/source"
        batch_dir = self.root / "runs/batch"
        run_dir.mkdir(parents=True)
        batch_dir.mkdir()
        keys = {
            "signal_path": "signal.fits",
            "model_full_fits": "model.fits",
            "science_residual_fits": "residual.fits",
            "inner_usable_residual_fits": "inner.fits",
            "outer_usable_residual_fits": "outer.fits",
            "df_sbf_csv": "measurements.csv",
        }
        for name in [*keys.values(), "test_sbf_catalog_mask_mcut.fits", "test_psf_129.fits"]:
            (run_dir / name).write_bytes(b"metadata-only fixture")
        saved = {
            "status": "ok", "stem": "test", "signal_background_scalar": 0.0,
            "output_dir": "/old/course_work-SBF/runs/source",
            **{key: f"/old/course_work-SBF/runs/source/{name}" for key, name in keys.items()},
        }
        source_path = batch_dir / "NGC_3379_result.json"
        source_path.write_text(json.dumps(saved), encoding="utf-8")
        before = source_path.read_bytes()
        with patch("sbf_paths.PROJECT_ROOT", self.root):
            source = inspect_source("NGC 3379", batch_dir)
        self.assertTrue(all(Path(value).is_relative_to(self.root) for value in source["paths"].values()))
        self.assertEqual(source_path.read_bytes(), before)

    def test_spectral_table_reader_accepts_relative_result_paths(self):
        from sbf2_normalized_winsor_core import load_result_tables

        (self.root / "runs").mkdir()
        (self.root / "runs/fit.csv").write_text("P0\n0.9\n", encoding="utf-8")
        with patch("sbf_paths.PROJECT_ROOT", self.root):
            tables = load_result_tables({"table_paths": {"fit": "runs/fit.csv"}})
        self.assertEqual(tables["fit"]["P0"].tolist(), [0.9])


if __name__ == "__main__":
    unittest.main()
