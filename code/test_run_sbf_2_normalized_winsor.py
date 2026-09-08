"""RU: маленькие регрессионные тесты FFT-ядра. EN: no science files are used."""

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from sbf2_normalized_winsor_core import (
    ExperimentConfig,
    radial_mean_sem,
    robust_mag_scatter,
    weighted_fit,
)


class SpectralCoreTests(unittest.TestCase):
    def test_adopted_settings_remain_fixed(self):
        config = ExperimentConfig(normalized_sigma=4.0)
        self.assertEqual(config.candidate_branch, "normalized_full_3p5")
        self.assertEqual(config.union_branch, "normalized_union_4")
        self.assertEqual(config.kmins, (0.01, 0.03, 0.04))
        self.assertEqual((config.kmax, config.k_bins), (0.25, 80))
        self.assertEqual((config.e_realizations, config.random_seed), (64, 1489))

    def test_weighted_fit_recovers_known_amplitude_and_white_floor(self):
        expectation = np.linspace(0.05, 0.5, 20)
        error = np.linspace(0.01, 0.02, 20)
        fit = weighted_fit(0.9 * expectation + 0.03, error, expectation)
        self.assertAlmostEqual(fit["P0"], 0.9, places=12)
        self.assertAlmostEqual(fit["P1"], 0.03, places=12)
        design = np.column_stack([expectation, np.ones(20)]) / error[:, None]
        covariance = np.linalg.pinv(design.T @ design)
        self.assertAlmostEqual(fit["P0_sigma"], np.sqrt(covariance[0, 0]))

    def test_radial_sem_and_minimum_support(self):
        plan = {
            "valid": np.ones(5, dtype=bool),
            "ids": np.array([0, 0, 0, 1, 1]),
            "n_bins": 2,
        }
        mean, sem, count = radial_mean_sem(np.array([1, 2, 3, 8, 9]), plan, 3)
        np.testing.assert_array_equal(count, [3, 2])
        self.assertEqual(mean[0], 2)
        self.assertAlmostEqual(sem[0], 1 / np.sqrt(3))
        self.assertTrue(np.isnan(mean[1]) and np.isnan(sem[1]))

    def test_window_scatter_is_not_the_annular_half_difference(self):
        scatter, method = robust_mag_scatter([28.0, 28.01, 28.02])
        self.assertAlmostEqual(scatter, 1.4826 * 0.01)
        self.assertEqual(method, "k-window MAD")


class DiagnosticPsfCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile only the small read-only validator, never a scientific cell.
        notebook = json.loads(
            Path(__file__).with_name("sbf-2-systematics.ipynb").read_text()
        )
        tree = ast.parse("".join(notebook["cells"][34]["source"]))
        validator = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "read_compatible_psf"
        )
        namespace = {"np": np, "fits": fits}
        exec(compile(ast.Module(body=[validator], type_ignores=[]),
                     "<PSF cache validator>", "exec"), namespace)
        cls.read_cache = staticmethod(namespace["read_compatible_psf"])

    def setUp(self):
        self.expected = {
            "GALAXY": "test", "FILTER": "F150W", "APERNAME": "NRCA1_FULL",
            "OPDCORR": "R2024052602", "OPDDAYS": 0.65,
            "DETX": 1024, "DETY": 1024, "NLAMBDA": 7,
            "FFTOVER": 4, "DETOVER": 1, "DISTORT": True,
            "STAMPSZ": 257, "SRCARG": "None", "SPECTRUM": "test spectrum",
            "EXTUSED": "DET_DIST",
        }
        self.header = fits.Header(self.expected)
        self.header.update({
            "RAWSUM": 0.99, "STPSFVER": "old-version",
            "POPPYVER": "old-poppy", "SYNPHOT": "old-synphot",
        })
        self.image = np.full((257, 257), 1 / 257**2)

    def test_compatible_cache_is_read_only_and_preserves_generation_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.fits"
            fits.writeto(path, self.image, self.header)
            before, mtime = path.read_bytes(), path.stat().st_mtime_ns
            image, header = self.read_cache(path, self.expected)
            np.testing.assert_array_equal(image, self.image)
            self.assertEqual(header["STPSFVER"], "old-version")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_mismatched_generation_contract_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.fits"
            fits.writeto(path, self.image, self.header)
            for key, value in {
                "OPDCORR": "different", "FILTER": "F090W", "DETX": 12,
                "NLAMBDA": 11, "FFTOVER": 2, "DETOVER": 2,
                "DISTORT": False, "EXTUSED": "DET_SAMP", "OPDDAYS": 0.7,
            }.items():
                with self.subTest(key=key), contextlib.redirect_stdout(io.StringIO()):
                    expected = {**self.expected, key: value}
                    self.assertIsNone(self.read_cache(path, expected))

    def test_unproven_legacy_and_invalid_pixels_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.fits"
            for header, image in [
                (fits.Header(), self.image),
                (self.header, np.ones((129, 129))),
                (self.header, self.image * 2),
                (self.header, np.full((257, 257), np.nan)),
            ]:
                fits.writeto(path, image, header, overwrite=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(self.read_cache(path, self.expected))


if __name__ == "__main__":
    unittest.main()
