"""RU: маленькие регрессионные тесты FFT-ядра. EN: no science files are used."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sbf.sbf2_normalized_winsor_core import (
    ExperimentConfig,
    _file_fingerprint,
    radial_mean_sem,
    robust_mag_scatter,
    weighted_fit,
)


class SpectralCoreTests(unittest.TestCase):
    def test_directory_relocation_preserves_existing_source_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for old_name, new_name in [
                ("sbf2_go3055", "F150W/source"),
                ("sbf_f090w_go3055", "F090W"),
            ]:
                original = root / "runs" / old_name / "product.csv"
                original.parent.mkdir(parents=True)
                original.write_bytes(b"P0\n0.9\n")
                with patch("sbf.sbf2_normalized_winsor_core.PROJECT_ROOT", root):
                    before = _file_fingerprint(original, hash_small=True)
                    relocated = root / "runs" / new_name / original.name
                    relocated.parent.mkdir(parents=True)
                    original.rename(relocated)
                    self.assertEqual(_file_fingerprint(relocated, hash_small=True), before)
                    self.assertFalse(original.exists())

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


if __name__ == "__main__":
    unittest.main()
