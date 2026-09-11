"""RU/EN: independent Fourier-mode SEM; small synthetic arrays, no science data."""

import unittest
from dataclasses import asdict

import numpy as np

from sbf.sbf2_normalized_winsor_core import (
    EXPERIMENT_VERSION, RADIAL_SEM_METHOD, ExperimentConfig,
    _jsonable, _result_config_matches, radial_mean_sem, radial_plan, weighted_fit,
)


class IndependentFourierModesTests(unittest.TestCase):
    def test_old_results_cannot_be_reused_with_the_new_sem(self):
        config = ExperimentConfig()
        result = {"status": "ok", "version": EXPERIMENT_VERSION,
                  "config": _jsonable(asdict(config))}
        self.assertFalse(_result_config_matches(result, config))
        result["radial_sem_method"] = RADIAL_SEM_METHOD
        self.assertTrue(_result_config_matches(result, config))

    def test_self_conjugate_coordinates_for_odd_and_even_axes(self):
        for shape in [(9, 11), (9, 12), (10, 11), (10, 12)]:
            with self.subTest(shape=shape):
                plan = radial_plan(shape, 8)
                multiplicity = plan["mode_multiplicity"].reshape(shape)
                y, x = np.indices(shape)
                self_conjugate = ((2 * y) % shape[0] == 0) & ((2 * x) % shape[1] == 0)
                np.testing.assert_array_equal(multiplicity == 1, self_conjugate)
                self.assertEqual(int(self_conjugate.sum()), (1 + (shape[0] % 2 == 0)) * (1 + (shape[1] % 2 == 0)))

    def test_full_grid_mean_is_preserved_and_pair_sem_matches_half_plane(self):
        rng = np.random.default_rng(1409)
        for shape in [(63, 65), (63, 64), (64, 65), (64, 66)]:
            with self.subTest(shape=shape):
                power = np.abs(np.fft.fft2(rng.normal(size=shape))) ** 2
                plan = radial_plan(shape, 12)
                mean, sem, count = radial_mean_sem(power, plan, 3)
                flat = power.ravel()
                selected = plan["valid"]
                old_count = np.bincount(plan["ids"][selected], minlength=plan["n_bins"])
                old_sum = np.bincount(plan["ids"][selected], weights=flat[selected], minlength=plan["n_bins"])
                np.testing.assert_array_equal(count, old_count)
                np.testing.assert_array_equal(mean, old_sum / old_count)
                y, x = np.indices(shape)
                index = np.arange(flat.size).reshape(shape)
                representative = (index <= index[(-y) % shape[0], (-x) % shape[1]]).ravel()
                for bin_id in range(plan["n_bins"]):
                    in_bin = selected & (plan["ids"] == bin_id)
                    if np.any(plan["mode_multiplicity"][in_bin] == 1):
                        continue
                    independent = flat[in_bin & representative]
                    expected = independent.std(ddof=1) / np.sqrt(independent.size)
                    self.assertAlmostEqual(sem[bin_id] / expected, 1.0, places=12)
                    self.assertEqual(plan["independent_count"][bin_id], independent.size)
                    self.assertEqual(plan["effective_count"][bin_id], independent.size)

    def test_mixed_self_conjugate_and_paired_modes_keep_their_weights(self):
        plan = {
            "valid": np.ones(5, dtype=bool),
            "ids": np.zeros(5, dtype=int),
            "n_bins": 1,
            "mode_multiplicity": np.array([1, 2, 2, 2, 2]),
        }
        # Independent values 0, 3, 7 with full-grid multiplicities 1, 2, 2.
        mean, sem, count = radial_mean_sem(np.array([0., 3., 3., 7., 7.]), plan, 1)
        unique, weights = np.array([0., 3., 7.]), np.array([1., 2., 2.])
        weighted_mean = np.average(unique, weights=weights)
        neff = weights.sum() ** 2 / np.sum(weights**2)
        variance = np.sum(weights * (unique - weighted_mean)**2) / (
            weights.sum() - np.sum(weights**2) / weights.sum()
        )
        self.assertEqual(mean[0], weighted_mean)
        self.assertAlmostEqual(sem[0], np.sqrt(variance / neff))
        self.assertEqual(count[0], 5)

    def test_one_conjugate_pair_cannot_estimate_a_sem(self):
        plan = {"valid": np.ones(2, bool), "ids": np.zeros(2, int),
                "n_bins": 1, "mode_multiplicity": np.array([2, 2])}
        mean, sem, count = radial_mean_sem(np.array([3., 3.]), plan, 1)
        self.assertEqual(mean[0], 3.)
        self.assertTrue(np.isnan(sem[0]))
        self.assertEqual(count[0], 2)

    def test_saved_full_grid_sem_has_exact_pair_only_conversion(self):
        values = np.repeat([1., 2., 4., 8., 9.], 2)
        plan = {"valid": np.ones(10, bool), "ids": np.zeros(10, int),
                "n_bins": 1, "mode_multiplicity": np.full(10, 2)}
        _, sem, _ = radial_mean_sem(values, plan, 1)
        old_sem = values.std(ddof=1) / np.sqrt(values.size)
        factor = np.sqrt(2 * (values.size - 1) / (values.size - 2))
        self.assertAlmostEqual(sem[0], old_sem * factor)

    def test_chi_square_rescaling_compensates_inflated_fit_covariance(self):
        x = np.linspace(.1, 1., 30)
        y = 0.9 * x + .03 + .02 * np.sin(np.arange(x.size))
        small_error = np.full(x.size, .001)
        before = weighted_fit(y, small_error, x)
        after = weighted_fit(y, np.sqrt(2) * small_error, x)
        self.assertGreater(after["chi2_reduced"], 1.)
        self.assertAlmostEqual(before["P0"], after["P0"])
        self.assertAlmostEqual(before["P0_sigma"], after["P0_sigma"])
        self.assertAlmostEqual(before["chi2"] / after["chi2"], 2.)

    def test_low_chi_square_fit_does_retain_larger_measurement_errors(self):
        x = np.linspace(.1, 1., 30)
        y = 0.9 * x + .03
        before = weighted_fit(y, np.full(x.size, .01), x)
        after = weighted_fit(y, np.full(x.size, np.sqrt(2) * .01), x)
        self.assertLess(after["chi2_reduced"], 1.)
        self.assertAlmostEqual(after["P0_sigma"] / before["P0_sigma"], np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
