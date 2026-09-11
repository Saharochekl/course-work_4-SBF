"""Ошибки опубликованных сравнений / Published-comparison error regression tests."""

import unittest

import numpy as np
import pandas as pd

from figures.build_sbf2_article_tables import jensen2015_distances


class Jensen2015ErrorsTest(unittest.TestCase):
    def test_common_scale_added_once_without_changing_distance(self):
        reference = pd.DataFrame({
            "galaxy": ["NGC 1380"],
            "mu_f160w_jensen2015_reconstructed": [31.51817],
            "sigma_mu_f160w_jensen2015_reconstructed": [0.1284842518685174],
        })
        before = reference.copy(deep=True)
        row = jensen2015_distances(reference).iloc[0]
        self.assertEqual(row.mu_jensen2015_f160w, 31.51817)
        self.assertEqual(row.sigma_mu_jensen2015_f160w_common, 0.10)
        self.assertAlmostEqual(row.sigma_mu_jensen2015_f160w_total ** 2,
                               row.sigma_mu_jensen2015_f160w_internal ** 2 + 0.10 ** 2)
        self.assertAlmostEqual(row.D_jensen2015_f160w_mpc, 10 ** ((31.51817 - 25) / 5))
        self.assertAlmostEqual(row.sigma_D_jensen2015_f160w_total_mpc ** 2,
                               row.sigma_D_jensen2015_f160w_internal_mpc ** 2
                               + row.sigma_D_jensen2015_f160w_common_mpc ** 2)
        self.assertTrue(np.isfinite(row.sigma_D_jensen2015_f160w_total_mpc))
        pd.testing.assert_frame_equal(reference, before)


if __name__ == "__main__":
    unittest.main()
