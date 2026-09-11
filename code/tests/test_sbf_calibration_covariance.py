"""Synthetic checks only; no scientific notebook cells or FITS are executed."""

import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sbf.sbf_calibration_covariance import (
    constant_loo_distance_covariance, shared_anchor_covariance,
    validate_color_covariance,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "sbf-2-graph.ipynb"


def notebook_functions(index, names, namespace):
    """Load named definitions only, never execute a complete notebook cell."""
    notebook = json.loads(NOTEBOOK.read_text())
    source = ast.parse("".join(notebook["cells"][index]["source"]))
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(definitions) != len(names):
        raise AssertionError("Requested notebook definitions are missing")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)


class CovarianceTests(unittest.TestCase):
    def test_anchor_builder_preserves_mixed_and_literal_experiments(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = ast.parse("".join(notebook["cells"][30]["source"]))
        namespace = {"np": np}
        constants = {"CLUSTER_ANCHORS", "virgo_cluster_members", "fornax_cluster_members"}
        for node in source.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in constants:
                    namespace[name] = ast.literal_eval(node.value)
        notebook_functions(30, {"frame_for_distance_anchor"}, namespace)
        frame = pd.DataFrame({
            "galaxy": ["NGC 1380", "NGC 4374", "NGC 4697"],
            "mu_lit": [31.408, 31.112, 30.324], "sigma_mu_lit": 0.028,
            "mbar_F150W": 28., "sigma_mbar_internal": 0.02,
            "sigma_A_F090W_mag": 0.006, "sigma_A_F150W_mag": 0.003,
            "sigma_reddening_color_mag": 0.003,
            "cov_color_mbar_extinction": 0.000009,
        })
        build = namespace["frame_for_distance_anchor"]
        mixed = build("Paper III cluster means", frame)
        np.testing.assert_allclose(mixed.mu_anchor, [31.424, 31.055, 30.324])
        np.testing.assert_allclose(mixed.sigma_mu_anchor**2,
                                   [0.053**2 + 0.020**2, 0.085**2 + 0.013**2, 0.028**2])
        np.testing.assert_allclose(mixed.sigma_mu_anchor_shared_mag, [0.020, 0.013, 0])
        np.testing.assert_allclose(mixed.cov_color_Mbar_extinction, [0.000009, 0.000009, -0.000009])
        literal = build("Paper III cluster means", frame,
                        {"NGC 4697": (30.330, 0.036, "Paper III individual")})
        self.assertEqual(literal.loc[2, "mu_anchor"], 30.330)
        self.assertEqual(literal.loc[2, "sigma_mu_anchor"], 0.036)
        self.assertEqual(literal.loc[2, "anchor_covariance_group"], "")
        self.assertEqual(frame.loc[2, "mu_lit"], 30.324)

    def test_shared_cluster_mean_does_not_share_depth(self):
        labels = ["Fornax", "Fornax", "Virgo", ""]
        means = np.array([0.020, 0.020, 0.013, 0.0])
        depth = np.array([0.053, 0.053, 0.085, 0.036])
        shared = shared_anchor_covariance(labels, means)
        total = np.diag(depth**2) + shared
        self.assertAlmostEqual(total[0, 1], 0.020**2)
        self.assertEqual(total[0, 2], 0.0)
        np.testing.assert_allclose(np.diag(total), depth**2 + means**2)
        self.assertGreater(np.linalg.eigvalsh(total).min(), 0)

    def test_rejects_impossible_ring_color_budget(self):
        with self.assertRaisesRegex(ValueError, "Non-PSD"):
            validate_color_covariance([0.03], [0.0001], [-0.000016])
        validate_color_covariance([0.03], [0.004], [-0.000016])

    def test_shared_anchor_cannot_hide_invalid_local_covariance(self):
        shared = np.array([[0.09]])
        with self.assertRaisesRegex(ValueError, "Non-PSD"):
            validate_color_covariance([np.sqrt(0.09 + 0.0001)], [0.01], [0.001], shared)

    def test_loo_includes_target_training_cross_terms(self):
        n = 5
        covariance, weights = constant_loo_distance_covariance(
            np.full(n, 0.02), np.zeros(n), np.zeros(n), np.full(n, 0.03),
            np.full(n, 0.08), 0.08, 0.6021, 1.4156, 0.047,
        )
        np.testing.assert_allclose(np.diag(weights), 0)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        mean_weights = np.full(n, 1 / n)
        # For equal weights, mean(I-A)=0: independent population terms cancel
        # in the mean of the SAME complete calibration sample, scale does not.
        self.assertAlmostEqual(mean_weights @ covariance @ mean_weights, 0.047**2)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)

    def test_nonlinear_local_derivative(self):
        namespace = {"np": np}
        notebook_functions(12, {"relation_derivative"}, namespace)
        derivative = namespace["relation_derivative"]
        fit = {"model": "quadratic", "color_center": 0.5, "slope": 2., "curvature": 3.}
        self.assertAlmostEqual(derivative(fit, 0.7), 3.2)
        fit.update(model="broken", slope_change=4.)
        np.testing.assert_allclose(derivative(fit, [0.4, 0.6]), [2., 6.])

    def test_gls_constant_matches_fixed_scatter_solution(self):
        namespace = {
            "np": np, "minimize": minimize, "USE_COLOR_ERRORS_IN_FIT": True,
            "COLOR_MODELS": {"constant": "", "linear": "", "quadratic": "", "broken": ""},
            "shared_anchor_covariance": shared_anchor_covariance,
            "validate_color_covariance": validate_color_covariance,
        }
        notebook_functions(12, {"fit_relation"}, namespace)
        groups = np.array(["Fornax"] * 3 + ["Virgo"] * 5)
        mean_sigma = np.where(groups == "Fornax", 0.020, 0.013)
        depth = np.where(groups == "Fornax", 0.053, 0.085)
        frame = pd.DataFrame({
            "color_F090W_F150W": np.linspace(0.5, 0.7, 8),
            "Mbar_F150W": [-3.24, -3.21, -3.23, -3.18, -3.19, -3.20, -3.17, -3.18],
            "sigma_Mbar_internal": np.sqrt(0.02**2 + depth**2 + mean_sigma**2),
            "sigma_color_stat_mag": 0., "cov_color_Mbar_extinction": 0.,
            "environment": groups, "anchor_covariance_group": groups,
            "sigma_mu_anchor_shared_mag": mean_sigma,
        })
        result = namespace["fit_relation"](frame, model="constant")
        covariance = np.diag(0.02**2 + depth**2 + result["sigma_int"]**2)
        covariance += shared_anchor_covariance(groups, mean_sigma)
        weights = np.linalg.solve(covariance, np.ones(len(frame)))
        expected = weights @ frame["Mbar_F150W"] / weights.sum()
        self.assertTrue(result["use_shared_anchor_covariance"])
        self.assertAlmostEqual(result["intercept"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
