"""Checks of analysis helpers using toy arrays; no scientific notebook execution."""

import ast
import importlib
import json
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from astropy.io import fits
from sbf.sbf_paths import CODE_DIR, PROJECT_ROOT


class AnalysisToolsTests(unittest.TestCase):
    def test_builders_use_project_root_not_package_depth(self):
        for name in ("build_sbf2_article_tables", "build_go3055_article_figures",
                     "build_f090w_residual_montage", "build_f090w_appendix_diagnostics",
                     "publish_article_assets"):
            with self.subTest(module=name):
                module = importlib.import_module("figures." + name)
                self.assertEqual(module.ROOT, PROJECT_ROOT)

    def test_active_notebooks_do_not_require_legacy(self):
        for name in ("sbf-2.ipynb", "sbf-2-graph.ipynb", "sbf-f090w-graph.ipynb"):
            notebook = json.loads((CODE_DIR / name).read_text())
            source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                               if cell["cell_type"] == "code")
            with self.subTest(notebook=name):
                self.assertNotIn("sbf2_systematics", source)
                self.assertNotIn("sbf2_normalized_winsor_recovery", source)
                self.assertNotIn("review-2026-09-09", source)

    def test_imports_never_run_analysis_or_write_products(self):
        modules = [
            "build_sbf2_article_tables", "build_go3055_article_figures",
            "build_f090w_residual_montage", "build_f090w_appendix_diagnostics",
            "build_sbf_f090w_graph_notebook",
        ]
        with patch.object(pd, "read_csv", side_effect=AssertionError("CSV read on import")), \
             patch.object(fits, "open", side_effect=AssertionError("FITS read on import")), \
             patch.object(nbformat, "write", side_effect=AssertionError("Notebook write on import")), \
             patch.object(plt, "subplots", side_effect=AssertionError("Plot on import")):
            for module in modules:
                with self.subTest(module=module):
                    importlib.reload(importlib.import_module("figures." + module))

    def test_distance_conversion_and_quadrature(self):
        from figures.build_sbf2_article_tables import common_part, distance_from_modulus, distance_uncertainty
        self.assertEqual(distance_from_modulus(25), 1)
        self.assertEqual(distance_from_modulus(30), 10)
        self.assertAlmostEqual(distance_uncertainty(10, 0.1), np.log(10) / 5)
        np.testing.assert_allclose(common_part([5, 1], [3, 1 + 1e-15]), [4, 0])

    def test_generated_notebook_sources_match_saved_notebook(self):
        from figures.build_sbf_f090w_graph_notebook import OUTPUT, build_notebook
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        generated = build_notebook()
        self.assertEqual(len(saved["cells"]), len(generated.cells))
        for old, new in zip(saved["cells"], generated.cells):
            self.assertEqual(old["cell_type"], new.cell_type)
            self.assertEqual("".join(old["source"]), new.source)
            if new.cell_type == "code":
                ast.parse(new.source)

    def test_model_derivatives_on_toy_arrays(self):
        # Load only the two pure definitions, never execute a notebook cell.
        from figures.build_sbf_f090w_graph_notebook import build_notebook
        namespace = {"np": np, "COLOR_CENTER": 0.57, "EXP_SCALE": 0.05}
        for cell in build_notebook().cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name in {"model_basis", "model_derivative"}]
            if definitions:
                exec(compile(ast.Module(body=definitions, type_ignores=[]), "toy-model-definitions", "exec"), namespace)
        x = np.array([0.51, 0.57, 0.64])
        step = 1e-6  # Finite-difference step, much smaller than the toy color span.
        for model in ("constant", "linear", "quadratic", "cubic", "logarithmic", "log_quadratic", "exponential"):
            with self.subTest(model=model):
                basis = namespace["model_basis"]
                coefficients = np.arange(1, basis(model, x).shape[1] + 1, dtype=float)
                numerical = (basis(model, x + step) @ coefficients - basis(model, x - step) @ coefficients) / (2 * step)
                analytic = namespace["model_derivative"](model, x, coefficients)
                np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-7)

    def test_color_fit_rejects_missing_errors_instead_of_silent_fallback(self):
        from figures.build_sbf_f090w_graph_notebook import build_notebook
        namespace = {"np": np, "COLOR_CENTER": 0.57}
        for cell in build_notebook().cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "fit_model"]
            if definitions:
                exec(compile(ast.Module(body=definitions, type_ignores=[]), "input-validation", "exec"), namespace)
        frame = pd.DataFrame({
            "color_F090W_F150W": [0.53, 0.59], "Mbar_F090W": [-2.0, -1.9],
            "sigma_Mbar_F090W": [0.03, 0.03],
            "sigma_color_adopted_mag": [0.01, np.nan], "cov_color_Mbar": [0.0, 0.0],
        })
        with self.assertRaisesRegex(ValueError, "пропуски"):
            namespace["fit_model"](frame)


    def test_f150w_fit_rejects_invalid_values_and_keeps_annular_aliases(self):
        from scipy.optimize import minimize
        from sbf.sbf_calibration_covariance import shared_anchor_covariance, validate_color_covariance
        notebook = json.loads((CODE_DIR / "sbf-2-graph.ipynb").read_text())
        namespace = {"np": np, "minimize": minimize, "USE_COLOR_ERRORS_IN_FIT": True,
                     "COLOR_MODELS": {"constant", "linear", "quadratic", "broken"},
                     "shared_anchor_covariance": shared_anchor_covariance,
                     "validate_color_covariance": validate_color_covariance}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            definitions = [node for node in ast.parse("".join(cell["source"])).body
                           if isinstance(node, ast.FunctionDef) and node.name == "fit_relation"]
            if definitions:
                exec(compile(ast.Module(body=definitions, type_ignores=[]), "toy-f150-fit", "exec"), namespace)
        colors = np.linspace(0.50, 0.65, 14)
        frame = pd.DataFrame({
            "color_F090W_F150W": colors,
            "Mbar_F150W": -3.2 + 0.3 * (colors - 0.57) + 0.04 * np.sin(np.arange(14)),
            "sigma_Mbar_internal": 0.02, "sigma_color_stat_mag": 0.01,
            "cov_color_Mbar_extinction": 0.0, "environment": "Virgo",
        })
        fit = namespace["fit_relation"]
        reference = fit(frame, model="constant")
        aliases = frame.rename(columns={"sigma_Mbar_internal": "sigma_Mbar_total",
                                        "sigma_color_stat_mag": "sigma_color_total"})
        self.assertEqual(fit(aliases, model="constant")["intercept"], reference["intercept"])
        for column in frame.select_dtypes(include="number"):
            with self.subTest(column=column):
                invalid = frame.copy()
                invalid.loc[0, column] = np.nan
                with self.assertRaisesRegex(ValueError, "finite"):
                    fit(invalid)
        for column, bad in [("sigma_Mbar_internal", 0.0), ("sigma_color_stat_mag", -0.1)]:
            with self.subTest(column=column):
                invalid = frame.copy()
                invalid.loc[0, column] = bad
                with self.assertRaises(ValueError):
                    fit(invalid)

    def test_f150w_common_systematics_cell_is_idempotent(self):
        notebook = json.loads((CODE_DIR / "sbf-2-graph.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code"
                      and "psf_stamp_common_mag =" in "".join(cell["source"]))
        assignments = [node for node in ast.parse(source).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "common_systematics"
                               for target in node.targets)]
        namespace = {"np": np, "pd": pd, "TRGB_COMMON_ZEROPOINT_MAG": 0.047,
                     "NIRCAM_COMMON_ZEROPOINT_MAG": 0.011, "psf_stamp_common_mag": 0.017,
                     "common_systematics": pd.DataFrame({"component": ["TRGB", "NIRCam"]})}
        compiled = compile(ast.Module(body=assignments, type_ignores=[]), "toy-systematics-table", "exec")
        exec(compiled, namespace)
        once = namespace["common_systematics"].copy()
        exec(compiled, namespace)
        pd.testing.assert_frame_equal(namespace["common_systematics"], once)


if __name__ == "__main__":
    unittest.main()
