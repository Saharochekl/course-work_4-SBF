#!/usr/bin/env python3
"""Build the compact tables and adopted figures used by the GO-3055 article.

The SBF analysis itself is performed in sbf-2-graph.ipynb.  This script only
joins its machine-readable products to published Paper III quantities and
records software/provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "runs" / "sbf2_go3055" / "analysis" / "tables"
FIGURE_DIR = ROOT / "runs" / "sbf2_go3055" / "analysis" / "figures"
F090_TABLE_DIR = ROOT / "runs" / "sbf_f090w_go3055" / "analysis" / "tables"
RUN_DIR = ROOT / "runs" / "sbf2_go3055"
NORMALIZED_RUN_DIR = ROOT / "runs" / "sbf2_normalized_winsor"
ADOPTED_SBF_VERSION = "sbf2-normalized-winsor-v3"
ADOPTED_SBF_BRANCH = "normalized_full_3p5"
SCIENCE_FREEZE_GIT_HEAD = "d89e09482978a4df01324e3fd532ad2e3c03c924"
TRGB_IV_COMMON_SIGMA_MAG = 0.047
JENSEN_III_COMMON_SIGMA_MAG = 0.063
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distance_cell(distance: float, uncertainty: float) -> str:
    if not np.isfinite(distance):
        return r"\ldots"
    if not np.isfinite(uncertainty):
        return f"${distance:.2f}$"
    return f"${distance:.2f}\\pm{uncertainty:.2f}$"


def distance_from_modulus(mu: pd.Series | np.ndarray | float):
    return 10 ** ((mu - 25) / 5)


def distance_uncertainty(distance, sigma_mu):
    return np.log(10) / 5 * distance * sigma_mu


def common_part(total, internal):
    """Recover a common quadrature term without numerical round-off failures."""
    return np.sqrt(np.maximum(np.asarray(total) ** 2 - np.asarray(internal) ** 2, 0.0))


def environment_label(value: str) -> str:
    """Printed label: Virgo is a broad sky-region class, not strict membership."""
    return "Virgo region" if value == "Virgo" else value


# Paper III, Table 1 and Eq. (2).  These are HST/WFC3-IR F110W values,
# deliberately kept separate from the five legacy F160W reconstructions.
paper3 = pd.DataFrame(
    [
        ("NGC 1380", 28.60, 0.022, 1.283, 0.012),
        ("NGC 1399", 28.84, 0.023, 1.363, 0.012),
        ("NGC 1404", 28.73, 0.024, 1.325, 0.012),
        ("NGC 4472", 28.43, 0.021, 1.383, 0.012),
        ("NGC 4552", 28.31, 0.022, 1.349, 0.013),
        ("NGC 4636", 28.36, 0.021, 1.332, 0.013),
        ("NGC 4649", 28.52, 0.021, 1.426, 0.013),
        ("NGC 4697", 27.52, 0.021, 1.271, 0.013),
    ],
    columns=["galaxy", "mbar110_0", "sigma_mbar110", "g_z_ps_0", "sigma_g_z_ps_0"],
)
paper3["Mbar110"] = 1.86 * (paper3["g_z_ps_0"] - 1.30) - 2.760
paper3["sigma_color_term_mag"] = 1.86 * paper3["sigma_g_z_ps_0"]
paper3["sigma_slope_term_mag"] = 0.16 * np.abs(paper3["g_z_ps_0"] - 1.30)
paper3["sigma_population_mag"] = 0.060
paper3["mu_jensen_paper3_f110w"] = paper3["mbar110_0"] - paper3["Mbar110"]
paper3["sigma_mu_jensen_paper3_f110w_internal"] = np.sqrt(
    paper3["sigma_mbar110"] ** 2
    + paper3["sigma_color_term_mag"] ** 2
    + paper3["sigma_slope_term_mag"] ** 2
    + paper3["sigma_population_mag"] ** 2
)
paper3["sigma_mu_jensen_paper3_f110w_common"] = JENSEN_III_COMMON_SIGMA_MAG
paper3["sigma_mu_jensen_paper3_f110w_total"] = np.hypot(
    paper3["sigma_mu_jensen_paper3_f110w_internal"],
    paper3["sigma_mu_jensen_paper3_f110w_common"],
)
paper3["D_jensen_paper3_f110w_mpc"] = distance_from_modulus(
    paper3["mu_jensen_paper3_f110w"]
)
for component in ["internal", "common", "total"]:
    paper3[f"sigma_D_jensen_paper3_f110w_{component}_mpc"] = distance_uncertainty(
        paper3["D_jensen_paper3_f110w_mpc"],
        paper3[f"sigma_mu_jensen_paper3_f110w_{component}"],
    )
# Backward-compatible aliases now deliberately mean the full absolute error.
paper3["sigma_mu_jensen_paper3_f110w"] = paper3[
    "sigma_mu_jensen_paper3_f110w_total"
]
paper3["sigma_D_jensen_paper3_f110w_mpc"] = paper3[
    "sigma_D_jensen_paper3_f110w_total_mpc"
]

# The compact public F150W CSV predates the final error decomposition.  Central
# values come from it, while internal and absolute uncertainties come from the
# explicit leave-one-out prediction table.
distances = pd.read_csv(TABLE_DIR / "go3055_distances_mpc_all_models.csv")
predictions = pd.read_csv(TABLE_DIR / "go3055_distance_predictions_full_errors.csv")
f150 = predictions.loc[predictions["model"] == "constant"].copy()
f150["sigma_mu_sbf_common"] = common_part(
    f150["sigma_mu_sbf_absolute"], f150["sigma_mu_sbf_internal"]
)
f150["D_sbf_f150w_mpc"] = distance_from_modulus(f150["mu_sbf_loo"])
for component, column in [
    ("internal", "sigma_mu_sbf_internal"),
    ("common", "sigma_mu_sbf_common"),
    ("total", "sigma_mu_sbf_absolute"),
]:
    f150[f"sigma_D_sbf_f150w_{component}_mpc"] = distance_uncertainty(
        f150["D_sbf_f150w_mpc"], f150[column]
    )

master = pd.read_csv(TABLE_DIR / "go3055_master_measurements.csv")
trgb = master[["galaxy", "environment", "mu_lit", "sigma_mu_lit"]].copy()
trgb = trgb.rename(
    columns={"mu_lit": "mu_trgb_paper4", "sigma_mu_lit": "sigma_mu_trgb_paper4_internal"}
)
trgb["sigma_mu_trgb_paper4_common"] = TRGB_IV_COMMON_SIGMA_MAG
trgb["sigma_mu_trgb_paper4_total"] = np.hypot(
    trgb["sigma_mu_trgb_paper4_internal"], trgb["sigma_mu_trgb_paper4_common"]
)
trgb["D_trgb_paper4_mpc"] = distance_from_modulus(trgb["mu_trgb_paper4"])
for component in ["internal", "common", "total"]:
    trgb[f"sigma_D_trgb_paper4_{component}_mpc"] = distance_uncertainty(
        trgb["D_trgb_paper4_mpc"], trgb[f"sigma_mu_trgb_paper4_{component}"]
    )

f090_budget = pd.read_csv(F090_TABLE_DIR / "go3055_f090w_distance_error_budget.csv")
f090 = f090_budget.loc[f090_budget["model"] == "linear"].copy()
f090 = f090.rename(
    columns={
        "mu_sbf": "mu_sbf_f090w",
        "sigma_mu_internal": "sigma_mu_sbf_f090w_internal",
        "sigma_mu_total": "sigma_mu_sbf_f090w_total",
        "distance_sbf_mpc": "D_sbf_f090w_mpc",
        "sigma_distance_sbf_mpc": "sigma_D_sbf_f090w_total_mpc",
        "sigma_common_trgb_mag": "sigma_mu_sbf_f090w_common",
    }
)
for component in ["internal", "common"]:
    f090[f"sigma_D_sbf_f090w_{component}_mpc"] = distance_uncertainty(
        f090["D_sbf_f090w_mpc"], f090[f"sigma_mu_sbf_f090w_{component}"]
    )

jensen2015 = pd.read_csv(
    ROOT / "code" / "sbf2_batch_outputs" / "jensen2015_f160w_comparison.csv"
)[
    [
        "galaxy",
        "mu_f160w_jensen2015_reconstructed",
        "sigma_mu_f160w_jensen2015_reconstructed",
    ]
].rename(
    columns={
        "mu_f160w_jensen2015_reconstructed": "mu_jensen2015_f160w",
        "sigma_mu_f160w_jensen2015_reconstructed": "sigma_mu_jensen2015_f160w_total",
    }
)
jensen2015["D_jensen2015_f160w_mpc"] = distance_from_modulus(
    jensen2015["mu_jensen2015_f160w"]
)
jensen2015["sigma_D_jensen2015_f160w_total_mpc"] = distance_uncertainty(
    jensen2015["D_jensen2015_f160w_mpc"],
    jensen2015["sigma_mu_jensen2015_f160w_total"],
)

normalized_results = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((NORMALIZED_RUN_DIR / "batch" / "results").glob(
        "NGC_*_result.json"
    ))
]
if (
    len(normalized_results) != 14
    or any(row.get("status") != "ok" for row in normalized_results)
    or any(row.get("version") != ADOPTED_SBF_VERSION for row in normalized_results)
    or any(row.get("candidate_branch") != ADOPTED_SBF_BRANCH
           for row in normalized_results)
):
    raise RuntimeError("The 14-target normalized_full_3p5 result contract is incomplete")
comparison = trgb.merge(
    paper3[
        [
            "galaxy",
            "D_jensen_paper3_f110w_mpc",
            "sigma_D_jensen_paper3_f110w_internal_mpc",
            "sigma_D_jensen_paper3_f110w_common_mpc",
            "sigma_D_jensen_paper3_f110w_total_mpc",
            "mu_jensen_paper3_f110w",
            "sigma_mu_jensen_paper3_f110w_internal",
            "sigma_mu_jensen_paper3_f110w_common",
            "sigma_mu_jensen_paper3_f110w_total",
        ]
    ],
    on="galaxy",
    how="left",
    validate="one_to_one",
).merge(
    jensen2015, on="galaxy", how="left", validate="one_to_one"
).merge(
    f150[
        [
            "galaxy", "mu_sbf_loo", "sigma_mu_sbf_internal",
            "sigma_mu_sbf_common", "sigma_mu_sbf_absolute", "D_sbf_f150w_mpc",
            "sigma_D_sbf_f150w_internal_mpc", "sigma_D_sbf_f150w_common_mpc",
            "sigma_D_sbf_f150w_total_mpc",
        ]
    ],
    on="galaxy", how="left", validate="one_to_one",
).merge(
    f090[
        [
            "galaxy", "mu_sbf_f090w", "sigma_mu_sbf_f090w_internal",
            "sigma_mu_sbf_f090w_common", "sigma_mu_sbf_f090w_total",
            "D_sbf_f090w_mpc", "sigma_D_sbf_f090w_internal_mpc",
            "sigma_D_sbf_f090w_common_mpc", "sigma_D_sbf_f090w_total_mpc",
        ]
    ],
    on="galaxy", how="left", validate="one_to_one",
)

# A long-form machine-readable table preserves the uncertainty level for every
# method.  Missing Jensen (2015) internal/common components are intentional:
# the supplied reconstruction reports only its final combined uncertainty.
method_specs = [
    (
        "trgb_paper4", "TRGB Paper IV", "TRGB",
        "mu_trgb_paper4", "sigma_mu_trgb_paper4_internal",
        "sigma_mu_trgb_paper4_common", "sigma_mu_trgb_paper4_total",
        "D_trgb_paper4_mpc", "sigma_D_trgb_paper4_internal_mpc",
        "sigma_D_trgb_paper4_common_mpc", "sigma_D_trgb_paper4_total_mpc",
        "Individual Paper IV uncertainty plus the shared 0.047-mag TRGB scale",
    ),
    (
        "jensen_paper3_f110w", "Jensen Paper III", "F110W",
        "mu_jensen_paper3_f110w", "sigma_mu_jensen_paper3_f110w_internal",
        "sigma_mu_jensen_paper3_f110w_common", "sigma_mu_jensen_paper3_f110w_total",
        "D_jensen_paper3_f110w_mpc", "sigma_D_jensen_paper3_f110w_internal_mpc",
        "sigma_D_jensen_paper3_f110w_common_mpc", "sigma_D_jensen_paper3_f110w_total_mpc",
        "Measurement, color, slope and 0.060-mag population scatter; plus the 0.063-mag Paper III common scale, which already includes the 0.016-mag zero-point term",
    ),
    (
        "jensen2015_f160w", "Jensen et al. (2015)", "F160W",
        "mu_jensen2015_f160w", None, None, "sigma_mu_jensen2015_f160w_total",
        "D_jensen2015_f160w_mpc", None, None, "sigma_D_jensen2015_f160w_total_mpc",
        "Final combined uncertainty supplied by the existing five-object reconstruction; internal/common split unavailable",
    ),
    (
        "this_work_f150w", "This work", "F150W",
        "mu_sbf_loo", "sigma_mu_sbf_internal", "sigma_mu_sbf_common",
        "sigma_mu_sbf_absolute", "D_sbf_f150w_mpc",
        "sigma_D_sbf_f150w_internal_mpc", "sigma_D_sbf_f150w_common_mpc",
        "sigma_D_sbf_f150w_total_mpc",
        "Color-independent leave-one-out prediction; absolute error includes the shared 0.047-mag TRGB scale",
    ),
    (
        "this_work_f090w", "This work", "F090W",
        "mu_sbf_f090w", "sigma_mu_sbf_f090w_internal",
        "sigma_mu_sbf_f090w_common", "sigma_mu_sbf_f090w_total",
        "D_sbf_f090w_mpc", "sigma_D_sbf_f090w_internal_mpc",
        "sigma_D_sbf_f090w_common_mpc", "sigma_D_sbf_f090w_total_mpc",
        "Adopted linear-color leave-one-out prediction; absolute error includes the shared 0.047-mag TRGB scale",
    ),
]

long_rows = []
for row in comparison.itertuples(index=False):
    values = row._asdict()
    for (
        method, label, band, mu_col, mu_internal_col, mu_common_col, mu_total_col,
        distance_col, distance_internal_col, distance_common_col, distance_total_col,
        convention,
    ) in method_specs:
        long_rows.append(
            {
                "galaxy": values["galaxy"],
                "environment": values["environment"],
                "method": method,
                "method_label": label,
                "band": band,
                "mu_mag": values.get(mu_col, np.nan),
                "sigma_mu_internal_mag": values.get(mu_internal_col, np.nan) if mu_internal_col else np.nan,
                "sigma_mu_common_mag": values.get(mu_common_col, np.nan) if mu_common_col else np.nan,
                "sigma_mu_total_mag": values.get(mu_total_col, np.nan),
                "distance_mpc": values.get(distance_col, np.nan),
                "sigma_distance_internal_mpc": values.get(distance_internal_col, np.nan) if distance_internal_col else np.nan,
                "sigma_distance_common_mpc": values.get(distance_common_col, np.nan) if distance_common_col else np.nan,
                "sigma_distance_total_mpc": values.get(distance_total_col, np.nan),
                "uncertainty_convention": convention,
            }
        )
distance_long = pd.DataFrame(long_rows)
distance_long.to_csv(
    TABLE_DIR / "go3055_final_distances_error_components.csv", index=False
)

distance_main = comparison[
    [
        "galaxy", "environment",
        "D_trgb_paper4_mpc", "sigma_D_trgb_paper4_total_mpc",
        "D_jensen_paper3_f110w_mpc", "sigma_D_jensen_paper3_f110w_total_mpc",
        "D_jensen2015_f160w_mpc", "sigma_D_jensen2015_f160w_total_mpc",
        "D_sbf_f150w_mpc", "sigma_D_sbf_f150w_total_mpc",
        "D_sbf_f090w_mpc", "sigma_D_sbf_f090w_total_mpc",
    ]
].copy()
for filename in [
    "go3055_final_distance_comparison.csv",
    "go3055_distance_comparison_all_methods.csv",
]:
    distance_main.to_csv(TABLE_DIR / filename, index=False)

distance_tex = [
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"Galaxy & TRGB IV & Jensen III & Jensen 2015 & This work & This work \\",
    r" & & F110W & F160W & F150W & F090W \\",
    r"\midrule",
]
for row in comparison.itertuples(index=False):
    distance_tex.append(
        " & ".join(
            [
                row.galaxy.replace(" ", "~"),
                distance_cell(row.D_trgb_paper4_mpc, row.sigma_D_trgb_paper4_total_mpc),
                distance_cell(
                    row.D_jensen_paper3_f110w_mpc,
                    row.sigma_D_jensen_paper3_f110w_total_mpc,
                ),
                distance_cell(
                    row.D_jensen2015_f160w_mpc,
                    row.sigma_D_jensen2015_f160w_total_mpc,
                ),
                distance_cell(row.D_sbf_f150w_mpc, row.sigma_D_sbf_f150w_total_mpc),
                distance_cell(row.D_sbf_f090w_mpc, row.sigma_D_sbf_f090w_total_mpc),
            ]
        )
        + r" \\"
    )
distance_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
for filename in [
    "go3055_final_distance_comparison.tex",
    "go3055_distance_comparison_all_methods.tex",
]:
    (TABLE_DIR / filename).write_text("\n".join(distance_tex), encoding="utf-8")

convention_tex = [
    r"\begin{tabular}{lp{7.4cm}p{3.2cm}}",
    r"\toprule",
    r"Method & Included terms & Common term \\",
    r"\midrule",
    r"TRGB Paper IV & Published individual uncertainty & Add 0.047 mag \\",
    r"Jensen III F110W & $\overline m$, color, slope, and 0.060-mag population scatter & Add 0.063 mag \\",
    r"Jensen 2015 F160W & Published final combined uncertainty & Already included \\",
    r"This work F150W & Measurement, intrinsic scatter, and finite LOO calibration & Add 0.047 mag \\",
    r"This work F090W & Measurement, color, intrinsic scatter, and finite LOO calibration & Add 0.047 mag \\",
    r"\bottomrule",
    r"\end{tabular}",
    "",
]
(TABLE_DIR / "go3055_distance_uncertainty_conventions.tex").write_text(
    "\n".join(convention_tex), encoding="utf-8"
)

uncertainty_conventions = pd.DataFrame(
    [
        {
            "method": method,
            "method_label": label,
            "band": band,
            "uncertainty_convention": convention,
        }
        for method, label, band, *_, convention in method_specs
    ]
)
uncertainty_conventions.to_csv(
    TABLE_DIR / "go3055_distance_uncertainty_conventions.csv", index=False
)

jensen_paper3_errors = paper3[
    [
        "galaxy", "mu_jensen_paper3_f110w",
        "sigma_mu_jensen_paper3_f110w_internal",
        "sigma_mu_jensen_paper3_f110w_common",
        "sigma_mu_jensen_paper3_f110w_total",
        "D_jensen_paper3_f110w_mpc",
        "sigma_D_jensen_paper3_f110w_internal_mpc",
        "sigma_D_jensen_paper3_f110w_common_mpc",
        "sigma_D_jensen_paper3_f110w_total_mpc",
    ]
].copy()
jensen_paper3_errors.to_csv(
    TABLE_DIR / "go3055_jensen_paper3_f110w_final_errors.csv", index=False
)
jensen_tex = [
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Galaxy & $\mu_{110}$ & $\sigma_{\rm int}$ & $\sigma_{\rm common}$ & $\sigma_{\rm total}$ \\",
    r" & (mag) & (mag) & (mag) & (mag) \\",
    r"\midrule",
]
for row in jensen_paper3_errors.itertuples(index=False):
    jensen_tex.append(
        f"{row.galaxy.replace(' ', '~')} & {row.mu_jensen_paper3_f110w:.3f} & "
        f"{row.sigma_mu_jensen_paper3_f110w_internal:.3f} & "
        f"{row.sigma_mu_jensen_paper3_f110w_common:.3f} & "
        f"{row.sigma_mu_jensen_paper3_f110w_total:.3f} " + r"\\"
    )
jensen_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_jensen_paper3_f110w_final_errors.tex").write_text(
    "\n".join(jensen_tex), encoding="utf-8"
)

# Compact F090W calibration-point table.  These absolute magnitudes use the
# individual Paper IV TRGB anchors; the common 0.047-mag scale is kept separate
# from the listed per-object calibration-point uncertainty.
f090_master = pd.read_csv(F090_TABLE_DIR / "go3055_f090w_master.csv")
f090_measurements = comparison[["galaxy", "environment"]].merge(
    f090_master[
        [
            "galaxy", "color_F090W_F150W", "sigma_color_adopted_mag",
            "mbar_F090W_0", "sigma_mbar_internal", "Mbar_F090W",
            "sigma_Mbar_F090W", "annulus_difference_mag",
            "paper_iv_high_quality",
        ]
    ],
    on="galaxy", how="left", validate="one_to_one",
)
f090_measurements.to_csv(
    TABLE_DIR / "go3055_article_measurements_f090w.csv", index=False
)
f090_measurement_tex = [
    r"\begin{tabular}{llrrrrrrc}",
    r"\toprule",
    r"Galaxy & Env. & $C_{090-150,0}$ & $\overline m_{090,0}$ & $\sigma_{\overline m}$ & $\overline M_{090,0}$ & $\sigma_{\overline M}$ & $|\Delta\overline m|_{\rm ann}$ & HQ IV \\",
    r" & & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) & \\",
    r"\midrule",
]
for row in f090_measurements.itertuples(index=False):
    high_quality = "+" if bool(row.paper_iv_high_quality) else "-"
    f090_measurement_tex.append(
        f"{row.galaxy.replace(' ', '~')} & {environment_label(row.environment)} & "
        f"{row.color_F090W_F150W:.3f} & {row.mbar_F090W_0:.3f} & "
        f"{row.sigma_mbar_internal:.3f} & {row.Mbar_F090W:.3f} & "
        f"{row.sigma_Mbar_F090W:.3f} & {abs(row.annulus_difference_mag):.3f} & "
        f"${high_quality}$ " + r"\\"
    )
f090_measurement_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_article_measurements_f090w.tex").write_text(
    "\n".join(f090_measurement_tex), encoding="utf-8"
)

f090_models = pd.read_csv(F090_TABLE_DIR / "go3055_f090w_color_model_comparison.csv")
f090_precision = pd.read_csv(F090_TABLE_DIR / "go3055_f090w_constant_vs_color_precision.csv")
f090_models = (
    f090_models.loc[f090_models["model"].isin(["constant", "linear"])]
    .merge(
        f090_precision[["model", "loo_rms_mpc", "median_reported_sigma_mpc"]],
        on="model", how="left", validate="one_to_one",
    )
    .set_index("model")
    .loc[["constant", "linear"]]
    .reset_index()
)
f090_models.to_csv(
    TABLE_DIR / "go3055_f090w_constant_vs_linear_model_comparison.csv", index=False
)
f090_model_tex = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Model & $a$ & $b$ & $\sigma_{\rm int}$ & AICc & LOO RMS & Median $\sigma_D$ \\",
    r" & (mag) & & (mag) & & (mag) & (Mpc) \\",
    r"\midrule",
]
for row in f090_models.itertuples(index=False):
    f090_model_tex.append(
        f"{row.description} & ${row.intercept_mag:.3f}\\pm{row.intercept_sigma_bootstrap_mag:.3f}$ & "
        f"${row.slope_at_center:.2f}\\pm{row.slope_sigma_bootstrap:.2f}$ & "
        f"{row.sigma_int_mag:.3f} & {row.aicc:.2f} & {row.loo_rms_mag:.3f} & "
        f"{row.median_reported_sigma_mpc:.2f} " + r"\\"
    )
f090_model_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_f090w_constant_vs_linear_model_comparison.tex").write_text(
    "\n".join(f090_model_tex), encoding="utf-8"
)


error_budget = pd.read_csv(TABLE_DIR / "go3055_error_budget.csv")
error_definitions = [
    ("Power-spectrum fit and k-window", "sigma_measurement_mag"),
    ("STPSF realization scatter", "sigma_psf_diagnostic_mag"),
    (r"Residual sources $P_r$", "sigma_Pr_mag"),
    ("Sky subtraction", "sigma_sky_mag"),
    ("Paper IV TRGB (without reddening)", "sigma_mu_without_reddening_mag"),
    ("Shared foreground reddening", "sigma_reddening_color_mag"),
    ("Total individual calibration point", "sigma_Mbar_internal"),
]
error_summary = pd.DataFrame(
    [
        {
            "component": label,
            "column": column,
            "median_sigma_mag": error_budget[column].median(),
            "min_sigma_mag": error_budget[column].min(),
            "max_sigma_mag": error_budget[column].max(),
        }
        for label, column in error_definitions
    ]
)
error_summary.to_csv(TABLE_DIR / "go3055_individual_error_summary.csv", index=False)

error_tex = [r"\begin{tabular}{lrr}", r"\toprule", r"Component & Median & Range \\", r"\midrule"]
for row in error_summary.itertuples(index=False):
    error_tex.append(
        f"{row.component} & {row.median_sigma_mag:.4f} & "
        f"{row.min_sigma_mag:.4f}--{row.max_sigma_mag:.4f} \\\\"
    )
error_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_individual_error_summary.tex").write_text(
    "\n".join(error_tex), encoding="utf-8"
)

common = pd.read_csv(TABLE_DIR / "go3055_common_systematics.csv")
common_tex = [r"\begin{tabular}{lrp{4.5cm}}", r"\toprule", r"Component & $\sigma$ (mag) & Use \\", r"\midrule"]
common_labels = {
    "TRGB / NGC 4258 common zero point": "TRGB/NGC 4258 zero point",
    "JWST/NIRCam F150W absolute flux scale": "NIRCam F150W flux scale",
    "STPSF 129-pixel finite-stamp term": "STPSF finite-stamp term",
    "Combined absolute-Mbar zero point": r"Combined absolute $\overline M_{150}$ scale",
    "Same-filter SBF distance-scale zero point": "Same-filter SBF distance scale",
}
for row in common.itertuples(index=False):
    use = (
        "absolute calibration"
        if row.applies_to_absolute_Mbar
        else "F150W SBF distances"
    )
    common_tex.append(
        f"{common_labels[row.component]} & {row.sigma_mag:.4f} & {use} \\\\"
    )
common_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_common_systematics.tex").write_text(
    "\n".join(common_tex), encoding="utf-8"
)


predictions = pd.read_csv(TABLE_DIR / "go3055_distance_predictions_full_errors.csv")
predictions = predictions.merge(
    error_budget[["galaxy", "sigma_mbar_internal"]],
    on="galaxy",
    how="left",
    validate="many_to_one",
)

constant_predictions = predictions.loc[predictions["model"] == "constant"].copy()
master = pd.read_csv(TABLE_DIR / "go3055_master_measurements.csv")
constant_measurements = master.merge(
    constant_predictions[
        [
            "galaxy",
            "mu_sbf_loo",
            "delta_mu_sbf_minus_trgb",
            "sigma_mu_sbf_internal",
            "sigma_mu_sbf_absolute",
            "sigma_validation_residual_internal",
            "sigma_intrinsic_mag",
            "sigma_calibration_prediction_mag",
        ]
    ],
    on="galaxy",
    how="left",
    validate="one_to_one",
)

measurement_tex = [
    r"\begin{tabular}{llrrrrrrrc}",
    r"\toprule",
    r"Galaxy & Environment & $C_{090-150,0}$ & $\overline m_{150,0}$ &",
    r"$\overline M_{150,0}$ & $\sigma_{\overline M}$ &",
    r"$|\Delta\overline m|_{\rm ann}$ & $\mu_{\rm TRGB}$ &",
    r"$\Delta\mu_{\rm LOO,const}$ & HQ IV \\",
    r" & & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) & \\",
    r"\midrule",
]
for row in constant_measurements.itertuples(index=False):
    high_quality = "+" if str(row.paper_iv_high_quality).lower() == "true" else "-"
    measurement_tex.append(
        f"{row.galaxy} & {environment_label(row.environment)} & "
        f"{row.color_F090W_F150W:.3f} & "
        f"{row.mbar_F150W:.3f} & {row.Mbar_F150W:.3f} & "
        f"{row.sigma_Mbar_internal:.3f} & {row.annulus_delta_mag:.3f} & "
        f"{row.mu_lit:.3f} & {row.delta_mu_sbf_minus_trgb:+.3f} & "
        f"${high_quality}$ " + r"\\"
    )
measurement_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_article_measurements_constant.tex").write_text(
    "\n".join(measurement_tex), encoding="utf-8"
)

environment_style = {
    "Fornax": ("#dc2626", "s"),
    "Virgo": ("#2563eb", "o"),
    "Other": ("#4b5563", "^"),
}

fig, ax = plt.subplots(figsize=(7.2, 6.8))
limits = [
    min(constant_predictions["mu_trgb"].min(), constant_predictions["mu_sbf_loo"].min()) - 0.08,
    max(constant_predictions["mu_trgb"].max(), constant_predictions["mu_sbf_loo"].max()) + 0.08,
]
ax.plot(limits, limits, "k--", lw=1.5, label="1:1")
for environment, group in constant_predictions.groupby("environment", sort=False):
    color, marker = environment_style[environment]
    ax.errorbar(
        group["mu_trgb"], group["mu_sbf_loo"],
        xerr=group["sigma_mu_trgb"], yerr=group["sigma_mu_sbf_internal"],
        fmt=marker, color=color, mec="black", mew=0.6, ms=7,
        capsize=3, linestyle="none", label=environment_label(environment),
    )
for row in constant_predictions.itertuples(index=False):
    ax.annotate(row.galaxy.replace("NGC ", ""), (row.mu_trgb, row.mu_sbf_loo),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set(
    xlim=limits, ylim=limits,
    xlabel=r"$\mu_{\rm TRGB}$ (mag)",
    ylabel=r"$\mu_{\rm SBF,const}^{\rm LOO}$ (mag)",
    title="Color-independent leave-one-out SBF distance recovery",
)
ax.grid(alpha=0.22)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(FIGURE_DIR / f"go3055_sbf_vs_trgb_leave_one_out_constant.{suffix}", dpi=220)
plt.close(fig)

constant_predictions["realized_distance_error_percent"] = 100 * np.abs(
    10 ** (constant_predictions["delta_mu_sbf_minus_trgb"] / 5) - 1
)
constant_predictions["internal_distance_sigma_percent"] = 100 * (
    10 ** (constant_predictions["sigma_mu_sbf_internal"] / 5) - 1
)
constant_predictions["absolute_distance_sigma_percent"] = 100 * (
    10 ** (constant_predictions["sigma_mu_sbf_absolute"] / 5) - 1
)
x = np.arange(len(constant_predictions))
fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.8), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 1.25]})
axes[0].errorbar(
    x, constant_predictions["delta_mu_sbf_minus_trgb"],
    yerr=constant_predictions["sigma_validation_residual_internal"],
    fmt="o", color="#2563eb", ecolor="#9ca3af", capsize=3,
)
axes[0].axhline(0, color="black", lw=1)
axes[0].set(ylabel=r"$\mu_{\rm SBF,const}^{\rm LOO}-\mu_{\rm TRGB}$ (mag)",
            title="Adopted color-independent LOO residuals")
width = 0.25
axes[1].bar(x - width, constant_predictions["realized_distance_error_percent"],
            width, label="Realized absolute residual", color="#2563eb")
axes[1].bar(x, constant_predictions["internal_distance_sigma_percent"],
            width, label="Predicted internal 1-sigma", color="#f59e0b")
axes[1].bar(x + width, constant_predictions["absolute_distance_sigma_percent"],
            width, label="Including common TRGB scale", color="#16a34a")
axes[1].set(ylabel="Distance error (%)", xlabel="Galaxy")
axes[1].set_xticks(x, constant_predictions["galaxy"].str.replace("NGC ", "", regex=False),
                   rotation=45, ha="right")
for ax in axes:
    ax.grid(axis="y", alpha=0.22)
axes[1].legend(frameon=False, ncol=3, fontsize=8)
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(FIGURE_DIR / f"go3055_distance_accuracy_by_galaxy_constant.{suffix}", dpi=220)
plt.close(fig)

error_plot = constant_measurements.copy()
x = np.arange(len(error_plot))
fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True)
calibration_components = [
    ("sigma_measurement_mag", "Power spectrum", "#2563eb"),
    ("sigma_psf_diagnostic_mag", "PSF realization", "#f97316"),
    ("sigma_sky_mag", "Background", "#16a34a"),
    ("sigma_Pr_mag", r"Unresolved sources $P_r$", "#dc2626"),
    ("sigma_reddening_color_mag", "Foreground extinction", "#8b5cf6"),
    ("sigma_mu_without_reddening_mag", "Individual TRGB anchor", "#8c564b"),
]
width = 0.12
offsets = (np.arange(len(calibration_components)) - 2.5) * width
for offset, (column, label, color) in zip(offsets, calibration_components):
    axes[0].bar(x + offset, error_plot[column], width, label=label, color=color)
axes[0].plot(x, error_plot["sigma_Mbar_internal"], "kD", ms=4,
             label="Total individual calibration point")
axes[0].set(ylabel="Uncertainty (mag)", title="Calibration-point error budget")

distance_components = [
    ("sigma_mbar_internal", "Apparent SBF measurement", "#2563eb"),
    ("sigma_intrinsic_mag", "Intrinsic scatter", "#f97316"),
    ("sigma_calibration_prediction_mag", "Finite calibration", "#16a34a"),
]
width = 0.22
for offset, (column, label, color) in zip([-width, 0, width], distance_components):
    axes[1].bar(x + offset, error_plot[column], width, label=label, color=color)
axes[1].plot(x, error_plot["sigma_mu_sbf_internal"], "ko", ms=4,
             label="Total internal distance error")
axes[1].plot(x, error_plot["sigma_mu_sbf_absolute"], "k_", ms=12,
             label="Including common TRGB scale")
axes[1].set(ylabel="Distance-modulus uncertainty (mag)", xlabel="Galaxy",
            title="Adopted constant-calibration LOO distance budget")
axes[1].set_xticks(x, error_plot["galaxy"].str.replace("NGC ", "", regex=False),
                   rotation=45, ha="right")
for ax in axes:
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, fontsize=8)
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(FIGURE_DIR / f"go3055_error_budget_by_galaxy_constant.{suffix}", dpi=220)
plt.close(fig)

prediction_definitions = [
    ("Apparent SBF measurement", "sigma_mbar_internal"),
    ("Color propagated through calibration", "sigma_color_prediction_mag"),
    ("Foreground-extinction propagation", "sigma_extinction_prediction_mag"),
    ("Fitted intrinsic population scatter", "sigma_intrinsic_mag"),
    ("Finite 13-galaxy LOO calibration", "sigma_calibration_prediction_mag"),
    ("Total internal SBF distance", "sigma_mu_sbf_internal"),
    ("Total including common TRGB scale", "sigma_mu_sbf_absolute"),
    ("SBF--TRGB validation residual", "sigma_validation_residual_internal"),
]
prediction_summary_rows = []
for model in ["constant", "linear"]:
    model_rows = predictions.loc[predictions["model"] == model]
    for label, column in prediction_definitions:
        values = model_rows[column]
        prediction_summary_rows.append(
            {
                "model": model,
                "component": label,
                "column": column,
                "median_sigma_mag": values.median(),
                "min_sigma_mag": values.min(),
                "max_sigma_mag": values.max(),
            }
        )
prediction_summary = pd.DataFrame(prediction_summary_rows)
prediction_summary.to_csv(
    TABLE_DIR / "go3055_predictive_error_summary.csv", index=False
)

adopted_prediction_definitions = [
    item for item in prediction_definitions
    if item[1] != "sigma_color_prediction_mag"
]
prediction_tex = [
    r"\begin{tabular}{lc}",
    r"\toprule",
    r"Component & Adopted constant \\",
    r"\midrule",
]
for label, _ in adopted_prediction_definitions:
    row = prediction_summary.loc[
        (prediction_summary["model"] == "constant")
        & (prediction_summary["component"] == label)
    ].iloc[0]
    cell = (
        f"{row.median_sigma_mag:.4f} "
        f"({row.min_sigma_mag:.4f}--{row.max_sigma_mag:.4f})"
    )
    prediction_tex.append(f"{label} & {cell} " + r"\\")
prediction_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_predictive_error_summary.tex").write_text(
    "\n".join(prediction_tex), encoding="utf-8"
)


provenance_path = RUN_DIR / "campaign" / "run_provenance.json"
provenance = json.loads(provenance_path.read_text())
results = pd.read_csv(RUN_DIR / "batch" / "sbf2_batch_results.csv")

input_provenance = []
for role in ["signal_path", "color_path"]:
    for input_path in results[role].dropna():
        with fits.open(Path(input_path), memmap=True) as hdul:
            header = hdul[0].header
        input_provenance.append(
            {
                "role": role,
                "pipeline": str(header.get("CAL_VER", "unknown")),
                "crds": str(header.get("CRDS_VER", "unknown")),
                "context": str(header.get("CRDS_CTX", "unknown")),
            }
        )
input_provenance = pd.DataFrame(input_provenance)
input_combinations = (
    input_provenance.groupby(["pipeline", "crds", "context"], dropna=False)
    .size()
    .rename("file_count")
    .reset_index()
    .sort_values(["pipeline", "crds", "context"])
)
input_combinations.to_csv(
    TABLE_DIR / "go3055_input_calibration_provenance.csv", index=False
)

package_versions = provenance["environment"]["packages"]
software_rows = []
for row in input_combinations.itertuples(index=False):
    software_rows.append(
        (
            f"Input i2d products ({row.file_count}/28)",
            f"pipeline {row.pipeline}; CRDS {row.crds}; {row.context}",
        )
    )
software_rows.append(("Python", provenance["process"]["python_version"].split()[0]))
for package in ["numpy", "astropy", "photutils", "stpsf", "scipy", "matplotlib", "pandas"]:
    software_rows.append((package, str(package_versions.get(package, "unknown"))))
software_rows.extend(
    [
        (
            "Adopted F150W SBF contract",
            f"{ADOPTED_SBF_VERSION}; {ADOPTED_SBF_BRANCH}; kmin=0.04",
        ),
        (
            "Normalized SBF core SHA256",
            sha256(ROOT / "code" / "sbf2_normalized_winsor_core.py"),
        ),
        (
            "Normalized SBF runner SHA256",
            sha256(ROOT / "code" / "run_sbf_2_normalized_winsor.py"),
        ),
        ("Product-generation Git HEAD", provenance["git"]["head"]),
        ("Executed sbf-2.ipynb SHA256", provenance["template"]["sha256"]),
        ("Current sbf-2-graph.ipynb SHA256", sha256(ROOT / "code" / "sbf-2-graph.ipynb")),
        (
            "Current sbf-2-systematics.ipynb SHA256",
            sha256(ROOT / "code" / "sbf-2-systematics.ipynb"),
        ),
        (
            "Current sbf-f090w-graph.ipynb SHA256",
            sha256(ROOT / "code" / "sbf-f090w-graph.ipynb"),
        ),
        (
            "F090W runner SHA256",
            sha256(ROOT / "code" / "run_sbf_f090w.py"),
        ),
        (
            "F090W graph builder SHA256",
            sha256(ROOT / "code" / "build_sbf_f090w_graph_notebook.py"),
        ),
        ("Article-table builder SHA256", sha256(Path(__file__).resolve())),
        ("F150W science-freeze Git commit", SCIENCE_FREEZE_GIT_HEAD),
    ]
)
software = pd.DataFrame(software_rows, columns=["item", "version_or_identifier"])
software.to_csv(TABLE_DIR / "go3055_software_provenance.csv", index=False)

software_tex = [r"\begin{tabular}{lp{5.2cm}}", r"\toprule", r"Item & Version or identifier \\", r"\midrule"]
for row in software.itertuples(index=False):
    value = row.version_or_identifier.replace("_", r"\_")
    if len(value) > 24 and all(char in "0123456789abcdef" for char in value.lower()):
        value = r"\texttt{" + value[:12] + r"\ldots}"
    software_tex.append(f"{row.item} & {value} \\\\")
software_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_software_provenance.tex").write_text(
    "\n".join(software_tex), encoding="utf-8"
)

print(f"Wrote article tables to {TABLE_DIR}")
print(distance_main.to_string(index=False))
