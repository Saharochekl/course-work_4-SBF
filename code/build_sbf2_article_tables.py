#!/usr/bin/env python3
"""Build the compact tables and adopted figures used by the GO-3055 article.

The SBF analysis itself is performed in sbf-2-graph.ipynb.  This script only
joins its machine-readable products to published Paper III quantities and
records software/provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
RUN_DIR = ROOT / "runs" / "sbf2_go3055"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def distance_cell(distance: float, uncertainty: float) -> str:
    if not np.isfinite(distance):
        return r"\ldots"
    return f"${distance:.2f}\\pm{uncertainty:.2f}$"


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
paper3["sigma_Mbar110_formal"] = np.sqrt(
    (1.86 * paper3["sigma_g_z_ps_0"]) ** 2
    + (0.16 * (paper3["g_z_ps_0"] - 1.30)) ** 2
    + 0.016**2
)
paper3["mu_jensen_paper3_f110w"] = paper3["mbar110_0"] - paper3["Mbar110"]
paper3["sigma_mu_jensen_paper3_f110w"] = np.hypot(
    paper3["sigma_mbar110"], paper3["sigma_Mbar110_formal"]
)
paper3["D_jensen_paper3_f110w_mpc"] = 10 ** (
    (paper3["mu_jensen_paper3_f110w"] - 25) / 5
)
paper3["sigma_D_jensen_paper3_f110w_mpc"] = (
    np.log(10) / 5
    * paper3["D_jensen_paper3_f110w_mpc"]
    * paper3["sigma_mu_jensen_paper3_f110w"]
)

distances = pd.read_csv(TABLE_DIR / "go3055_distances_mpc.csv")
comparison = distances.merge(
    paper3[
        [
            "galaxy",
            "D_jensen_paper3_f110w_mpc",
            "sigma_D_jensen_paper3_f110w_mpc",
            "mu_jensen_paper3_f110w",
            "sigma_mu_jensen_paper3_f110w",
        ]
    ],
    on="galaxy",
    how="left",
    validate="one_to_one",
)
comparison.to_csv(TABLE_DIR / "go3055_distance_comparison_all_methods.csv", index=False)

distance_tex = [
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"Galaxy & Env. & TRGB IV & SBF const. & SBF linear & Jensen III & Jensen 2015 \\",
    r" & & & & & F110W & F160W \\",
    r"\midrule",
]
for row in comparison.itertuples(index=False):
    distance_tex.append(
        " & ".join(
            [
                row.galaxy.replace(" ", "~"),
                row.environment,
                distance_cell(row.D_trgb_mpc, row.sigma_D_trgb_mpc),
                distance_cell(row.D_sbf_constant_mpc, row.sigma_D_sbf_constant_mpc),
                distance_cell(row.D_sbf_linear_mpc, row.sigma_D_sbf_linear_mpc),
                distance_cell(
                    row.D_jensen_paper3_f110w_mpc,
                    row.sigma_D_jensen_paper3_f110w_mpc,
                ),
                distance_cell(row.D_jensen_f160w_mpc, row.sigma_D_jensen_f160w_mpc),
            ]
        )
        + r" \\"
    )
distance_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_distance_comparison_all_methods.tex").write_text(
    "\n".join(distance_tex), encoding="utf-8"
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
    "STPSF 129-pixel stamp normalization": "STPSF stamp normalization",
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
        f"{row.galaxy} & {row.environment} & {row.color_F090W_F150W:.3f} & "
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
        capsize=3, linestyle="none", label=environment,
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

prediction_tex = [
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Component & Constant & Linear \\",
    r"\midrule",
]
for label, _ in prediction_definitions:
    cells = []
    for model in ["constant", "linear"]:
        row = prediction_summary.loc[
            (prediction_summary["model"] == model)
            & (prediction_summary["component"] == label)
        ].iloc[0]
        if row.max_sigma_mag < 1.0e-4 and row.max_sigma_mag > 0:
            def scientific(value: float) -> str:
                exponent = int(np.floor(np.log10(abs(value))))
                mantissa = value / 10**exponent
                return rf"${mantissa:.1f}\times10^{{{exponent}}}$"

            cells.append(
                f"{scientific(row.median_sigma_mag)} "
                f"({scientific(row.min_sigma_mag)}--"
                f"{scientific(row.max_sigma_mag)})"
            )
        else:
            cells.append(
                f"{row.median_sigma_mag:.4f} "
                f"({row.min_sigma_mag:.4f}--{row.max_sigma_mag:.4f})"
            )
    prediction_tex.append(f"{label} & {cells[0]} & {cells[1]} " + r"\\")
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
        ("Product-generation Git HEAD", provenance["git"]["head"]),
        ("Executed sbf-2.ipynb SHA256", provenance["template"]["sha256"]),
        ("Current sbf-2-graph.ipynb SHA256", sha256(ROOT / "code" / "sbf-2-graph.ipynb")),
        (
            "Current article TeX SHA256",
            sha256(ROOT / "texts" / "paper_work" / "go3055_jwst_sbf_article_draft.tex"),
        ),
        ("Article-table builder SHA256", sha256(Path(__file__).resolve())),
        ("Current article-worktree Git HEAD", git_value("rev-parse", "HEAD")),
        (
            "Current article worktree clean",
            "yes" if not git_value("status", "--porcelain") else "no",
        ),
    ]
)
software = pd.DataFrame(software_rows, columns=["item", "version_or_identifier"])
software.to_csv(TABLE_DIR / "go3055_software_provenance.csv", index=False)

software_tex = [r"\begin{tabular}{lp{5.2cm}}", r"\toprule", r"Item & Version or identifier \\", r"\midrule"]
for row in software.itertuples(index=False):
    value = row.version_or_identifier.replace("_", r"\_")
    if len(value) > 24 and all(char in "0123456789abcdef" for char in value.lower()):
        value = r"\texttt{" + value[:12] + r"\ldots}"
    software_tex.append(f"{row.item} & {value} \\\\ ")
software_tex.extend([r"\bottomrule", r"\end{tabular}", ""])
(TABLE_DIR / "go3055_software_provenance.tex").write_text(
    "\n".join(software_tex), encoding="utf-8"
)

print(f"Wrote article tables to {TABLE_DIR}")
print(comparison[["galaxy", "D_trgb_mpc", "D_sbf_constant_mpc", "D_sbf_linear_mpc", "D_jensen_paper3_f110w_mpc", "D_jensen_f160w_mpc"]].to_string(index=False))
