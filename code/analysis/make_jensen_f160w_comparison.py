from pathlib import Path
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-sbf")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sbf2_batch_outputs"
FIG_DIR = ROOT / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CALIBRATION_INPUT = OUT_DIR / "coursework_calibration_input.csv"
COMPARISON_CSV = OUT_DIR / "jensen2015_f160w_comparison.csv"
SUMMARY_CSV = OUT_DIR / "jensen2015_f160w_comparison_summary.csv"
FIG_PDF = FIG_DIR / "coursework_jensen_f160w_comparison.pdf"
FIG_PNG = FIG_DIR / "coursework_jensen_f160w_comparison.png"


def weighted_mean(values, sigmas):
    w = 1.0 / np.square(sigmas)
    mean = np.sum(w * values) / np.sum(w)
    err = math.sqrt(1.0 / np.sum(w))
    wrms = math.sqrt(np.sum(w * np.square(values - mean)) / np.sum(w))
    return mean, err, wrms


def main():
    our = pd.read_csv(CALIBRATION_INPUT)

    # Jensen et al. (2015), ApJ 808, 91:
    # Table 3 gives extinction-corrected HST/WFC3 H160 apparent SBF magnitudes.
    # Table 4 gives ACS (g475-z850) colors and the optical-SBF reference moduli.
    # Equation (4) gives the red-galaxy H160 calibration:
    # M160 = (-3.699 +/- 0.028) + (2.13 +/- 0.27) * [(g-z) - 1.4].
    jensen = pd.DataFrame(
        [
            ("NGC 1380", 27.80, 0.05, 1.391, 0.007, 31.632, 0.075),
            ("NGC 1399", 28.03, 0.04, 1.490, 0.005, 31.596, 0.091),
            ("NGC 1404", 27.94, 0.06, 1.471, 0.006, 31.526, 0.072),
            ("NGC 4472", 27.59, 0.05, 1.514, 0.006, 31.116, 0.075),
            ("NGC 4649", 27.63, 0.05, 1.554, 0.006, 31.082, 0.079),
        ],
        columns=[
            "galaxy",
            "m160_ab_jensen2015",
            "sigma_m160_ab_jensen2015",
            "g475_z850_jensen2015",
            "sigma_g475_z850_jensen2015",
            "mu_reference_acs_sbf_jensen2015",
            "sigma_mu_reference_acs_sbf_jensen2015",
        ],
    )

    m0 = -3.699
    sigma_m0 = 0.028
    slope = 2.13
    sigma_slope = 0.27
    pivot = 1.4
    calibration_scatter = 0.114

    color_delta = jensen["g475_z850_jensen2015"] - pivot
    jensen["M160_calib_jensen2015"] = m0 + slope * color_delta
    jensen["sigma_M160_calib_no_scatter"] = np.sqrt(
        sigma_m0**2
        + np.square(color_delta * sigma_slope)
        + np.square(slope * jensen["sigma_g475_z850_jensen2015"])
    )
    jensen["sigma_M160_calib_with_scatter"] = np.sqrt(
        np.square(jensen["sigma_M160_calib_no_scatter"]) + calibration_scatter**2
    )
    jensen["mu_f160w_jensen2015_reconstructed"] = (
        jensen["m160_ab_jensen2015"] - jensen["M160_calib_jensen2015"]
    )
    jensen["sigma_mu_f160w_jensen2015_reconstructed"] = np.sqrt(
        np.square(jensen["sigma_m160_ab_jensen2015"])
        + np.square(jensen["sigma_M160_calib_with_scatter"])
    )

    cols = [
        "galaxy",
        "mbar_150",
        "sigma_mbar_150",
        "color_F090W_F150W",
        "mu_sbf_clean_linear",
        "sigma_mu_sbf_clean_linear",
        "mu_lit",
        "sigma_mu_lit",
    ]
    cmp = our[cols].merge(jensen, on="galaxy", how="inner")
    cmp["delta_mu_f150w_minus_f160w"] = (
        cmp["mu_sbf_clean_linear"] - cmp["mu_f160w_jensen2015_reconstructed"]
    )
    cmp["sigma_delta_mu"] = np.sqrt(
        np.square(cmp["sigma_mu_sbf_clean_linear"])
        + np.square(cmp["sigma_mu_f160w_jensen2015_reconstructed"])
    )
    cmp["distance_ratio_f150w_over_f160w"] = 10 ** (
        cmp["delta_mu_f150w_minus_f160w"] / 5.0
    )

    mean_delta, mean_delta_err, wrms_delta = weighted_mean(
        cmp["delta_mu_f150w_minus_f160w"].to_numpy(),
        cmp["sigma_delta_mu"].to_numpy(),
    )
    distance_ratio = 10 ** (mean_delta / 5.0)
    summary = pd.DataFrame(
        [
            {
                "n_overlap": len(cmp),
                "weighted_mean_delta_mu_f150w_minus_f160w": mean_delta,
                "sigma_weighted_mean_delta_mu": mean_delta_err,
                "weighted_rms_delta_mu": wrms_delta,
                "mean_distance_ratio_f150w_over_f160w": distance_ratio,
                "mean_distance_offset_percent": (distance_ratio - 1.0) * 100.0,
                "jensen2015_h160_calibration_scatter_mag": calibration_scatter,
            }
        ]
    )

    cmp.to_csv(COMPARISON_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [1.05, 1.0]})

    x = cmp["mu_f160w_jensen2015_reconstructed"]
    y = cmp["mu_sbf_clean_linear"]
    xerr = cmp["sigma_mu_f160w_jensen2015_reconstructed"]
    yerr = cmp["sigma_mu_sbf_clean_linear"]
    ax0.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=5.0, capsize=2.0, lw=0.8, color="#2f6f9f")
    lo = min(x.min(), y.min()) - 0.12
    hi = max(x.max(), y.max()) + 0.12
    ax0.plot([lo, hi], [lo, hi], color="black", lw=1.1)
    for _, row in cmp.iterrows():
        ax0.annotate(row["galaxy"].replace("NGC ", ""), (row["mu_f160w_jensen2015_reconstructed"], row["mu_sbf_clean_linear"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel(r"$\mu_{\rm SBF,F160W}$ Jensen et al. 2015 [mag]")
    ax0.set_ylabel(r"$\mu_{\rm SBF,F150W}$ this work [mag]")
    ax0.set_title("Distance moduli")
    ax0.grid(alpha=0.15)

    order = cmp.sort_values("delta_mu_f150w_minus_f160w")
    yy = np.arange(len(order))
    ax1.errorbar(
        order["delta_mu_f150w_minus_f160w"],
        yy,
        xerr=order["sigma_delta_mu"],
        fmt="o",
        ms=5.0,
        capsize=2.0,
        lw=0.8,
        color="#2f6f9f",
    )
    ax1.axvline(0, color="black", lw=1.0)
    ax1.axvline(mean_delta, color="#c95b2b", lw=1.2)
    ax1.axvspan(mean_delta - mean_delta_err, mean_delta + mean_delta_err, color="#c95b2b", alpha=0.15)
    ax1.set_yticks(yy)
    ax1.set_yticklabels(order["galaxy"])
    ax1.set_xlabel(r"$\mu_{150}-\mu_{160,\rm Jensen}$ [mag]")
    ax1.set_title("Residuals")
    ax1.grid(axis="x", alpha=0.15)
    summary_text = (
        rf"$\langle\Delta\mu\rangle={mean_delta:+.3f}\pm{mean_delta_err:.3f}$ mag"
        + "\n"
        + rf"WRMS = {wrms_delta:.3f} mag"
    )
    ax1.text(
        0.03,
        0.97,
        summary_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )

    fig.suptitle("JWST F150W SBF versus HST F160W SBF", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved {COMPARISON_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {FIG_PDF}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
