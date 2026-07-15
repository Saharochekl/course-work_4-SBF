from pathlib import Path
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

CALIBRATION_INPUT = OUT_DIR / "coursework_calibration_input.csv"
FIT_SUMMARY = OUT_DIR / "coursework_fit_summary.csv"
FIG_PDF = FIG_DIR / "coursework_Mbar150_color_calibration.pdf"
FIG_PNG = FIG_DIR / "coursework_Mbar150_color_calibration.png"


STYLE = {
    "clean": {"marker": "o", "color": "#2f6f9f", "mfc": "#2f6f9f", "mec": "#2f6f9f"},
    "flagged": {"marker": "s", "color": "#c95b2b", "mfc": "white", "mec": "#c95b2b"},
    "fit": "#111111",
    "band": "#9a9a9a",
}

LABEL_OFFSETS = {
    "NGC 1380": (7, -5),
    "NGC 1399": (7, 8),
    "NGC 1404": (7, -2),
    "NGC 1549": (-31, -2),
    "NGC 3379": (7, -7),
    "NGC 4374": (7, -5),
    "NGC 4406": (7, -5),
    "NGC 4472": (-32, 7),
    "NGC 4486": (7, 7),
    "NGC 4552": (7, -2),
    "NGC 4621": (7, 2),
    "NGC 4636": (7, -12),
    "NGC 4649": (-18, -13),
    "NGC 4697": (7, 4),
}


def main():
    calib = pd.read_csv(CALIBRATION_INPUT)
    fit = pd.read_csv(FIT_SUMMARY).iloc[0]

    plot = calib[np.isfinite(calib["Mbar_150"]) & np.isfinite(calib["sigma_Mbar_150"])].copy()
    plot["is_clean"] = plot["is_clean_effective"].astype(bool)

    pivot = float(fit["pivot_color"])
    intercept = float(fit["intercept"])
    slope = float(fit["slope"])
    sigma_int = float(fit["sigma_int"])
    wrms = float(fit["in_sample_wrms"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.9))

    for is_clean, label in [(True, "adopted sample"), (False, "pipeline-warning objects")]:
        sub = plot[plot["is_clean"].eq(is_clean)]
        style = STYLE["clean" if is_clean else "flagged"]
        ax.errorbar(
            sub["color_F090W_F150W"],
            sub["Mbar_150"],
            yerr=sub["sigma_Mbar_150"],
            fmt=style["marker"],
            ms=5.2,
            capsize=2.1,
            lw=0.8,
            color=style["color"],
            mec=style["mec"],
            mfc=style["mfc"],
            mew=1.0,
            alpha=0.96,
            label=label,
            zorder=3,
        )

    xgrid = np.linspace(
        plot["color_F090W_F150W"].min() - 0.012,
        plot["color_F090W_F150W"].max() + 0.012,
        300,
    )
    ygrid = intercept + slope * (xgrid - pivot)
    ax.fill_between(
        xgrid,
        ygrid - sigma_int,
        ygrid + sigma_int,
        color=STYLE["band"],
        alpha=0.18,
        lw=0,
        label=rf"intrinsic scatter $\sigma_{{int}}={sigma_int:.3f}$ mag",
        zorder=1,
    )
    ax.plot(xgrid, ygrid, color=STYLE["fit"], lw=1.35, label="weighted linear fit", zorder=2)

    for _, row in plot.iterrows():
        dx, dy = LABEL_OFFSETS.get(row["galaxy"], (6, 4))
        ax.annotate(
            row["galaxy_label"],
            (row["color_F090W_F150W"], row["Mbar_150"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2,
            ha="left" if dx >= 0 else "right",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=4,
        )

    fit_text = (
        rf"$\bar M_{{150}}={intercept:.2f}{slope:+.2f}(C-{pivot:.2f})$" "\n"
        rf"clean WRMS = {wrms:.3f} mag"
    )
    ax.text(0.03, 0.05, fit_text, transform=ax.transAxes, fontsize=8, va="bottom")

    ax.invert_yaxis()
    ax.set_xlim(plot["color_F090W_F150W"].min() - 0.022, plot["color_F090W_F150W"].max() + 0.022)
    ax.set_xlabel(r"$C=F090W-F150W$ [mag]")
    ax.set_ylabel(r"$\bar M_{150}$ [mag]")
    ax.set_title("JWST F150W SBF calibration", pad=7)
    ax.legend(loc="upper right", fontsize=7.6, handlelength=1.6)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=220)
    print(f"Saved: {FIG_PDF}")
    print(f"Saved: {FIG_PNG}")


if __name__ == "__main__":
    main()
