#!/usr/bin/env python3
"""Парные рисунки краткой статьи / Paired figures for the concise manuscript.

Read completed tables only: no fitting, new measurements or table writes.
Run from code/: py -m figures.build_short_article_figures [--cmyk]
"""

import argparse
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from figures import build_go3055_article_figures as full


# Resolve paths from this script, not a particular checkout or working directory.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "texts/paper_work/materials/pasa/short"
TABLES = {band: ROOT / f"runs/{band}/analysis/tables" for band in ("F150W", "F090W")}
# Official pas.cls text width, in TeX points; all fonts are final-size >=8 pt.
WIDTH = 515 / 72.27
GROUPS = full.GROUP_STYLES
# Colour distinguishes budget terms; hatching additionally identifies shared scale.
COMPONENTS = [
    ("Spectrum", "sigma_measurement_mag", "#4c92c3", ""),
    ("Background", "sigma_sky_mag", "#9e9e9e", ""),
    ("PSF", "sigma_psf_mag", "#ff963f", ""),
    (r"Unresolved sources $P_r$", "sigma_Pr_mag", "#55ad55", ""),
    ("Extinction", "sigma_extinction_mag", "#9a73c1", ""),
    ("Colour", "sigma_color_measurement_mag", "#e377c2", ""),
    ("Intrinsic scatter", "sigma_intrinsic_mag", "#f2c14e", ""),
    ("Finite calibration", "sigma_calibration_mag", "#17a589", ""),
    ("Shared TRGB scale", "sigma_common_trgb_mag", "#8c564b", "///"),
]


def style():
    """Оформление, не статистика / Rendering choices, not analysis settings."""
    full.PASA = True  # Reuse its 8-pt collision-aware galaxy-label placement only.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "grid.color": "0.9",
        "grid.alpha": 1, "grid.linewidth": 0.45,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def group_handles():
    return [Line2D([], [], color=colour, marker=marker, ls="", ms=4.5, label=group)
            for group, (colour, marker) in GROUPS.items()]


def points(ax, frame, x, y, xerr, yerr, environment):
    groups = frame.galaxy.map(environment)
    for group, (colour, marker) in GROUPS.items():
        part = frame.loc[groups.eq(group)]
        ax.errorbar(part[x], part[y], xerr=part[xerr], yerr=part[yerr],
                    fmt=marker, ms=4.5, color=colour, mec="black", mew=0.35,
                    elinewidth=0.8, capsize=2, zorder=4)
    ax._galaxy_labels = list(frame[["galaxy", x, y]].itertuples(index=False, name=None))
    ax.minorticks_on()


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    full.place_galaxy_labels(fig)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches=None)
    plt.close(fig)
    print((OUT / f"{stem}.pdf").relative_to(ROOT))


def calibration(frames, fits, environment):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.8))
    for index, (band, ax) in enumerate(zip(frames, axes)):
        frame, models = frames[band], fits[band]
        f150 = band == "F150W"
        adopted = "constant" if f150 else "linear"
        alternative = "linear" if f150 else "constant"
        row, other = models.loc[adopted], models.loc[alternative]
        intercept = "intercept" if f150 else "intercept_mag"
        slope = "slope" if f150 else "slope_at_center"
        scatter = "sigma_int" if f150 else "sigma_int_mag"
        x = "color_F090W_F150W"
        grid = np.linspace(frame[x].min() - 0.013, frame[x].max() + 0.013, 500)
        dx = grid - frame[x].median()
        curve = row[intercept] + row[slope] * dx
        ax.fill_between(grid, curve - row[scatter], curve + row[scatter],
                        color="0.88", linewidth=0, zorder=1)
        ax.plot(grid, curve, color="black", lw=1.4, zorder=3)
        ax.plot(grid, other[intercept] + other[slope] * dx,
                color="0.35", lw=1.2, ls="--", zorder=2)
        points(ax, frame, x, f"Mbar_{band}",
               "sigma_color_total" if f150 else "sigma_color_adopted_mag",
               "sigma_Mbar_internal" if f150 else "sigma_Mbar_F090W", environment)
        ax.set(title=f"({chr(97 + index)}) {band}",
               xlabel=r"$(F090W-F150W)_0$ [mag]",
               ylabel=rf"$\overline{{M}}_{{{band[1:4]}}}$ [mag]")
        ax.invert_yaxis()
        ax.margins(y=0.20)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    handles = group_handles() + [
        Line2D([], [], color="black", lw=1.4, label="Adopted calibration"),
        Line2D([], [], color="0.35", lw=1.2, ls="--", label="Alternative calibration"),
        Patch(facecolor="0.88", label=r"Adopted $\pm\sigma_{\rm int}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               columnspacing=1.2, handlelength=2.3)
    fig.tight_layout(pad=0.6, w_pad=1.2, rect=(0, 0.16, 1, 1))
    save(fig, "calibration")


def recovery(loo, environment):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.8))
    # Identical scales retain an honest visual comparison of the two filters.
    low = min((f.mu_trgb - f.sigma_mu_trgb).min() for f in loo.values()) - 0.12
    high = max((f.mu_sbf + f.sigma_internal).max() for f in loo.values()) + 0.12
    for index, (band, ax) in enumerate(zip(loo, axes)):
        frame = loo[band]
        points(ax, frame, "mu_trgb", "mu_sbf", "sigma_mu_trgb", "sigma_internal", environment)
        ax.plot([low, high], [low, high], ls="--", color="0.3", lw=1)
        ax.set(xlim=(low, high), ylim=(low, high), title=f"({chr(97 + index)}) {band}",
               xlabel=r"$\mu_{\rm TRGB}$ [mag]",
               ylabel=r"$\mu_{\rm SBF}^{\rm LOO}$ [mag]")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(handles=group_handles(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(pad=0.7, w_pad=1.2, rect=(0, 0.08, 1, 1))
    save(fig, "distance_recovery")


def variance(frames, components, environment, stem):
    # Both panels keep the same row order; left-panel variance sets the ranking.
    first = frames["F150W"].set_index("galaxy")
    order = sum(first[column] ** 2 for _, column, _, _ in components).sort_values(ascending=False).index
    limit = max(sum(frame[column] ** 2 for _, column, _, _ in components).max()
                for frame in frames.values()) * 1.06
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.25))
    for index, (band, ax) in enumerate(zip(frames, axes)):
        frame = frames[band].set_index("galaxy").loc[order]
        left = np.zeros(len(frame))
        for label, column, colour, hatch in components:
            values = frame[column].to_numpy(float) ** 2
            ax.barh(np.arange(len(frame)), values, left=left, height=0.73,
                    color=colour, hatch=hatch, edgecolor="0.2" if hatch else "none",
                    linewidth=0.25, label=label)
            left += values
        ax.set(title=f"({chr(97 + index)}) {band}", xlabel=r"Variance [mag$^2$]",
               yticks=np.arange(len(frame)), yticklabels=order)
        ax.set_ylim(len(frame) - 0.4, -0.6)
        ax.set_xlim(0, limit)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        ax.xaxis.set_major_formatter(formatter)
        for tick, galaxy in zip(ax.get_yticklabels(), order):
            tick.set_color(GROUPS[environment[galaxy]][0])
    handles = [Patch(facecolor=colour, hatch=hatch,
                     edgecolor="0.2" if hatch else "none", label=label)
               for label, _, colour, hatch in components]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               columnspacing=1, handlelength=2)
    bottom = 0.16 if len(components) <= 5 else 0.20
    # Finalise scientific-notation offsets before calculating layout extents.
    fig.canvas.draw()
    fig.tight_layout(pad=0.6, w_pad=1.2, rect=(0, bottom, 1, 0.98))
    save(fig, stem)


def export_cmyk(paths):
    """CMYK для подачи / Optional vector-preserving journal colour export."""
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError("The --cmyk export requires Ghostscript (gs).")
    work = ROOT / "texts/paper_work/build/short-figure-export"
    work.mkdir(parents=True, exist_ok=True)
    for path in paths:
        source, target = work / f"{path.stem}-rgb.pdf", work / f"{path.stem}-cmyk.pdf"
        shutil.copyfile(path, source)
        subprocess.run([
            ghostscript, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4", "-sColorConversionStrategy=CMYK",
            "-dProcessColorModel=/DeviceCMYK", "-dEmbedAllFonts=true", "-dSubsetFonts=true",
            "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
            "-dDownsampleMonoImages=false", f"-sOutputFile={target}", str(source),
        ], check=True)
        shutil.copyfile(target, path)
        print(f"CMYK: {path.relative_to(ROOT)}")


def main(cmyk=False):
    style()
    f150 = pd.read_csv(TABLES["F150W"] / "go3055_master_measurements.csv")
    f090 = pd.read_csv(TABLES["F090W"] / "go3055_f090w_master.csv")
    environment = f150.set_index("galaxy").environment.replace({"Virgo": "Virgo region"})
    fits = {
        "F150W": pd.read_csv(TABLES["F150W"] / "go3055_color_model_comparison.csv").set_index("model"),
        "F090W": pd.read_csv(TABLES["F090W"] / "go3055_f090w_color_model_comparison.csv").set_index("model"),
    }
    d150 = pd.read_csv(TABLES["F150W"] / "go3055_leave_one_out_distances.csv")
    d150 = d150.loc[d150.model.eq("constant")].rename(columns={
        "mu_sbf_loo": "mu_sbf", "sigma_mu_sbf_internal": "sigma_internal"})
    d090 = pd.read_csv(TABLES["F090W"] / "go3055_f090w_distance_error_budget.csv")
    d090 = d090.loc[d090.model.eq("linear")].rename(columns={"sigma_mu_internal": "sigma_internal"})
    b150 = pd.read_csv(TABLES["F150W"] / "go3055_error_budget.csv").rename(columns={
        "sigma_psf_diagnostic_mag": "sigma_psf_mag", "sigma_A_F150W_mag": "sigma_extinction_mag"})
    b090 = f090.rename(columns={"sigma_A_F090W": "sigma_extinction_mag"})
    p150 = b150.merge(d150[["galaxy", "sigma_intrinsic_mag", "sigma_calibration_prediction_mag"]],
                       on="galaxy", validate="one_to_one").rename(columns={
                           "sigma_calibration_prediction_mag": "sigma_calibration_mag"})
    p150["sigma_color_measurement_mag"] = 0.0  # The adopted F150W model has no colour term.
    p150["sigma_common_trgb_mag"] = 0.047  # Shared Paper IV scale, never divided by sqrt(N).
    # Verify that graphical stacks reproduce existing absolute distance variances.
    for frame, target, column in ((p150, d150, "sigma_mu_sbf_absolute"), (d090, d090, "sigma_mu_total")):
        rows = frame.set_index("galaxy").sort_index()
        total = sum(rows[c] ** 2 for _, c, _, _ in COMPONENTS)
        np.testing.assert_allclose(total, target.set_index("galaxy").sort_index()[column] ** 2)
    calibration({"F150W": f150, "F090W": f090}, fits, environment)
    recovery({"F150W": d150, "F090W": d090}, environment)
    variance({"F150W": b150, "F090W": b090}, COMPONENTS[:5], environment, "measurement_variance")
    variance({"F150W": p150, "F090W": d090}, COMPONENTS, environment, "distance_variance")
    # Preserve the existing same-eight-galaxy plot exactly; no new regressions.
    same8 = OUT / "same8_colour.pdf"
    shutil.copyfile(ROOT / "texts/paper_work/materials/pasa/figures/go3055_article_same8_three_band_color_comparison.pdf", same8)
    print(same8.relative_to(ROOT))
    if cmyk:
        export_cmyk([OUT / f"{stem}.pdf" for stem in (
            "calibration", "distance_recovery", "measurement_variance", "distance_variance", "same8_colour")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmyk", action="store_true", help="Export journal CMYK PDFs with Ghostscript; retain vector artwork.")
    main(cmyk=parser.parse_args().cmyk)
