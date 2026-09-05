#!/usr/bin/env python3
"""Build matched publication figures for the GO-3055 F150W/F090W analysis.

The script reads only completed analysis tables.  It does not execute either
measurement pipeline and does not rewrite any scientific table.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
F150_TABLES = ROOT / "runs/sbf2_go3055/analysis/tables"
F090_TABLES = ROOT / "runs/sbf_f090w_go3055/analysis/tables"
F150_FIGURES = ROOT / "runs/sbf2_go3055/analysis/figures"
F090_FIGURES = ROOT / "runs/sbf_f090w_go3055/analysis/figures"
PAPER3_F110 = ROOT / "code/sbf2_batch_outputs/jensen2025_paper3_f110w_calibration.csv"

BLUE = "#2c7fb8"
ORANGE = "#d95f0e"
BLACK = "#111111"
GROUP_STYLES = {
    "Fornax": ("#d62728", "s"),
    "Virgo region": ("#2463c5", "o"),
    "Other": ("#62626b", "^"),
}
# Match the broad environment classes in the measurement tables; this is
# deliberately not the seven-member strict Virgo subset used in the step test.
ENVIRONMENT = pd.read_csv(F150_TABLES / "go3055_master_measurements.csv").set_index("galaxy")["environment"].replace({"Virgo": "Virgo region"})
COMPONENT_COLORS = {
    "Power spectrum": "#4c92c3",
    "Background": "#9e9e9e",
    "PSF": "#ff963f",
    r"Unresolved sources $P_r$": "#55ad55",
    "Foreground extinction": "#9a73c1",
    "Color": "#e377c2",
    "Intrinsic scatter": "#f2c14e",
    "Finite calibration": "#17a589",
    "Shared TRGB scale": "#8c564b",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "axes.linewidth": 1.4,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 3.5,
        "ytick.minor.size": 3.5,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
        "savefig.facecolor": "white",
    }
)


def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    place_galaxy_labels(fig)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def group_legend(ax, **kwargs):
    handles = [Line2D([], [], color=color, marker=marker, ls="", label=group)
               for group, (color, marker) in GROUP_STYLES.items()]
    return ax.legend(handles=handles, frameon=False, **kwargs)


def plot_points(ax, frame, x, y, yerr, xerr=None, annotate=True):
    groups = frame["galaxy"].map(ENVIRONMENT)
    for group, (color, marker) in GROUP_STYLES.items():
        part = frame.loc[groups.eq(group)]
        if part.empty:
            continue
        ax.errorbar(part[x], part[y], yerr=part[yerr],
                    xerr=part[xerr] if xerr else None, fmt=marker, ms=6.5,
                    color=color, ecolor=color, mec=BLACK, mew=0.5,
                    capsize=3, lw=1.2, label=group, zorder=4)
    if annotate:
        pending = getattr(ax, "_galaxy_labels", [])
        pending.extend((row.galaxy, getattr(row, x), getattr(row, y))
                       for row in frame.itertuples(index=False))
        ax._galaxy_labels = pending


def place_galaxy_labels(fig):
    """Place NGC numbers after layout, avoiding points, text and panel edges."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        points = getattr(ax, "_galaxy_labels", [])
        if not points:
            continue
        occupied = [text.get_window_extent(renderer).expanded(1.08, 1.12)
                    for text in ax.texts if text.get_visible()]
        if ax.get_legend() is not None:
            occupied.append(ax.get_legend().get_window_extent(renderer))
        positions = ax.transData.transform([(x, y) for _, x, y in points])
        occupied.extend(Bbox.from_bounds(x-5, y-5, 10, 10) for x, y in positions)
        for galaxy, x, y in points:
            label = ax.annotate(galaxy.replace("NGC ", ""), (x, y),
                xytext=(6, 6), textcoords="offset points", fontsize=9,
                zorder=6, bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.3),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5))
            candidates = [(sx*dx, sy*dy) for dx, dy in
                          ((6, 6), (6, 15), (20, 6), (20, 18), (6, 30), (36, 6), (36, 25))
                          for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))]
            best, best_cost = None, float("inf")
            for dx, dy in candidates:
                label.set_position((dx, dy))
                label.set_ha("left" if dx > 0 else "right")
                label.set_va("bottom" if dy > 0 else "top")
                label.update_positions(renderer)
                # Text extent only; the annotation extent also contains its arrow.
                box = matplotlib.text.Text.get_window_extent(label, renderer).expanded(1.08, 1.12)
                overflow = not (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1))
                cost = 10000*overflow + 1000*sum(box.overlaps(b) for b in occupied) + np.hypot(dx, dy)
                if cost < best_cost:
                    best, best_cost = (dx, dy, box), cost
            dx, dy, box = best
            label.set_position((dx, dy))
            label.set_ha("left" if dx > 0 else "right")
            label.set_va("bottom" if dy > 0 else "top")
            occupied.append(box)
        ax._galaxy_labels = []


def make_calibration_plots(f150, f090):
    f150_model = pd.read_csv(F150_TABLES / "go3055_color_model_comparison.csv")
    f150_fit = f150_model.loc[f150_model["model"] == "constant"].iloc[0]
    f150_linear = f150_model.loc[f150_model["model"] == "linear"].iloc[0]

    f090_models = pd.read_csv(F090_TABLES / "go3055_f090w_color_model_comparison.csv")
    f090_fit = f090_models.loc[f090_models["model"] == "linear"].iloc[0]

    configurations = [
        {
            "frame": f150,
            "directory": F150_FIGURES,
            "stem": "go3055_f150w_article_calibration",
            "band": "F150W",
            "y": "Mbar_F150W",
            "yerr": "sigma_Mbar_internal",
            "intercept": f150_fit["intercept"],
            "slope": 0.0,
            "center": float(f150["color_F090W_F150W"].median()),
            "sigma_int": f150_fit["sigma_int"],
            "label": "adopted constant calibration",
        },
        {
            "frame": f150,
            "directory": F150_FIGURES,
            "stem": "go3055_f150w_article_calibration_linear",
            "band": "F150W",
            "y": "Mbar_F150W",
            "yerr": "sigma_Mbar_internal",
            "intercept": f150_linear["intercept"],
            "slope": f150_linear["slope"],
            "center": float(f150["color_F090W_F150W"].median()),
            "sigma_int": f150_linear["sigma_int"],
            "label": "linear color test",
        },
        {
            "frame": f090,
            "directory": F090_FIGURES,
            "stem": "go3055_f090w_article_calibration",
            "band": "F090W",
            "y": "Mbar_F090W",
            "yerr": "sigma_Mbar_F090W",
            "intercept": f090_fit["intercept_mag"],
            "slope": f090_fit["slope_at_center"],
            "center": float(f090["color_F090W_F150W"].median()),
            "sigma_int": f090_fit["sigma_int_mag"],
            "label": "adopted linear calibration",
        },
    ]

    for cfg in configurations:
        frame = cfg["frame"].copy()
        x = frame["color_F090W_F150W"].to_numpy(float)
        grid = np.linspace(x.min() - 0.012, x.max() + 0.012, 500)
        dx = grid - cfg["center"]
        line = cfg["intercept"] + cfg["slope"] * dx

        # A uniform intrinsic-scatter strip, not a confidence interval for the
        # fitted line and not the full uncertainty of a future distance.
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        ax.fill_between(
            grid,
            line - cfg["sigma_int"],
            line + cfg["sigma_int"],
            color="0.86",
            alpha=0.65,
            linewidth=0,
            label=rf"$\pm\sigma_{{\rm int}}={cfg['sigma_int']:.3f}$ mag",
            zorder=1,
        )
        ax.plot(grid, line, color=BLACK, lw=2.2, label=cfg["label"], zorder=3)
        plot_points(ax, frame, "color_F090W_F150W", cfg["y"], cfg["yerr"])
        ax.set_title(f"JWST {cfg['band']} SBF calibration")
        ax.set_xlabel(r"$(F090W-F150W)_0$ [mag]")
        ax.set_ylabel(rf"$\overline{{M}}_{{{cfg['band'][1:4]}}}$ [mag]")
        ax.invert_yaxis()
        ax.minorticks_on()
        if cfg["slope"] == 0:
            equation = rf"$\overline{{M}}_{{150}}={cfg['intercept']:.3f}$ mag"
        else:
            equation = (
                rf"$\overline{{M}}_{{{cfg['band'][1:4]}}}={cfg['intercept']:.3f}"
                rf"+{cfg['slope']:.3f}(C-{cfg['center']:.3f})$"
            )
        ax.text(0.025, 0.035, equation, transform=ax.transAxes, fontsize=12)
        ax.margins(y=0.18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        fig.tight_layout()
        save_figure(fig, cfg["directory"], cfg["stem"])


def make_recovery_plot(frame, band, directory, stem, y_sigma):
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    plot_points(ax, frame, "mu_trgb", "mu_sbf", y_sigma, "sigma_mu_trgb")
    low = min(frame["mu_trgb"].min(), frame["mu_sbf"].min()) - 0.08
    high = max(frame["mu_trgb"].max(), frame["mu_sbf"].max()) + 0.08
    ax.plot([low, high], [low, high], color=BLACK, ls="--", lw=1.7, label="1:1")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{band} leave-one-out distance recovery")
    ax.set_xlabel(r"$\mu_{\rm TRGB}$ [mag]")
    ax.set_ylabel(rf"$\mu_{{\rm SBF,{band[1:4]}}}^{{\rm LOO}}$ [mag]")
    ax.minorticks_on()
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, directory, stem)


def make_residual_plot(frame, band, directory, stem, error_column, loo_rms):
    ordered = frame.sort_values("delta_mu_sbf_minus_trgb", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    groups = ordered["galaxy"].map(ENVIRONMENT)
    for group, (color, marker) in GROUP_STYLES.items():
        selected = groups.eq(group)
        ax.errorbar(ordered.loc[selected, "delta_mu_sbf_minus_trgb"], y[selected],
                    xerr=ordered.loc[selected, error_column], fmt=marker, ms=7,
                    color=color, capsize=3, lw=1.3, label=group)
    ax.axvline(0, color=BLACK, ls="--", lw=1.7)
    ax.set_yticks(y, ordered["galaxy"])
    ax.invert_yaxis()
    ax.set_title(f"{band} leave-one-out distance residuals")
    ax.set_xlabel(r"$\mu_{\rm SBF}^{\rm LOO}-\mu_{\rm TRGB}$ [mag]")
    ax.text(0.025, 0.965, f"LOO RMS = {loo_rms:.3f} mag", transform=ax.transAxes, va="top")
    ax.minorticks_on()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, directory, stem)


def make_stacked_variance(frame, components, title, directory, stem):
    frame = frame.reset_index(drop=True)
    values = pd.DataFrame({label: frame[column].to_numpy(float) ** 2 for label, column in components})
    order = values.sum(axis=1).sort_values(ascending=False).index
    values = values.loc[order].reset_index(drop=True)
    labels = frame.loc[order, "galaxy"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    y = np.arange(len(values))
    left = np.zeros(len(values))
    for label, _ in components:
        ax.barh(y, values[label], left=left, height=0.76, color=COMPONENT_COLORS[label], label=label)
        left += values[label].to_numpy()
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(r"variance contribution [mag$^2$]")
    ax.minorticks_on()
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for tick, galaxy in zip(ax.get_yticklabels(), labels):
        tick.set_color(GROUP_STYLES[ENVIRONMENT[galaxy]][0])
    group_handles = [Line2D([], [], color=c, marker=m, ls="", label=g)
                     for g, (c, m) in GROUP_STYLES.items()]
    fig.legend(handles=group_handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, directory, stem)


def make_annulus_plot(frame, fit_rows, band, directory, stem):
    inner = frame.loc[frame["annulus"] == "inner"].sort_values("galaxy")
    outer = frame.loc[frame["annulus"] == "outer"].sort_values("galaxy")
    paired = inner[["galaxy", "color", "Mbar"]].merge(
        outer[["galaxy", "color", "Mbar"]], on="galaxy", suffixes=("_inner", "_outer")
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for row in paired.itertuples(index=False):
        ax.plot(
            [row.color_inner, row.color_outer],
            [row.Mbar_inner, row.Mbar_outer],
            color="0.74",
            lw=1.0,
            zorder=1,
        )
    for annulus, marker, color, label in (
        ("inner", "o", BLUE, "inner annulus"),
        ("outer", "s", ORANGE, "outer annulus"),
    ):
        sample = frame.loc[frame["annulus"] == annulus]
        ax.errorbar(
            sample["color"],
            sample["Mbar"],
            yerr=sample["sigma_Mbar"],
            fmt=marker,
            ms=6.5,
            color=color,
            ecolor=color,
            capsize=3,
            lw=1.2,
            label=label,
            zorder=3,
        )
        fit = fit_rows.loc[fit_rows["annulus"] == annulus].iloc[0]
        grid = np.linspace(frame["color"].min() - 0.012, frame["color"].max() + 0.012, 400)
        line = fit["intercept"] + fit["slope"] * (grid - fit["color_center"])
        ax.plot(grid, line, color=color, ls="-" if annulus == "inner" else "--", lw=2.0)
    ax.set_title(f"{band} annulus-matched color and SBF")
    ax.set_xlabel(r"local $(F090W-F150W)_0$ [mag]")
    ax.set_ylabel(rf"local $\overline{{M}}_{{{band[1:4]}}}$ [mag]")
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, directory, stem)


def make_f160_comparison(frame, band, directory, stem):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.25))

    plot_points(axes[0], frame, "g475_z850", "mbar_difference", "sigma_mbar_difference", "sigma_g475_z850")
    axes[0].set_title("Distance-free fluctuation color", fontsize=15)
    axes[0].set_xlabel(r"$(g_{475}-z_{850})_0$ [mag]", fontsize=13)
    axes[0].set_ylabel(rf"$\overline{{m}}_{{{band[1:4]}}}-\overline{{m}}_{{160}}$ [mag]", fontsize=13)

    plot_points(axes[1], frame, "Mbar_F160W_individual", "Mbar_target", "sigma_Mbar_target", "sigma_Mbar_F160W_individual")
    axes[1].set_title("Absolute SBF with the same TRGB anchor", fontsize=15)
    axes[1].set_xlabel(r"$\overline{M}_{160}$ [mag]", fontsize=13)
    axes[1].set_ylabel(rf"$\overline{{M}}_{{{band[1:4]}}}$ [mag]", fontsize=13)
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()

    low = min(frame["mu_F160W_jensen2015"].min(), frame["mu_target"].min()) - 0.08
    high = max(frame["mu_F160W_jensen2015"].max(), frame["mu_target"].max()) + 0.08
    plot_points(axes[2], frame, "mu_F160W_jensen2015", "mu_target", "sigma_mu_target", "sigma_mu_F160W_jensen2015")
    axes[2].plot([low, high], [low, high], color=BLACK, lw=1.7)
    axes[2].set_xlim(low, high)
    axes[2].set_ylim(low, high)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("SBF distance comparison", fontsize=15)
    axes[2].set_xlabel(r"$\mu_{160}$, Jensen et al. (2015) [mag]", fontsize=13)
    axes[2].set_ylabel(rf"$\mu_{{{band[1:4]}}}^{{\rm LOO}}$ [mag]", fontsize=13)

    for ax in axes:
        ax.minorticks_on()
        ax.margins(0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"JWST {band} versus HST/WFC3 F160W", fontsize=18)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, directory, stem)


def make_aperture_luminosity_plot(frame, band, y, yerr, directory, stem):
    x = frame["M150_model_aperture"].to_numpy(float)
    values = frame[y].to_numpy(float)
    sigma = frame[yerr].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x - np.median(x)))
    weights = 1.0 / sigma**2
    coefficients = np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * values))
    residual = values - design @ coefficients
    scatter = np.sqrt(np.sum(residual**2) / (len(frame) - 2))
    grid = np.linspace(x.min() - 0.3, x.max() + 0.3, 400)
    grid_design = np.column_stack((np.ones(len(grid)), grid - np.median(x)))
    line = grid_design @ coefficients
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.fill_between(grid, line - scatter, line + scatter, color="0.86", alpha=0.65, linewidth=0,
                    label=rf"$\pm s_{{\rm res}}={scatter:.3f}$ mag")
    ax.plot(grid, line, color=BLACK, lw=2.0, label="weighted linear diagnostic")
    plot_points(ax, frame, "M150_model_aperture", y, yerr)
    ax.set_title(f"{band} SBF versus host aperture magnitude")
    ax.set_xlabel(r"F150W model magnitude within $32.8''$ [mag]")
    ax.set_ylabel(rf"$\overline{{M}}_{{{band[1:4]}}}$ [mag]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.margins(y=0.14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    save_figure(fig, directory, stem)


def make_same8_three_band_color_comparison(f150, f090):
    """Compare three SBF bands against both available colors for the same objects."""
    colors = pd.read_csv(F150_TABLES / "go3055_color_pairs_vs_jensen.csv")
    paper3 = pd.read_csv(PAPER3_F110)
    paper3["Mbar_F110W"] = paper3["m110_0"] - paper3["mu_cluster"]
    paper3["sigma_Mbar_F110W"] = np.hypot(
        paper3["sigma_m110_0"], paper3["sigma_mu_cluster"]
    )
    common = (
        colors.merge(
            f150[["galaxy", "Mbar_F150W", "sigma_Mbar_internal"]],
            on="galaxy",
            how="inner",
        )
        .merge(
            f090[["galaxy", "Mbar_F090W", "sigma_Mbar_F090W"]],
            on="galaxy",
            how="inner",
        )
        .merge(
            paper3[["galaxy", "Mbar_F110W", "sigma_Mbar_F110W"]],
            on="galaxy",
            how="inner",
        )
        .sort_values("galaxy")
    )
    if len(common) != 8:
        raise ValueError(f"Expected eight common galaxies, found {len(common)}")

    rows = [
        ("F090W", "Mbar_F090W", "sigma_Mbar_F090W"),
        ("F150W", "Mbar_F150W", "sigma_Mbar_internal"),
        ("Paper III F110W", "Mbar_F110W", "sigma_Mbar_F110W"),
    ]
    columns = [
        (r"$(F090W-F150W)_0$", "color_F090W_F150W", "sigma_color_total"),
        (r"$(g_{475}-z_{850})_0$", "g_z_ps_0", "sigma_g_z_ps_0"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 14.4), sharex="col")
    for row_index, (band, y_name, yerr_name) in enumerate(rows):
        for column_index, (color_label, x_name, xerr_name) in enumerate(columns):
            ax = axes[row_index, column_index]
            x = common[x_name].to_numpy(float)
            y = common[y_name].to_numpy(float)
            yerr = common[yerr_name].to_numpy(float)
            center = float(np.median(x))
            design = np.column_stack((np.ones(len(x)), x - center))
            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            residual_variance = np.sum(residual**2) / (len(x) - 2)
            grid = np.linspace(x.min() - 0.04 * np.ptp(x), x.max() + 0.04 * np.ptp(x), 500)
            grid_design = np.column_stack((np.ones(len(grid)), grid - center))
            line = grid_design @ coefficients
            scatter = np.sqrt(residual_variance)

            ax.fill_between(
                grid,
                line - scatter,
                line + scatter,
                color="0.88",
                linewidth=0,
                label=r"$\pm s_{\rm res}$",
            )
            ax.plot(grid, line, color=BLACK, lw=2.0, label="linear fit")
            plot_points(ax, common, x_name, y_name, yerr_name, xerr_name)
            ax.text(0.03, 0.04, rf"$s_{{\rm res}}={scatter:.3f}$ mag", transform=ax.transAxes, fontsize=10)
            if row_index == 0:
                ax.set_title(color_label)
            if column_index == 0:
                wavelength = band.split()[0][1:4] if band.startswith("F") else "110"
                ax.set_ylabel(f"{band}\n" + rf"$\overline{{M}}_{{{wavelength}}}$ [mag]")
            if row_index == 2:
                ax.set_xlabel(f"color {color_label} [mag]")
            ax.invert_yaxis()
            ax.minorticks_on()
            ax.margins(x=0.12, y=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Same-galaxy color comparison across three SBF bands", fontsize=20)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    save_figure(
        fig,
        F150_FIGURES,
        "go3055_article_same8_three_band_color_comparison",
    )


def make_cross_band_plot(f150, f090):
    common = f090[["galaxy", "Mbar_F090W", "sigma_Mbar_F090W", "mbar_F090W_0", "sigma_mbar_internal"]].merge(
        f150[["galaxy", "Mbar_F150W", "sigma_Mbar_internal", "mbar_F150W", "sigma_mbar_internal", "color_F090W_F150W", "sigma_color_total"]],
        on="galaxy", suffixes=("_090", "_150"))
    common["fluctuation_color"] = common["mbar_F090W_0"] - common["mbar_F150W"]
    common["sigma_fluctuation_color"] = np.hypot(common["sigma_mbar_internal_090"], common["sigma_mbar_internal_150"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    specifications = [
        ("Mbar_F150W", "Mbar_F090W", "sigma_Mbar_F090W", "sigma_Mbar_internal",
         r"$\overline{M}_{150,0}$ [mag]", r"$\overline{M}_{090,0}$ [mag]", "Shared Paper IV distance anchors"),
        ("color_F090W_F150W", "fluctuation_color", "sigma_fluctuation_color", "sigma_color_total",
         r"$(F090W-F150W)_0$ [mag]", r"$\overline{m}_{090,0}-\overline{m}_{150,0}$ [mag]", "Distance-free fluctuation color"),
    ]
    for ax, (x, y, sy, sx, xlabel, ylabel, title) in zip(axes, specifications):
        plot_points(ax, common, x, y, sy, sx)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.invert_yaxis()
        ax.margins(0.18)
        ax.minorticks_on()
    axes[0].invert_xaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, F090_FIGURES, "go3055_f090w_article_vs_f150w_sbf")


def make_winsorization_plot(f090):
    base = ROOT / "runs/sbf2_normalized_winsor/batch/aggregates"
    clipping = {"F150W": pd.read_csv(base / "all_galaxies_clipping.csv")}
    annuli = {"F150W": pd.read_csv(base / "all_galaxies_combined_annuli.csv")}
    paths = [json.loads(Path(p).read_text())["table_paths"] for p in f090["final_result_path"]]
    clipping["F090W"] = pd.concat([pd.read_csv(p["clipping"]) for p in paths])
    annuli["F090W"] = pd.concat([pd.read_csv(p["combined_annuli"]) for p in paths])
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for row, band in enumerate(("F150W", "F090W")):
        clip = clipping[band].loc[clipping[band]["branch"].eq("normalized_full_3p5")]
        measured = annuli[band].loc[np.isclose(annuli[band]["requested_kmin"], 0.04)]
        adopted = measured.loc[measured["branch"].eq("normalized_full_3p5")].set_index("galaxy")
        plain = measured.loc[measured["branch"].eq("no_winsor")].set_index("galaxy")
        for col, ring in enumerate(("inner", "outer")):
            ax = axes[row, col]
            part = clip.loc[clip["ring"].eq(ring), ["galaxy", "changed_fraction"]].copy()
            part["percent"] = 100*part["changed_fraction"]
            part["delta"] = part["galaxy"].map(adopted[f"mbar_{ring}"] - plain[f"mbar_{ring}"])
            part["zero_error"] = 0.0
            plot_points(ax, part, "percent", "delta", "zero_error")
            ax.axhline(0, color=BLACK, lw=1, ls="--")
            ax.set(title=f"{band}: {ring} annulus", xlabel="Winsorized pixels (%)",
                   ylabel=r"$\overline{m}_{3.5\sigma}-\overline{m}_{\rm none}$ [mag]")
            ax.margins(x=0.2, y=0.3)
            ax.minorticks_on()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, F090_FIGURES, "go3055_f150w_f090w_article_winsorization")


def make_psf_size_plot(frame, band, delta_column, directory, stem):
    frame = frame.sort_values("galaxy").reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(y, frame[delta_column], color=BLUE, height=0.65)
    ax.axvline(0.0, color=BLACK, lw=1.5)
    ax.set_yticks(y, frame["galaxy"])
    ax.invert_yaxis()
    ax.set_xlim(-0.001, 0.018)
    ax.set_title(f"{band} PSF stamp-size sensitivity")
    ax.set_xlabel(r"$\overline{m}_{129}-\overline{m}_{257}$ [mag]")
    ax.minorticks_on()
    fig.tight_layout()
    save_figure(fig, directory, stem)


def main():
    f150 = pd.read_csv(F150_TABLES / "go3055_master_measurements.csv")
    f090 = pd.read_csv(F090_TABLES / "go3055_f090w_master.csv")
    f150_budget = pd.read_csv(F150_TABLES / "go3055_error_budget.csv")
    f150_loo = pd.read_csv(F150_TABLES / "go3055_leave_one_out_distances.csv")
    f150_loo = f150_loo.loc[f150_loo["model"] == "constant"].copy()
    f090_budget = pd.read_csv(F090_TABLES / "go3055_f090w_distance_error_budget.csv")
    f090_loo = f090_budget.loc[f090_budget["model"] == "linear"].copy()

    make_calibration_plots(f150, f090)

    f150_annuli = pd.read_csv(F150_TABLES / "go3055_annulus_local_measurements.csv").rename(
        columns={"color_F090W_F150W": "color", "Mbar_F150W": "Mbar"}
    )
    f150_annuli["sigma_Mbar"] = f150_annuli["sigma_Mbar"]
    f150_annulus_fits = pd.read_csv(F150_TABLES / "go3055_annulus_local_fit_summary.csv")
    make_annulus_plot(
        f150_annuli,
        f150_annulus_fits,
        "F150W",
        F150_FIGURES,
        "go3055_f150w_article_annulus_comparison",
    )

    f090_annulus_rows = []
    for row in f090.itertuples(index=False):
        reddening = row.A_F090W - row.A_F150W
        for annulus in ("inner", "outer"):
            sigma_spectrum = getattr(row, f"{annulus}_sigma_spectrum_no_psf_mag")
            sigma_psf = getattr(row, f"{annulus}_sigma_psf_mag")
            sigma_pr = getattr(row, f"{annulus}_sigma_Pr_mag")
            sigma_sky = getattr(row, f"{annulus}_sigma_sky_mag")
            sigma_mbar = np.sqrt(sigma_spectrum**2 + sigma_psf**2 + sigma_pr**2 + sigma_sky**2)
            f090_annulus_rows.append(
                {
                    "galaxy": row.galaxy,
                    "annulus": annulus,
                    "color": getattr(row, f"color_{annulus}_observed") - reddening,
                    "Mbar": getattr(row, f"mbar_{annulus}_observed") - row.A_F090W - row.mu_lit,
                    "sigma_Mbar": np.hypot(sigma_mbar, row.sigma_mu_without_reddening_mag),
                }
            )
    f090_annuli = pd.DataFrame(f090_annulus_rows)
    f090_ring_summary = pd.read_csv(F090_TABLES / "go3055_f090w_annulus_comparison.csv")
    center_090 = float(f090["color_F090W_F150W"].median())
    f090_fit_rows = []
    for annulus, measurement in (("inner", "Inner annulus"), ("outer", "Outer annulus")):
        sample = f090_annuli.loc[f090_annuli["annulus"] == annulus]
        slope = float(
            f090_ring_summary.loc[f090_ring_summary["measurement"] == measurement, "linear_slope"].iloc[0]
        )
        weights = 1.0 / sample["sigma_Mbar"].to_numpy(float) ** 2
        intercept = np.average(sample["Mbar"] - slope * (sample["color"] - center_090), weights=weights)
        f090_fit_rows.append(
            {"annulus": annulus, "intercept": intercept, "slope": slope, "color_center": center_090}
        )
    make_annulus_plot(
        f090_annuli,
        pd.DataFrame(f090_fit_rows),
        "F090W",
        F090_FIGURES,
        "go3055_f090w_article_annulus_comparison",
    )

    overlap = pd.read_csv(F150_TABLES / "go3055_current_f150w_vs_jensen2015_f160w.csv")
    f150_f160 = overlap.assign(
        mbar_difference=overlap["m150_minus_m160"],
        sigma_mbar_difference=overlap["sigma_m150_minus_m160"],
        Mbar_target=overlap["Mbar_F150W"],
        sigma_Mbar_target=overlap["sigma_Mbar_internal"],
        mu_target=overlap["mu_sbf_loo"],
        sigma_mu_target=overlap["sigma_mu_sbf_internal"],
    )
    make_f160_comparison(
        f150_f160,
        "F150W",
        F150_FIGURES,
        "go3055_f150w_article_f160w_comparison",
    )

    f090_overlap = f090.merge(
        overlap[
            [
                "galaxy",
                "m160_ab",
                "sigma_m160_ab",
                "g475_z850",
                "sigma_g475_z850",
                "Mbar_F160W_individual",
                "sigma_Mbar_F160W_individual",
                "mu_F160W_jensen2015",
                "sigma_mu_F160W_jensen2015",
            ]
        ],
        on="galaxy",
        how="inner",
    ).merge(
        f090_loo[["galaxy", "mu_sbf", "sigma_mu_internal"]], on="galaxy", how="inner"
    )
    f090_f160 = f090_overlap.assign(
        mbar_difference=f090_overlap["mbar_F090W_0"] - f090_overlap["m160_ab"],
        sigma_mbar_difference=np.hypot(
            f090_overlap["sigma_mbar_internal"], f090_overlap["sigma_m160_ab"]
        ),
        Mbar_target=f090_overlap["Mbar_F090W"],
        sigma_Mbar_target=f090_overlap["sigma_Mbar_F090W"],
        mu_target=f090_overlap["mu_sbf"],
        sigma_mu_target=f090_overlap["sigma_mu_internal"],
    )
    make_f160_comparison(
        f090_f160,
        "F090W",
        F090_FIGURES,
        "go3055_f090w_article_f160w_comparison",
    )

    f090_luminosity = f090.merge(
        f150[["galaxy", "M150_model_aperture"]], on="galaxy", how="inner"
    )
    make_aperture_luminosity_plot(
        f150,
        "F150W",
        "Mbar_F150W",
        "sigma_Mbar_internal",
        F150_FIGURES,
        "go3055_f150w_article_aperture_luminosity",
    )
    make_aperture_luminosity_plot(
        f090_luminosity,
        "F090W",
        "Mbar_F090W",
        "sigma_Mbar_F090W",
        F090_FIGURES,
        "go3055_f090w_article_aperture_luminosity",
    )
    make_same8_three_band_color_comparison(f150, f090)
    make_cross_band_plot(f150, f090)
    make_winsorization_plot(f090)

    f150_psf_size = pd.read_csv(F150_TABLES / "go3055_psf_129_vs_257_sensitivity.csv")
    f090_psf_size = pd.read_csv(F090_TABLES / "go3055_f090w_psf_129_vs_257.csv")
    make_psf_size_plot(
        f150_psf_size,
        "F150W",
        "delta_mbar_mag",
        F150_FIGURES,
        "go3055_f150w_article_psf_size",
    )
    make_psf_size_plot(
        f090_psf_size,
        "F090W",
        "delta_mbar_129_minus_257_mag",
        F090_FIGURES,
        "go3055_f090w_article_psf_size",
    )

    flags = f150[["galaxy", "include_residual_clean"]]
    f150_recovery = f150_loo.merge(flags, on="galaxy", how="left").rename(
        columns={"mu_sbf_loo": "mu_sbf"}
    )
    f090_recovery = f090_loo.merge(flags, on="galaxy", how="left")
    make_recovery_plot(
        f150_recovery,
        "F150W",
        F150_FIGURES,
        "go3055_f150w_article_loo_recovery",
        "sigma_mu_sbf_internal",
    )
    make_recovery_plot(
        f090_recovery,
        "F090W",
        F090_FIGURES,
        "go3055_f090w_article_loo_recovery",
        "sigma_mu_internal",
    )
    make_residual_plot(
        f150_recovery,
        "F150W",
        F150_FIGURES,
        "go3055_f150w_article_loo_residuals",
        "sigma_validation_residual_internal",
        float(np.sqrt(np.mean(f150_recovery["delta_mu_sbf_minus_trgb"] ** 2))),
    )
    make_residual_plot(
        f090_recovery,
        "F090W",
        F090_FIGURES,
        "go3055_f090w_article_loo_residuals",
        "sigma_delta_validation",
        float(np.sqrt(np.mean(f090_recovery["delta_mu_sbf_minus_trgb"] ** 2))),
    )

    f150_measurement = f150_budget.rename(
        columns={
            "sigma_psf_diagnostic_mag": "sigma_psf_mag",
            "sigma_A_F150W_mag": "sigma_extinction_mag",
        }
    )
    f090_measurement = f090[
        [
            "galaxy",
            "sigma_measurement_mag",
            "sigma_sky_mag",
            "sigma_psf_mag",
            "sigma_Pr_mag",
            "sigma_A_F090W",
        ]
    ].rename(columns={"sigma_A_F090W": "sigma_extinction_mag"})
    measurement_components = [
        ("Power spectrum", "sigma_measurement_mag"),
        ("Background", "sigma_sky_mag"),
        ("PSF", "sigma_psf_mag"),
        (r"Unresolved sources $P_r$", "sigma_Pr_mag"),
        ("Foreground extinction", "sigma_extinction_mag"),
    ]
    make_stacked_variance(
        f150_measurement,
        measurement_components,
        "F150W SBF measurement error budget",
        F150_FIGURES,
        "go3055_f150w_article_measurement_error_budget",
    )
    make_stacked_variance(
        f090_measurement,
        measurement_components,
        "F090W SBF measurement error budget",
        F090_FIGURES,
        "go3055_f090w_article_measurement_error_budget",
    )

    f150_predictive = f150_measurement.merge(
        f150_loo[
            ["galaxy", "sigma_intrinsic_mag", "sigma_calibration_prediction_mag"]
        ].rename(columns={"sigma_calibration_prediction_mag": "sigma_calibration_mag"}),
        on="galaxy",
        how="inner",
    )
    f150_predictive["sigma_color_measurement_mag"] = 0.0
    f150_predictive["sigma_common_trgb_mag"] = 0.047
    f090_predictive = f090_loo.copy()
    predictive_components = [
        ("Power spectrum", "sigma_measurement_mag"),
        ("Background", "sigma_sky_mag"),
        ("PSF", "sigma_psf_mag"),
        (r"Unresolved sources $P_r$", "sigma_Pr_mag"),
        ("Foreground extinction", "sigma_extinction_mag"),
        ("Color", "sigma_color_measurement_mag"),
        ("Intrinsic scatter", "sigma_intrinsic_mag"),
        ("Finite calibration", "sigma_calibration_mag"),
        ("Shared TRGB scale", "sigma_common_trgb_mag"),
    ]
    make_stacked_variance(
        f150_predictive,
        predictive_components,
        "F150W full distance-error budget",
        F150_FIGURES,
        "go3055_f150w_article_distance_error_budget",
    )
    make_stacked_variance(
        f090_predictive,
        predictive_components,
        "F090W full distance-error budget",
        F090_FIGURES,
        "go3055_f090w_article_distance_error_budget",
    )

    print("Publication figures written:")
    for directory in (F150_FIGURES, F090_FIGURES):
        for path in sorted(directory.glob("*article*.pdf")):
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
