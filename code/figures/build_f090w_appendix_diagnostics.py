#!/usr/bin/env python3
"""Диагностика из кэша / Build diagnostics from saved GO-3055 products.

The two passbands share plotting conventions; no source extraction, isophotal
fitting, or SBF measurement is rerun. Histograms read galaxy pixels; PSF checks
read the small saved PSF stamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures.publish_article_assets import publish_figure
from figures.pasa_style import WIDTH_INCH as PASA_WIDTH, TICK_PT, LABEL_PT, TITLE_PT
from sbf.sbf_paths import PROJECT_ROOT, load_project_json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib.patches import Patch, Rectangle


ROOT = PROJECT_ROOT
RUN = ROOT / "runs/F090W"
PRODUCTS = RUN / "products"
FIGURES = RUN / "analysis/figures"
PASA_FIGURES = ROOT / "texts/paper_work/materials/pasa/figures"
# All 14 calibrators; representative targets span clean/structured residuals.
GALAXIES = [
    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 1549",
    "NGC 3379", "NGC 4374", "NGC 4406", "NGC 4472",
    "NGC 4486", "NGC 4552", "NGC 4621", "NGC 4636",
    "NGC 4649", "NGC 4697",
]
REPRESENTATIVE = ["NGC 3379", "NGC 1380", "NGC 4486"]
ADOPTED_BRANCH = "normalized_full_3p5"  # Accepted operation order, not refitted here.
KMIN, KMAX = 0.04, 0.25  # Published spectral window, excluding strongest low-k structure.


# Sampling radii reach the edge of the adopted 129-pixel PSF stamp.
PSF_RADII_PIXEL = np.array([1, 2, 3, 5, 10, 20, 32, 48, 64], dtype=float)
# Histogram colors mark nested thresholds, not separate fitted populations.
zone_colors = {"blue": "#2563eb", "red": "#dc2626", "yellow": "#facc15", "orange": "#f97316"}


def save_figure(fig, stem, directory=FIGURES, *, pasa=False):
    """Сохранить выбранный рисунок / Save a figure and refresh its article copy."""
    if pasa:
        PASA_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(PASA_FIGURES / f"{stem}.pdf")
        plt.close(fig)
        return
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        publish_figure(path)
    plt.close(fig)


def normalized_values(source, expected_limits):
    """Восстановить нормированные значения / Reconstruct and verify saved support."""
    output_dir = Path(source["output_dir"])
    mask_path = output_dir / f"{source['stem']}_sbf_catalog_mask_mcut.fits"
    with fits.open(source["signal_path"], memmap=True) as signal_hdul, \
            fits.open(source["model_full_fits"], memmap=True) as model_hdul, \
            fits.open(mask_path, memmap=True) as mask_hdul:
        science = signal_hdul["SCI"].data
        model = model_hdul[0].data
        mask = np.asarray(mask_hdul[0].data, dtype=bool)
        valid = ~mask & np.isfinite(science) & np.isfinite(model) & (model > 0)
        model_values = np.asarray(model[valid], dtype=np.float64)
        residual = np.asarray(
            np.asarray(science[valid], dtype=np.float64)
            - float(source["signal_background_scalar"])
            - model_values,
            dtype=np.float32,
        )
        values = residual / np.sqrt(model_values)

    # Five iterations match the production robust threshold estimator. 2e-7
    # allows float32 reconstruction round-off, not a physical extra uncertainty.
    thresholds = {}
    for sigma in (3.0, 3.5, 4.0):
        _, median, scale = sigma_clipped_stats(values, sigma=sigma, maxiters=5)
        thresholds[sigma] = (float(median - sigma * scale), float(median + sigma * scale))
    if (
        values.size != int(expected_limits["n_pixels"])
        or not np.isclose(thresholds[3.5][0], expected_limits["lower"], atol=2e-7, rtol=0)
        or not np.isclose(thresholds[3.5][1], expected_limits["upper"], atol=2e-7, rtol=0)
    ):
        raise RuntimeError("Reconstructed normalized histogram does not match saved limits")
    return values, thresholds



def draw_histogram(ax, values, thresholds, band, *, pasa=False):
    """Равные бины, точные цветовые пороги / Equal bins, exact colour thresholds.

    Цвет меняется внутри бина без пересчёта его высоты: Y — число пикселей
    во всём бине, не в цветной части. / A colour boundary can cross a bin;
    its height remains the whole-bin pixel count, not a sub-bin count.
    """
    low35, high35 = thresholds[3.5]
    center = 0.5 * (low35 + high35)
    scale = (high35 - low35) / 7.0
    # ±6 sigma and 150 bins keep the tail overview readable. Values outside the
    # plotted range enter edge bins; the scientific residual is never modified.
    shown_min, shown_max = center - 6 * scale, center + 6 * scale
    shown = np.clip(values, shown_min, shown_max)
    edges = np.linspace(shown_min, shown_max, 151)
    counts, edges = np.histogram(shown, bins=edges)
    ax.stairs(counts, edges, fill=True, color=zone_colors["orange"], linewidth=0)
    # Clip identical coloured layers, not statistical bins. Inserting threshold
    # edges into the histogram would create narrow bins and false count dips.
    for sigma, colour in ((4.0, "yellow"), (3.5, "red"), (3.0, "blue")):
        layer = ax.stairs(counts, edges, fill=True, color=zone_colors[colour], linewidth=0)
        low, high = np.clip(thresholds[sigma], shown_min, shown_max)
        layer.set_clip_path(Rectangle(
            (low, 0), high - low, 1, transform=ax.get_xaxis_transform(),
        ))
    affected = 100 * np.mean((values < low35) | (values > high35))
    ax.text(
        0.98, 0.92, rf"$3.5\sigma$ affected: {affected:.2f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    ax.set(
        title=band,
        xlabel=(r"Normalised residual [$(\mathrm{MJy\ sr}^{-1})^{1/2}$]"
                if pasa else r"Normalized residual [$(\mathrm{MJy\ sr}^{-1})^{1/2}$]"),
        ylabel="Number of pixels",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))



def make_power_spectrum_comparison(result, band, output_directory, stem, *, pasa=False):
    """Сравнить готовые P(k) / Plot saved fits without fitting again."""
    spectra = pd.read_csv(result["table_paths"]["power_spectra"])
    fit_summary = pd.read_csv(result["table_paths"]["fit_summary"])
    fig, axes = plt.subplots(
        2, 2, figsize=(PASA_WIDTH, 4.9) if pasa else (11.0, 7.0), sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    for column, ring in enumerate(("inner", "outer")):
        top, bottom = axes[0, column], axes[1, column]
        for branch, point_color, fit_color, label in [
            ("no_winsor", "0.65", "0.45", "No winsorisation" if pasa else "No winsorization"),
            (ADOPTED_BRANCH, "black", "#dc2626", r"Adopted $3.5\sigma$"),
        ]:
            data = spectra[
                spectra["ring"].eq(ring) & spectra["branch"].eq(branch)
            ].sort_values("k")
            fit = fit_summary[
                fit_summary["ring"].eq(ring)
                & fit_summary["branch"].eq(branch)
                & np.isclose(fit_summary["requested_kmin"], KMIN)
            ].iloc[0]
            selected = data["k"].between(KMIN, KMAX)
            k = data.loc[selected, "k"].to_numpy(float)
            power = data.loc[selected, "Pk"].to_numpy(float)
            error = data.loc[selected, "Pk_error"].to_numpy(float)
            expectation = data.loc[selected, "E_median"].to_numpy(float)
            model = float(fit["P0"]) * expectation + float(fit["P1"])
            top.errorbar(
                k, power, yerr=error, fmt="o", markersize=3.2,
                color=point_color, ecolor=point_color, elinewidth=0.7,
                capsize=0, label=label, zorder=2,
            )
            top.plot(k, model, color=fit_color, linewidth=1.8, zorder=3)
            bottom.plot(k, (power - model) / error, color=fit_color, linewidth=1.2)
        top.set_title(f"{ring.capitalize()} annulus")
        top.set_ylabel(r"$P(k)$")
        bottom.axhline(0, color="0.35", linewidth=0.8)
        bottom.set(xlabel=r"$k$ (pixel$^{-1}$)", ylabel="Residual / error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"NGC 1380: {band} power-spectrum fits", fontsize=10 if pasa else 15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, stem, output_directory, pasa=pasa)



def annular_weights(result, galaxy):
    """Принятые обратные дисперсии / Normalize the saved inverse-variance weights."""
    combined = pd.read_csv(result["table_paths"]["combined_annuli"])
    row = combined[
        combined["galaxy"].eq(galaxy)
        & combined["branch"].eq(ADOPTED_BRANCH)
        & np.isclose(combined["requested_kmin"], KMIN)
    ].iloc[0]
    weights = np.array([1 / row["sigma_inner"] ** 2, 1 / row["sigma_outer"] ** 2])
    return weights / weights.sum()



def psf_normalization_sensitivity(
    results, psf_paths, weights_by_galaxy, band, size_table_path, size_column, output_directory, stem,
    *, pasa=False,
):
    """Энергия и сдвиги PSF / Encircled energy and saved PSF-induced shifts."""
    growth = []
    field_shifts = []
    for galaxy in GALAXIES:
        with fits.open(psf_paths[galaxy], memmap=False) as hdul:
            for hdu in hdul[1:]:
                psf_image = np.asarray(hdu.data, dtype=float)
                yy, xx = np.indices(psf_image.shape)
                rr = np.hypot(
                    xx - (psf_image.shape[1] - 1) / 2,
                    yy - (psf_image.shape[0] - 1) / 2,
                )
                total = psf_image.sum()
                growth.append([
                    psf_image[rr <= radius].sum() / total for radius in PSF_RADII_PIXEL
                ])

        fit_table = pd.read_csv(results[galaxy]["table_paths"]["fit_per_psf"])
        selected = fit_table[
            fit_table["branch"].eq(ADOPTED_BRANCH)
            & np.isclose(fit_table["requested_kmin"], KMIN)
        ]
        nominal_id = selected.loc[
            ~selected["psf_id"].str.contains("field"), "psf_id"
        ].iloc[0]
        weights = weights_by_galaxy[galaxy]
        nominal = selected[selected["psf_id"].eq(nominal_id)].set_index("ring")
        nominal_mbar = np.dot(weights, nominal.loc[["inner", "outer"], "mbar"])
        for psf_id in selected.loc[
            selected["psf_id"].str.contains("field"), "psf_id"
        ].unique():
            field = selected[selected["psf_id"].eq(psf_id)].set_index("ring")
            field_mbar = np.dot(weights, field.loc[["inner", "outer"], "mbar"])
            field_shifts.append(field_mbar - nominal_mbar)

    growth = np.asarray(growth)
    size_table = pd.read_csv(size_table_path)
    fig, axes = plt.subplots(1, 3, figsize=(PASA_WIDTH, 2.9) if pasa else (13.5, 4.3))
    axes[0].fill_between(
        PSF_RADII_PIXEL, growth.min(axis=0), growth.max(axis=0),
        color="#93c5fd", alpha=0.55, label=f"range of {len(growth)} PSFs",
    )
    axes[0].plot(PSF_RADII_PIXEL, np.median(growth, axis=0), "o-", color="black", linewidth=1.3)
    axes[0].set_xscale("log")
    axes[0].set(
        title=f"{band}: encircled energy" if pasa else f"{band} STPSF curve of growth",
        xlabel="Radius (pixel)", ylabel="Encircled energy", ylim=(0.5, 1.01),
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(size_table["galaxy"], size_table[size_column], color="#ef4444", width=0.65)
    axes[1].axhline(0, color="0.35", linewidth=0.8)
    wavelength = "150" if band == "F150W" else "090"
    axes[1].set(
        title="129 versus 257 pixels" if pasa else "Finite stamp-size test",
        xlabel="Galaxy",
        ylabel=(rf"$\Delta\overline{{m}}_{{{wavelength}}}$ (mag)" if pasa
                else rf"$\Delta\overline{{m}}_{{{wavelength}}}$: 129 minus 257 (mag)"),
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].hist(field_shifts, bins=12, color="#8b5cf6", edgecolor="white")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set(
        title="Detector position" if pasa else "Detector-position sensitivity",
        xlabel=(rf"$\Delta\overline{{m}}_{{{wavelength}}}$ (mag)" if pasa
                else rf"$\Delta\overline{{m}}_{{{wavelength}}}$ from nominal PSF (mag)"),
        ylabel="Number of field PSFs",
    )
    fig.tight_layout()
    save_figure(fig, stem, output_directory, pasa=pasa)



def main(argv=None):
    """Построить семь используемых рисунков / Render seven selected diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasa", action="store_true", help="PASA PDF only; no legacy figures, tables or manifests are changed")
    args = parser.parse_args(argv)
    if args.pasa:
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": LABEL_PT,
            "axes.labelsize": LABEL_PT, "axes.titlesize": TITLE_PT,
            "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": TICK_PT,
            "axes.linewidth": 0.7, "pdf.fonttype": 42,
        })
    results = {"F150W": {}, "F090W": {}}
    sources = {"F150W": {}, "F090W": {}}
    psf_paths = {"F150W": {}, "F090W": {}}
    weights = {"F150W": {}, "F090W": {}}
    master090 = pd.read_csv(RUN / "analysis/tables/go3055_f090w_master.csv").set_index("galaxy")

    for galaxy in GALAXIES:
        slug = galaxy.replace(" ", "_")
        manifest = load_project_json(PRODUCTS / slug / "products.json")
        result_paths = {
            "F090W": manifest["final_result"],
            "F150W": ROOT / "runs/F150W/spectra/batch" / f"{slug}_result.json",
        }
        for band, result_path in result_paths.items():
            result = load_project_json(result_path)
            if result["status"] != "ok" or result["candidate_branch"] != ADOPTED_BRANCH:
                raise RuntimeError(f"{galaxy}: unexpected {band} final result state")
            results[band][galaxy] = result
            sources[band][galaxy] = load_project_json(result["source_result"])

        psf_paths["F090W"][galaxy] = manifest["products"]["psf_129"]
        candidates = sorted(Path(sources["F150W"][galaxy]["output_dir"]).glob("*_psf_129.fits"))
        if len(candidates) != 1:
            raise RuntimeError(f"{galaxy}: expected one F150W 129-pixel PSF library")
        psf_paths["F150W"][galaxy] = candidates[0]
        weights["F150W"][galaxy] = annular_weights(results["F150W"][galaxy], galaxy)
        # Keep exactly the published F090W weights (not a fresh estimate).
        weights["F090W"][galaxy] = master090.loc[galaxy, ["weight_inner", "weight_outer"]].to_numpy(float)

    legend = [
        Patch(color=zone_colors["blue"], label=r"retained at $3\sigma$"),
        Patch(color=zone_colors["red"], label=r"capped at $3\sigma$"),
        Patch(color=zone_colors["yellow"], label=r"capped at $3.5\sigma$"),
        Patch(color=zone_colors["orange"], label=r"capped at $4\sigma$"),
    ]
    for galaxy in REPRESENTATIVE:
        fig, axes = plt.subplots(2, 1, figsize=(PASA_WIDTH, 5.6) if args.pasa else (9.0, 8.0))
        for ax, band in zip(axes, ("F150W", "F090W")):
            values, thresholds = normalized_values(
                sources[band][galaxy], results[band][galaxy]["candidate_limits"]
            )
            draw_histogram(ax, values, thresholds, band, pasa=args.pasa)
            del values
        axes[0].legend(handles=legend, frameon=False, ncol=2, fontsize=8)
        fig.suptitle(
            f"{galaxy}: normalised full-support pixel distributions" if args.pasa
            else f"{galaxy}: normalized full-support pixel distributions",
            fontsize=10 if args.pasa else 15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        stem = galaxy.lower().replace(" ", "_")
        save_figure(fig, f"{stem}_f150w_f090w_normalized_full_pixel_histograms", pasa=args.pasa)

    settings = (
        ("F150W", ROOT / "runs/F150W/analysis",
         "go3055_psf_129_vs_257_sensitivity.csv", "delta_mbar_mag"),
        ("F090W", RUN / "analysis",
         "go3055_f090w_psf_129_vs_257.csv", "delta_mbar_129_minus_257_mag"),
    )
    for band, analysis, size_table, size_column in settings:
        make_power_spectrum_comparison(
            results[band]["NGC 1380"], band, analysis / "figures",
            f"ngc_1380_{band.lower()}_pk_fit_comparison",
            pasa=args.pasa,
        )
        psf_normalization_sensitivity(
            results[band], psf_paths[band], weights[band], band,
            analysis / "tables" / size_table, size_column, analysis / "figures",
            f"go3055_{band.lower()}_psf_normalization_sensitivity",
            pasa=args.pasa,
        )
    print("Built three paired histograms and two P(k)/PSF comparisons from saved products.")


if __name__ == "__main__":
    main()
