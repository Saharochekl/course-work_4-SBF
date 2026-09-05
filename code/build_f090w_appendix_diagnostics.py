#!/usr/bin/env python3
"""Build F090W pipeline diagnostics from already saved GO-3055 products."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/sbf_f090w_go3055"
PRODUCTS = RUN / "products"
FIGURES = RUN / "analysis/figures"
GALAXIES = [
    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 1549",
    "NGC 3379", "NGC 4374", "NGC 4406", "NGC 4472",
    "NGC 4486", "NGC 4552", "NGC 4621", "NGC 4636",
    "NGC 4649", "NGC 4697",
]
REPRESENTATIVE = ["NGC 3379", "NGC 1380", "NGC 4486"]
ADOPTED_BRANCH = "normalized_full_3p5"
KMIN, KMAX = 0.04, 0.25


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


FIGURES.mkdir(parents=True, exist_ok=True)
manifests = {}
final_results = {}
for galaxy in GALAXIES:
    manifest_path = PRODUCTS / galaxy.replace(" ", "_") / "products.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = Path(manifest["final_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok" or result.get("candidate_branch") != ADOPTED_BRANCH:
        raise RuntimeError(f"{galaxy}: unexpected final result state")
    manifests[galaxy] = manifest
    final_results[galaxy] = result


# 1. Direct all-galaxy analogue of the adopted F150W winsorization diagnostic.
clipping_frames = []
annulus_frames = []
for galaxy in GALAXIES:
    paths = final_results[galaxy]["table_paths"]
    clipping_frames.append(pd.read_csv(paths["clipping"]))
    annulus_frames.append(pd.read_csv(paths["combined_annuli"]))
clipping = pd.concat(clipping_frames, ignore_index=True)
annuli = pd.concat(annulus_frames, ignore_index=True)

clip_adopted = clipping[clipping["branch"].eq(ADOPTED_BRANCH)].copy()
mbar_adopted = annuli[
    annuli["branch"].eq(ADOPTED_BRANCH)
    & np.isclose(annuli["requested_kmin"], KMIN)
].copy()
mbar_none = annuli[
    annuli["branch"].eq("no_winsor")
    & np.isclose(annuli["requested_kmin"], KMIN)
].copy()

rows = []
for galaxy in GALAXIES:
    adopted = mbar_adopted[mbar_adopted["galaxy"].eq(galaxy)].iloc[0]
    no_winsor = mbar_none[mbar_none["galaxy"].eq(galaxy)].iloc[0]
    for ring in ("inner", "outer"):
        clip_row = clip_adopted[
            clip_adopted["galaxy"].eq(galaxy)
            & clip_adopted["ring"].eq(ring)
        ].iloc[0]
        rows.append({
            "galaxy": galaxy,
            "ring": ring,
            "changed_percent": 100 * float(clip_row["changed_fraction"]),
            "delta_mbar": float(adopted[f"mbar_{ring}"] - no_winsor[f"mbar_{ring}"]),
        })
winsor = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
colors = {"inner": "#2563eb", "outer": "#f59e0b"}
for ring in ("inner", "outer"):
    part = winsor[winsor["ring"].eq(ring)]
    axes[0].scatter(
        part["changed_percent"], part["delta_mbar"],
        s=48, color=colors[ring], label=ring.capitalize(), zorder=3,
    )
    for row in part.itertuples(index=False):
        axes[0].annotate(
            row.galaxy.replace("NGC ", ""),
            (row.changed_percent, row.delta_mbar),
            xytext=(3, 3), textcoords="offset points", fontsize=7,
        )
axes[0].axhline(0, color="0.35", linewidth=0.8)
axes[0].set(
    title=r"Normalized-space $3.5\sigma$ sensitivity",
    xlabel="Winsorized pixels (%)",
    ylabel=r"$\Delta\overline{m}_{090}$: adopted minus no winsor (mag)",
)
axes[0].legend(frameon=False)

x = np.arange(len(GALAXIES))
width = 0.38
for offset, ring in [(-width / 2, "inner"), (width / 2, "outer")]:
    values = (
        winsor[winsor["ring"].eq(ring)]
        .set_index("galaxy").loc[GALAXIES, "changed_percent"]
    )
    axes[1].bar(x + offset, values, width, color=colors[ring], label=ring.capitalize())
axes[1].set(
    title="Fraction of winsorized values",
    ylabel="Winsorized pixels (%)",
    xticks=x,
    xticklabels=GALAXIES,
)
axes[1].tick_params(axis="x", rotation=55, labelsize=8)
axes[1].legend(frameon=False)
fig.tight_layout()
save_figure(fig, "go3055_f090w_winsorization_diagnostics")


# 2. Actual normalized-space pixel distributions for three representative targets.
zone_colors = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "yellow": "#facc15",
    "orange": "#f97316",
}
legend = [
    Patch(color=zone_colors["blue"], label=r"retained at $3\sigma$"),
    Patch(color=zone_colors["red"], label=r"capped at $3\sigma$"),
    Patch(color=zone_colors["yellow"], label=r"capped at $3.5\sigma$"),
    Patch(color=zone_colors["orange"], label=r"capped at $4\sigma$"),
]

for galaxy in REPRESENTATIVE:
    manifest = manifests[galaxy]
    source = json.loads(Path(manifest["source_result"]).read_text(encoding="utf-8"))
    out_dir = Path(source["output_dir"])
    mask_path = out_dir / f"{source['stem']}_sbf_catalog_mask_mcut.fits"

    with fits.open(source["signal_path"], memmap=True) as signal_hdul, \
            fits.open(source["model_full_fits"], memmap=True) as model_hdul, \
            fits.open(mask_path, memmap=True) as mask_hdul:
        science = signal_hdul["SCI"].data
        model = model_hdul[0].data
        catalog_mask = mask_hdul[0].data
        valid = (
            ~np.asarray(catalog_mask, dtype=bool)
            & np.isfinite(science)
            & np.isfinite(model)
            & (model > 0)
        )
        model_values = np.asarray(model[valid], dtype=np.float64)
        raw = np.asarray(
            (np.asarray(science[valid], dtype=np.float64)
             - float(source["signal_background_scalar"]))
            - model_values,
            dtype=np.float32,
        )
        normalized = raw / np.sqrt(model_values)

    limits = {}
    for sigma in (3.0, 3.5, 4.0):
        _, median, scale = sigma_clipped_stats(normalized, sigma=sigma, maxiters=5)
        limits[sigma] = (
            float(median - sigma * scale),
            float(median + sigma * scale),
        )

    expected = final_results[galaxy]["candidate_limits"]
    lower35, upper35 = limits[3.5]
    if (
        normalized.size != int(expected["n_pixels"])
        or not np.isclose(lower35, expected["lower"], rtol=0, atol=2e-7)
        or not np.isclose(upper35, expected["upper"], rtol=0, atol=2e-7)
    ):
        raise RuntimeError(f"{galaxy}: reconstructed normalized thresholds do not close")

    center = 0.5 * (lower35 + upper35)
    scale = (upper35 - lower35) / 7.0
    shown_min, shown_max = center - 6 * scale, center + 6 * scale
    shown = np.clip(normalized, shown_min, shown_max)
    sigma_edges = np.array([
        limits[4.0][0], lower35, limits[3.0][0],
        limits[3.0][1], upper35, limits[4.0][1],
    ])
    regular_edges = np.linspace(shown_min, shown_max, 181)
    edges = np.unique(np.r_[regular_edges, sigma_edges])
    edges = edges[(edges >= shown_min) & (edges <= shown_max)]
    counts, edges = np.histogram(shown, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    zones = np.full(centers.size, "orange", dtype=object)
    zones[(centers >= limits[4.0][0]) & (centers <= limits[4.0][1])] = "yellow"
    zones[(centers >= lower35) & (centers <= upper35)] = "red"
    zones[(centers >= limits[3.0][0]) & (centers <= limits[3.0][1])] = "blue"
    affected = 100 * np.mean((normalized < lower35) | (normalized > upper35))

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.bar(
        centers, counts, width=np.diff(edges),
        color=[zone_colors[zone] for zone in zones], linewidth=0,
    )
    ax.set(
        title=f"{galaxy}: F090W normalized full support",
        xlabel=r"Normalized residual [$(\mathrm{MJy\ sr}^{-1})^{1/2}$]",
        ylabel="Number of pixels",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(
        0.98, 0.96, rf"Pixels capped at $3.5\sigma$: {affected:.2f}%",
        transform=ax.transAxes, ha="right", va="top",
    )
    ax.legend(handles=legend, frameon=False)
    fig.tight_layout()
    stem = galaxy.lower().replace(" ", "_")
    save_figure(fig, f"{stem}_f090w_normalized_full_pixel_histogram")


# 3. Saved P(k) fits: adopted normalized-full branch versus no winsorization.
galaxy = "NGC 1380"
paths = final_results[galaxy]["table_paths"]
spectra = pd.read_csv(paths["power_spectra"])
fits_table = pd.read_csv(paths["fit_summary"])

fig, axes = plt.subplots(
    2, 2, figsize=(11.0, 7.0), sharex="col",
    gridspec_kw={"height_ratios": [3.0, 1.25]},
)
for column, ring in enumerate(("inner", "outer")):
    top, bottom = axes[0, column], axes[1, column]
    for branch, point_color, fit_color, label, marker in [
        ("no_winsor", "0.65", "0.45", "No winsorization", "o"),
        (ADOPTED_BRANCH, "black", "#dc2626", r"Adopted $3.5\sigma$", "o"),
    ]:
        data = spectra[
            spectra["ring"].eq(ring) & spectra["branch"].eq(branch)
        ].sort_values("k")
        fit = fits_table[
            fits_table["ring"].eq(ring)
            & fits_table["branch"].eq(branch)
            & np.isclose(fits_table["requested_kmin"], KMIN)
        ].iloc[0]
        selected = data["k"].between(KMIN, KMAX)
        k = data.loc[selected, "k"].to_numpy(float)
        pk = data.loc[selected, "Pk"].to_numpy(float)
        error = data.loc[selected, "Pk_error"].to_numpy(float)
        expectation = data.loc[selected, "E_median"].to_numpy(float)
        model = float(fit["P0"]) * expectation + float(fit["P1"])

        top.errorbar(
            k, pk, yerr=error, fmt=marker, markersize=3.2,
            color=point_color, ecolor=point_color, elinewidth=0.7,
            capsize=0, label=label, zorder=2,
        )
        top.plot(k, model, color=fit_color, linewidth=1.8, zorder=3)
        bottom.plot(k, (pk - model) / error, color=fit_color, linewidth=1.2)

    top.set_title(f"{ring.capitalize()} annulus")
    top.set_ylabel(r"$P(k)$")
    bottom.axhline(0, color="0.35", linewidth=0.8)
    bottom.set(xlabel=r"$k$ (pixel$^{-1}$)", ylabel="Residual / error")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("NGC 1380: F090W power-spectrum fits", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(fig, "ngc_1380_f090w_pk_fit_comparison")


# 4. Direct measurement-level P_r sensitivity from saved P0 and P_r values.
master = pd.read_csv(RUN / "analysis/tables/go3055_f090w_master.csv").set_index("galaxy")
pr_rows = []
for galaxy in GALAXIES:
    fit_table = pd.read_csv(final_results[galaxy]["table_paths"]["fit_summary"])
    selected = fit_table[
        fit_table["branch"].eq(ADOPTED_BRANCH)
        & np.isclose(fit_table["requested_kmin"], KMIN)
    ].set_index("ring")
    weights = {
        "inner": float(master.loc[galaxy, "weight_inner"]),
        "outer": float(master.loc[galaxy, "weight_outer"]),
    }
    weighted = {}
    for multiplier in (0.0, 1.0, 2.0):
        value = 0.0
        for ring in ("inner", "outer"):
            row = selected.loc[ring]
            p0, pr, mbar = float(row["P0"]), float(row["Pr"]), float(row["mbar"])
            zero_point = mbar + 2.5 * np.log10(p0 - pr)
            value += weights[ring] * (-2.5 * np.log10(p0 - multiplier * pr) + zero_point)
        weighted[multiplier] = value
    pr_rows.append({
        "galaxy": galaxy,
        "zero": weighted[0.0] - weighted[1.0],
        "double": weighted[2.0] - weighted[1.0],
    })
pr_sensitivity = pd.DataFrame(pr_rows).set_index("galaxy").loc[GALAXIES]

fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(GALAXIES))
ax.plot(x, 1000 * pr_sensitivity["zero"], "o-", color="#2563eb", label=r"$P_r=0$")
ax.plot(x, 1000 * pr_sensitivity["double"], "s-", color="#dc2626", label=r"$P_r=2P_r^{\rm adopted}$")
ax.axhline(0, color="0.35", linewidth=0.8)
ax.set(
    title=r"F090W measurement sensitivity to unresolved-source power $P_r$",
    ylabel=r"Change in weighted $\overline{m}_{090}$ (mmag)",
    xticks=x,
    xticklabels=GALAXIES,
)
ax.tick_params(axis="x", rotation=55, labelsize=8)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
save_figure(fig, "go3055_f090w_Pr_measurement_sensitivity")

print("Built F090W appendix diagnostics from saved products only:")
print("  all-galaxy winsorization diagnostic")
print("  three normalized pixel histograms")
print("  NGC 1380 saved P(k) fit comparison")
print("  direct P_r measurement sensitivity")


# 5. Strictly matched F150W/F090W winsorization panels.
def winsor_frame(clipping_table, annuli_table):
    clip = clipping_table[clipping_table["branch"].eq(ADOPTED_BRANCH)]
    adopted = annuli_table[
        annuli_table["branch"].eq(ADOPTED_BRANCH)
        & np.isclose(annuli_table["requested_kmin"], KMIN)
    ].set_index("galaxy")
    plain = annuli_table[
        annuli_table["branch"].eq("no_winsor")
        & np.isclose(annuli_table["requested_kmin"], KMIN)
    ].set_index("galaxy")
    records = []
    for target in GALAXIES:
        for ring in ("inner", "outer"):
            row = clip[clip["galaxy"].eq(target) & clip["ring"].eq(ring)].iloc[0]
            records.append({
                "galaxy": target,
                "ring": ring,
                "changed_percent": 100 * float(row["changed_fraction"]),
                "delta_mbar": float(
                    adopted.loc[target, f"mbar_{ring}"]
                    - plain.loc[target, f"mbar_{ring}"]
                ),
            })
    return pd.DataFrame(records)


f150_base = ROOT / "runs/sbf2_normalized_winsor/batch/aggregates"
f150_winsor = winsor_frame(
    pd.read_csv(f150_base / "all_galaxies_clipping.csv"),
    pd.read_csv(f150_base / "all_galaxies_combined_annuli.csv"),
)
paired_winsor = {"F150W": f150_winsor, "F090W": winsor}

fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), sharex="col", sharey="col")
for row_index, (band, table) in enumerate(paired_winsor.items()):
    scatter_ax, fraction_ax = axes[row_index]
    for ring in ("inner", "outer"):
        part = table[table["ring"].eq(ring)]
        scatter_ax.scatter(
            part["changed_percent"], part["delta_mbar"],
            s=42, color=colors[ring], label=ring.capitalize(), zorder=3,
        )
        for point in part.itertuples(index=False):
            if abs(point.delta_mbar) >= 0.02:
                scatter_ax.annotate(
                    point.galaxy.replace("NGC ", ""),
                    (point.changed_percent, point.delta_mbar),
                    xytext=(3, 3), textcoords="offset points", fontsize=7,
                )
        values = part.set_index("galaxy").loc[GALAXIES, "changed_percent"]
        offset = -width / 2 if ring == "inner" else width / 2
        fraction_ax.bar(
            x + offset, values, width, color=colors[ring], label=ring.capitalize(),
        )
    scatter_ax.axhline(0, color="0.35", linewidth=0.8)
    scatter_ax.set_ylabel(
        rf"{band}  $\Delta\overline{{m}}$" + "\n(adopted minus none; mag)"
    )
    fraction_ax.set_ylabel("Winsorized pixels (%)")
    scatter_ax.legend(frameon=False)
    fraction_ax.legend(frameon=False)

axes[0, 0].set_title(r"Effect of normalized-space $3.5\sigma$ winsorization")
axes[0, 1].set_title("Fraction of affected pixels")
axes[1, 0].set_xlabel("Winsorized pixels (%)")
axes[1, 1].set(xticks=x, xticklabels=GALAXIES)
axes[1, 1].tick_params(axis="x", rotation=55, labelsize=8)
fig.tight_layout()
save_figure(fig, "go3055_f150w_f090w_winsorization_diagnostics")


# 6. Same normalized-space pixel diagnostic in both filters.
def normalized_values(source, expected_limits):
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


def draw_histogram(ax, values, thresholds, band):
    low35, high35 = thresholds[3.5]
    center = 0.5 * (low35 + high35)
    scale = (high35 - low35) / 7.0
    shown_min, shown_max = center - 6 * scale, center + 6 * scale
    shown = np.clip(values, shown_min, shown_max)
    exact_edges = np.array([
        thresholds[4.0][0], low35, thresholds[3.0][0],
        thresholds[3.0][1], high35, thresholds[4.0][1],
    ])
    edges = np.unique(np.r_[np.linspace(shown_min, shown_max, 151), exact_edges])
    edges = edges[(edges >= shown_min) & (edges <= shown_max)]
    counts, edges = np.histogram(shown, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    zones = np.full(centers.size, "orange", dtype=object)
    zones[(centers >= thresholds[4.0][0]) & (centers <= thresholds[4.0][1])] = "yellow"
    zones[(centers >= low35) & (centers <= high35)] = "red"
    zones[(centers >= thresholds[3.0][0]) & (centers <= thresholds[3.0][1])] = "blue"
    ax.bar(
        centers, counts, width=np.diff(edges),
        color=[zone_colors[zone] for zone in zones], linewidth=0,
    )
    affected = 100 * np.mean((values < low35) | (values > high35))
    ax.text(
        0.98, 0.92, rf"$3.5\sigma$ affected: {affected:.2f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    ax.set(
        title=band,
        xlabel=r"Normalized residual [$(\mathrm{MJy\ sr}^{-1})^{1/2}$]",
        ylabel="Number of pixels",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


for galaxy in REPRESENTATIVE:
    f150_result = json.loads(
        (ROOT / "runs/sbf2_normalized_winsor/batch"
         / f"{galaxy.replace(' ', '_')}_result.json").read_text(encoding="utf-8")
    )
    sources = {
        "F150W": json.loads(Path(f150_result["source_result"]).read_text(encoding="utf-8")),
        "F090W": json.loads(Path(manifests[galaxy]["source_result"]).read_text(encoding="utf-8")),
    }
    saved_limits = {
        "F150W": f150_result["candidate_limits"],
        "F090W": final_results[galaxy]["candidate_limits"],
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0))
    for ax, band in zip(axes, ("F150W", "F090W")):
        values, thresholds = normalized_values(sources[band], saved_limits[band])
        draw_histogram(ax, values, thresholds, band)
        del values
    axes[0].legend(handles=legend, frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"{galaxy}: normalized full-support pixel distributions", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    stem = galaxy.lower().replace(" ", "_")
    save_figure(fig, f"{stem}_f150w_f090w_normalized_full_pixel_histograms")


# 7. F090W PSF normalization checks available from the saved products.
radii = np.array([1, 2, 3, 5, 10, 20, 32, 48, 64], dtype=float)
growth_curves = []
detector_shifts = []
for galaxy in GALAXIES:
    psf_path = Path(manifests[galaxy]["products"]["psf_129"])
    with fits.open(psf_path, memmap=False) as hdul:
        for hdu in hdul[1:]:
            image = np.asarray(hdu.data, dtype=float)
            yy, xx = np.indices(image.shape)
            rr = np.hypot(xx - (image.shape[1] - 1) / 2, yy - (image.shape[0] - 1) / 2)
            total = image.sum()
            growth_curves.append([image[rr <= radius].sum() / total for radius in radii])

    fit_table = pd.read_csv(final_results[galaxy]["table_paths"]["fit_per_psf"])
    selected = fit_table[
        fit_table["branch"].eq(ADOPTED_BRANCH)
        & np.isclose(fit_table["requested_kmin"], KMIN)
    ]
    nominal_id = selected.loc[~selected["psf_id"].str.contains("field"), "psf_id"].iloc[0]
    weights = master.loc[galaxy, ["weight_inner", "weight_outer"]].to_numpy(float)
    nominal = selected[selected["psf_id"].eq(nominal_id)].set_index("ring")
    nominal_mbar = np.dot(weights, nominal.loc[["inner", "outer"], "mbar"])
    for psf_id in selected.loc[selected["psf_id"].str.contains("field"), "psf_id"].unique():
        field = selected[selected["psf_id"].eq(psf_id)].set_index("ring")
        field_mbar = np.dot(weights, field.loc[["inner", "outer"], "mbar"])
        detector_shifts.append(field_mbar - nominal_mbar)

growth_curves = np.asarray(growth_curves)
size_test = pd.read_csv(RUN / "analysis/tables/go3055_f090w_psf_129_vs_257.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
axes[0].fill_between(
    radii, growth_curves.min(axis=0), growth_curves.max(axis=0),
    color="#93c5fd", alpha=0.55, label="range of 70 PSFs",
)
axes[0].plot(radii, np.median(growth_curves, axis=0), "o-", color="black", linewidth=1.3)
axes[0].set_xscale("log")
axes[0].set(
    title="F090W STPSF curve of growth",
    xlabel="Radius (pixel)", ylabel="Encircled energy", ylim=(0.5, 1.01),
)
axes[0].legend(frameon=False, fontsize=8)

axes[1].bar(
    size_test["galaxy"], size_test["delta_mbar_129_minus_257_mag"],
    color="#ef4444", width=0.65,
)
axes[1].axhline(0, color="0.35", linewidth=0.8)
axes[1].set(
    title="Finite stamp-size test",
    xlabel="Galaxy", ylabel=r"$\Delta\overline{m}_{090}$: 129 minus 257 (mag)",
)
axes[1].tick_params(axis="x", rotation=25)

axes[2].hist(detector_shifts, bins=12, color="#8b5cf6", edgecolor="white")
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set(
    title="Detector-position sensitivity",
    xlabel=r"$\Delta\overline{m}_{090}$ from nominal PSF (mag)",
    ylabel="Number of field PSFs",
)
fig.tight_layout()
save_figure(fig, "go3055_f090w_psf_normalization_sensitivity")

print("  matched F150W/F090W winsorization diagnostic")
print("  three matched F150W/F090W normalized pixel histograms")
print("  F090W PSF normalization sensitivity (70 PSFs; 129/257 test; 56 field shifts)")


# 8. Matched current-product F150W diagnostics for the article appendix.
f150_results = {}
f150_sources = {}
for galaxy in GALAXIES:
    result_path = (
        ROOT / "runs/sbf2_normalized_winsor/batch"
        / f"{galaxy.replace(' ', '_')}_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok" or result.get("candidate_branch") != ADOPTED_BRANCH:
        raise RuntimeError(f"{galaxy}: unexpected F150W final result state")
    f150_results[galaxy] = result
    f150_sources[galaxy] = json.loads(
        Path(result["source_result"]).read_text(encoding="utf-8")
    )


def make_power_spectrum_comparison(result, band, output_directory, stem):
    spectra = pd.read_csv(result["table_paths"]["power_spectra"])
    fit_summary = pd.read_csv(result["table_paths"]["fit_summary"])
    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 7.0), sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    for column, ring in enumerate(("inner", "outer")):
        top, bottom = axes[0, column], axes[1, column]
        for branch, point_color, fit_color, label in [
            ("no_winsor", "0.65", "0.45", "No winsorization"),
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
    fig.suptitle(f"NGC 1380: {band} power-spectrum fits", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


make_power_spectrum_comparison(
    f150_results["NGC 1380"],
    "F150W",
    ROOT / "runs/sbf2_go3055/analysis/figures",
    "ngc_1380_f150w_pk_fit_comparison",
)


def annular_weights(result, galaxy):
    combined = pd.read_csv(result["table_paths"]["combined_annuli"])
    row = combined[
        combined["galaxy"].eq(galaxy)
        & combined["branch"].eq(ADOPTED_BRANCH)
        & np.isclose(combined["requested_kmin"], KMIN)
    ].iloc[0]
    weights = np.array([1 / row["sigma_inner"] ** 2, 1 / row["sigma_outer"] ** 2])
    return weights / weights.sum()


def pr_measurement_sensitivity(results, band, output_directory, stem):
    rows = []
    for galaxy in GALAXIES:
        result = results[galaxy]
        fits_table = pd.read_csv(result["table_paths"]["fit_summary"])
        selected = fits_table[
            fits_table["branch"].eq(ADOPTED_BRANCH)
            & np.isclose(fits_table["requested_kmin"], KMIN)
        ].set_index("ring")
        weights = annular_weights(result, galaxy)
        weighted = {}
        for multiplier in (0.0, 1.0, 2.0):
            ring_values = []
            for ring in ("inner", "outer"):
                row = selected.loc[ring]
                p0, pr, mbar = float(row["P0"]), float(row["Pr"]), float(row["mbar"])
                zero_point = mbar + 2.5 * np.log10(p0 - pr)
                ring_values.append(-2.5 * np.log10(p0 - multiplier * pr) + zero_point)
            weighted[multiplier] = float(np.dot(weights, ring_values))
        rows.append({
            "galaxy": galaxy,
            "zero": weighted[0.0] - weighted[1.0],
            "double": weighted[2.0] - weighted[1.0],
        })

    sensitivity = pd.DataFrame(rows).set_index("galaxy").loc[GALAXIES]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    positions = np.arange(len(GALAXIES))
    ax.plot(
        positions, 1000 * sensitivity["zero"], "o-", color="#2563eb",
        label=r"$P_r=0$",
    )
    ax.plot(
        positions, 1000 * sensitivity["double"], "s-", color="#dc2626",
        label=r"$P_r=2P_r^{\rm adopted}$",
    )
    ax.axhline(0, color="0.35", linewidth=0.8)
    wavelength = "150" if band == "F150W" else "090"
    ax.set(
        title=rf"{band} measurement sensitivity to unresolved-source power $P_r$",
        ylabel=rf"Change in weighted $\overline{{m}}_{{{wavelength}}}$ (mmag)",
        xticks=positions,
        xticklabels=GALAXIES,
    )
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


pr_measurement_sensitivity(
    f150_results,
    "F150W",
    ROOT / "runs/sbf2_go3055/analysis/figures",
    "go3055_f150w_Pr_measurement_sensitivity",
)


def psf_normalization_sensitivity(
    results, sources, band, size_table_path, size_column, output_directory, stem,
):
    growth = []
    field_shifts = []
    for galaxy in GALAXIES:
        source = sources[galaxy]
        psf_candidates = sorted(Path(source["output_dir"]).glob("*_psf_129.fits"))
        if len(psf_candidates) != 1:
            raise RuntimeError(f"{galaxy}: expected one F150W 129-pixel PSF library")
        with fits.open(psf_candidates[0], memmap=False) as hdul:
            for hdu in hdul[1:]:
                psf_image = np.asarray(hdu.data, dtype=float)
                yy, xx = np.indices(psf_image.shape)
                rr = np.hypot(
                    xx - (psf_image.shape[1] - 1) / 2,
                    yy - (psf_image.shape[0] - 1) / 2,
                )
                total = psf_image.sum()
                growth.append([
                    psf_image[rr <= radius].sum() / total for radius in radii
                ])

        fit_table = pd.read_csv(results[galaxy]["table_paths"]["fit_per_psf"])
        selected = fit_table[
            fit_table["branch"].eq(ADOPTED_BRANCH)
            & np.isclose(fit_table["requested_kmin"], KMIN)
        ]
        nominal_id = selected.loc[
            ~selected["psf_id"].str.contains("field"), "psf_id"
        ].iloc[0]
        weights = annular_weights(results[galaxy], galaxy)
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    axes[0].fill_between(
        radii, growth.min(axis=0), growth.max(axis=0),
        color="#93c5fd", alpha=0.55, label=f"range of {len(growth)} PSFs",
    )
    axes[0].plot(radii, np.median(growth, axis=0), "o-", color="black", linewidth=1.3)
    axes[0].set_xscale("log")
    axes[0].set(
        title=f"{band} STPSF curve of growth",
        xlabel="Radius (pixel)", ylabel="Encircled energy", ylim=(0.5, 1.01),
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(size_table["galaxy"], size_table[size_column], color="#ef4444", width=0.65)
    axes[1].axhline(0, color="0.35", linewidth=0.8)
    wavelength = "150" if band == "F150W" else "090"
    axes[1].set(
        title="Finite stamp-size test",
        xlabel="Galaxy",
        ylabel=rf"$\Delta\overline{{m}}_{{{wavelength}}}$: 129 minus 257 (mag)",
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].hist(field_shifts, bins=12, color="#8b5cf6", edgecolor="white")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set(
        title="Detector-position sensitivity",
        xlabel=rf"$\Delta\overline{{m}}_{{{wavelength}}}$ from nominal PSF (mag)",
        ylabel="Number of field PSFs",
    )
    fig.tight_layout()
    fig.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


psf_normalization_sensitivity(
    f150_results,
    f150_sources,
    "F150W",
    ROOT / "runs/sbf2_go3055/analysis/tables/go3055_psf_129_vs_257_sensitivity.csv",
    "delta_mbar_mag",
    ROOT / "runs/sbf2_go3055/analysis/figures",
    "go3055_f150w_psf_normalization_sensitivity",
)

print("  matched current-product F150W P(k), P_r, and PSF diagnostics")
