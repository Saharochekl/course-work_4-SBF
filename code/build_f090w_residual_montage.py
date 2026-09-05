#!/usr/bin/env python3
"""Build the publication montage of the final F090W FFT-input residuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "runs/sbf_f090w_go3055/products"
FIGURES = ROOT / "runs/sbf_f090w_go3055/analysis/figures"
GALAXIES = [
    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 1549",
    "NGC 3379", "NGC 4374", "NGC 4406", "NGC 4472",
    "NGC 4486", "NGC 4552", "NGC 4621", "NGC 4636",
    "NGC 4649", "NGC 4697",
]
RADII_ARCSEC = (8.2, 16.4, 32.8)


panels = []
for galaxy in GALAXIES:
    manifest_path = PRODUCTS / galaxy.replace(" ", "_") / "products.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing product manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    residual_path = Path(manifest["products"]["normalized_full"])
    model_path = Path(manifest["products"]["model"])
    if not residual_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing F090W input for {galaxy}: "
            f"residual={residual_path.exists()}, model={model_path.exists()}"
        )
    if "_02_normalized_full_clip_3p5sigma.fits" not in residual_path.name:
        raise ValueError(f"Unexpected normalized-full product for {galaxy}: {residual_path}")

    with fits.open(model_path, memmap=True) as hdul:
        header = hdul[0].header
        ny, nx = hdul[0].data.shape
        x0 = float(header["SBFXCEN"])
        y0 = float(header["SBFYCEN"])
        pixel_scale = float(np.sqrt(header["PIXAR_A2"]))

    half_size = int(np.ceil(RADII_ARCSEC[-1] / pixel_scale)) + 2
    x1 = max(0, int(np.floor(x0)) - half_size)
    x2 = min(nx, int(np.floor(x0)) + half_size + 1)
    y1 = max(0, int(np.floor(y0)) - half_size)
    y2 = min(ny, int(np.floor(y0)) + half_size + 1)

    with fits.open(residual_path, memmap=True) as hdul:
        if hdul[0].data.shape != (ny, nx):
            raise ValueError(f"Model/residual shape mismatch for {galaxy}")
        normalized = np.asarray(hdul[0].data[y1:y2:2, x1:x2:2], dtype=float)

    display_scale = pixel_scale * 2
    panel_x0 = (x0 - x1) / 2
    panel_y0 = (y0 - y1) / 2
    yy, xx = np.mgrid[:normalized.shape[0], :normalized.shape[1]]
    radius = np.hypot(xx - panel_x0, yy - panel_y0) * display_scale
    working = np.full_like(normalized, np.nan)
    for r_in, r_out in zip(RADII_ARCSEC[:-1], RADII_ARCSEC[1:]):
        ring = (
            np.isfinite(normalized)
            & (radius >= r_in)
            & (radius < r_out)
        )
        if not np.any(ring):
            raise ValueError(f"No usable pixels in {galaxy}, {r_in}-{r_out} arcsec")
        working[ring] = normalized[ring] - np.mean(normalized[ring])

    scale = float(np.nanpercentile(np.abs(working), 99.3))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid display stretch for {galaxy}: {scale}")
    panels.append((galaxy, working, scale, panel_x0, panel_y0, display_scale))
    print(f"{galaxy}: {residual_path.name}; stretch=±{scale:.4g}")


FIGURES.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(4, 4, figsize=(13.2, 13.2))
gray = matplotlib.colormaps["gray"].copy()
gray.set_bad("black")

for ax, (galaxy, image, scale, x0, y0, pixel_scale) in zip(axes.flat, panels):
    ax.set_facecolor("black")
    ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        cmap=gray,
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )
    for radius_arcsec, color in zip(
        RADII_ARCSEC, ("#22d3ee", "#f59e0b", "#ef4444")
    ):
        ax.add_patch(
            plt.Circle(
                (x0, y0),
                radius_arcsec / pixel_scale,
                fill=False,
                color=color,
                linewidth=0.8,
            )
        )
    ax.set_title(galaxy, fontsize=10)
    ax.set(xticks=[], yticks=[])

for ax in axes.flat[len(panels):]:
    ax.axis("off")

fig.suptitle(
    r"Final normalized F090W FFT inputs: full-support 3.5$\sigma$ winsorization",
    fontsize=17,
)
fig.text(
    0.5,
    0.012,
    "Cyan/orange/red: 8.2, 16.4, and 32.8 arcsec; black: masked pixels",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0.025, 1, 0.965])

png_path = FIGURES / "go3055_f090w_final_working_residuals.png"
pdf_path = FIGURES / "go3055_f090w_final_working_residuals.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved {png_path.relative_to(ROOT)}")
print(f"Saved {pdf_path.relative_to(ROOT)}")
