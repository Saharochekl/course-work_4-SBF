#!/usr/bin/env python3
"""Монтаж готовых остатков / Montage of saved FFT-input residuals.

The default remains F090W; --pasa also supports the adopted F150W products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures.publish_article_assets import publish_figure
from figures.pasa_style import WIDTH_INCH as PASA_WIDTH, LABEL_PT
from sbf.sbf_paths import PROJECT_ROOT, load_project_json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = PROJECT_ROOT
PRODUCTS = ROOT / "runs/F090W/products"
FIGURES = ROOT / "runs/F090W/analysis/figures"
PASA_FIGURES = ROOT / "texts/paper_work/materials/pasa/figures"
# Fixed GO-3055 calibration sample; montage never makes a quality selection.
GALAXIES = [
    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 1549",
    "NGC 3379", "NGC 4374", "NGC 4406", "NGC 4472",
    "NGC 4486", "NGC 4552", "NGC 4621", "NGC 4636",
    "NGC 4649", "NGC 4697",
]
RADII_ARCSEC = (8.2, 16.4, 32.8)  # Adopted circular annuli, matching Jensen geometry.


def main(argv=None):
    """Запустить построение явно / Run this saved-product workflow explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasa", action="store_true", help="PASA PDF only; leave the existing figures and manifest unchanged")
    parser.add_argument("--band", choices=("F090W", "F150W"), default="F090W")
    args = parser.parse_args(argv)
    if args.band == "F150W" and not args.pasa:
        parser.error("--band F150W is available only with --pasa")
    if args.pasa:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": LABEL_PT, "pdf.fonttype": 42})
    panels = []
    for galaxy in GALAXIES:
        slug = galaxy.replace(" ", "_")
        if args.band == "F150W":
            final = load_project_json(ROOT / "runs/F150W/spectra/batch" / f"{slug}_result.json")
            source = load_project_json(final["source_result"])
            residual_path = Path(final["full_normalized_residual_fits"])
            model_path = Path(source["model_full_fits"])
        else:
            manifest_path = PRODUCTS / slug / "products.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing product manifest: {manifest_path}")
            manifest = load_project_json(manifest_path)
            residual_path = Path(manifest["products"]["normalized_full"])
            model_path = Path(manifest["products"]["model"])
        if not residual_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Missing {args.band} input for {galaxy}: "
                f"residual={residual_path.exists()}, model={model_path.exists()}"
            )
        if not args.pasa and "_02_normalized_full_clip_3p5sigma.fits" not in residual_path.name:
            raise ValueError(f"Unexpected normalized-full product for {galaxy}: {residual_path}")

        with fits.open(model_path, memmap=True) as hdul:
            header = hdul[0].header
            ny, nx = hdul[0].data.shape
            x0 = float(header["SBFXCEN"])
            y0 = float(header["SBFYCEN"])
            pixel_scale = float(np.sqrt(header["PIXAR_A2"]))

        # Two-pixel border and 2x display subsampling save rendering memory only.
        half_size = int(np.ceil(RADII_ARCSEC[-1] / pixel_scale)) + 2
        x1 = max(0, int(np.floor(x0)) - half_size)
        x2 = min(nx, int(np.floor(x0)) + half_size + 1)
        y1 = max(0, int(np.floor(y0)) - half_size)
        y2 = min(ny, int(np.floor(y0)) + half_size + 1)

        with fits.open(residual_path, memmap=True) as hdul:
            # PASA reads actual products, not the retired alias filename.
            if args.pasa and hdul[0].header.get("SBFBRNCH") != "normalized_full_3p5":
                raise ValueError(f"Unexpected residual branch for {galaxy}: {residual_path}")
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

        # Robust display stretch, not clipping of the measured residual product.
        scale = float(np.nanpercentile(np.abs(working), 99.3))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid display stretch for {galaxy}: {scale}")
        panels.append((galaxy, working, scale, panel_x0, panel_y0, display_scale))
        print(f"{galaxy}: {residual_path.name}; stretch=±{scale:.4g}")


    output_dir = PASA_FIGURES if args.pasa else FIGURES
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(PASA_WIDTH, 7.4) if args.pasa else (13.2, 13.2))
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
        ax.set_title(galaxy, fontsize=9 if args.pasa else 10)
        ax.set(xticks=[], yticks=[])

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        rf"{args.band}: normalised working residuals, full-support $3.5\sigma$ winsorisation"
        if args.pasa else r"Final normalized F090W FFT inputs: full-support 3.5$\sigma$ winsorization",
        fontsize=10 if args.pasa else 17,
    )
    fig.text(
        0.5,
        0.012,
        "Cyan/orange/red: 8.2, 16.4, and 32.8 arcsec; black: masked pixels",
        ha="center",
        fontsize=8 if args.pasa else 10,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])

    stem = "go3055_final_working_residuals" if args.band == "F150W" else "go3055_f090w_final_working_residuals"
    pdf_path = output_dir / f"{stem}.pdf"
    if args.pasa:
        fig.savefig(pdf_path, dpi=300)
        plt.close(fig)
        print(f"Saved {pdf_path.relative_to(ROOT)}")
        return

    png_path = FIGURES / "go3055_f090w_final_working_residuals.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    publish_figure(png_path)
    publish_figure(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path.relative_to(ROOT)}")
    print(f"Saved {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
