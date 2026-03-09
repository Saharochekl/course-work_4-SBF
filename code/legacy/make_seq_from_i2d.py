#!/usr/bin/env python3
# make_seq_from_i2d.py
# Извлекает SCI (и при наличии WHT) из JWST *_i2d.fits,
# приводит к общему WCS и пишет файлы как base_00001.fit ... для Siril.

import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd

def load_sci_wcs(path):
    with fits.open(path) as hdul:
        sci = hdul['SCI'].data.astype(np.float32)
        hdr = hdul['SCI'].header
        wcs = WCS(hdr)
        wht = hdul['WHT'].data.astype(np.float32) if 'WHT' in hdul else None
    return sci, wcs, wht

def reproject_to_common_grid(files, outdir, base="stack", exact=False):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # 1) читаем данные
    tiles = []
    weights = []
    for f in files:
        sci, wcs, wht = load_sci_wcs(f)
        tiles.append((sci, wcs))
        weights.append(wht if wht is not None else np.ones_like(sci, dtype=np.float32))

    # 2) общий WCS/размер (auto_rotate=True уменьшает пустоты)
    wcs_out, shape_out = find_optimal_celestial_wcs([t[1] for t in tiles], auto_rotate=True)  #  [oai_citation:3‡reproject.readthedocs.io](https://reproject.readthedocs.io/en/stable/mosaicking.html)

    # 3a) просто выровнять и сохранить по одному кадру (для Siril)
    for i, (arr, wcs) in enumerate(tiles, 1):
        rp_func = reproject_exact if exact else reproject_interp
        arr_rp, _ = rp_func((arr, wcs), wcs_out, shape_out=shape_out)
        # Siril ловит паттерн basename + индекс → seq.  [oai_citation:4‡siril.readthedocs.io](https://siril.readthedocs.io/en/stable/Sequences.html)
        fits.PrimaryHDU(arr_rp, header=wcs_out.to_header()).writeto(
            outdir / f"{base}_{i:05d}.fit", overwrite=True)

    # 3b) дополнительно можно сразу сделать суммарную мозаику с весами WHT
    mosaic, footprint = reproject_and_coadd(
        tiles, wcs_out, shape_out=shape_out,
        input_weights=weights,                 # веса из WHT, если были.  [oai_citation:5‡reproject.readthedocs.io](https://reproject.readthedocs.io/en/stable/api/reproject.mosaicking.reproject_and_coadd.html?utm_source=chatgpt.com)
        reproject_function=reproject_exact if exact else reproject_interp,
        match_background=True
    )
    fits.PrimaryHDU(mosaic, header=wcs_out.to_header()).writeto(
        outdir / f"{base}_mosaic.fit", overwrite=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="пути к *_i2d.fits (4 детектора)")
    ap.add_argument("--out", default="../data/siril_seq", help="куда писать")
    ap.add_argument("--base", default="jwst_i2d", help="basename для последовательности")
    ap.add_argument("--exact", action="store_true", help="flux-conserving reprojection")
    args = ap.parse_args()
    reproject_to_common_grid(args.inputs, args.out, base=args.base, exact=args.exact)