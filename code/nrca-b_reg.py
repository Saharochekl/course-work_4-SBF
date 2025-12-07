#!/usr/bin/env python3
# wcs_register_by_folder.py
# pip install astropy reproject tqdm
import argparse
import re
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm

# По умолчанию: скрипт лежит в ../code, данные в ../data
# Базируем пути относительно расположения этого файла, чтобы запускать из папки code
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IN_DIR = (_SCRIPT_DIR / "../data/sorted_i2d").resolve()
DEFAULT_OUT_DIR = (_SCRIPT_DIR / "../data/nrc_seq").resolve()

SW_SCALE_ARCSEC = 0.031  # NIRCam short channel pixel scale. 0.031″/px.   [oai_citation:4‡jwst-docs.stsci.edu](https://jwst-docs.stsci.edu/jwst-near-infrared-camera?utm_source=chatgpt.com)

def list_sw_i2d(folder: Path, module: str):
    """Возвращает SW-файлы (nrca1..4/nrcb1..4), исключая *long, только *_i2d.fits в данной папке."""
    mod = module.lower()
    pat = re.compile(rf"nrc{mod}[1-4].*_i2d\.fits$", re.I)
    return sorted([p for p in folder.glob("*_i2d.fits") if pat.search(p.name)])

def north_up_wcs(centers_deg, scale_arcsec):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ra = np.mean([c[0] for c in centers_deg]); dec = np.mean([c[1] for c in centers_deg])
    w.wcs.crval = [ra, dec]
    pix = scale_arcsec / 3600.0  # deg/px
    w.wcs.cdelt = np.array([-pix, pix])  # North up, East left
    w.wcs.crpix = [1.0, 1.0]     # временно; потом сдвинем под bbox
    return w

def footprint_bbox(files, w_out):
    xmin=ymin=+1e30; xmax=ymax=-1e30
    for f in files:
        with fits.open(f, memmap=True) as hdul:
            h = hdul["SCI"].header; ny, nx = hdul["SCI"].data.shape
            wi = WCS(h)
            xs = np.array([0, nx, 0, nx]) - 0.5
            ys = np.array([0, 0, ny, ny]) - 0.5
            world = wi.pixel_to_world(xs, ys)
            xr, yr = w_out.world_to_pixel(world)
            xmin = min(xmin, xr.min()); xmax = max(xmax, xr.max())
            ymin = min(ymin, yr.min()); ymax = max(ymax, yr.max())
    width  = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))
    w2 = w_out.deepcopy()
    w2.wcs.crpix = w2.wcs.crpix + [1 - xmin, 1 - ymin]
    return w2, (height, width)

def filter_key(hdr):
    """Ключ фильтра: предпочитаем FILTER; если его нет/"N/A", используем PUPIL или PUPIL-FILTER."""
    pupil = (hdr.get('PUPIL') or '').strip().upper()
    filtr = (hdr.get('FILTER') or '').strip().upper()
    if filtr and filtr != 'N/A':
        return filtr
    if pupil and pupil != 'CLEAR':
        return f"{pupil}-{filtr}" if filtr else pupil
    return filtr or 'UNKNOWN'

def run(in_dir: Path, out_dir: Path, modules: list[str]):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Собираем все *_i2d.fits под in_dir и находим уникальные папки, где есть нужные детекторы
    candidate_files = list(in_dir.rglob('*_i2d.fits'))
    folders = sorted({p.parent for p in candidate_files})

    for folder in folders:
        for mod in modules:
            files = list_sw_i2d(folder, mod)
            if not files:
                continue
            # группировка по FILTER
            groups: dict[str, list[Path]] = {}
            for f in files:
                with fits.open(f, memmap=True) as hdul:
                    if 'SCI' not in hdul:
                        continue
                    hdr = hdul['SCI'].header
                    if hdr.get('WCSAXES') is None:
                        continue
                    key = filter_key(hdr)
                    groups.setdefault(key, []).append(f)

            if not groups:
                continue

            # относительный путь этой папки внутри IN, чтобы зеркально положить в OUT
            try:
                rel = folder.relative_to(in_dir)
            except ValueError:
                rel = Path('.')

            for filt, flist in groups.items():
                # центры для WCS
                centers = []
                for f in flist:
                    with fits.open(f, memmap=True) as hdul:
                        h = hdul['SCI'].header
                        ny, nx = hdul['SCI'].data.shape
                        w = WCS(h)
                        ra, dec = w.wcs_pix2world([[nx/2, ny/2]], 0)[0]
                        centers.append((ra, dec))
                w0 = north_up_wcs(centers, SW_SCALE_ARCSEC)
                w_out, shape_out = footprint_bbox(flist, w0)

                # Выходная папка: ../data/wcs_seq/<rel>/wcs_seq_<FILTER>_<MOD>
                seq_dir = (out_dir / rel / f"wcs_seq_{filt}_{mod.upper()}").resolve()
                seq_dir.mkdir(parents=True, exist_ok=True)
                base = f"{filt.lower()}_{mod.lower()}_wcs"
                print(f"\n[{rel if rel != Path('.') else folder.name}] {mod.upper()} {filt}: {len(flist)} файлов → {seq_dir}")

                for i, f in enumerate(tqdm(sorted(flist), desc=f"{folder.name}:{mod}:{filt}"), 1):
                    with fits.open(f, memmap=True) as hdul:
                        sci = hdul['SCI'].data.astype(np.float32)
                        w_in = WCS(hdul['SCI'].header)
                        arr, fp = reproject_interp((sci, w_in), w_out, shape_out=shape_out, return_footprint=True)
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        h = w_out.to_header()
                        h['SRC_FILE'] = f.name
                        h['FILTER']   = filter_key(hdul['SCI'].header)
                        h['MODULE']   = mod.upper()
                        fits.PrimaryHDU(arr, header=h).writeto(seq_dir / f"{base}_{i:05d}.fits", overwrite=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WCS-регистрация внутри папок для NIRCam SW (nrca1..4 / nrcb1..4) с относительными путями ../data/... от скрипта")
    ap.add_argument("--in", dest="in_dir", default=str(DEFAULT_IN_DIR), help="корень с входными *_i2d.fits (по умолчанию ../data/sorted_i2d от скрипта)")
    ap.add_argument("--out", dest="out_dir", default=str(DEFAULT_OUT_DIR), help="корень для выходных последовательностей (по умолчанию ../data/wcs_seq от скрипта)")
    ap.add_argument("--modules", default="a,b", help="модули через запятую: a,b")
    args = ap.parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    modules = [m.strip().lower() for m in args.modules.split(',') if m.strip()]
    run(in_dir, out_dir, modules)