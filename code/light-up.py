#!/usr/bin/env python3
# smooth_fits.py — без смены WCS/ориентации: NaN->0 + лёгкое сглаживание
# deps: astropy, numpy  (tqdm опционально)
from pathlib import Path
import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve

# базовые пути: скрипт лежит в ../code, данные — в ../data
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IN  = (_SCRIPT_DIR / "../data").resolve()
DEFAULT_OUT = (_SCRIPT_DIR / "../data/smoothed").resolve()

def clean_primary_header(h):
    """Убираем ключи, нежелательные в PrimaryHDU, но оставляем WCS и полезные метаданные."""
    drop = {"XTENSION","PCOUNT","GCOUNT","EXTNAME","EXTVER","ZIMAGE","ZBITPIX",
            "ZNAXIS","ZCMPTYPE","ZNAME1","ZVAL1","NAXIS1","NAXIS2"}  # NAXIS* оставит astropy сам
    hdr = h.copy()
    for k in list(hdr.keys()):
        if k in drop:
            del hdr[k]
    return hdr

def pick_images(hdul, which="sci"):
    """Возвращает список (data, header, tag) для сохранения.
       which: 'sci' — все SCI; 'primary' — только первичный образ (если он *image*);
              'auto' — SCI если есть, иначе первый image HDU."""
    out = []
    if which.lower() == "sci":
        for h in hdul:
            if getattr(h, "name", "") == "SCI" and getattr(h, "data", None) is not None:
                tag = f"SCI{h.header.get('EXTVER','')}".rstrip()
                out.append((h.data, h.header, tag))
        return out
    if which.lower() == "primary":
        h = hdul[0]
        if getattr(h, "data", None) is not None:
            out.append((h.data, h.header, "PRI"))
        return out
    # auto
    scis = [h for h in hdul if getattr(h, "name", "") == "SCI" and getattr(h, "data", None) is not None]
    if scis:
        for h in scis:
            tag = f"SCI{h.header.get('EXTVER','')}".rstrip()
            out.append((h.data, h.header, tag))
        return out
    # иначе первый image HDU
    for h in hdul:
        if getattr(h, "data", None) is not None:
            out.append((h.data, h.header, "IMG"))
            break
    return out

def smooth_array(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """NaN/Inf -> 0, затем гаусс сглаживание kernel sigma_px; возвращает float32."""
    a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if sigma_px > 0:
        # компактное сглаживание без FFT и без смены WCS
        k = Gaussian2DKernel(x_stddev=float(sigma_px))
        a = convolve(a, k, boundary="extend", normalize_kernel=True, preserve_nan=False).astype(np.float32, copy=False)
        # лишний раз прибьём возможные численные хвосты
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return a

def process_one(fin: Path, fout_dir: Path, sigma_px: float, which: str, overwrite: bool):
    # игнорим мусорные метафайлы macOS
    if fin.name.startswith("._"):
        return False, "[skip appledouble]"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(fin, memmap=True) as hdul:
                items = pick_images(hdul, which=which)
                if not items:
                    return False, "[no image HDU]"
                wrote_any = False
                for data, hdr, tag in items:
                    sm = smooth_array(data, sigma_px)
                    hdr_out = clean_primary_header(hdr)
                    hdr_out.add_history(f"NaN->0; Gaussian smooth sigma={sigma_px}px; no reprojection")
                    out = fout_dir / (fin.stem + (f"_SM_{tag}" if tag else "_SM") + ".fits")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fits.PrimaryHDU(data=sm, header=hdr_out).writeto(out, overwrite=overwrite, output_verify="ignore")
                    wrote_any = True
                return wrote_any, "ok"
    except Exception as e:
        return False, f"[err {type(e).__name__}] {e}"

def main():
    ap = argparse.ArgumentParser(
        prog="smooth_fits.py",
        description="Пакетное сглаживание FITS без смены WCS/ориентации: NaN→0 + Gaussian(σ пикс).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  py smooth_fits.py --in ../data/sci_only --out ../data/smoothed --which sci --sigma-px 1.2\n"
            "  py smooth_fits.py --which auto --sigma-px 0.8 --overwrite\n"
        ),
    )
    ap.add_argument("--in", dest="in_dir", default=str(DEFAULT_IN), help="Корень с FITS (рекурсивно).")
    ap.add_argument("--out", dest="out_dir", default=str(DEFAULT_OUT), help="Куда писать результат.")
    ap.add_argument("--which", choices=["sci", "primary", "auto"], default="sci",
                    help="Какое HDU брать: все SCI; только primary; или auto=SCI если есть, иначе первый image HDU.")
    ap.add_argument("--sigma-px", type=float, default=1.0, help="Сигма гаусс-сглаживания в пикселях. 0 — без сглаживания.")
    ap.add_argument("--glob", default="*.fits", help="Маска поиска файлов внутри --in (рекурсивно).")
    ap.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие выходные файлы.")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    files = list(in_dir.rglob(args.glob))
    if not files:
        print("[!] Ничего не найдено по маске.")
        return

    ok = fail = 0
    for f in files:
        wrote, msg = process_one(f, out_dir, args.sigma_px, args.which, args.overwrite)
        if wrote:
            ok += 1
        else:
            fail += 1
        print(f"{'OK ' if wrote else 'SKP'} {f.name} {msg}")
    print(f"\nDone. processed={len(files)} ok={ok} skipped/failed={fail}")

if __name__ == "__main__":
    main()