#!/usr/bin/env python3
# JWST NIRCam i2d -> WCS-aligned sequences for Siril
# Groups by FILTER, reprojects each SCI plane to a common celestial WCS and writes
# f"seq_<FILTER>/<filter>_wcs_00001.fits" style sequences.
# Now with CLI flags.

# deps: python3 -m pip install astropy reproject tqdm

import os, re, glob
from pathlib import Path
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent

# -------------------------- helpers --------------------------

def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return (p if p.is_absolute() else (_SCRIPT_DIR / p)).resolve()


def filt_key(hdr):
    """Сформировать ключ фильтра из JWST-хедера: CLEAR-F090W -> F090W, иначе PUPIL-FILTER."""
    pupil = (hdr.get('PUPIL') or '').strip().upper()
    filtr = (hdr.get('FILTER') or '').strip().upper()
    if filtr and filtr != 'N/A':
        return filtr
    if pupil and pupil != 'CLEAR':
        return f"{pupil}-{filtr}" if filtr else pupil
    return filtr or 'UNKNOWN'


def choose_files(in_dir: Path, level: str, modules: str, include_long: bool) -> list[str]:
    """Подобрать i2d/CAL/RATE файлы NIRCam по шаблону имени детектора."""
    if level == 'any':
        suffix = '*.fits'
    else:
        suffix = f"*_{level}.fits"
    cand = glob.glob(str(in_dir / '**' / suffix), recursive=True)

    want_a = 'a' in {m.strip().lower() for m in modules.split(',') if m.strip()}
    want_b = 'b' in {m.strip().lower() for m in modules.split(',') if m.strip()}

    picked = []
    for f in cand:
        name = os.path.basename(f).lower()
        ok = False
        if want_a and re.search(r'_nrca[1-4]_', name):
            ok = True
        if want_b and re.search(r'_nrcb[1-4]_', name):
            ok = True
        if include_long:
            if want_a and '_nrcalong_' in name:
                ok = True
            if want_b and '_nrcblong_' in name:
                ok = True
        if ok:
            picked.append(f)
    return picked


# -------------------------- main --------------------------

def main():
    class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    ap = argparse.ArgumentParser(
        prog='WCS-all.py',
        description=(
            'Репроекция JWST/NIRCam продуктов на общий небесный WCS, группировка по FILTER,\n'
            'вывод готовых для Siril последовательностей.'
        ),
        formatter_class=_Fmt,
        epilog=(
            'Примеры:\n'
            '  py WCS-all.py --in ../data/sorted_i2d --out ../data/wcs_seq --level i2d --modules a,b --skip-long\n'
            '  py WCS-all.py --filters F090W F150W --overwrite\n'
        ),
    )
    ap.add_argument('--in', dest='in_dir', default='../data/sorted_i2d', help='Корень, где лежат *_<level>.fits.')
    ap.add_argument('--out', dest='out_dir', default='../data/wcs_seq', help='Куда писать последовательности.')
    ap.add_argument('--level', choices=['i2d', 'cal', 'rate', 'any'], default='i2d',
                    help='Суффикс уровня продукта по имени файла.')
    ap.add_argument('--modules', default='a,b', help='Какие модули брать: a,b / a / b.')
    ap.add_argument('--skip-long', action='store_true', help='Не брать nrcalong/nrcblong.')
    ap.add_argument('--filters', nargs='*', default=None,
                    help='Ограничить обработку указанными фильтрами (пример: F090W F150W).')
    ap.add_argument('--overwrite', action='store_true', help='Перезаписывать выходные файлы.')
    args = ap.parse_args()

    in_dir  = _resolve(args.in_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = choose_files(in_dir, args.level, args.modules, include_long=(not args.skip_long))
    if not files:
        print('[!] Ничего не найдено по заданным параметрам')
        return

    # собрать кандидатов: только детекторы a/b, только те, где есть SCI + валидный WCS
    groups: dict[str, list[str]] = {}
    for f in files:
        try:
            with fits.open(f, memmap=True) as hdul:
                if 'SCI' not in hdul:
                    continue
                hdr = hdul['SCI'].header
                if hdr.get('WCSAXES') is None:
                    continue
                key = filt_key(hdr)
                groups.setdefault(key, []).append(f)
        except Exception:
            # битый файл пропускаем молча, чтобы не мешал батчу
            continue

    if args.filters:
        keep = {s.upper() for s in args.filters}
        groups = {k: v for k, v in groups.items() if k.upper() in keep}

    if not groups:
        print('[!] Нет групп для обработки после фильтров/WCS-проверок')
        return

    print('Групп найдено:', {k: len(v) for k, v in groups.items()})

    for key, flist in groups.items():
        # 1) общий WCS/размер через reproject.find_optimal_celestial_wcs
        try:
            w_out, shape_out = find_optimal_celestial_wcs([
                (fits.getdata(f, 'SCI'), WCS(fits.getheader(f, 'SCI'))) for f in flist
            ])
        except Exception as e:
            print(f"[skip group] {key}: {e}")
            continue

        # 2) репроекция каждого кадра на общий грид
        seq_dir = out_dir / f"seq_{key}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        out_base = f"{key.lower()}_wcs"
        idx = 0
        for f in tqdm(sorted(flist), desc=f"Reproject {key}"):
            try:
                with fits.open(f, memmap=True) as hdul:
                    sci = hdul['SCI'].data.astype(np.float32)
                    w_in = WCS(hdul['SCI'].header)
                    arr, fp = reproject_interp((sci, w_in), w_out, shape_out=shape_out, return_footprint=True)
                    # Siril не любит NaN/Inf
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                    # создаём PrimaryHDU с целевым WCS + полезная мета
                    hdr_out = w_out.to_header(relax=True)
                    # Siril compatibility: ensure PC/CD or PC+CDELT present, WCSNAME/EQUINOX retained
                    if not any(k in hdr_out for k in ("CD1_1", "PC1_1")):
                        # we keep north-up, so identity PC with CDELT is fine
                        hdr_out["PC1_1"] = 1.0
                        hdr_out["PC1_2"] = 0.0
                        hdr_out["PC2_1"] = 0.0
                        hdr_out["PC2_2"] = 1.0
                    # some software expects EQUINOX and WCSNAME even with RADESYS=ICRS
                    hdr_out.setdefault("EQUINOX", 2000.0)
                    hdr_out.setdefault("WCSNAME", "ASTROM")
                    # optional CROTA2 for old readers
                    hdr_out.setdefault("CROTA2", 0.0)

                    hdr_out['SRC_FILE'] = os.path.basename(f)
                    hdr_out['FILTER']   = (hdul['SCI'].header.get('FILTER') or '')
                    hdr_out['PUPIL']    = (hdul['SCI'].header.get('PUPIL') or '')
                    hdr_out['BUNIT']    = hdul['SCI'].header.get('BUNIT', 'MJy/sr')

                    idx += 1
                    out_name = seq_dir / f"{out_base}_{idx:05d}.fits"
                    from astropy.io.fits import PrimaryHDU, ImageHDU, HDUList
                    hpri = PrimaryHDU()  # minimal primary
                    himg = ImageHDU(data=arr, header=hdr_out, name='SCI')
                    HDUList([hpri, himg]).writeto(out_name, overwrite=args.overwrite, output_verify='ignore')
            except Exception as e:
                print(f"[skip file] {os.path.basename(f)}: {e}")
                continue

    print('[OK] Готово: см. подпапки', out_dir)


if __name__ == '__main__':
    main()