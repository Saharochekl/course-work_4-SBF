#!/usr/bin/env python3
# group_i2d_by_detector.py
# Группирует JWST NIRCam *_i2d.fits по 4 детекторам (A1–A4, B1–B4) в общие папки.
# По умолчанию создаёт symlink'и, можно --copy для физического копирования.

import re
import argparse
from pathlib import Path
import shutil
import sys

PAT = re.compile(
    r'^(?P<prefix>jw\d{11}_\d{5}_\d{5})(?:-seg\d{3})?_(?P<mod>nrc[ab])(?P<det>long|[1-5])_i2d\.fits$',
    re.IGNORECASE
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/mastDownload/JWST",
                    help="корень, где лежат файлы из MAST")
    ap.add_argument("--out", default="../data/sorted_i2d",
                    help="куда складывать сгруппированные файлы")
    ap.add_argument("--copy", action="store_true",
                    help="копировать вместо симлинков")
    ap.add_argument("--dry", action="store_true",
                    help="только показать, что будет сделано")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.rglob("*_i2d.fits"))
    if not files:
        print("Файлов *_i2d.fits не найдено", file=sys.stderr)
        sys.exit(1)

    grouped = 0
    skipped = 0
    mosaics = 0

    for f in files:
        name = f.name

        # быстрая диагностика паттерна в --dry режиме
        if args.dry and not PAT.match(name) and name.lower().endswith('_i2d.fits') and mosaics < 5:
            print(f"[skip:pattern] {name}")

        m = PAT.match(name)
        if not m:
            # сюда попадут ассоциации вида jw01685-o009_t005_nircam_clear-f090w_i2d.fits
            mosaics += 1
            # при желании можно раскидывать их по фильтрам; пока просто пропускаем
            continue

        prefix = m.group("prefix")        # jw01685009001_03101_00001
        mod    = m.group("mod").lower()   # nrca или nrcb
        det    = m.group("det").lower()   # 1..4 или long

        # Папка-группа: общий префикс + модуль
        # Пример: jw01685009001_03101_00001_nrca
        group_dir = out / f"{prefix}_{mod}"
        # long кладём в свою подгруппу, чтобы не мешалось к 1–4
        if det == "long":
            group_dir = out / f"{prefix}_{mod}long"

        group_dir.mkdir(parents=True, exist_ok=True)
        dst = group_dir / name

        if args.dry:
            action = "COPY " if args.copy else "LINK "
            print(f"{action}{f}  ->  {dst}")
            grouped += 1
            continue

        try:
            if dst.exists():
                skipped += 1
            else:
                if args.copy:
                    shutil.copy2(f, dst)
                else:
                    try:
                        dst.symlink_to(f.resolve())
                    except OSError:
                        # на случай ограничений FS — fallback на копирование
                        shutil.copy2(f, dst)
                grouped += 1
        except Exception as e:
            print(f"[WARN] {f}: {e}", file=sys.stderr)

    print(f"[Итог] сгруппировано: {grouped}, пропущено (уже есть): {skipped}, мозаики-ассоциации (пропущены): {mosaics}")

if __name__ == "__main__":
    main()