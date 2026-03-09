import re, argparse, shutil
from pathlib import Path

# Карта T### -> имя объекта (по APT-PDF JWST GO-3055, раздел Targets)
T2NAME = {
    "001": "NGC 1380",
    "002": "NGC 1399",
    "003": "NGC 1404",
    "004": "MESSIER-084",
    "005": "MESSIER-086",
    "006": "MESSIER-049",
    "007": "MESSIER-087",
    "008": "MESSIER-089",
    "009": "MESSIER-059",
    "010": "MESSIER-060",
    "011": "NGC 4697",
    "012": "NGC 1549",
    "013": "MESSIER-105",
    "014": "NGC 4636",
}

PAT = re.compile(
    r"^(jw03055)-o(\d{3})_t(\d{3})_[a-z0-9]+_[^-]+-[a-z0-9]+_(?:i2d|cal|rate)\.(fits|seq)$",
    re.IGNORECASE,
)

def main():
    ap = argparse.ArgumentParser(description="Разложить продукты JWST GO-3055 по папкам объектов.")
    ap.add_argument("--src", default=".", help="Папка с файлами")
    ap.add_argument("--dst", default="by_object", help="Куда складывать")
    ap.add_argument("--dry-run", action="store_true", help="Только показать план")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst_root = (Path(args.dst)).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.name.lower().startswith("jw03055-")
                   and p.suffix.lower() in (".fits", ".seq"))

    moves = []
    for p in files:
        m = PAT.match(p.name)
        if not m:
            continue
        tnum = m.group(3)  # '001'..'014'
        target = T2NAME.get(tnum, f"_UNKNOWN_T{tnum}")
        target_dir = dst_root / target
        target_dir.mkdir(parents=True, exist_ok=True)
        moves.append((p, target_dir / p.name))

    for src_p, dst_p in moves:
        if args.dry_run:
            print(f"[PLAN] {src_p.name} -> {dst_p}")
        else:
            if src_p.resolve() == dst_p.resolve():
                continue
            shutil.move(str(src_p), str(dst_p))
            print(f"[MOVE] {src_p.name} -> {dst_p}")

    print(f"\n[DONE] всего: {len(moves)} {'(dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()