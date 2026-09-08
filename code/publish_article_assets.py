#!/usr/bin/env python3
"""Copy the article's selected figures beside its TeX, without recomputing them.

Run from code/: ``py publish_article_assets.py``; add ``--check`` for a
read-only check. Scientific producer outputs stay in runs/ for the notebooks.
"""

import argparse
import filecmp
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("texts/paper_work/materials/figure_sources.json")


def figure_sources(root=ROOT):
    return json.loads((Path(root) / MANIFEST).read_text(encoding="utf-8"))


def publish_figure(source, root=ROOT):
    """Refresh a saved figure only if the article manifest selects it."""
    root = Path(root).resolve()
    source = Path(source)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    for asset in figure_sources(root):
        if source == (root / asset["source"]).resolve():
            destination = root / asset["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or not filecmp.cmp(source, destination, shallow=False):
                shutil.copy2(source, destination)
            return destination
    return None


def publish_article_assets(root=ROOT, check=False):
    """Synchronize selected images, or verify saved copies without writing."""
    root = Path(root).resolve()
    issues = []
    count = 0
    for asset in figure_sources(root):
        source, destination = root / asset["source"], root / asset["destination"]
        if check:
            if not destination.is_file():
                issues.append(f"Missing article figure: {asset['destination']}")
            elif source.is_file() and not filecmp.cmp(source, destination, shallow=False):
                issues.append(f"Outdated article copy: {asset['destination']}")
            else:
                count += 1
        elif source.is_file():
            publish_figure(source, root)
            count += 1
        else:
            issues.append(f"Missing producer figure: {asset['source']}")
    if issues:
        raise FileNotFoundError("\n".join(issues))
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check copies without writing")
    args = parser.parse_args()
    count = publish_article_assets(check=args.check)
    print(f"{'Checked' if args.check else 'Published'} {count} article figures.")
