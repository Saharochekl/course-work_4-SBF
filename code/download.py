#!/usr/bin/env python3
"""Download/check inputs: py download.py images|opd [options].

RU: без --download проверяет локальные файлы, не скачивает.
EN: delegates to the image/OPD downloader; its options pass through unchanged.
"""
import argparse
import subprocess
import sys

from sbf.sbf_paths import CODE_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["images", "opd"])
    argv = sys.argv[1:] if argv is None else argv
    # Parse only the dispatcher argument: images/opd --help belongs to its module.
    args = parser.parse_args(argv[:1])
    options = argv[1:]
    module = "download_go3055_go7763" if args.kind == "images" else "download_wss_opds"
    return subprocess.call([sys.executable, "-m", f"sbf.{module}", *options], cwd=CODE_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
