#!/usr/bin/env python3
"""Запуск обработки / Run the adopted F150W and F090W pipelines from code/."""

import argparse
import shlex
import subprocess
import sys

from sbf.sbf_paths import CODE_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=["F150W", "F090W", "both"], default="both")
    parser.add_argument("--stage", choices=["source", "spectra", "all"], default="all")
    parser.add_argument("--galaxies", nargs="+", help='Например / e.g. "NGC 4636"')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Только показать команды / print commands only")
    mode.add_argument("--check", action="store_true", help="Проверить входы без моделирования и FFT / validate without science")
    args = parser.parse_args(argv)
    if args.filter in {"F090W", "both"} and args.stage != "all":
        parser.error("F090W поддерживает --stage all: готовые этапы пропускаются по кэшу / completed stages are cached")
    return args


def build_commands(args):
    modules = []
    if args.filter in {"F150W", "both"}:
        if args.stage in {"source", "all"}:
            modules.append("sbf.run_sbf_2_batch")
        if args.stage in {"spectra", "all"}:
            modules.append("sbf.run_sbf_2_normalized_winsor")
    if args.filter in {"F090W", "both"}:
        modules.append("sbf.run_sbf_f090w")
    commands = []
    for module in modules:
        command = [sys.executable, "-m", module]
        if args.galaxies:
            command += ["--galaxies", *args.galaxies]
        if args.check:
            command.append("--dry-run")
        commands.append(command)
    return commands


def main(argv=None):
    args = parse_args(argv)
    for command in build_commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=CODE_DIR)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
