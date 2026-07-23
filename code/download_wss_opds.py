#!/usr/bin/env python3
"""Prepare local JWST WSS OPDs for fully offline STPSF calculations.

The default mode only inventories local ``*_i2d.fits`` files (including usable
partial headers from the program manifests) and checks the project-local OPD
cache.  Pass ``--download`` while network access is available to retrieve the
smallest representative set of measured WSS OPDs that keeps every known
science date within the configured maximum age.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits
from astropy.time import Time
from stpsf import mast_wss

from download_go3055_go7763 import partial_path, read_products, restart_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "wss_opd"
FITS_HEADER_PROBE_BYTES = 2880 * 32
USER_AGENT = "course-work-SBF/wss-opd-1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def header_time(header) -> Time | None:
    date_obs = str(header.get("DATE-OBS", "")).strip()
    if not date_obs:
        return None
    time_obs = str(header.get("TIME-OBS", "00:00:00")).strip()
    try:
        return Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
    except Exception:
        return None


def opd_time(path: Path) -> Time | None:
    try:
        if path.stat().st_size <= 0 or path.stat().st_size % 2880:
            return None
        with fits.open(path, memmap=False) as hdul:
            hdul.verify("exception")
            for hdu in hdul:
                if hdu.data is not None:
                    _ = hdu.data.shape
            return header_time(hdul[0].header)
    except Exception:
        return None


def group_science_headers(
    header_rows: list[tuple[str, fits.Header]],
) -> list[dict[str, object]]:
    by_day: dict[str, dict[str, object]] = {}
    seen_sources = set()
    for source, header in header_rows:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        if str(header.get("TELESCOP", "")).strip().upper() != "JWST":
            continue
        observation_time = header_time(header)
        if observation_time is None:
            continue
        day = observation_time.isot[:10]
        row = by_day.setdefault(
            day,
            {
                "date": day,
                "times_mjd": [],
                "files": [],
                "programs": set(),
            },
        )
        row["times_mjd"].append(float(observation_time.mjd))
        row["files"].append(source)
        row["programs"].add(str(header.get("PROGRAM", "")).lstrip("0") or "0")

    rows = []
    for day, row in sorted(by_day.items()):
        mjd = float(np.median(np.asarray(row.pop("times_mjd"), dtype=float)))
        rows.append(
            {
                **row,
                "date": day,
                "science_time_mjd": mjd,
                "science_time_isot": Time(mjd, format="mjd", scale="utc").isot,
                "programs": sorted(row["programs"]),
                "file_count": len(row["files"]),
            }
        )
    return rows


def parse_fits_header_prefix(payload: bytes) -> fits.Header:
    """Parse a primary FITS header from a small HTTP range response."""
    end_offset = None
    for offset in range(0, len(payload) - 79, 80):
        if payload[offset : offset + 8] == b"END     ":
            end_offset = offset + 80
            break
    if end_offset is None:
        raise ValueError(
            f"FITS END card not found in first {len(payload)} bytes"
        )
    text = payload[:end_offset].decode("ascii", errors="replace")
    return fits.Header.fromstring(text, sep="")


def remote_product_header(product, timeout: float) -> fits.Header:
    request = Request(
        product.url,
        headers={
            "Range": f"bytes=0-{FITS_HEADER_PROBE_BYTES - 1}",
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(FITS_HEADER_PROBE_BYTES)
    return parse_fits_header_prefix(payload)


def science_dates(
    data_root: Path,
    *,
    programs: set[str],
    include_remote_manifest_headers: bool = False,
    timeout: float = 30.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Inventory science dates, including tiny header probes for missing files."""
    header_rows: list[tuple[str, fits.Header]] = []
    inventory: list[dict[str, object]] = []

    # Completed files outside the two manifests are still useful for the local
    # OPD inventory.
    for path in sorted(data_root.glob("*/*_i2d.fits")):
        try:
            header = fits.getheader(path, 0)
            program = str(header.get("PROGRAM", "")).strip().lstrip("0") or "0"
            if program in programs:
                header_rows.append((str(path.resolve()), header))
        except Exception:
            continue

    products, _ = read_products(programs, data_root)
    for product in products:
        source = None
        header = None
        error = None
        for candidate in (
            product.destination,
            partial_path(product.destination),
            restart_path(product.destination),
        ):
            if not candidate.is_file():
                continue
            try:
                header = fits.getheader(candidate, 0)
                source = str(candidate.resolve())
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if header is None and include_remote_manifest_headers:
            try:
                header = remote_product_header(product, timeout)
                source = f"remote-header:{product.product_uri}"
                error = None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if header is not None:
            header_rows.append((source, header))
        inventory.append(
            {
                "program": product.program,
                "obsid": product.obsid,
                "target": product.target,
                "filter": product.filter_name,
                "product": product.file_name,
                "product_uri": product.product_uri,
                "header_source": source,
                "date_obs": None if header is None else header.get("DATE-OBS"),
                "time_obs": None if header is None else header.get("TIME-OBS"),
                "error": error,
            }
        )
    return group_science_headers(header_rows), inventory


def local_match(output_dir: Path, science_mjd: float) -> dict[str, object]:
    candidates = []
    for path in sorted(output_dir.glob("*.fits")):
        time_value = opd_time(path)
        if time_value is None:
            continue
        signed_delta = float(time_value.mjd - science_mjd)
        candidates.append((abs(signed_delta), signed_delta, path, time_value))
    if not candidates:
        return {
            "opd_path": None,
            "opd_time_isot": None,
            "opd_delta_days": None,
        }
    _, signed_delta, path, time_value = min(candidates, key=lambda item: item[0])
    return {
        "opd_path": str(path.resolve()),
        "opd_time_isot": time_value.isot,
        "opd_delta_days": signed_delta,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--program", choices=("3055", "7763", "both"), default="both"
    )
    parser.add_argument("--max-delta-days", type=float, default=30.0)
    parser.add_argument("--header-timeout", type=float, default=30.0)
    parser.add_argument(
        "--remote-manifest-headers",
        action="store_true",
        help="Probe missing archive products for dates; normally unnecessary when local GO-7763 dates span the campaign.",
    )
    parser.add_argument(
        "--coverage-order",
        choices=("median-first", "chronological"),
        default="median-first",
        help="Median-first minimises the number of OPDs needed for a permitted age window.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    status_file = (args.status_file or output_dir / "download_status.json").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    programs = {"3055", "7763"} if args.program == "both" else {args.program}

    rows, product_header_inventory = science_dates(
        data_root,
        programs=programs,
        include_remote_manifest_headers=(args.download and args.remote_manifest_headers),
        timeout=args.header_timeout,
    )
    if not rows:
        print(f"Нет завершённых JWST i2d в {data_root}")
        return 1

    if args.coverage_order == "median-first" and rows:
        median_mjd = float(np.median([row["science_time_mjd"] for row in rows]))
        rows.sort(key=lambda row: abs(float(row["science_time_mjd"]) - median_mjd))
    print(
        f"Найдено дат наблюдений: {len(rows)}; GO-{','.join(sorted(programs))}; "
        f"порядок={args.coverage_order}; локальный каталог: {output_dir}"
    )
    failures = []
    for index, row in enumerate(rows, start=1):
        science_time = Time(row["science_time_mjd"], format="mjd", scale="utc")
        before = local_match(output_dir, float(science_time.mjd))
        before_delta = before["opd_delta_days"]
        needs_download = before_delta is None or abs(float(before_delta)) > args.max_delta_days
        print(
            f"[{index}/{len(rows)}] {row['science_time_isot']} "
            f"GO-{','.join(row['programs'])}: local_delta={before_delta}"
        )
        error = None
        downloaded_path = None
        if args.download and needs_download:
            try:
                downloaded_path = mast_wss.get_opd_at_time(
                    science_time,
                    choice="closest",
                    verbose=True,
                    output_path=str(output_dir),
                )
                print(f"  сохранено: {downloaded_path}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  ОШИБКА: {error}")

        after = local_match(output_dir, float(science_time.mjd))
        after_delta = after["opd_delta_days"]
        ready = after_delta is not None and abs(float(after_delta)) <= args.max_delta_days
        row.update(
            {
                "local_before": before,
                "download_attempted": bool(args.download and needs_download),
                "downloaded_path": str(downloaded_path) if downloaded_path else None,
                "error": error,
                "local_after": after,
                "ready_offline": ready,
            }
        )
        if not ready:
            failures.append(row["date"])

    payload = {
        "updated_at": utc_now(),
        "mode": "download" if args.download else "dry-run",
        "programs": sorted(programs),
        "coverage_order": args.coverage_order,
        "remote_manifest_headers": bool(args.remote_manifest_headers),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "max_delta_days": args.max_delta_days,
        "date_count": len(rows),
        "ready_count": len(rows) - len(failures),
        "failed_dates": failures,
        "manifest_product_count": len(product_header_inventory),
        "manifest_headers_missing": [
            row for row in product_header_inventory if not row["date_obs"]
        ],
        "manifest_header_failures": [
            row for row in product_header_inventory if row["error"]
        ],
        "manifest_headers": product_header_inventory,
        "dates": rows,
    }
    status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Готово локально: {payload['ready_count']}/{payload['date_count']}; "
        f"отчёт: {status_file}"
    )
    if not args.download:
        print("Сеть не использовалась. Для загрузки добавьте --download.")
    require_manifest_coverage = bool(args.download and args.remote_manifest_headers)
    return 0 if not failures and (
        not require_manifest_coverage or not payload["manifest_header_failures"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
