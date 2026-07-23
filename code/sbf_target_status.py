#!/usr/bin/env python3
"""Small, human-readable status ledger for the SBF target queue.

The ledger deliberately identifies work by archive observation and filter pair,
not by a notebook or file hash.  Hashes may still be kept in result provenance,
but editing a notebook must not silently turn a completed target back into work.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from astropy.io import fits


TARGET_STATUS_COLUMNS = (
    "program",
    "obsid",
    "galaxy",
    "signal_filter",
    "color_filter",
    "status",
    "method",
    "quantity",
    "result_value",
    "result_unit",
    "selected_region",
    "selection_method",
    "qc",
    "result_json",
    "error",
    "updated_at",
)

TARGET_STATUS_VALUES = frozenset(
    {"pending", "running", "done", "failed", "skipped"}
)

PRIMARY_QUANTITY = "apparent_sbf_magnitude"

SBF3_FITS_KEYS = (
    "clean_model_fits",
    "clean_isophotes_fits",
    "full_residual_fits",
    "working_residual_fits",
    "working_annuli_residual_fits",
)

LEGACY_SBF2_FITS_KEYS = (
    "model_full_fits",
    "science_residual_fits",
    "science_residual_raw_fits",
    "inner_usable_residual_fits",
    "outer_usable_residual_fits",
)


def canonical_program(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("GO-"):
        text = text[3:]
    elif text.startswith("GO"):
        text = text[2:]
    text = text.strip()
    return str(int(text)) if text.isdigit() else text


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _field(target: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if target.get(name) not in (None, ""):
            return target.get(name)
    return ""


def target_status_key(target: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the stable, SHA-independent identity of one science job."""
    return (
        canonical_program(_field(target, "program")),
        _compact_text(_field(target, "obsid")).casefold(),
        _compact_text(_field(target, "galaxy", "name", "target")).casefold(),
        _compact_text(_field(target, "signal_filter")).upper(),
        _compact_text(_field(target, "color_filter")).upper(),
    )


def status_row_for_target(target: Mapping[str, Any]) -> dict[str, str]:
    return {
        "program": canonical_program(_field(target, "program")),
        "obsid": _compact_text(_field(target, "obsid")),
        "galaxy": _compact_text(_field(target, "galaxy", "name", "target")),
        "signal_filter": _compact_text(_field(target, "signal_filter")).upper(),
        "color_filter": _compact_text(_field(target, "color_filter")).upper(),
        "status": "pending",
        "method": "",
        "quantity": "",
        "result_value": "",
        "result_unit": "",
        "selected_region": "",
        "selection_method": "",
        "qc": "",
        "result_json": "",
        "error": "",
        "updated_at": utc_timestamp(),
    }


def utc_timestamp(now: float | None = None) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now)
    )


def read_target_status(path: str | os.PathLike[str]) -> dict[tuple[str, ...], dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return {}
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {name: str(raw.get(name) or "") for name in TARGET_STATUS_COLUMNS}
            # One unreleased development revision used ``result`` as the JSON
            # locator.  Accept it on read so that an interrupted local trial
            # can be resumed, but always write the explicit new schema.
            row["result_json"] = row["result_json"] or str(
                raw.get("result") or ""
            )
            status = row["status"].strip().lower()
            if status not in TARGET_STATUS_VALUES:
                continue
            row["status"] = status
            key = target_status_key(row)
            if all(key):
                rows[key] = row
    return rows


def write_target_status(
    path: str | os.PathLike[str],
    rows: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> Path:
    """Atomically rewrite the complete ledger; the parent is its only writer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=TARGET_STATUS_COLUMNS)
    writer.writeheader()
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            canonical_program(row.get("program")),
            _compact_text(row.get("obsid")),
            _compact_text(row.get("galaxy")).casefold(),
            _compact_text(row.get("signal_filter")),
            _compact_text(row.get("color_filter")),
        ),
    )
    for row in ordered:
        writer.writerow({name: row.get(name, "") for name in TARGET_STATUS_COLUMNS})

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ensure_target_rows(
    rows: dict[tuple[str, ...], dict[str, str]],
    targets: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, ...], dict[str, str]]:
    for target in targets:
        key = target_status_key(target)
        if not all(key):
            raise ValueError(f"incomplete target status identity: {target!r}")
        rows.setdefault(key, status_row_for_target(target))
    return rows


def update_target_status(
    rows: dict[tuple[str, ...], dict[str, str]],
    target: Mapping[str, Any],
    status: str,
    *,
    method: str | None = None,
    quantity: str | None = None,
    result_value: Any | None = None,
    result_unit: str | None = None,
    selected_region: str | None = None,
    selection_method: str | None = None,
    qc: str | None = None,
    result_json: str | os.PathLike[str] | None = None,
    error: str | None = None,
    updated_at: str | None = None,
) -> dict[str, str]:
    normalized_status = str(status).strip().lower()
    if normalized_status not in TARGET_STATUS_VALUES:
        raise ValueError(f"unknown target status: {status!r}")
    key = target_status_key(target)
    row = rows.setdefault(key, status_row_for_target(target))
    row["status"] = normalized_status
    if method is not None:
        row["method"] = str(method)
    if quantity is not None:
        row["quantity"] = str(quantity)
    if result_value is not None:
        row["result_value"] = str(result_value)
    if result_unit is not None:
        row["result_unit"] = str(result_unit)
    if selected_region is not None:
        row["selected_region"] = str(selected_region)
    if selection_method is not None:
        row["selection_method"] = str(selection_method)
    if qc is not None:
        row["qc"] = str(qc)
    if result_json is not None:
        row["result_json"] = (
            str(Path(result_json).resolve()) if str(result_json) else ""
        )
    if error is not None:
        row["error"] = str(error)
    row["updated_at"] = updated_at or utc_timestamp()
    return row


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "ok"}


def _fits_header_is_readable(path: Path) -> bool:
    try:
        with fits.open(path, memmap=True) as hdul:
            if not hdul:
                return False
            hdul.verify("exception")
            for hdu in hdul:
                _ = hdu.header
                if hdu.data is not None and hdu.data.size:
                    # Touch the tail without loading the whole image.  A file
                    # truncated after a valid header must never suppress rerun.
                    _ = hdu.data.reshape(-1)[-1]
        return True
    except Exception:
        return False


def result_method(result: Mapping[str, Any]) -> str:
    family = str(result.get("template_family") or "").strip().lower()
    if family:
        return family
    if all(result.get(key) for key in SBF3_FITS_KEYS):
        return "sbf3"
    if all(result.get(key) for key in LEGACY_SBF2_FITS_KEYS):
        return "sbf2_legacy"
    return "unknown"


def measurement_method(result: Mapping[str, Any]) -> str:
    """Return the scientific estimator, falling back to the pipeline family."""
    return str(
        result.get("recommended_measurement_method")
        or result.get("measurement_method")
        or result.get("status_result_method")
        or result_method(result)
    )


def science_status_fields(result: Mapping[str, Any]) -> dict[str, str]:
    """Extract the compact scientific result shown in the target ledger."""
    value = result.get("recommended_mbar_selected")
    if not _finite(value):
        value = result.get("recommended_mbar_weighted")

    quantity = str(
        result.get("recommended_primary_quantity")
        or result.get("primary_quantity")
        or "apparent_sbf_magnitude"
    )
    unit = str(
        result.get("recommended_primary_unit")
        or result.get("primary_unit")
        or "AB mag"
    )
    selected_region = str(
        result.get("recommended_selected_region")
        or result.get("selected_sbf_region")
        or ""
    )
    selection_method = str(
        result.get("recommended_selection_method")
        or result.get("selected_sbf_selection_method")
        or ""
    )
    if not selected_region:
        if _truthy(result.get("recommended_uses_two_annuli")):
            selected_region = "circular_inner_lit+circular_outer_lit"
        elif _finite(result.get("recommended_mbar_inner")):
            selected_region = "circular_inner_lit"
        elif _finite(result.get("recommended_mbar_outer")):
            selected_region = "circular_outer_lit"
    if not selection_method:
        selection_method = (
            "inverse_variance_weighted_two_annuli"
            if "+" in selected_region
            else str(result.get("recommended_method_id") or "recommended_annulus")
        )
    return {
        "quantity": quantity,
        "result_value": repr(float(value)) if _finite(value) else "",
        "result_unit": unit,
        "selected_region": selected_region,
        "selection_method": selection_method,
    }


def validate_reusable_result(
    result_path: str | os.PathLike[str],
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate a result without comparing notebook hashes.

    This is intentionally a cheap restart gate: read FITS headers and the two
    numerical CSVs, but never hash multi-gigabyte inputs or products.
    """
    path = Path(result_path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if result.get("status") != "ok":
        return None
    if _compact_text(result.get("galaxy")).casefold() != target_status_key(target)[2]:
        return None

    target_signal = target_status_key(target)[3]
    target_color = target_status_key(target)[4]
    result_signal = _compact_text(result.get("signal_filter") or "F150W").upper()
    result_color = _compact_text(result.get("color_filter") or "F090W").upper()
    if (result_signal, result_color) != (target_signal, target_color):
        return None

    method = result_method(result)
    fits_keys = SBF3_FITS_KEYS if method == "sbf3" else LEGACY_SBF2_FITS_KEYS
    for key in fits_keys:
        value = result.get(key)
        if not value:
            return None
        artifact = Path(value)
        if not artifact.is_file() or not _fits_header_is_readable(artifact):
            return None
    for key in ("df_sbf_csv", "annulus_summary_csv"):
        value = result.get(key)
        if not value or not Path(value).is_file() or Path(value).stat().st_size <= 0:
            return None
    if not _finite(result.get("recommended_mbar_weighted")):
        return None

    adopted = dict(result)
    adopted["result_json"] = str(path.resolve())
    adopted.setdefault("template_family", "sbf2" if method == "sbf2_legacy" else method)
    adopted.setdefault("signal_filter", result_signal)
    adopted.setdefault("color_filter", result_color)
    adopted["status_result_method"] = method
    adopted["status_adopted_without_sha"] = True
    return adopted


def legacy_result_path(
    target: Mapping[str, Any], legacy_root: str | os.PathLike[str]
) -> Path | None:
    if canonical_program(_field(target, "program")) != "3055":
        return None
    if target_status_key(target)[3:] != ("F150W", "F090W"):
        return None
    galaxy = _compact_text(_field(target, "galaxy", "name", "target"))
    slug = "".join(ch if ch.isalnum() else "_" for ch in galaxy).strip("_")
    return Path(legacy_root) / f"{slug}_result.json"


def find_legacy_reusable_result(
    target: Mapping[str, Any], legacy_root: str | os.PathLike[str]
) -> dict[str, Any] | None:
    path = legacy_result_path(target, legacy_root)
    return None if path is None else validate_reusable_result(path, target)


def _read_csv_rows(path: Any) -> list[dict[str, str]]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def annulus_qc(result: Mapping[str, Any]) -> str:
    """Return compact PASS/WARN/FAIL diagnostics from already-written tables."""
    measurements = _read_csv_rows(result.get("df_sbf_csv"))
    summary = _read_csv_rows(result.get("annulus_summary_csv"))
    if not measurements or not summary:
        return "fail:missing_annulus_tables"

    kmin = float(result.get("recommended_kmin", 0.04))
    kmax = float(result.get("recommended_kmax", 0.25))
    known_regions = ("circular_inner_lit", "circular_outer_lit")
    selected_region = str(
        result.get("recommended_selected_region")
        or result.get("selected_sbf_region")
        or ""
    )
    regions = (
        (selected_region,)
        if selected_region in known_regions
        else known_regions
    )

    def same_window(row: Mapping[str, Any]) -> bool:
        return (
            _finite(row.get("kmin"))
            and _finite(row.get("kmax"))
            and math.isclose(float(row["kmin"]), kmin, abs_tol=1e-9)
            and math.isclose(float(row["kmax"]), kmax, abs_tol=1e-9)
        )

    main = {
        region: next(
            (
                row
                for row in measurements
                if row.get("region") == region and same_window(row)
            ),
            None,
        )
        for region in regions
    }
    failures: list[str] = []
    warnings: list[str] = []
    selected_qc = str(
        result.get("recommended_selected_region_qc_status") or ""
    ).strip().upper()
    if selected_qc == "FAIL":
        failures.append("selected_region_qc_fail")
    elif selected_qc == "WARN":
        warnings.append("selected_region_qc_warn")
    for region, row in main.items():
        short = "inner" if "inner" in region else "outer"
        if row is None:
            failures.append(f"{short}_missing")
            continue
        if "measurement_ok" in row and not _truthy(row.get("measurement_ok")):
            failures.append(f"{short}_measurement_not_ok")
        if not _finite(row.get("mbar_spec")):
            failures.append(f"{short}_mbar_nonfinite")
        if not _finite(row.get("P_fluc")) or float(row["P_fluc"]) <= 0.0:
            failures.append(f"{short}_Pfluc_invalid")
        n_use = row.get("n_use", row.get("usable_pixels"))
        if not _finite(n_use) or float(n_use) < 5000:
            failures.append(f"{short}_too_few_pixels")
        if _finite(row.get("usable_fraction")) and float(row["usable_fraction"]) < 0.50:
            warnings.append(f"{short}_coverage<0.50")
        if _finite(row.get("Pr_over_P0")) and float(row["Pr_over_P0"]) > 0.20:
            warnings.append(f"{short}_Pr/P0>0.20")
        if _finite(row.get("corr")) and float(row["corr"]) < 0.30:
            warnings.append(f"{short}_corr<0.30")

        values = [
            float(item["mbar_spec"])
            for item in measurements
            if item.get("region") == region
            and _finite(item.get("mbar_spec"))
            and (
                "measurement_ok" not in item or _truthy(item.get("measurement_ok"))
            )
        ]
        if len(values) >= 2 and max(values) - min(values) > 0.20:
            warnings.append(f"{short}_k_span>0.20mag")

    inner = main.get(known_regions[0])
    outer = main.get(known_regions[1])
    if (
        inner is not None
        and outer is not None
        and _finite(inner.get("mbar_spec"))
        and _finite(outer.get("mbar_spec"))
        and abs(float(inner["mbar_spec"]) - float(outer["mbar_spec"])) > 0.20
    ):
        warnings.append("annulus_delta>0.20mag")

    if failures:
        return "fail:" + "|".join(dict.fromkeys(failures))
    if warnings:
        return "warn:" + "|".join(dict.fromkeys(warnings))
    return "pass"


def reusable_result_from_status(
    rows: Mapping[tuple[str, ...], Mapping[str, str]],
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
    row = rows.get(target_status_key(target))
    if not row or row.get("status") != "done" or not row.get("result_json"):
        return None
    return validate_reusable_result(row["result_json"], target)
