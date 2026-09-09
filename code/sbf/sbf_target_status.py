#!/usr/bin/env python3
"""RU: читаемый CSV-журнал и проверка готовых F150W-продуктов.
EN: readable target ledger and validation of completed F150W products.

The ledger deliberately identifies work by archive observation and filter pair,
not by a notebook or file hash.  Hashes may still be kept in result provenance,
but editing a notebook must not silently turn a completed target back into work.
"""

from __future__ import annotations

import csv
import io
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from astropy.io import fits
from sbf.sbf_campaign_runtime import atomic_write_text
from sbf.sbf_paths import load_project_json, project_path


# Fixed CSV contract: human-readable identity, state, result and provenance.
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

# Ledger label distinguishes the measured apparent SBF from calibrated distance.
PRIMARY_QUANTITY = "apparent_sbf_magnitude"

# Required full-frame and two-ring products; missing/truncated files force rerun.
REQUIRED_SBF2_FITS_KEYS = (
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

    return atomic_write_text(destination, buffer.getvalue())


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
    """The active status validator accepts the F150W/SBF-2 product contract."""
    family = str(result.get("template_family") or "sbf2").strip().lower()
    return family if family == "sbf2" else "unknown"


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
        or PRIMARY_QUANTITY
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

    This is intentionally a cheap restart gate: check FITS headers/payload tails
    and the presence of both CSVs, but never hash multi-gigabyte products.
    """
    path = project_path(result_path)
    try:
        result = load_project_json(path)
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
    if method != "sbf2":
        return None
    for key in REQUIRED_SBF2_FITS_KEYS:
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
    adopted.setdefault("template_family", "sbf2")
    adopted.setdefault("signal_filter", result_signal)
    adopted.setdefault("color_filter", result_color)
    adopted["status_result_method"] = method
    adopted["status_adopted_without_sha"] = True
    return adopted



def reusable_result_from_status(
    rows: Mapping[tuple[str, ...], Mapping[str, str]],
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
    row = rows.get(target_status_key(target))
    if not row or row.get("status") != "done" or not row.get("result_json"):
        return None
    return validate_reusable_result(row["result_json"], target)
