#!/usr/bin/env python3
"""Offline, resumable GO-3055 F090W SBF campaign.

This is deliberately separate from the frozen F150W runners.  For every
galaxy it performs two durable stages:

1. build the F090W galaxy model, source mask, ``P_r`` inputs and a local
   129x129 F090W PSF ensemble;
2. measure the adopted ``normalized_full_3p5`` branch, reading the saved
   normalized FITS back before the FFT.

The public completion marker is a plain ``target_status.csv``.  Internal
fingerprints are used only to protect numerical caches; they never decide the
human-readable status by themselves.
"""

from __future__ import annotations

import argparse
import atexit
import builtins
import gc
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_PROJECT_CACHE = Path(__file__).resolve().parent.parent / "runs" / ".runtime_cache"
for _cache_directory in (
    _PROJECT_CACHE / "matplotlib",
    _PROJECT_CACHE / "xdg",
    _PROJECT_CACHE / "astropy",
):
    _cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_CACHE / "xdg"))
os.environ.setdefault("ASTROPY_CACHE_DIR", str(_PROJECT_CACHE / "astropy"))
warnings.filterwarnings(
    "ignore", message=r"XDG_CACHE_HOME is set to .* takes precedence.*"
)

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.data import CacheMissingWarning

warnings.filterwarnings("ignore", category=CacheMissingWarning)

from run_sbf_2_batch import (
    SBF2_REQUIRED_FITS_KEYS,
    SBF2_REQUIRED_TABLE_KEYS,
    acquire_campaign_lock,
    execute_template_for_target,
    fits_is_readable,
    input_fingerprint,
    local_target_files,
    read_targets_from_csv,
    release_campaign_lock,
    validate_offline_dependencies,
)
from sbf090_pipeline_support import (
    F090W_INNER_MASK_GUARD_METHOD,
    F090W_INNER_MASK_GUARD_TARGETS,
    F090W_ISOPHOTE_METHOD,
    F090W_ISOPHOTE_QC_LIMITS,
    F090W_MASK_METHOD,
    F090W_MIN_WORKING_ISOPHOTES,
    F090W_SOURCE_SCHEMA,
    F090W_PSF_SIZE,
    build_f090w_template,
    isophote_sequence_qc,
    load_f090w_psf_cache,
)
from sbf2_normalized_winsor_core import (
    ExperimentConfig,
    galaxy_slug,
    inspect_source,
    load_result_tables,
    process_target,
)
from sbf_campaign_runtime import atomic_write_json, atomic_write_text


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "targets_go3055_f090w_manifest.csv"
DEFAULT_BASE_NOTEBOOK = SCRIPT_DIR / "sbf-2.ipynb"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "sbf_f090w_go3055"
DEFAULT_STPSF_DATA = Path.home() / "data" / "stpsf-data"
DEFAULT_WSS_OPD = DEFAULT_DATA_ROOT / "wss_opd"
SIGNAL_FILTER = "F090W"
AUXILIARY_FILTER = "F150W"
MAIN_KMIN = 0.04
MAIN_KMAX = 0.25
ESTIMATED_CAMPAIGN_GB = 50.0
FINAL_TABLE_KEYS = {
    "power_spectra", "fit_per_psf", "fit_summary", "clipping",
    "production_closure", "combined_annuli",
}
SOURCE_SCHEMA_REPAIR_TARGETS = {
    "NGC 1380", "NGC 1399", "NGC 1404", "NGC 4374", "NGC 4406",
    "NGC 4472", "NGC 4486", "NGC 4552", "NGC 4621", "NGC 4636",
    "NGC 4649", "NGC 4697", "NGC 1549", "NGC 3379",
}

STATUS_COLUMNS = [
    "galaxy",
    "status",
    "stage",
    "attempt",
    "started_at",
    "updated_at",
    "finished_at",
    "message",
    "pid",
    "source_result",
    "final_result",
]


_original_print = builtins.print
_LOG_GALAXY = "campaign"
_LOG_QUEUE_INDEX = 0
_LOG_QUEUE_TOTAL = 14
_LEADING_TIMESTAMP = re.compile(
    r"^(?:\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2}\]\s*)+"
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_log_context(galaxy: str, queue_index: int, queue_total: int) -> None:
    """Set the prefix used by every subsequent line printed in this process."""

    global _LOG_GALAXY, _LOG_QUEUE_INDEX, _LOG_QUEUE_TOTAL
    _LOG_GALAXY = str(galaxy)
    _LOG_QUEUE_INDEX = int(queue_index)
    _LOG_QUEUE_TOTAL = int(queue_total)


def timestamped_print(*args, **kwargs) -> None:
    """Print every physical line with time, galaxy and queue position."""

    separator = kwargs.pop("sep", " ")
    ending = kwargs.pop("end", "\n")
    stream = kwargs.pop("file", None)
    flush = kwargs.pop("flush", True)
    if kwargs:
        unknown = next(iter(kwargs))
        raise TypeError(f"invalid keyword argument for print(): {unknown!r}")
    if separator is None:
        separator = " "
    if ending is None:
        ending = "\n"

    text = separator.join(str(value) for value in args)
    lines = text.splitlines() or [""]
    prefix = (
        f"[{time.strftime('%H:%M:%S')}, {_LOG_GALAXY}, "
        f"{_LOG_QUEUE_INDEX}/{_LOG_QUEUE_TOTAL}]"
    )
    formatted = "\n".join(
        f"{prefix} {_LEADING_TIMESTAMP.sub('', line)}" for line in lines
    )
    print_kwargs = {"end": ending, "flush": flush}
    if stream is not None:
        print_kwargs["file"] = stream
    _original_print(formatted, **print_kwargs)


def free_space_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def campaign_paths(run_root: Path) -> dict[str, Path]:
    run_root = Path(run_root).resolve()
    return {
        "root": run_root,
        "status": run_root / "target_status.csv",
        "source_batch": run_root / "source_batch",
        "source_products": run_root / "source_products",
        "final": run_root / "final",
        "products": run_root / "products",
        "logs": run_root / "logs",
        "campaign_log": run_root / "campaign.log",
        "generated_template": run_root / "generated" / "sbf-f090w.ipynb",
        "summary": run_root / "f090w_sbf_results.csv",
        "input_report": run_root / "input_readiness.csv",
    }


def read_targets(manifest: Path, data_root: Path) -> list[dict[str, Any]]:
    targets = read_targets_from_csv(manifest, data_root)
    if len(targets) != 14:
        raise RuntimeError(
            f"F090W GO-3055 manifest must contain 14 targets, found {len(targets)}"
        )
    names = []
    for target in targets:
        pair = (target["signal_filter"], target["color_filter"])
        if pair != (SIGNAL_FILTER, AUXILIARY_FILTER):
            raise RuntimeError(
                f"{target['name']}: filter pair {pair}, expected "
                f"{(SIGNAL_FILTER, AUXILIARY_FILTER)}"
            )
        if target["name"] in names:
            raise RuntimeError(f"duplicate target: {target['name']}")
        names.append(target["name"])
    return targets


def select_targets(
    targets: list[dict[str, Any]], galaxies: list[str] | None
) -> list[dict[str, Any]]:
    if not galaxies:
        return targets
    wanted = {" ".join(name.upper().split()) for name in galaxies}
    selected = [
        target for target in targets
        if " ".join(target["name"].upper().split()) in wanted
    ]
    found = {" ".join(target["name"].upper().split()) for target in selected}
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError("unknown galaxies: " + ", ".join(missing))
    return selected


def inspect_local_inputs(
    targets: list[dict[str, Any]], data_root: Path
) -> pd.DataFrame:
    rows = []
    for target in targets:
        files = local_target_files(target, data_root)
        row: dict[str, Any] = {
            "galaxy": target["name"],
            "signal_filter": target["signal_filter"],
            "auxiliary_filter": target["color_filter"],
            "signal_path": str(files["signal"]),
            "auxiliary_path": str(files["color"]),
            "ready": False,
            "message": "",
        }
        try:
            for role, expected in (
                ("signal", SIGNAL_FILTER), ("color", AUXILIARY_FILTER)
            ):
                path = files[role]
                if not path.is_file():
                    raise FileNotFoundError(path)
                header = fits.getheader(path, 0)
                actual = str(header.get("FILTER", "")).strip().upper()
                if actual != expected:
                    raise RuntimeError(
                        f"{role} FILTER={actual}, expected {expected}"
                    )
                with fits.open(path, memmap=True) as hdul:
                    if "SCI" not in hdul or hdul["SCI"].data is None:
                        raise RuntimeError(f"{role}: SCI extension is absent")
                    _ = hdul["SCI"].data.reshape(-1)[-1]
            row["ready"] = True
            row["message"] = "ok"
        except Exception as error:
            row["message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def _read_status(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=STATUS_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in STATUS_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[STATUS_COLUMNS].copy()


def _write_status(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame[STATUS_COLUMNS].to_csv(buffer, index=False)
    atomic_write_text(path, buffer.getvalue())


def update_status(
    status_path: Path,
    galaxy: str,
    status: str,
    *,
    stage: str,
    message: str = "",
    source_result: str = "",
    final_result: str = "",
    increment_attempt: bool = False,
) -> None:
    frame = _read_status(status_path)
    if galaxy not in set(frame.get("galaxy", [])):
        frame.loc[len(frame)] = {column: "" for column in STATUS_COLUMNS}
        frame.loc[len(frame) - 1, "galaxy"] = galaxy
    index = frame.index[frame["galaxy"].eq(galaxy)][0]
    now = timestamp()
    previous_attempt = int(frame.at[index, "attempt"] or 0)
    frame.at[index, "status"] = status
    frame.at[index, "stage"] = stage
    frame.at[index, "attempt"] = str(
        previous_attempt + 1 if increment_attempt else previous_attempt
    )
    if status == "running" and not frame.at[index, "started_at"]:
        frame.at[index, "started_at"] = now
    frame.at[index, "updated_at"] = now
    if status == "ok":
        frame.at[index, "finished_at"] = now
    elif status in {"pending", "running", "interrupted", "failed"}:
        frame.at[index, "finished_at"] = ""
    frame.at[index, "message"] = message
    frame.at[index, "pid"] = str(os.getpid()) if status == "running" else ""
    if source_result:
        frame.at[index, "source_result"] = source_result
    if final_result:
        frame.at[index, "final_result"] = final_result
    _write_status(status_path, frame)


def source_result_path(paths: dict[str, Path], galaxy: str) -> Path:
    return paths["source_batch"] / f"{galaxy_slug(galaxy)}_result.json"


def final_result_path(paths: dict[str, Path], galaxy: str) -> Path:
    return paths["final"] / "batch" / f"{galaxy_slug(galaxy)}_result.json"


def invalidate_completion_markers(
    paths: dict[str, Path], galaxy: str, *, source: bool
) -> None:
    """Invalidate only small completion markers before a forced rebuild."""
    if source:
        source_result_path(paths, galaxy).unlink(missing_ok=True)
    batch_root = paths["final"] / "batch"
    if batch_root.is_dir():
        for marker in batch_root.rglob(f"{galaxy_slug(galaxy)}_result.json"):
            marker.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


F090W_QC_METRIC_KEYS = (
    "n_isophotes",
    "max_sma_px",
    "median_center_shift_px",
    "max_center_shift_px",
    "quality_n_isophotes",
    "quality_sma_min_px",
    "quality_sma_max_px",
    "stop2_count",
    "stop2_fraction",
    "max_consecutive_stop2",
    "frozen_stop_count",
    "frozen_stop_fraction",
    "max_consecutive_frozen_stop",
    "singular_stop_count",
    "max_center_step_px",
    "max_eps_step",
    "max_pa_step_rad",
    "max_shape_step",
    "max_intensity_rise_fraction",
)


def _selected_isophote_qc(attempts_path: Path) -> dict[str, Any]:
    """Recheck the one selected full isophote solution from its durable CSV."""

    attempts = pd.read_csv(attempts_path)
    if "selected" not in attempts or "phase" not in attempts:
        return {
            "passed": False,
            "reason": "attempt table has no selected full solution",
            "method": F090W_ISOPHOTE_METHOD,
        }
    selected_flag = (
        attempts["selected"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes"})
    )
    selected = attempts.loc[
        selected_flag & attempts["phase"].astype(str).eq("full")
    ]
    if len(selected) != 1:
        return {
            "passed": False,
            "reason": f"expected one selected full solution, found {len(selected)}",
            "method": F090W_ISOPHOTE_METHOD,
        }
    index = int(selected.index[0])
    row = selected.iloc[0]
    missing = [key for key in F090W_QC_METRIC_KEYS if key not in row.index]
    if missing:
        return {
            "passed": False,
            "reason": "selected solution misses QC metrics: " + ", ".join(missing),
            "method": F090W_ISOPHOTE_METHOD,
        }

    details: dict[str, Any] = {}
    integer_keys = {
        "n_isophotes", "quality_n_isophotes", "stop2_count",
        "max_consecutive_stop2", "frozen_stop_count",
        "max_consecutive_frozen_stop", "singular_stop_count",
    }
    for key in F090W_QC_METRIC_KEYS:
        value = row[key]
        details[key] = (
            int(value) if key in integer_keys and pd.notna(value)
            else float(value) if pd.notna(value)
            else np.nan
        )
    required_sma = float(row.get("required_sma_px", np.nan))
    passed, reason = isophote_sequence_qc(
        details,
        required_sma_px=required_sma,
        min_isophotes=F090W_MIN_WORKING_ISOPHOTES,
        max_median_center_shift_px=F090W_ISOPHOTE_QC_LIMITS[
            "max_median_center_shift_px"
        ],
        max_center_shift_px=F090W_ISOPHOTE_QC_LIMITS["max_center_shift_px"],
        max_stop2_fraction=F090W_ISOPHOTE_QC_LIMITS["max_stop2_fraction"],
        max_consecutive_stop2=F090W_ISOPHOTE_QC_LIMITS[
            "max_consecutive_stop2"
        ],
        max_frozen_stop_fraction=F090W_ISOPHOTE_QC_LIMITS[
            "max_frozen_stop_fraction"
        ],
        max_consecutive_frozen_stop=F090W_ISOPHOTE_QC_LIMITS[
            "max_consecutive_frozen_stop"
        ],
        max_singular_stop_count=F090W_ISOPHOTE_QC_LIMITS[
            "max_singular_stop_count"
        ],
        max_center_step_px=F090W_ISOPHOTE_QC_LIMITS["max_center_step_px"],
        max_eps_step=F090W_ISOPHOTE_QC_LIMITS["max_eps_step"],
        max_pa_step_rad=None,
        max_shape_step=F090W_ISOPHOTE_QC_LIMITS["max_shape_step"],
        max_intensity_rise_fraction=F090W_ISOPHOTE_QC_LIMITS[
            "max_intensity_rise_fraction"
        ],
    )
    status_ok = str(row.get("status", "")).strip().lower() == "working"
    passed = bool(passed and status_ok)
    if not status_ok:
        reason = "; ".join(filter(None, [reason, "selected status is not working"]))

    json_metrics = {
        key: (None if not np.isfinite(float(value)) else value)
        for key, value in details.items()
    }
    return {
        "passed": passed,
        "reason": reason,
        "method": F090W_ISOPHOTE_METHOD,
        "attempt_row": index,
        "dataset": str(row.get("dataset", "")),
        "start_sma_px": float(row.get("start_sma_px", np.nan)),
        "step_px": float(row.get("step_px", np.nan)),
        "fix_center": str(row.get("fix_center", "")).lower() == "true",
        "seed_source": str(row.get("seed_source", "")),
        "seed_eps": float(row.get("seed_eps", np.nan)),
        "seed_pa_rad": float(row.get("seed_pa_rad", np.nan)),
        "required_sma_px": required_sma,
        "metrics": json_metrics,
    }


def source_result_valid(
    paths: dict[str, Path], galaxy: str
) -> tuple[bool, dict[str, Any] | None, str]:
    result_path = source_result_path(paths, galaxy)
    result = _read_json(result_path)
    if result is None or result.get("status") != "ok":
        return False, result, "source result is absent or incomplete"
    if (
        str(result.get("signal_filter", "")).upper() != SIGNAL_FILTER
        or str(result.get("color_filter", "")).upper() != AUXILIARY_FILTER
    ):
        return False, result, "source result has the wrong filter pair"
    for role in ("signal", "color"):
        path_text = result.get(f"{role}_path")
        if not path_text:
            return False, result, f"source result misses {role}_path"
        current = input_fingerprint(Path(path_text))
        if result.get(f"{role}_fingerprint") != current:
            return False, result, f"{role} input changed after source stage"
    required_fits = [result.get(key) for key in SBF2_REQUIRED_FITS_KEYS]
    required_tables = [result.get(key) for key in SBF2_REQUIRED_TABLE_KEYS]
    if any(not value for value in required_fits + required_tables):
        return False, result, "source result misses required product paths"
    for value in required_fits:
        readable, error = fits_is_readable(value)
        if not readable:
            return False, result, f"unreadable source FITS: {value}: {error}"
    for value in required_tables:
        try:
            if pd.read_csv(value).empty:
                raise ValueError("empty table")
        except Exception as error:
            return False, result, f"unreadable source table: {value}: {error}"
    source_schema = int(result.get("f090_source_schema", 1))
    if galaxy in SOURCE_SCHEMA_REPAIR_TARGETS and source_schema < F090W_SOURCE_SCHEMA:
        return (
            False,
            result,
            "legacy F090W source must be rebuilt with the current isophote method",
        )
    if source_schema >= F090W_SOURCE_SCHEMA:
        if result.get("f090_isophote_method") != F090W_ISOPHOTE_METHOD:
            return False, result, "source result uses a different isophote method"
        if result.get("f090_mask_method") != F090W_MASK_METHOD:
            return False, result, "source result uses a different contaminant-mask method"
        diagnostics = result.get("f090_diagnostic_tables", {})
        expected = {
            "center", "isophote_attempts", "isophotes",
            "external_contaminants",
        }
        if set(diagnostics) != expected:
            return False, result, "source result misses F090W centre/isophote diagnostics"
        for name, value in diagnostics.items():
            try:
                table = pd.read_csv(value)
                if name != "external_contaminants" and table.empty:
                    raise ValueError("empty table")
            except Exception as error:
                return False, result, f"unreadable F090W {name} table: {value}: {error}"
        try:
            table_qc = _selected_isophote_qc(
                Path(diagnostics["isophote_attempts"])
            )
        except Exception as error:
            return False, result, f"cannot recheck F090W isophote QC: {error}"
        if not table_qc["passed"]:
            return (
                False,
                result,
                "F090W isophote geometry QC did not pass: "
                + str(table_qc.get("reason", "")),
            )
        saved_qc = result.get("f090_isophote_qc", {})
        if (
            not bool(result.get("f090_isophote_qc_passed", False))
            or not bool(saved_qc.get("passed", False))
            or saved_qc.get("method") != F090W_ISOPHOTE_METHOD
        ):
            return False, result, "F090W isophote geometry QC did not pass"
        contaminant_path = result.get("f090_external_contaminant_mask_fits")
        readable, error = fits_is_readable(contaminant_path or "")
        if not readable:
            return False, result, f"unreadable F090W contaminant mask: {error}"
        if galaxy in F090W_INNER_MASK_GUARD_TARGETS:
            guard = result.get("f090_isophote_inner_mask_guard", {})
            if (
                not bool(guard.get("enabled", False))
                or guard.get("method") != F090W_INNER_MASK_GUARD_METHOD
                or bool(guard.get("affects_sbf_measurement_mask", True))
            ):
                return False, result, "required F090W inner isophote-mask guard is absent"
    psf_path = Path(result["output_dir"]) / f"{result['stem']}_psf_129.fits"
    cache = load_f090w_psf_cache(
        psf_path,
        Path(result["signal_path"]),
        Path(result["output_dir"]),
        str(result["stem"]),
        expected_filter=SIGNAL_FILTER,
        expected_size=F090W_PSF_SIZE,
    )
    if cache is None:
        return False, result, f"invalid F090W PSF cache: {psf_path}"
    return True, result, "ok"


def _candidate_ring_paths(result: dict[str, Any]) -> dict[str, Path]:
    candidate = result.get("candidate_branch", "normalized_full_3p5")
    ring_paths = {
        str(item.get("ring")): Path(item["path"])
        for item in result.get("normalized_fits", [])
        if item.get("branch") == candidate and item.get("ring") in {"inner", "outer"}
    }
    return ring_paths


def serialized_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return exactly the representation written to a JSON result file."""
    return json.loads(json.dumps(asdict(config)))


def final_result_valid(
    paths: dict[str, Path], galaxy: str, config: ExperimentConfig
) -> tuple[bool, dict[str, Any] | None, str]:
    result_path = final_result_path(paths, galaxy)
    result = _read_json(result_path)
    if result is None or result.get("status") != "ok":
        return False, result, "final result is absent or incomplete"
    source_ok, _, source_message = source_result_valid(paths, galaxy)
    if not source_ok:
        return False, result, f"source stage is not reusable: {source_message}"
    # JSON has no tuples: ``kmins`` is written as a list.  Compare the two
    # configurations only after the same JSON normalization, otherwise every
    # completed target would be rejected on restart.
    expected_config = serialized_config(config)
    if result.get("config") != expected_config:
        return False, result, "final result uses different settings"
    try:
        current_source_key = inspect_source(
            galaxy, paths["source_batch"]
        )["source_key"]
    except Exception as error:
        return False, result, f"source products are not valid: {error}"
    if result.get("source_key") != current_source_key:
        return False, result, "final result belongs to older source products"
    full_path = Path(result.get("full_normalized_residual_fits", ""))
    readable, error = fits_is_readable(full_path)
    if not readable:
        return False, result, f"unreadable normalized full FITS: {error}"
    ring_paths = _candidate_ring_paths(result)
    if set(ring_paths) != {"inner", "outer"}:
        return False, result, "inner/outer normalized FITS are absent"
    for ring, path in ring_paths.items():
        readable, error = fits_is_readable(path)
        if not readable:
            return False, result, f"unreadable {ring} FITS: {error}"
    table_paths = result.get("table_paths", {})
    if set(table_paths) != FINAL_TABLE_KEYS:
        return False, result, "final result misses required spectral tables"
    for value in table_paths.values():
        try:
            if pd.read_csv(value).empty:
                raise ValueError("empty table")
        except Exception as table_error:
            return False, result, f"unreadable final table: {table_error}"
    return True, result, "ok"


def sync_status(
    targets: list[dict[str, Any]], paths: dict[str, Path], config: ExperimentConfig
) -> pd.DataFrame:
    old = _read_status(paths["status"])
    old_rows = {
        row["galaxy"]: row.to_dict() for _, row in old.iterrows()
        if row.get("galaxy")
    }
    rows = []
    for target in targets:
        galaxy = target["name"]
        previous = old_rows.get(galaxy, {})
        final_ok, _, final_message = final_result_valid(paths, galaxy, config)
        source_ok, _, source_message = source_result_valid(paths, galaxy)
        previous_status = str(previous.get("status", ""))
        previous_pid = str(previous.get("pid", ""))
        active_pid = _pid_is_alive(previous_pid) if previous_status == "running" else False
        if final_ok:
            status, stage, message = "ok", "complete", "verified products found"
            pid = ""
        elif active_pid:
            status = "running"
            stage = str(previous.get("stage", "running"))
            message = f"worker pid {previous_pid} is still active"
            pid = previous_pid
        elif source_ok:
            status = (
                "interrupted" if previous_status == "running"
                else "failed" if previous_status == "failed"
                else "pending"
            )
            stage = "spectral measurement"
            message = final_message
            pid = ""
        else:
            status = (
                "interrupted" if previous_status == "running"
                else previous_status if previous_status in {"failed", "interrupted"}
                else "pending"
            )
            stage = "model/mask/PSF"
            message = source_message
            pid = ""
        rows.append({
            "galaxy": galaxy,
            "status": status,
            "stage": stage,
            "attempt": str(previous.get("attempt", "0") or "0"),
            "started_at": str(previous.get("started_at", "")),
            "updated_at": timestamp(),
            "finished_at": (
                str(previous.get("finished_at", "")) if final_ok else ""
            ),
            "message": message,
            "pid": pid,
            "source_result": str(source_result_path(paths, galaxy)) if source_ok else "",
            "final_result": str(final_result_path(paths, galaxy)) if final_ok else "",
        })
    frame = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    _write_status(paths["status"], frame)
    return frame


def _pid_is_alive(value: str | int | None) -> bool:
    try:
        pid = int(value or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    return True


def _atomic_symlink(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(source, destination.parent))
    os.replace(temporary, destination)


def _paper_iv_row(galaxy: str, data_root: Path) -> pd.Series:
    frame = pd.read_csv(Path(data_root) / "go3055_paper_iv_metadata.csv")
    selected = frame[frame["galaxy"].eq(galaxy)]
    if len(selected) != 1:
        raise RuntimeError(f"{galaxy}: Paper IV metadata row is absent")
    return selected.iloc[0]


def build_science_row(
    galaxy: str, final_result: dict[str, Any], data_root: Path
) -> dict[str, Any]:
    tables = load_result_tables(final_result)
    combined = tables["combined_annuli"]
    candidate = final_result.get("candidate_branch", "normalized_full_3p5")
    selected = combined[
        combined["branch"].eq(candidate)
        & np.isclose(combined["requested_kmin"], MAIN_KMIN)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"{galaxy}: expected one adopted k-window row")
    row = selected.iloc[0]
    clipping = tables["clipping"]
    clipping = clipping[
        clipping["branch"].eq(candidate)
    ].set_index("ring")
    if not {"inner", "outer"}.issubset(clipping.index):
        raise RuntimeError(f"{galaxy}: clipping summary misses a ring")
    metadata = _paper_iv_row(galaxy, data_root)
    extinction = float(metadata["A_F090W"])
    sigma_extinction = float(metadata["sigma_A_F090W"])

    inner = float(row["mbar_inner"])
    outer = float(row["mbar_outer"])
    weighted = float(row["mbar_weighted"])
    sigma_inner = float(row["sigma_inner"])
    sigma_outer = float(row["sigma_outer"])
    sigma_weighted = float(row["sigma_adopted_internal"])
    return {
        "galaxy": galaxy,
        "filter": SIGNAL_FILTER,
        "kmin": MAIN_KMIN,
        "kmax": MAIN_KMAX,
        "mbar_inner_observed": inner,
        "sigma_inner_internal": sigma_inner,
        "mbar_inner_0": inner - extinction,
        "sigma_inner_0": float(np.hypot(sigma_inner, sigma_extinction)),
        "mbar_outer_observed": outer,
        "sigma_outer_internal": sigma_outer,
        "mbar_outer_0": outer - extinction,
        "sigma_outer_0": float(np.hypot(sigma_outer, sigma_extinction)),
        "mbar_weighted_observed": weighted,
        "sigma_weighted_internal": sigma_weighted,
        "mbar_weighted_0": weighted - extinction,
        "sigma_weighted_0": float(np.hypot(sigma_weighted, sigma_extinction)),
        "A_F090W": extinction,
        "sigma_A_F090W": sigma_extinction,
        "clipped_inner_percent": 100.0 * float(
            clipping.loc["inner", "changed_fraction"]
        ),
        "clipped_outer_percent": 100.0 * float(
            clipping.loc["outer", "changed_fraction"]
        ),
    }


def materialize_products(
    paths: dict[str, Path], galaxy: str, final_result: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    source_ok, source_result, message = source_result_valid(paths, galaxy)
    if not source_ok or source_result is None:
        raise RuntimeError(f"{galaxy}: source product gate failed: {message}")
    ring_paths = _candidate_ring_paths(final_result)
    product_dir = paths["products"] / galaxy_slug(galaxy)
    stem = str(source_result["stem"])
    stable = {
        "model": product_dir / f"{stem}_01_model.fits",
        "normalized_full": product_dir
        / f"{stem}_02_normalized_full_clip_3p5sigma.fits",
        "normalized_inner": product_dir
        / f"{stem}_03_normalized_inner_fft_input.fits",
        "normalized_outer": product_dir
        / f"{stem}_04_normalized_outer_fft_input.fits",
        "psf_129": product_dir / f"{stem}_05_psf_129.fits",
    }
    sources = {
        "model": Path(source_result["model_full_fits"]),
        "normalized_full": Path(final_result["full_normalized_residual_fits"]),
        "normalized_inner": ring_paths["inner"],
        "normalized_outer": ring_paths["outer"],
        "psf_129": Path(source_result["output_dir"]) / f"{stem}_psf_129.fits",
    }
    for name in stable:
        _atomic_symlink(sources[name], stable[name])
        readable, error = fits_is_readable(stable[name])
        if not readable:
            raise RuntimeError(f"{galaxy}: published {name} is unreadable: {error}")

    diagnostic_products: dict[str, Path] = {}
    if int(source_result.get("f090_source_schema", 1)) >= F090W_SOURCE_SCHEMA:
        diagnostics = source_result["f090_diagnostic_tables"]
        diagnostic_names = (
            "center", "isophote_attempts", "isophotes",
            "external_contaminants",
        )
        for index, name in enumerate(diagnostic_names, start=6):
            destination = product_dir / f"{stem}_{index:02d}_{name}.csv"
            _atomic_symlink(Path(diagnostics[name]), destination)
            table = pd.read_csv(destination)
            if name != "external_contaminants" and table.empty:
                raise RuntimeError(f"{galaxy}: published {name} table is empty")
            diagnostic_products[name] = destination

    science_row = build_science_row(galaxy, final_result, data_root)
    per_target_csv = product_dir / f"{stem}_f090w_sbf_summary.csv"
    buffer = io.StringIO()
    pd.DataFrame([science_row]).round(3).to_csv(
        buffer, index=False, float_format="%.3f"
    )
    atomic_write_text(per_target_csv, buffer.getvalue())
    manifest = {
        "galaxy": galaxy,
        "filter": SIGNAL_FILTER,
        "created_at": timestamp(),
        "source_schema": int(source_result["f090_source_schema"]),
        "isophote_method": source_result["f090_isophote_method"],
        "mask_method": source_result["f090_mask_method"],
        "isophote_qc": source_result["f090_isophote_qc"],
        "products": {name: str(path) for name, path in stable.items()},
        "diagnostics": {
            name: str(path) for name, path in diagnostic_products.items()
        },
        "sources": {name: str(path.resolve()) for name, path in sources.items()},
        "summary_csv": str(per_target_csv.resolve()),
        "source_result": str(source_result_path(paths, galaxy)),
        "final_result": str(final_result_path(paths, galaxy)),
        "notes": {
            "normalized_full": (
                "SCI-background-model divided by sqrt(model), then winsorized "
                "at 3.5 sigma over the full valid model support"
            ),
            "normalized_inner_outer": (
                "exact ring FFT inputs after ring-specific mean subtraction; "
                "PRIMARY is NaN outside the ring"
            ),
        },
    }
    atomic_write_json(product_dir / "products.json", manifest, sort_keys=False)
    return science_row


def rebuild_global_summary(
    targets: list[dict[str, Any]], paths: dict[str, Path],
    config: ExperimentConfig, data_root: Path,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        galaxy = target["name"]
        valid, result, _ = final_result_valid(paths, galaxy, config)
        if not valid or result is None:
            continue
        rows.append(materialize_products(paths, galaxy, result, data_root))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        order = {target["name"]: index for index, target in enumerate(targets)}
        frame["_order"] = frame["galaxy"].map(order)
        frame = frame.sort_values("_order").drop(columns="_order")
    buffer = io.StringIO()
    frame.round(3).to_csv(buffer, index=False, float_format="%.3f")
    atomic_write_text(paths["summary"], buffer.getvalue())
    return frame


def run_source_stage(
    target: dict[str, Any], args: argparse.Namespace,
    paths: dict[str, Path], template: Path,
) -> dict[str, Any]:
    galaxy = target["name"]
    files = local_target_files(target, args.data_root)
    output_dir = paths["source_products"] / galaxy_slug(galaxy)
    output_dir.mkdir(parents=True, exist_ok=True)
    psf_path = output_dir / f"{files['signal'].stem}_psf_129.fits"
    if args.rebuild_psf:
        psf_path.unlink(missing_ok=True)
    print(f"{galaxy}: building F090W model/mask/P_r/PSF")
    result = execute_template_for_target(
        template_path=template,
        galaxy=galaxy,
        signal_path=files["signal"],
        color_path=files["color"],
        batch_root=paths["source_batch"],
        signal_filter=SIGNAL_FILTER,
        color_filter=AUXILIARY_FILTER,
        output_dir=output_dir,
        job_id=f"f090w-{galaxy_slug(galaxy).lower()}",
        worker_log_path=paths["logs"] / f"{galaxy_slug(galaxy)}.log",
        cell_timings_path=paths["logs"] / f"{galaxy_slug(galaxy)}_cells.jsonl",
    )
    diagnostic_tables = {
        "center": output_dir / f"{files['signal'].stem}_sbf_center.csv",
        "isophote_attempts": output_dir / f"{files['signal'].stem}_sbf_isophote_attempts.csv",
        "isophotes": output_dir / f"{files['signal'].stem}_sbf_isophotes.csv",
        "external_contaminants": output_dir / f"{files['signal'].stem}_sbf_external_contaminants.csv",
    }
    result["f090_source_schema"] = F090W_SOURCE_SCHEMA
    result["f090_isophote_method"] = F090W_ISOPHOTE_METHOD
    result["f090_mask_method"] = F090W_MASK_METHOD
    result["f090_isophote_inner_mask_guard"] = {
        "enabled": galaxy in F090W_INNER_MASK_GUARD_TARGETS,
        "method": (
            F090W_INNER_MASK_GUARD_METHOD
            if galaxy in F090W_INNER_MASK_GUARD_TARGETS
            else "none"
        ),
        "radius_definition": "inner_sbf_boundary",
        "affects_sbf_measurement_mask": False,
    }
    result["f090_diagnostic_tables"] = {
        key: str(path.resolve()) for key, path in diagnostic_tables.items()
    }
    isophote_qc = _selected_isophote_qc(diagnostic_tables["isophote_attempts"])
    result["f090_isophote_qc"] = isophote_qc
    result["f090_isophote_qc_passed"] = bool(isophote_qc["passed"])
    if not isophote_qc["passed"]:
        raise RuntimeError(
            f"{galaxy}: selected isophotes failed the durable QC: "
            f"{isophote_qc.get('reason', '')}"
        )
    result["f090_external_contaminant_mask_fits"] = str(
        (output_dir / f"{files['signal'].stem}_sbf_external_contaminant_mask.fits").resolve()
    )
    # The inherited executor keeps its provenance-rich JSON name.  Publish a
    # second, stable per-galaxy JSON atomically: this is the deliberately
    # simple source-stage completion marker used by restart logic and by the
    # normalized spectral core.
    atomic_write_json(
        source_result_path(paths, galaxy), result, sort_keys=False
    )
    valid, validated_result, message = source_result_valid(paths, galaxy)
    if not valid or validated_result is None:
        raise RuntimeError(f"{galaxy}: source completion gate failed: {message}")
    return validated_result


def run_worker(args: argparse.Namespace) -> int:
    paths = campaign_paths(args.run_root)
    targets = read_targets(args.manifest, args.data_root)
    selected = select_targets(targets, [args.galaxy])
    if len(selected) != 1:
        raise RuntimeError("worker requires exactly one galaxy")
    target = selected[0]
    galaxy = target["name"]
    set_log_context(galaxy, args.queue_index, args.queue_total)
    config = ExperimentConfig(
        normalized_sigma=3.5,
        kmins=(0.01, 0.03, 0.04),
        kmax=0.25,
        k_bins=80,
        e_realizations=args.e_realizations,
        random_seed=1489,
        fft_workers=args.fft_workers,
        min_modes_per_bin=10,
        save_ring_fft_fits=True,
        save_all_branch_fits=False,
    )
    log_path = paths["logs"] / f"{galaxy_slug(galaxy)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        tee_out = Tee(sys.stdout, log_handle)
        tee_err = Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            builtins.print = timestamped_print
            update_status(
                paths["status"], galaxy, "running", stage="starting",
                increment_attempt=True,
            )
            try:
                force_source = bool(args.force_source or args.rebuild_psf)
                force_spectra = bool(
                    args.force_spectra
                    or args.rebuild_input_cache
                    or args.rebuild_expectation_cache
                )
                if force_source:
                    invalidate_completion_markers(
                        paths, galaxy, source=True
                    )
                elif force_spectra:
                    invalidate_completion_markers(
                        paths, galaxy, source=False
                    )
                source_ok, _, _ = source_result_valid(paths, galaxy)
                if force_source or not source_ok:
                    update_status(
                        paths["status"], galaxy, "running",
                        stage="model/mask/PSF",
                    )
                    run_source_stage(target, args, paths, args.generated_template)
                else:
                    print(f"{galaxy}: valid source products reused")

                update_status(
                    paths["status"], galaxy, "running",
                    stage="normalized residual + FFT",
                    source_result=str(source_result_path(paths, galaxy)),
                )
                result = process_target(
                    galaxy=galaxy,
                    source_batch_root=paths["source_batch"],
                    output_root=paths["final"],
                    config=config,
                    force=(
                        force_spectra
                    ),
                    rebuild_input_cache=args.rebuild_input_cache,
                    rebuild_expectation_cache=args.rebuild_expectation_cache,
                )
                valid, validated_result, message = final_result_valid(
                    paths, galaxy, config
                )
                if not valid or validated_result is None:
                    raise RuntimeError(
                        f"{galaxy}: final completion gate failed: {message}"
                    )
                materialize_products(
                    paths, galaxy, validated_result, args.data_root
                )
                update_status(
                    paths["status"], galaxy, "ok", stage="complete",
                    message="all four science FITS, PSF and CSV verified",
                    source_result=str(source_result_path(paths, galaxy)),
                    final_result=str(final_result_path(paths, galaxy)),
                )
                print(f"{galaxy}: complete -> {result['run_dir']}")
                return 0
            except KeyboardInterrupt:
                update_status(
                    paths["status"], galaxy, "interrupted",
                    stage="interrupted", message="KeyboardInterrupt",
                )
                return 130
            except BaseException as error:
                failure = {
                    "galaxy": galaxy,
                    "timestamp": timestamp(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(
                    paths["logs"] / f"{galaxy_slug(galaxy)}_failure.json",
                    failure,
                    sort_keys=False,
                )
                update_status(
                    paths["status"], galaxy, "failed", stage="failed",
                    message=failure["error"],
                )
                print(failure["traceback"], file=sys.stderr)
                return 1
            finally:
                try:
                    import matplotlib.pyplot as plt

                    plt.close("all")
                except Exception:
                    pass
                gc.collect()


def worker_command(
    args: argparse.Namespace,
    galaxy: str,
    template: Path,
    queue_index: int,
    queue_total: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--galaxy", galaxy,
        "--queue-index", str(queue_index),
        "--queue-total", str(queue_total),
        "--manifest", str(args.manifest),
        "--data-root", str(args.data_root),
        "--run-root", str(args.run_root),
        "--base-notebook", str(args.base_notebook),
        "--generated-template", str(template),
        "--stpsf-data-dir", str(args.stpsf_data_dir),
        "--wss-opd-dir", str(args.wss_opd_dir),
        "--e-realizations", str(args.e_realizations),
        "--fft-workers", str(args.fft_workers),
    ]
    for flag, enabled in (
        ("--force-source", args.force_source),
        ("--force-spectra", args.force_spectra),
        ("--rebuild-psf", args.rebuild_psf),
        ("--rebuild-input-cache", args.rebuild_input_cache),
        ("--rebuild-expectation-cache", args.rebuild_expectation_cache),
    ):
        if enabled:
            command.append(flag)
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--base-notebook", default=str(DEFAULT_BASE_NOTEBOOK))
    parser.add_argument("--generated-template", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stpsf-data-dir", default=str(DEFAULT_STPSF_DATA))
    parser.add_argument("--wss-opd-dir", default=str(DEFAULT_WSS_OPD))
    parser.add_argument("--galaxies", nargs="+", default=None)
    parser.add_argument("--galaxy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--queue-total", type=int, default=14, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-spectra", action="store_true")
    parser.add_argument("--rebuild-psf", action="store_true")
    parser.add_argument("--rebuild-input-cache", action="store_true")
    parser.add_argument("--rebuild-expectation-cache", action="store_true")
    parser.add_argument("--e-realizations", type=int, default=64)
    parser.add_argument("--fft-workers", type=int, default=-1)
    parser.add_argument(
        "--min-free-gb", type=float, default=20.0,
        help="safely stop before the next target below this free-space reserve",
    )
    args = parser.parse_args(argv)
    for name in (
        "manifest", "data_root", "run_root", "base_notebook",
        "stpsf_data_dir", "wss_opd_dir",
    ):
        setattr(args, name, resolve_project_path(getattr(args, name)))
    if args.generated_template:
        args.generated_template = resolve_project_path(args.generated_template)
    else:
        args.generated_template = campaign_paths(args.run_root)["generated_template"]
    if args.e_realizations < 1:
        parser.error("--e-realizations must be positive")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb cannot be negative")
    if args.worker and not args.galaxy:
        parser.error("--worker requires --galaxy")
    if args.worker and not 1 <= args.queue_index <= args.queue_total:
        parser.error("--worker requires 1 <= --queue-index <= --queue-total")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = campaign_paths(args.run_root)
    for path in (
        paths["root"], paths["source_batch"], paths["source_products"],
        paths["products"], paths["logs"], paths["generated_template"].parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["STPSF_PATH"] = str(args.stpsf_data_dir)
    os.environ["SBF_WSS_OPD_DIR"] = str(args.wss_opd_dir)
    os.environ["SBF_PSF_MAX_OPD_DELTA_DAYS"] = "7.0"

    parent_log_handle = None
    if not args.worker:
        set_log_context("campaign", 0, 14)
        builtins.print = timestamped_print
        campaign_lock = acquire_campaign_lock(paths["root"])
        atexit.register(release_campaign_lock, campaign_lock)
        parent_log_handle = paths["campaign_log"].open(
            "a", encoding="utf-8", buffering=1
        )
        sys.stdout = Tee(sys.stdout, parent_log_handle)
        sys.stderr = Tee(sys.stderr, parent_log_handle)
        print("=" * 88)
        print(f"F090W campaign invocation; cwd={Path.cwd()}")

    if not args.worker:
        build_f090w_template(args.base_notebook, paths["generated_template"])
    template = args.generated_template
    if args.worker:
        set_log_context(args.galaxy, args.queue_index, args.queue_total)
        return run_worker(args)

    targets = read_targets(args.manifest, args.data_root)
    selected = select_targets(targets, args.galaxies)
    readiness = inspect_local_inputs(targets, args.data_root)
    buffer = io.StringIO()
    readiness.to_csv(buffer, index=False)
    atomic_write_text(paths["input_report"], buffer.getvalue())
    print(readiness[["galaxy", "signal_filter", "auxiliary_filter", "ready", "message"]].to_string(index=False))
    if not readiness["ready"].all():
        print("Not all local inputs are ready", file=sys.stderr)
        return 2

    dependency_args = SimpleNamespace(
        stpsf_data_dir=str(args.stpsf_data_dir),
        wss_opd_dir=str(args.wss_opd_dir),
        no_download=True,
        max_opd_delta_days=7.0,
    )
    validate_offline_dependencies(
        dependency_args, targets, args.data_root, paths["source_batch"]
    )
    config = ExperimentConfig(
        normalized_sigma=3.5,
        kmins=(0.01, 0.03, 0.04),
        kmax=0.25,
        k_bins=80,
        e_realizations=args.e_realizations,
        random_seed=1489,
        fft_workers=args.fft_workers,
        min_modes_per_bin=10,
        save_ring_fft_fits=True,
        save_all_branch_fits=False,
    )
    status = sync_status(targets, paths, config)
    print(f"Status table: {paths['status']}")
    print(status[["galaxy", "status", "stage", "attempt", "pid"]].to_string(index=False))
    print(f"Generated F090W template: {template}")
    print("Inputs: 14/14 F090W + F150W local; no downloads will be attempted")
    print("PSF: local STPSF F090W, 129x129, nearest WSS OPD + 4 field offsets")
    print(
        f"Disk: free={free_space_gb(paths['root']):.1f} GB; "
        f"reserve={args.min_free_gb:.1f} GB; full campaign estimate "
        f"~{ESTIMATED_CAMPAIGN_GB:.0f} GB"
    )
    if args.dry_run:
        print("Dry run complete: no galaxy model, PSF or FFT was calculated")
        return 0
    active = status[status["status"].eq("running")]
    if not active.empty:
        print(
            "Another F090W worker is still active; refusing a duplicate run:\n"
            + active[["galaxy", "stage", "pid"]].to_string(index=False),
            file=sys.stderr,
        )
        return 3

    selected_names = {target["name"] for target in selected}
    failures = 0
    disk_stop = False
    for index, target in enumerate(targets, start=1):
        galaxy = target["name"]
        if galaxy not in selected_names:
            continue
        set_log_context(galaxy, index, len(targets))
        final_ok, result, _ = final_result_valid(paths, galaxy, config)
        force_requested = any((
            args.force_source,
            args.force_spectra,
            args.rebuild_psf,
            args.rebuild_input_cache,
            args.rebuild_expectation_cache,
        ))
        if final_ok and not force_requested:
            materialize_products(paths, galaxy, result, args.data_root)
            update_status(
                paths["status"], galaxy, "ok", stage="complete",
                message="verified products reused",
                source_result=str(source_result_path(paths, galaxy)),
                final_result=str(final_result_path(paths, galaxy)),
            )
            print("already complete")
            continue

        available_gb = free_space_gb(paths["root"])
        if available_gb < args.min_free_gb:
            message = (
                f"disk guard: {available_gb:.1f} GB free, "
                f"reserve is {args.min_free_gb:.1f} GB"
            )
            update_status(
                paths["status"], galaxy, "pending", stage="disk guard",
                message=message,
            )
            print(message + "; campaign stopped safely", file=sys.stderr)
            disk_stop = True
            break

        command = worker_command(args, galaxy, template, index, len(targets))
        print("starting")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
            update_status(
                paths["status"], galaxy, "interrupted",
                stage="interrupted", message="parent interrupted",
            )
            print("Interrupted. Restart the same command to continue")
            return 130
        if returncode != 0:
            failures += 1
            print(f"worker exited with code {returncode}", file=sys.stderr)
            if args.stop_on_error:
                break
        rebuild_global_summary(targets, paths, config, args.data_root)

    set_log_context("campaign", 0, len(targets))
    summary = rebuild_global_summary(targets, paths, config, args.data_root)
    final_status = sync_status(targets, paths, config)
    print(final_status[["galaxy", "status", "stage", "attempt", "message"]].to_string(index=False))
    print(f"Final F090W table ({len(summary)}/14): {paths['summary']}")
    return 4 if disk_stop else 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
