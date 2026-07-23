#!/usr/bin/env python3
"""Wait for the SBF first pass and build a reproducible final report.

The script is deliberately independent from ``run_sbf_batch.py``: it only reads
the campaign ledger/database and result products.  It never stops, restarts, or
modifies the running batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CAMPAIGN_ROOT = PROJECT_ROOT / "runs" / "sbf3_go3055_go7763" / "campaign"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "sbf3_go3055_go7763" / "final_analysis"
DEFAULT_GO3055_ROOT = SCRIPT_DIR / "sbf2_batch_outputs"

# Matplotlib otherwise tries to write into ~/.matplotlib, which is undesirable
# for an unattended run and may be read-only in some environments.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


PLOT_DPI = 170
CSV_FIELDS = [
    "program",
    "obsid",
    "galaxy",
    "status",
    "method_generation",
    "signal_filter",
    "color_filter",
    "color_name",
    "mbar",
    "sigma_mbar",
    "absolute_mbar",
    "sigma_absolute_mbar",
    "external_distance_modulus",
    "color",
    "color_sigma",
    "mbar_inner",
    "mbar_outer",
    "inner_outer_delta",
    "selected_region",
    "selection_method",
    "qc_status",
    "qc_reasons",
    "k_stability_mag",
    "Pr_over_P0",
    "uses_two_annuli",
    "result_json",
    "worker_log",
    "clean_model_fits",
    "clean_isophotes_fits",
    "full_residual_fits",
    "working_residual_fits",
    "working_annuli_residual_fits",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_path: Path | None = None) -> None:
    line = f"[{local_stamp()}] {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_value(row.get(key)) for key in fields})


def clean_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def safe_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # retained in the report instead of aborting all analysis
        return {"status": "invalid", "_read_error": f"{type(exc).__name__}: {exc}"}


def latest_run_id(database: Path) -> str | None:
    if not database.exists():
        return None
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return None if row is None else str(row[0])


def first_pass_state(database: Path) -> dict[str, Any]:
    """Return completion of the first real attempt for every non-reused job."""
    if not database.exists():
        return {"complete": False, "reason": "campaign database is absent"}
    run_id = latest_run_id(database)
    if run_id is None:
        return {"complete": False, "reason": "campaign database has no runs"}

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        run = connection.execute(
            "SELECT state, created_at, soft_stop_at, deadline_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        jobs = connection.execute(
            """
            SELECT j.target, j.state, j.attempt_count,
                   a.state AS first_attempt_state, a.ended_at AS first_attempt_ended_at
            FROM jobs AS j
            LEFT JOIN attempts AS a
              ON a.run_id = j.run_id AND a.job_id = j.job_id AND a.attempt_no = 1
            WHERE j.run_id = ?
            ORDER BY j.queue_position, j.target
            """,
            (run_id,),
        ).fetchall()

    unfinished = []
    for job in jobs:
        # SUCCEEDED with attempt_count=0 means an accepted pre-existing result.
        accepted = job["state"] == "SUCCEEDED"
        first_attempt_finished = job["first_attempt_ended_at"] is not None
        if not accepted and not first_attempt_finished:
            unfinished.append(str(job["target"]))

    states = Counter(str(job["state"]) for job in jobs)
    return {
        "complete": not unfinished and bool(jobs),
        "reason": "complete" if not unfinished and jobs else "first pass still running",
        "run_id": run_id,
        "run_state": None if run is None else run["state"],
        "job_count": len(jobs),
        "job_states": dict(states),
        "unfinished_count": len(unfinished),
        "unfinished_targets": unfinished,
    }


def wait_for_first_pass(
    database: Path, status_csv: Path, poll_seconds: float, log_path: Path
) -> dict[str, Any]:
    previous_signature: tuple[Any, ...] | None = None
    while True:
        state = first_pass_state(database)
        status_counts: Counter[str] = Counter()
        if status_csv.exists():
            status_counts.update(row.get("status", "unknown") for row in read_csv(status_csv))
        signature = (
            state.get("complete"),
            state.get("unfinished_count"),
            tuple(sorted(status_counts.items())),
        )
        if signature != previous_signature:
            unfinished = state.get("unfinished_targets", [])
            preview = ", ".join(unfinished[:6]) or "none"
            log(
                "GO-7763: "
                + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
                + f"; first-pass unfinished={state.get('unfinished_count', '?')} ({preview})",
                log_path,
            )
            previous_signature = signature
        if state.get("complete"):
            log("First pass is complete; starting final analysis.", log_path)
            return state
        if state.get("job_count") and state.get("run_state") not in {None, "RUNNING"}:
            state["trigger"] = "batch_stopped_before_first_pass_completion"
            log(
                f"Batch state is {state.get('run_state')} with unfinished first-pass targets; "
                "building an explicitly incomplete report instead of waiting forever.",
                log_path,
            )
            return state
        time.sleep(max(5.0, poll_seconds))


def find_result_jsons(root: Path) -> list[Path]:
    return sorted(root.glob("*_result.json"))


def load_go3055(go3055_root: Path) -> list[dict[str, Any]]:
    calibration_path = go3055_root / "coursework_calibration_input.csv"
    calibration_rows = read_csv(calibration_path) if calibration_path.exists() else []
    calibration = {row["galaxy"].strip(): row for row in calibration_rows}
    records: list[dict[str, Any]] = []

    for result_path in find_result_jsons(go3055_root):
        data = safe_json(result_path)
        galaxy = str(data.get("galaxy") or result_path.stem.replace("_result", "")).strip()
        cal = calibration.get(galaxy, {})
        mbar = as_float(data.get("recommended_mbar_weighted"))
        inner = as_float(data.get("recommended_mbar_inner"))
        outer = as_float(data.get("recommended_mbar_outer"))
        qc = str(first_present(cal, "quality_flag_effective", "quality_flag") or "unknown").upper()
        records.append(
            {
                "program": "3055",
                "obsid": "",
                "galaxy": galaxy,
                "status": "done" if data.get("status") == "ok" else "failed",
                "method_generation": "sbf2_two_annuli_trgb",
                "signal_filter": "F150W",
                "color_filter": "F090W",
                "color_name": "F090W-F150W",
                "mbar": mbar,
                "sigma_mbar": as_float(
                    first_present(data, "recommended_sigma_adopted", "recommended_sigma_weighted_formal")
                ),
                "absolute_mbar": as_float(cal.get("Mbar_150")),
                "sigma_absolute_mbar": as_float(cal.get("sigma_Mbar_150")),
                "external_distance_modulus": as_float(cal.get("mu_lit")),
                "color": as_float(data.get("color_F090W_F150W")),
                "color_sigma": as_float(data.get("color_sigma_proxy")),
                "mbar_inner": inner,
                "mbar_outer": outer,
                "inner_outer_delta": inner - outer,
                "selected_region": "inner+outer_weighted",
                "selection_method": "inverse_variance_two_annuli",
                "qc_status": qc,
                "qc_reasons": str(cal.get("quality_flag") or ""),
                "k_stability_mag": math.nan,
                "Pr_over_P0": np.nanmean(
                    [
                        as_float(data.get("recommended_Pr_over_P0_inner")),
                        as_float(data.get("recommended_Pr_over_P0_outer")),
                    ]
                ),
                "uses_two_annuli": True,
                "result_json": str(result_path.resolve()),
                "worker_log": "",
                "clean_model_fits": str(data.get("model_full_fits") or ""),
                "clean_isophotes_fits": "",
                "full_residual_fits": str(data.get("science_residual_raw_fits") or ""),
                "working_residual_fits": str(data.get("science_residual_fits") or ""),
                "working_annuli_residual_fits": "",
                "error": str(data.get("_read_error") or ""),
                "fit_sample": str(cal.get("fit_sample") or ""),
                "is_clean_effective": as_bool(cal.get("is_clean_effective")),
                "mu_residual": as_float(cal.get("mu_resid_clean_linear")),
            }
        )
    return records


def first_attempt_failures(database: Path) -> dict[str, dict[str, str]]:
    run_id = latest_run_id(database)
    if run_id is None:
        return {}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT j.target, j.obsid, a.state, a.error, a.log_path
            FROM jobs AS j
            JOIN attempts AS a ON a.run_id = j.run_id AND a.job_id = j.job_id
            WHERE j.run_id = ? AND a.attempt_no = 1 AND a.state != 'SUCCEEDED'
            """,
            (run_id,),
        ).fetchall()
    failures: dict[str, dict[str, str]] = {}
    for row in rows:
        worker_log = str(row["log_path"] or "")
        error = str(row["error"] or "first attempt failed")
        log_path = Path(worker_log)
        if log_path.exists():
            try:
                with log_path.open("rb") as handle:
                    handle.seek(max(0, log_path.stat().st_size - 512_000))
                    tail = handle.read().decode("utf-8", errors="replace")
                if "isolist too short" in tail and "isolist too short" not in error:
                    error += "; isolist too short after all starts and datasets"
                elif "No meaningful fit was possible" in tail and "No meaningful fit" not in error:
                    error += "; No meaningful fit was possible"
            except OSError:
                pass
        failures[str(row["target"])] = {
            "obsid": str(row["obsid"] or ""),
            "error": error,
            "worker_log": worker_log,
        }
    return failures


def load_go7763(status_csv: Path, database: Path) -> list[dict[str, Any]]:
    status_rows = read_csv(status_csv)
    failures = first_attempt_failures(database)
    records: list[dict[str, Any]] = []

    for row in status_rows:
        if str(row.get("program")) != "7763":
            continue
        galaxy = row.get("galaxy", "").strip()
        result_path = Path(row["result_json"]) if row.get("result_json") else None
        data = safe_json(result_path) if result_path and result_path.exists() else {}
        # If a retry is already running, preserve the scientifically useful first-pass
        # failure in the snapshot instead of pretending the target is unfinished.
        status = row.get("status", "unknown")
        error = row.get("error", "")
        worker_log = str(data.get("worker_log_path") or "")
        if galaxy in failures and not data:
            if status in {"running", "pending"}:
                status = "failed"
            error = failures[galaxy]["error"]
            worker_log = failures[galaxy]["worker_log"]

        uses_two = as_bool(data.get("recommended_uses_two_annuli"))
        method_generation = "sbf3_legacy_two_annuli" if uses_two else "sbf3_single_annulus_qc_v1"
        mbar = as_float(
            first_present(data, "recommended_mbar_selected", "recommended_mbar_weighted")
            or row.get("result_value")
        )
        inner = as_float(data.get("recommended_mbar_inner"))
        outer = as_float(data.get("recommended_mbar_outer"))
        qc_status = str(data.get("recommended_selected_region_qc_status") or "")
        qc_reasons = str(data.get("recommended_selected_region_qc_reasons") or row.get("qc") or "")
        if uses_two and not qc_status:
            qc_status = "LEGACY"
            qc_reasons = "accepted result from the previous two-annulus selection"
        if status != "done" and not qc_status:
            qc_status = "FAILED"

        records.append(
            {
                "program": "7763",
                "obsid": row.get("obsid", ""),
                "galaxy": galaxy,
                "status": status,
                "method_generation": method_generation if data else "sbf3",
                "signal_filter": row.get("signal_filter", "F150W"),
                "color_filter": row.get("color_filter", "F115W"),
                "color_name": str(data.get("color_name") or "F115W-F150W"),
                "mbar": mbar,
                "sigma_mbar": as_float(
                    first_present(data, "recommended_sigma_selected", "recommended_sigma_adopted", "recommended_sigma_weighted_formal")
                ),
                "absolute_mbar": math.nan,
                "sigma_absolute_mbar": math.nan,
                "external_distance_modulus": math.nan,
                "color": as_float(first_present(data, "selected_color_index", "color_index", "color_F115W_F150W")),
                "color_sigma": as_float(data.get("color_sigma_proxy")),
                "mbar_inner": inner,
                "mbar_outer": outer,
                "inner_outer_delta": inner - outer,
                "selected_region": str(
                    first_present(data, "recommended_selected_region", "selected_sbf_region")
                    or row.get("selected_region")
                    or ""
                ),
                "selection_method": str(
                    first_present(data, "recommended_selection_method", "selected_sbf_selection_method")
                    or row.get("selection_method")
                    or ""
                ),
                "qc_status": qc_status,
                "qc_reasons": qc_reasons,
                "k_stability_mag": as_float(data.get("recommended_selected_region_k_stability_mag")),
                "Pr_over_P0": as_float(
                    first_present(data, "recommended_Pr_over_P0_selected", "recommended_Pr_over_P0_inner")
                ),
                "uses_two_annuli": uses_two,
                "result_json": str(result_path.resolve()) if result_path else "",
                "worker_log": worker_log,
                "clean_model_fits": str(data.get("clean_model_fits") or ""),
                "clean_isophotes_fits": str(data.get("clean_isophotes_fits") or ""),
                "full_residual_fits": str(data.get("full_residual_fits") or ""),
                "working_residual_fits": str(data.get("working_residual_fits") or ""),
                "working_annuli_residual_fits": str(data.get("working_annuli_residual_fits") or ""),
                "error": str(error or data.get("_read_error") or ""),
            }
        )
    return records


def finite_rows(records: Iterable[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    return [row for row in records if math.isfinite(as_float(row.get(x_key))) and math.isfinite(as_float(row.get(y_key)))]


def fit_linear(
    model_id: str,
    records: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    sigma_key: str,
    quantity: str,
    caveat: str,
    pivot: float,
    primary_method: str,
) -> dict[str, Any]:
    sample = finite_rows(records, x_key, y_key)
    if len(sample) < 3:
        return {"model_id": model_id, "n": len(sample), "status": "insufficient_data", "caveat": caveat}
    x = np.asarray([as_float(row[x_key]) for row in sample])
    y = np.asarray([as_float(row[y_key]) for row in sample])
    sigma = np.asarray([as_float(row.get(sigma_key)) for row in sample])
    design = np.column_stack([np.ones_like(x), x - pivot])
    good_sigma = np.isfinite(sigma) & (sigma > 0)
    if np.count_nonzero(good_sigma) >= 3:
        fallback = float(np.median(sigma[good_sigma]))
        sigma = np.where(good_sigma, sigma, fallback)
        weight = 1.0 / np.square(sigma)
    else:
        sigma = np.ones_like(x)
        weight = np.ones_like(x)
    weighted_normal = design.T @ (weight[:, None] * design)
    weighted_covariance = np.linalg.pinv(weighted_normal)
    weighted_coefficients = weighted_covariance @ (design.T @ (weight * y))
    weighted_residual = y - design @ weighted_coefficients
    dof = max(1, len(y) - 2)
    chi2_red = float(np.sum(np.square(weighted_residual / sigma)) / dof)

    ordinary_coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    ordinary_residual = y - design @ ordinary_coefficients
    ordinary_variance = float(np.sum(np.square(ordinary_residual)) / dof)
    ordinary_covariance = np.linalg.pinv(design.T @ design) * ordinary_variance

    if primary_method == "weighted_least_squares":
        coefficients = weighted_coefficients
        errors = np.sqrt(np.maximum(np.diag(weighted_covariance), 0))
        residual = weighted_residual
    elif primary_method == "ordinary_least_squares":
        coefficients = ordinary_coefficients
        errors = np.sqrt(np.maximum(np.diag(ordinary_covariance), 0))
        residual = ordinary_residual
    else:
        raise ValueError(f"unknown primary fit method: {primary_method}")
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    robust = stats.theilslopes(y, x)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    median_sigma = float(np.median(sigma))
    intrinsic_proxy = math.sqrt(max(0.0, rms * rms - median_sigma * median_sigma))
    return {
        "model_id": model_id,
        "status": "ok",
        "quantity": quantity,
        "primary_fit_method": primary_method,
        "x_key": x_key,
        "y_key": y_key,
        "n": len(sample),
        "pivot_color": pivot,
        "intercept_at_pivot": float(coefficients[0]),
        "intercept_error": float(errors[0]),
        "slope": float(coefficients[1]),
        "slope_error": float(errors[1]),
        "ordinary_intercept_at_pivot": float(ordinary_coefficients[0]),
        "ordinary_intercept_error": float(np.sqrt(max(ordinary_covariance[0, 0], 0))),
        "ordinary_slope": float(ordinary_coefficients[1]),
        "ordinary_slope_error": float(np.sqrt(max(ordinary_covariance[1, 1], 0))),
        "weighted_intercept_at_pivot": float(weighted_coefficients[0]),
        "weighted_intercept_error_formal": float(np.sqrt(max(weighted_covariance[0, 0], 0))),
        "weighted_slope": float(weighted_coefficients[1]),
        "weighted_slope_error_formal": float(np.sqrt(max(weighted_covariance[1, 1], 0))),
        "rms": rms,
        "median_reported_sigma": median_sigma,
        "intrinsic_scatter_proxy": intrinsic_proxy,
        "chi2_red": chi2_red,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "theil_sen_slope": float(robust.slope),
        "theil_sen_intercept_at_zero": float(robust.intercept),
        "galaxies": "; ".join(row["galaxy"] for row in sample),
        "caveat": caveat,
    }


def model_prediction(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return as_float(model["intercept_at_pivot"]) + as_float(model["slope"]) * (
        x - as_float(model["pivot_color"])
    )


def build_models(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    go3055 = [
        row
        for row in records
        if row["program"] == "3055" and row["status"] == "done" and row.get("is_clean_effective")
    ]
    go7763_new = [
        row
        for row in records
        if row["program"] == "7763"
        and row["status"] == "done"
        and row["method_generation"] == "sbf3_single_annulus_qc_v1"
    ]
    go7763_pass = [row for row in go7763_new if row.get("qc_status") == "PASS"]
    return [
        fit_linear(
            "go3055_trgb_absolute_clean",
            go3055,
            "color",
            "absolute_mbar",
            "sigma_absolute_mbar",
            "absolute F150W SBF magnitude",
            "TRGB-anchored GO-3055 calibration; F090W-F150W color only.",
            0.4,
            "weighted_least_squares",
        ),
        fit_linear(
            "go7763_virgo_single_annulus_all",
            go7763_new,
            "color",
            "mbar",
            "sigma_mbar",
            "apparent F150W SBF magnitude",
            "Common-cluster trend, not an independent distance calibration; F115W-F150W color only.",
            0.15,
            "ordinary_least_squares",
        ),
        fit_linear(
            "go7763_virgo_single_annulus_pass",
            go7763_pass,
            "color",
            "mbar",
            "sigma_mbar",
            "apparent F150W SBF magnitude",
            "PASS-only sensitivity check; fixed angular annuli remain a systematic limitation.",
            0.15,
            "ordinary_least_squares",
        ),
    ]


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_campaign_status(records: list[dict[str, Any]], path: Path) -> None:
    programs = ["3055", "7763"]
    statuses = ["done", "failed", "running", "pending"]
    colors = {"done": "#2a9d55", "failed": "#d1495b", "running": "#e9a23b", "pending": "#8b95a5"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(programs))
    for status in statuses:
        values = [sum(row["program"] == program and row["status"] == status for row in records) for program in programs]
        bars = ax.bar(programs, values, bottom=bottom, label=status, color=colors[status])
        for bar, value, base in zip(bars, values, bottom):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(value), ha="center", va="center")
        bottom += values
    ax.set_ylabel("Number of targets")
    ax.set_title("SBF campaign status snapshot")
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, path)


def scatter_with_model(
    ax: plt.Axes,
    sample: list[dict[str, Any]],
    model: dict[str, Any],
    y_key: str,
    title: str,
    ylabel: str,
) -> None:
    sample = finite_rows(sample, "color", y_key)
    for row in sample:
        color = "#2878b5" if row.get("qc_status") in {"PASS", "CLEAN"} else "#d95f02"
        marker = "o" if not row.get("uses_two_annuli") else "s"
        ax.errorbar(
            row["color"],
            row[y_key],
            yerr=as_float(row.get("sigma_mbar" if y_key == "mbar" else "sigma_absolute_mbar")),
            fmt=marker,
            ms=4.5,
            color=color,
            alpha=0.82,
        )
    if model.get("status") == "ok" and sample:
        xmin = min(row["color"] for row in sample)
        xmax = max(row["color"] for row in sample)
        grid = np.linspace(xmin, xmax, 200)
        fit_label = {
            "weighted_least_squares": "weighted linear fit",
            "ordinary_least_squares": "ordinary linear fit",
        }.get(str(model.get("primary_fit_method")), "linear fit")
        ax.plot(grid, model_prediction(model, grid), color="black", lw=1.6, label=fit_label)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(sample[0]["color_name"] if sample else "color")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def plot_color_models(records: list[dict[str, Any]], models: list[dict[str, Any]], path: Path) -> None:
    model_map = {model["model_id"]: model for model in models}
    go3055 = [row for row in records if row["program"] == "3055" and row.get("is_clean_effective")]
    go7763 = [
        row
        for row in records
        if row["program"] == "7763" and row["method_generation"] == "sbf3_single_annulus_qc_v1" and row["status"] == "done"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))
    scatter_with_model(
        axes[0], go3055, model_map["go3055_trgb_absolute_clean"], "absolute_mbar", "GO-3055: TRGB calibration", r"$\overline{M}_{150}$ (AB mag)"
    )
    scatter_with_model(
        axes[1], go7763, model_map["go7763_virgo_single_annulus_all"], "mbar", "GO-7763: Virgo trend", r"$\overline{m}_{150}$ (AB mag)"
    )
    fig.suptitle("Separate color relations: filters are not interchangeable", y=1.01)
    save_figure(fig, path)


def plot_ring_comparison(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, program in zip(axes, ["3055", "7763"]):
        sample = finite_rows(
            [row for row in records if row["program"] == program and row["status"] == "done"],
            "mbar_inner",
            "mbar_outer",
        )
        if sample:
            inner = np.asarray([row["mbar_inner"] for row in sample])
            outer = np.asarray([row["mbar_outer"] for row in sample])
            limits = [float(min(inner.min(), outer.min()) - 0.1), float(max(inner.max(), outer.max()) + 0.1)]
            ax.plot(limits, limits, "k--", lw=1)
            ax.scatter(inner, outer, c=["#2878b5" if row.get("qc_status") in {"PASS", "CLEAN"} else "#d95f02" for row in sample], s=28)
            median_delta = np.median(inner - outer)
            ax.text(0.03, 0.96, f"n={len(sample)}; median inner-outer={median_delta:+.3f} mag", transform=ax.transAxes, va="top", fontsize=9)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        ax.set_title(f"GO-{program}")
        ax.set_xlabel("inner annulus mbar (mag)")
        ax.set_ylabel("outer annulus mbar (mag)")
        ax.grid(alpha=0.2)
    fig.suptitle("Annulus consistency")
    save_figure(fig, path)


def plot_annulus_delta(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, program in zip(axes, ["3055", "7763"]):
        sample = finite_rows(
            [row for row in records if row["program"] == program and row["status"] == "done"],
            "color",
            "inner_outer_delta",
        )
        ax.axhline(0, color="black", lw=1)
        ax.axhspan(-0.2, 0.2, color="#2a9d55", alpha=0.10)
        ax.scatter(
            [row["color"] for row in sample],
            [row["inner_outer_delta"] for row in sample],
            c=["#2878b5" if row.get("qc_status") in {"PASS", "CLEAN"} else "#d95f02" for row in sample],
            s=28,
        )
        ax.set_title(f"GO-{program}")
        ax.set_xlabel(sample[0]["color_name"] if sample else "color")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("inner - outer mbar (mag)")
    fig.suptitle("Fixed-annulus systematic check")
    save_figure(fig, path)


def plot_model_residuals(records: list[dict[str, Any]], models: list[dict[str, Any]], path: Path) -> None:
    model_map = {model["model_id"]: model for model in models}
    panels = [
        (
            "3055",
            "go3055_trgb_absolute_clean",
            "absolute_mbar",
            lambda row: row.get("is_clean_effective"),
            "GO-3055 absolute calibration",
        ),
        (
            "7763",
            "go7763_virgo_single_annulus_all",
            "mbar",
            lambda row: row.get("method_generation") == "sbf3_single_annulus_qc_v1",
            "GO-7763 apparent Virgo trend",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, (program, model_id, y_key, predicate, title) in zip(axes, panels):
        model = model_map[model_id]
        sample = finite_rows(
            [row for row in records if row["program"] == program and row["status"] == "done" and predicate(row)],
            "color",
            y_key,
        )
        if model.get("status") == "ok" and sample:
            x = np.asarray([row["color"] for row in sample])
            y = np.asarray([row[y_key] for row in sample])
            residual = y - model_prediction(model, x)
            ax.scatter(x, residual, c="#2878b5", s=28)
            threshold = 2 * as_float(model["rms"])
            ax.axhspan(-threshold, threshold, color="#2a9d55", alpha=0.10)
            for row, xv, rv in zip(sample, x, residual):
                if abs(rv) > threshold:
                    ax.annotate(row["galaxy"], (xv, rv), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(title)
        ax.set_xlabel(sample[0]["color_name"] if sample else "color")
        ax.set_ylabel("fit residual (mag)")
        ax.grid(alpha=0.2)
    save_figure(fig, path)


def plot_qc_diagnostics(records: list[dict[str, Any]], path: Path) -> None:
    sample = [
        row
        for row in finite_rows(records, "Pr_over_P0", "sigma_mbar")
        if row["program"] == "7763" and row["status"] == "done" and row["method_generation"] == "sbf3_single_annulus_qc_v1"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].scatter(
        [row["Pr_over_P0"] for row in sample],
        [row["sigma_mbar"] for row in sample],
        c=["#2878b5" if row.get("qc_status") == "PASS" else "#d95f02" for row in sample],
        s=30,
    )
    axes[0].axvline(0.2, color="#d1495b", ls="--", lw=1)
    axes[0].set_xlabel("Pr/P0")
    axes[0].set_ylabel("reported mbar uncertainty (mag)")
    axes[0].set_title("Compact-source correction")
    axes[0].grid(alpha=0.2)

    delta_sample = finite_rows(
        [row for row in records if row["program"] == "7763" and row["status"] == "done"],
        "k_stability_mag",
        "inner_outer_delta",
    )
    axes[1].scatter(
        [row["k_stability_mag"] for row in delta_sample],
        [abs(row["inner_outer_delta"]) for row in delta_sample],
        c=["#2878b5" if row.get("qc_status") == "PASS" else "#d95f02" for row in delta_sample],
        s=30,
    )
    axes[1].axhline(0.2, color="#d1495b", ls="--", lw=1)
    axes[1].set_xlabel("k-window stability (mag)")
    axes[1].set_ylabel("|inner - outer| (mag)")
    axes[1].set_title("Two independent instability indicators")
    axes[1].grid(alpha=0.2)
    save_figure(fig, path)


def annotate_model_residuals(records: list[dict[str, Any]], models: list[dict[str, Any]]) -> None:
    definitions = [
        ("3055", "go3055_trgb_absolute_clean", "absolute_mbar", lambda row: row.get("is_clean_effective")),
        ("7763", "go7763_virgo_single_annulus_all", "mbar", lambda row: row.get("method_generation") == "sbf3_single_annulus_qc_v1"),
    ]
    model_map = {model["model_id"]: model for model in models}
    for program, model_id, y_key, predicate in definitions:
        model = model_map[model_id]
        if model.get("status") != "ok":
            continue
        for row in records:
            if row["program"] != program or row["status"] != "done" or not predicate(row):
                continue
            x = as_float(row.get("color"))
            y = as_float(row.get(y_key))
            if math.isfinite(x) and math.isfinite(y):
                row["model_id"] = model_id
                row["model_residual"] = y - float(model_prediction(model, np.asarray([x]))[0])


def build_review_queue(records: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_rms = {model["model_id"]: as_float(model.get("rms")) for model in models}
    queue: list[dict[str, Any]] = []
    for row in records:
        reasons: list[str] = []
        score = 0
        if row["status"] != "done":
            reasons.append("processing_failed")
            score += 100
            if "isolist too short" in str(row.get("error")):
                reasons.append("isophote_family_too_short")
        else:
            if str(row.get("qc_status")) not in {"PASS", "CLEAN"}:
                reasons.append(f"qc={row.get('qc_status') or 'unknown'}")
                score += 25
            delta = abs(as_float(row.get("inner_outer_delta")))
            if math.isfinite(delta) and delta > 0.2:
                reasons.append(f"abs_inner_outer_delta={delta:.3f}mag")
                score += min(30, int(delta * 50))
            pr_ratio = as_float(row.get("Pr_over_P0"))
            if math.isfinite(pr_ratio) and pr_ratio > 0.2:
                reasons.append(f"Pr_over_P0={pr_ratio:.3f}")
                score += 20
            stability = as_float(row.get("k_stability_mag"))
            if math.isfinite(stability) and stability > 0.15:
                reasons.append(f"k_stability={stability:.3f}mag")
                score += 20
            residual = abs(as_float(row.get("model_residual")))
            rms = model_rms.get(str(row.get("model_id")), math.nan)
            if math.isfinite(residual) and math.isfinite(rms) and rms > 0 and residual > 2 * rms:
                reasons.append(f"model_outlier={residual:.3f}mag")
                score += 30
            if row.get("method_generation") == "sbf3_legacy_two_annuli":
                reasons.append("legacy_two_annulus_result")
                score += 10
            artifact_keys = [
                "clean_model_fits",
                "clean_isophotes_fits",
                "full_residual_fits",
                "working_residual_fits",
                "working_annuli_residual_fits",
            ]
            if row["program"] == "7763" and any(not Path(str(row.get(key) or "")).exists() for key in artifact_keys):
                reasons.append("missing_required_fits")
                score += 50
        if reasons:
            queue.append(
                {
                    "priority": score,
                    "program": row["program"],
                    "obsid": row["obsid"],
                    "galaxy": row["galaxy"],
                    "status": row["status"],
                    "reasons": "; ".join(reasons),
                    "qc_status": row.get("qc_status", ""),
                    "inner_outer_delta": row.get("inner_outer_delta"),
                    "model_residual": row.get("model_residual", ""),
                    "result_json": row.get("result_json", ""),
                    "worker_log": row.get("worker_log", ""),
                    "clean_model_fits": row.get("clean_model_fits", ""),
                    "clean_isophotes_fits": row.get("clean_isophotes_fits", ""),
                    "full_residual_fits": row.get("full_residual_fits", ""),
                    "working_residual_fits": row.get("working_residual_fits", ""),
                    "working_annuli_residual_fits": row.get("working_annuli_residual_fits", ""),
                    "error": row.get("error", ""),
                }
            )
    return sorted(queue, key=lambda row: (-int(row["priority"]), row["program"], row["galaxy"]))


def center_from_products(record: dict[str, Any]) -> tuple[float, float] | None:
    result_path = Path(str(record.get("result_json") or ""))
    if not result_path.exists():
        return None
    data = safe_json(result_path)
    out_dir = Path(str(data.get("out_dir") or result_path.parent))
    candidates = sorted(out_dir.glob("*_02_center_adopted_center.csv"))
    if not candidates:
        return None
    rows = read_csv(candidates[0])
    if not rows:
        return None
    x = as_float(rows[0].get("x_pixel"))
    y = as_float(rows[0].get("y_pixel"))
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def fits_cutout(path: Path, center: tuple[float, float] | None, half_size: int = 1200) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        source = hdul[0].data
        if source is None or source.ndim != 2:
            raise ValueError("expected a 2-D primary FITS image")
        height, width = source.shape
        if center is None:
            x, y = width / 2, height / 2
        else:
            x, y = center
        x0 = max(0, int(round(x)) - half_size)
        x1 = min(width, int(round(x)) + half_size)
        y0 = max(0, int(round(y)) - half_size)
        y1 = min(height, int(round(y)) + half_size)
        step = max(1, int(math.ceil(max(x1 - x0, y1 - y0) / 650)))
        return np.asarray(source[y0:y1:step, x0:x1:step], dtype=np.float32).copy()


def make_review_sheets(
    records: list[dict[str, Any]], review_queue: list[dict[str, Any]], output_dir: Path, maximum: int
) -> int:
    by_key = {(row["program"], row["galaxy"]): row for row in records}
    candidates = [
        item
        for item in review_queue
        if item["program"] == "7763" and item["status"] == "done" and item.get("clean_model_fits")
    ][:maximum]
    output_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    stages = [
        ("clean_model_fits", "Model"),
        ("clean_isophotes_fits", "Isophotes"),
        ("full_residual_fits", "All residuals"),
        ("working_residual_fits", "Compact sources masked"),
        ("working_annuli_residual_fits", "Working annuli"),
    ]
    for item in candidates:
        record = by_key[(item["program"], item["galaxy"])]
        center = center_from_products(record)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
        usable = 0
        for ax, (key, title) in zip(axes, stages):
            path = Path(str(record.get(key) or ""))
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            try:
                image = fits_cutout(path, center)
                finite = image[np.isfinite(image)]
                if finite.size < 10:
                    raise ValueError("too few finite pixels")
                if key in {"full_residual_fits", "working_residual_fits", "working_annuli_residual_fits"}:
                    scale = float(np.nanpercentile(np.abs(finite), 98.5))
                    vmin, vmax = -scale, scale
                else:
                    vmin, vmax = np.nanpercentile(finite, [2, 99.5])
                ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                usable += 1
            except Exception as exc:
                ax.text(0.5, 0.5, f"read error\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        fig.suptitle(f"GO-7763 {item['galaxy']} | {item['reasons']}", fontsize=11)
        slug = item["galaxy"].replace(" ", "_").replace("/", "_")
        save_figure(fig, output_dir / f"{slug}_diagnostic.png")
        if usable:
            made += 1
    return made


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def format_model(model: dict[str, Any]) -> str:
    if model.get("status") != "ok":
        return f"insufficient data (n={model.get('n', 0)})"
    return (
        f"y = {model['intercept_at_pivot']:.4f} +/- {model['intercept_error']:.4f} "
        f"+ ({model['slope']:.4f} +/- {model['slope_error']:.4f}) "
        f"[color - {model['pivot_color']:.4f}]; n={model['n']}, RMS={model['rms']:.4f} mag "
        f"({model['primary_fit_method']})"
    )


def build_report(
    records: list[dict[str, Any]],
    models: list[dict[str, Any]],
    review_queue: list[dict[str, Any]],
    state: dict[str, Any],
    output_root: Path,
    sheets_made: int,
) -> str:
    counts = Counter((row["program"], row["status"]) for row in records)
    go7763_done = [row for row in records if row["program"] == "7763" and row["status"] == "done"]
    go7763_new = [row for row in go7763_done if row["method_generation"] == "sbf3_single_annulus_qc_v1"]
    go7763_legacy = [row for row in go7763_done if row["method_generation"] == "sbf3_legacy_two_annuli"]
    deltas = np.asarray([as_float(row["inner_outer_delta"]) for row in go7763_new])
    deltas = deltas[np.isfinite(deltas)]
    failures = [row for row in records if row["program"] == "7763" and row["status"] == "failed"]
    flagged_successes = [item for item in review_queue if item["status"] == "done"]
    iso_failures = sum("isolist too short" in str(row.get("error")) for row in failures)
    model_map = {model["model_id"]: model for model in models}

    top_review = review_queue[:15]
    review_rows = [
        [item["priority"], f"GO-{item['program']}", item["galaxy"], item["status"], item["reasons"]]
        for item in top_review
    ]
    report = f"""# Итоговый автоматический отчёт SBF

Сформирован: `{utc_now()}`. Снимок первого прохода GO-7763: **{'завершён' if state.get('complete') else 'ещё не завершён'}**.

## Жёсткий вывод

- GO-3055 даёт настоящую абсолютную калибровку: 14 измерений, привязанных к внешним модулям расстояния TRGB. Основная чистая выборка содержит {model_map['go3055_trgb_absolute_clean'].get('n', 0)} объектов.
- GO-7763 даёт плотную выборку одного скопления для проверки цветового наклона и систематики метода: готово {len(go7763_done)} из 74, из них {len(go7763_new)} новым однокольцевым выбором и {len(go7763_legacy)} старым двухкольцевым методом.
- Единую регрессию GO-3055 + GO-7763 строить нельзя: цвета `F090W-F150W` и `F115W-F150W` не взаимозаменяемы. На графике они стоят рядом только как две отдельные задачи.
- Автоматика не заменяет просмотр изображений. Отдельно отмечены {len(failures)} неудачных обработок и {len(flagged_successes)} успешных, но подозрительных измерений; для {sheets_made} наиболее приоритетных успешных целей уже собраны пятикадровые диагностические листы.

## Состояние обработки

{markdown_table([
    ['GO-3055', counts[('3055', 'done')], counts[('3055', 'failed')], counts[('3055', 'running')], counts[('3055', 'pending')]],
    ['GO-7763', counts[('7763', 'done')], counts[('7763', 'failed')], counts[('7763', 'running')], counts[('7763', 'pending')]],
], ['Программа', 'Готово', 'Падение', 'В работе', 'Ожидает'])}

У GO-7763 падений первого прохода: {len(failures)}. Из них с явным `isolist too short`: {iso_failures}. Это структурный сбой построения изофот, а не случайная сетевая ошибка; слепой повтор той же геометрии не считается исправлением.

## Модели

### GO-3055: абсолютная TRGB-калибровка

`{format_model(model_map['go3055_trgb_absolute_clean'])}`

Здесь `y = Mbar(F150W)`, цвет — `F090W-F150W`. Это модель, которую можно применять как калибровочную в пределах её выборки и фильтров, с сохранением оговорок по морфологии и диапазону цвета.

### GO-7763: Virgo, новый однокольцевой метод

Все новые успешные измерения:

`{format_model(model_map['go7763_virgo_single_annulus_all'])}`

Только внутренний QC=PASS:

`{format_model(model_map['go7763_virgo_single_annulus_pass'])}`

Здесь `y = mbar(F150W)`, цвет — `F115W-F150W`. Это **не независимая калибровка расстояния**: модель проверяет наклон внутри Virgo и одновременно содержит глубину скопления, морфологию и систематику фиксированных колец.

## Кольца и систематика

Для нового GO-7763 медиана `inner - outer` = {float(np.median(deltas)) if deltas.size else math.nan:+.4f} mag; объектов с `|inner - outer| > 0.2 mag`: {int(np.count_nonzero(np.abs(deltas) > 0.2)) if deltas.size else 0} из {len(deltas)}. Если эта доля велика, итог нельзя лечить выбором одного удобного кольца: нужны кольца, масштабированные по размеру/профилю галактики.

## Что смотреть глазами

{markdown_table(review_rows, ['Приоритет', 'Программа', 'Галактика', 'Статус', 'Причины']) if review_rows else 'Автоматических флагов нет.'}

Полная очередь лежит в [`manual_review_queue.csv`](manual_review_queue.csv), диагностические листы — в [`review_sheets/`](review_sheets/).

## Графики и таблицы

- [`01_campaign_status.png`](plots/01_campaign_status.png) — состояние обеих программ.
- [`02_color_models.png`](plots/02_color_models.png) — две отдельные цветовые модели.
- [`03_inner_vs_outer.png`](plots/03_inner_vs_outer.png) — согласие колец.
- [`04_annulus_delta_vs_color.png`](plots/04_annulus_delta_vs_color.png) — систематика колец против цвета.
- [`05_model_residuals.png`](plots/05_model_residuals.png) — остатки моделей и выбросы.
- [`06_qc_diagnostics.png`](plots/06_qc_diagnostics.png) — Pr/P0, формальная ошибка, стабильность k и расхождение колец.
- [`all_results.csv`](all_results.csv) — единая машиночитаемая таблица, но с явным разделением цветов и методов.
- [`model_summary.csv`](model_summary.csv) — коэффициенты, ошибки, RMS и ранговые тесты.
- [`analysis_summary.json`](analysis_summary.json) — полный снимок для воспроизводимости.
- [`campaign_attempts.csv`](campaign_attempts.csv), [`campaign_events.csv`](campaign_events.csv), [`campaign_artifacts.csv`](campaign_artifacts.csv) — история обработки из SQLite.
- [`campaign_resource_summary.csv`](campaign_resource_summary.csv) — максимальная память/своп и минимальный свободный диск по целям.

## Ограничения, которые нельзя замазывать автоматикой

1. Старые двухкольцевые и новые однокольцевые результаты GO-7763 методически неоднородны; старые девять не входят в основную Virgo-регрессию.
2. Фиксированные угловые кольца дают размер-зависимую систематику. До адаптивной геометрии это предварительная линейка, не финальная шкала расстояний.
3. Успешный численный результат не доказывает хорошую маску, центр и изофоты. Именно поэтому сохранена очередь ручной проверки и пятикадровые листы.
4. GO-7763 сам по себе не проверяет абсолютное расстояние без принятого общего модуля Virgo или независимых расстояний для отдельных объектов.
"""
    return report


def export_query(connection: sqlite3.Connection, query: str, params: tuple[Any, ...], path: Path) -> int:
    cursor = connection.execute(query, params)
    fields = [column[0] for column in cursor.description]
    rows = [dict(zip(fields, row)) for row in cursor.fetchall()]
    write_csv(path, rows, fields)
    return len(rows)


def export_campaign_snapshot(database: Path, status_csv: Path, output_root: Path) -> dict[str, int]:
    run_id = latest_run_id(database)
    if run_id is None:
        return {}
    status_rows = read_csv(status_csv)
    status_fields = list(status_rows[0]) if status_rows else ["program", "galaxy", "status"]
    write_csv(output_root / "target_status_snapshot.csv", status_rows, status_fields)
    with sqlite3.connect(database) as connection:
        counts = {
            "jobs": export_query(
                connection,
                """
                SELECT j.run_id, j.job_id, j.program, j.obsid, j.target, j.state,
                       j.queue_position, j.attempt_count, j.started_at, j.ended_at, j.last_error
                FROM jobs AS j WHERE j.run_id = ? ORDER BY j.queue_position, j.target
                """,
                (run_id,),
                output_root / "campaign_jobs.csv",
            ),
            "attempts": export_query(
                connection,
                """
                SELECT a.attempt_id, a.run_id, a.job_id, j.program, j.obsid, j.target,
                       a.attempt_no, a.state, a.pid, a.started_at, a.ended_at,
                       CASE WHEN a.ended_at IS NOT NULL THEN a.ended_at-a.started_at END AS duration_seconds,
                       a.exit_code, a.log_path, a.error, a.command_json, a.metadata_json
                FROM attempts AS a JOIN jobs AS j ON j.run_id=a.run_id AND j.job_id=a.job_id
                WHERE a.run_id = ? ORDER BY j.queue_position, a.attempt_no
                """,
                (run_id,),
                output_root / "campaign_attempts.csv",
            ),
            "events": export_query(
                connection,
                """
                SELECT e.event_id, e.run_id, e.job_id, j.target, e.attempt_id,
                       e.created_at, e.level, e.event_type, e.message, e.payload_json
                FROM events AS e LEFT JOIN jobs AS j ON j.run_id=e.run_id AND j.job_id=e.job_id
                WHERE e.run_id = ? ORDER BY e.event_id
                """,
                (run_id,),
                output_root / "campaign_events.csv",
            ),
            "artifacts": export_query(
                connection,
                """
                SELECT ar.artifact_id, ar.run_id, ar.job_id, j.target, ar.attempt_id,
                       ar.kind, ar.path, ar.size_bytes, ar.sha256, ar.verified,
                       ar.created_at, ar.updated_at, ar.metadata_json
                FROM artifacts AS ar JOIN jobs AS j ON j.run_id=ar.run_id AND j.job_id=ar.job_id
                WHERE ar.run_id = ? ORDER BY j.queue_position, ar.artifact_id
                """,
                (run_id,),
                output_root / "campaign_artifacts.csv",
            ),
            "resource_summaries": export_query(
                connection,
                """
                SELECT r.run_id, r.job_id, j.target, COUNT(*) AS sample_count,
                       MAX(r.process_rss_bytes) AS max_process_rss_bytes,
                       MAX(r.children_rss_bytes) AS max_children_rss_bytes,
                       MAX(r.swap_used_bytes) AS max_swap_used_bytes,
                       MIN(r.ram_available_bytes) AS min_ram_available_bytes,
                       MIN(r.disk_free_bytes) AS min_disk_free_bytes,
                       MIN(r.sampled_at) AS first_sample_at,
                       MAX(r.sampled_at) AS last_sample_at
                FROM resource_samples AS r
                LEFT JOIN jobs AS j ON j.run_id=r.run_id AND j.job_id=r.job_id
                WHERE r.run_id = ?
                GROUP BY r.run_id, r.job_id, j.target
                ORDER BY j.target
                """,
                (run_id,),
                output_root / "campaign_resource_summary.csv",
            ),
        }
    return counts


def run_analysis(args: argparse.Namespace, state: dict[str, Any], log_path: Path) -> Path:
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    plots = output_root / "plots"
    log("Reading GO-3055 and GO-7763 result ledgers.", log_path)
    records = load_go3055(args.go3055_root) + load_go7763(args.status_csv, args.database)
    models = build_models(records)
    annotate_model_residuals(records, models)
    review_queue = build_review_queue(records, models)

    snapshot_counts = export_campaign_snapshot(args.database, args.status_csv, output_root)

    write_csv(output_root / "all_results.csv", records, CSV_FIELDS + ["model_id", "model_residual"])
    write_csv(
        output_root / "successful_results.csv",
        [row for row in records if row["status"] == "done"],
        CSV_FIELDS + ["model_id", "model_residual"],
    )
    write_csv(
        output_root / "failed_results.csv",
        [row for row in records if row["status"] != "done"],
        CSV_FIELDS,
    )
    review_fields = list(review_queue[0]) if review_queue else ["priority", "program", "galaxy", "status", "reasons"]
    write_csv(output_root / "manual_review_queue.csv", review_queue, review_fields)
    model_fields = sorted({key for model in models for key in model})
    write_csv(output_root / "model_summary.csv", models, model_fields)

    log("Building summary plots.", log_path)
    plot_campaign_status(records, plots / "01_campaign_status.png")
    plot_color_models(records, models, plots / "02_color_models.png")
    plot_ring_comparison(records, plots / "03_inner_vs_outer.png")
    plot_annulus_delta(records, plots / "04_annulus_delta_vs_color.png")
    plot_model_residuals(records, models, plots / "05_model_residuals.png")
    plot_qc_diagnostics(records, plots / "06_qc_diagnostics.png")

    sheets_made = 0
    if not args.no_review_sheets and args.max_review_sheets > 0:
        log(f"Building up to {args.max_review_sheets} diagnostic review sheets.", log_path)
        sheets_made = make_review_sheets(records, review_queue, output_root / "review_sheets", args.max_review_sheets)

    summary = {
        "generated_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "campaign_first_pass": state,
        "record_counts": {
            f"GO-{program}_{status}": count
            for (program, status), count in sorted(Counter((row["program"], row["status"]) for row in records).items())
        },
        "models": models,
        "manual_review_count": len(review_queue),
        "review_sheets_made": sheets_made,
        "campaign_snapshot_row_counts": snapshot_counts,
        "scientific_separation": {
            "GO-3055": "F090W-F150W, TRGB-anchored absolute calibration",
            "GO-7763": "F115W-F150W, apparent Virgo relation",
            "combined_color_fit_allowed": False,
        },
    }
    (output_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_report(records, models, review_queue, state, output_root, sheets_made)
    report_path = output_root / "README.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"Analysis complete: {report_path}", log_path)
    return report_path


def path_argument(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for GO-7763 first-pass completion, then analyze GO-3055 + GO-7763."
    )
    parser.add_argument("--campaign-root", type=path_argument, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--status-csv", type=path_argument, default=None)
    parser.add_argument("--database", type=path_argument, default=None)
    parser.add_argument("--go3055-root", type=path_argument, default=DEFAULT_GO3055_ROOT)
    parser.add_argument("--output-root", type=path_argument, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--analyze-now",
        action="store_true",
        help="Build a provisional report immediately instead of waiting (for testing).",
    )
    parser.add_argument("--no-review-sheets", action="store_true")
    parser.add_argument("--max-review-sheets", type=int, default=12)
    args = parser.parse_args(argv)
    if args.status_csv is None:
        args.status_csv = args.campaign_root / "target_status.csv"
    if args.database is None:
        args.database = args.campaign_root / "campaign_state.sqlite"
    for path, label in [
        (args.status_csv, "target status CSV"),
        (args.database, "campaign SQLite database"),
        (args.go3055_root, "GO-3055 result directory"),
    ]:
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "monitor.log"
    log(f"Monitor started; project={PROJECT_ROOT}", log_path)
    log(f"Campaign database={args.database}", log_path)
    log(f"Output directory={args.output_root}", log_path)
    try:
        if args.analyze_now:
            state = first_pass_state(args.database)
            log("--analyze-now: creating a provisional report without waiting.", log_path)
        else:
            state = wait_for_first_pass(args.database, args.status_csv, args.poll_seconds, log_path)
        run_analysis(args, state, log_path)
    except KeyboardInterrupt:
        log("Monitor interrupted by user; the processing batch was not touched.", log_path)
        return 130
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}", log_path)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
