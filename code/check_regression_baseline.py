#!/usr/bin/env python3
"""Compare existing compact GO-3055 outputs with the frozen baseline.

This script does not rerun the SBF notebook and does not read large FITS files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_DIR = SCRIPT_DIR / "regression_baseline"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "sbf2_batch_outputs"


def read_csv_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"duplicate {key}={value!r} in {path}")
        result[value] = row
    return result


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def compare_float(
    failures: list[str],
    label: str,
    expected: str | float,
    actual: str | float,
    tolerance: float,
) -> float:
    expected_f = float(expected)
    actual_f = float(actual)
    delta = abs(actual_f - expected_f)
    if not math.isfinite(delta) or delta > tolerance:
        failures.append(
            f"{label}: expected={expected_f:.12g}, actual={actual_f:.12g}, "
            f"|delta|={delta:.6g} > {tolerance:.6g}"
        )
    return delta


def compare_exact(
    failures: list[str], label: str, expected: object, actual: object
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected={expected!r}, actual={actual!r}")


def check_targets(
    baseline_dir: Path, outputs_dir: Path, config: dict
) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    maxima = {"magnitude": 0.0, "sigma": 0.0, "color": 0.0, "k_window": 0.0}
    baseline = read_csv_by(baseline_dir / "baseline_targets.csv", "galaxy")
    current = read_csv_by(outputs_dir / "sbf2_batch_results.csv", "galaxy")
    calibration = read_csv_by(
        outputs_dir / "coursework_calibration_input.csv", "galaxy"
    )
    tolerances = config["tolerances"]

    compare_exact(failures, "target set", set(baseline), set(current))
    compare_exact(failures, "calibration target set", set(baseline), set(calibration))

    for galaxy in sorted(set(baseline) & set(current) & set(calibration)):
        expected = baseline[galaxy]
        actual = current[galaxy]
        actual_cal = calibration[galaxy]
        prefix = f"{galaxy}"

        compare_exact(failures, f"{prefix}.status", expected["status"], actual["status"])
        compare_exact(
            failures,
            f"{prefix}.signal_product",
            expected["signal_product"],
            Path(actual["f150w_path"]).name,
        )
        compare_exact(
            failures,
            f"{prefix}.color_product",
            expected["color_product"],
            Path(actual["f090w_path"]).name,
        )

        for field in [
            "recommended_mbar_inner",
            "recommended_mbar_outer",
            "recommended_mbar_weighted",
        ]:
            delta = compare_float(
                failures,
                f"{prefix}.{field}",
                expected[field],
                actual[field],
                tolerances["target_magnitude_abs"],
            )
            maxima["magnitude"] = max(maxima["magnitude"], delta)

        delta = compare_float(
            failures,
            f"{prefix}.recommended_sigma_adopted",
            expected["recommended_sigma_adopted"],
            actual["recommended_sigma_adopted"],
            tolerances["target_sigma_abs"],
        )
        maxima["sigma"] = max(maxima["sigma"], delta)

        delta = compare_float(
            failures,
            f"{prefix}.color_F090W_F150W",
            expected["color_F090W_F150W"],
            actual["color_F090W_F150W"],
            tolerances["target_color_abs"],
        )
        maxima["color"] = max(maxima["color"], delta)

        for field in ["recommended_kmin", "recommended_kmax"]:
            delta = compare_float(
                failures,
                f"{prefix}.{field}",
                expected[field],
                actual[field],
                tolerances["k_window_abs"],
            )
            maxima["k_window"] = max(maxima["k_window"], delta)

        compare_exact(
            failures,
            f"{prefix}.quality_flag_effective",
            expected["quality_flag_effective"],
            actual_cal["quality_flag_effective"],
        )
        for field in [
            "is_clean_effective",
            "recommended_is_main_window",
            "recommended_uses_two_annuli",
        ]:
            source = actual_cal if field == "is_clean_effective" else actual
            compare_exact(
                failures,
                f"{prefix}.{field}",
                parse_bool(expected[field]),
                parse_bool(source[field]),
            )

    return failures, maxima


def check_global_fit(
    outputs_dir: Path, config: dict
) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    maxima = {"fit": 0.0, "model_wrms": 0.0}
    fit_path = outputs_dir / "coursework_fit_summary.csv"
    with fit_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return [f"{fit_path}: expected one row, found {len(rows)}"], maxima
    current = rows[0]
    expected = config["fit"]
    tolerances = config["tolerances"]

    compare_exact(
        failures, "fit.n_total", config["target_count"], int(current["n_total"])
    )
    compare_exact(
        failures,
        "fit.n_clean_effective",
        config["clean_target_count"],
        int(current["n_clean_effective"]),
    )

    field_tolerances = {
        "pivot_color": tolerances["target_color_abs"],
        "intercept": tolerances["fit_intercept_abs"],
        "slope": tolerances["fit_slope_abs"],
        "sigma_int": tolerances["fit_scatter_abs"],
        "in_sample_wrms": tolerances["fit_scatter_abs"],
        "leave_one_out_wrms": tolerances["fit_scatter_abs"],
    }
    for field, tolerance in field_tolerances.items():
        delta = compare_float(
            failures, f"fit.{field}", expected[field], current[field], tolerance
        )
        maxima["fit"] = max(maxima["fit"], delta)

    model_rows = read_csv_by(outputs_dir / "coursework_model_comparison.csv", "model")
    compare_exact(
        failures,
        "model comparison set",
        set(config["model_comparison"]),
        set(model_rows),
    )
    for model in sorted(set(config["model_comparison"]) & set(model_rows)):
        for field in ["in_sample_wrms", "leave_one_out_wrms"]:
            delta = compare_float(
                failures,
                f"model.{model}.{field}",
                config["model_comparison"][model][field],
                model_rows[model][field],
                tolerances["model_wrms_abs"],
            )
            maxima["model_wrms"] = max(maxima["model_wrms"], delta)

    return failures, maxima


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    config = json.loads((baseline_dir / "baseline_calibration.json").read_text())

    target_failures, target_maxima = check_targets(baseline_dir, outputs_dir, config)
    fit_failures, fit_maxima = check_global_fit(outputs_dir, config)
    failures = target_failures + fit_failures

    if failures:
        print(f"REGRESSION FAILED: {len(failures)} difference(s)")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "REGRESSION OK: "
        f"{config['target_count']} targets; "
        f"max |delta mbar|={target_maxima['magnitude']:.3g} mag; "
        f"max |delta color|={target_maxima['color']:.3g} mag; "
        f"max |delta fit|={fit_maxima['fit']:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

