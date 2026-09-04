"""Small helpers for the offline GO-3055 F090W SBF campaign.

The frozen F150W notebook remains untouched.  At run time we make a private
F090W template which keeps its galaxy modelling and masking cells, changes the
foreground-extinction correction to F090W, omits the deferred colour block and
adds a validated, interrupt-safe PSF cache.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from sbf_campaign_runtime import atomic_write_text


F090W_FILTER = "F090W"
F090W_PSF_SIZE = 129
F090W_PSF_COUNT = 5
F090W_SOURCE_SCHEMA = 3
F090W_ISOPHOTE_METHOD = "f090_sersic_seed_multistart_universal_qc_v2"
F090W_MASK_METHOD = "external_large_bright_guarded_v1"
F090W_INNER_MASK_GUARD_METHOD = "ignore_compact_premask_inside_inner_sbf_radius_v1"
F090W_INNER_MASK_GUARD_TARGETS = frozenset({"NGC 4636"})
F090W_MIN_WORKING_ISOPHOTES = 10
F090W_ISOPHOTE_QC_LIMITS = {
    "max_median_center_shift_px": 50.0,
    "max_center_shift_px": 15.0,
    "max_stop2_fraction": 0.30,
    "max_consecutive_stop2": 8,
    "max_frozen_stop_fraction": 0.25,
    "max_consecutive_frozen_stop": 20,
    "max_singular_stop_count": 0,
    "max_center_step_px": 10.0,
    "max_eps_step": 0.03,
    "max_shape_step": 0.05,
    "max_intensity_rise_fraction": 0.05,
}


def is_large_external_contaminant(
    *,
    area_pixels: int,
    min_radius_pixels: float,
    peak_snr: float,
    compact_max_area: int,
    core_guard_radius_pixels: float,
    min_peak_snr: float,
) -> bool:
    """Select a large, bright source that does not touch the galaxy core."""

    values = np.asarray([min_radius_pixels, peak_snr], dtype=float)
    return bool(
        int(area_pixels) > int(compact_max_area)
        and np.all(np.isfinite(values))
        and float(min_radius_pixels) > float(core_guard_radius_pixels)
        and float(peak_snr) > float(min_peak_snr)
    )


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def isophote_sequence_diagnostics(
    items: Any,
    center_x: float,
    center_y: float,
    *,
    sma_min: float | None = None,
    sma_max: float | None = None,
) -> dict[str, Any]:
    """Summarise convergence and smoothness of an isophote sequence.

    Position angles are axial: 0 and pi describe the same major axis.  Their
    adjacent differences are therefore wrapped with period pi, rather than
    subtracted naively.
    """

    if items is None or len(items) == 0:
        return {
            "n_isophotes": 0,
            "max_sma_px": np.nan,
            "median_center_shift_px": np.nan,
            "max_center_shift_px": np.nan,
            "quality_n_isophotes": 0,
            "quality_sma_min_px": np.nan,
            "quality_sma_max_px": np.nan,
            "stop2_count": 0,
            "stop2_fraction": np.nan,
            "max_consecutive_stop2": 0,
            "frozen_stop_count": 0,
            "frozen_stop_fraction": np.nan,
            "max_consecutive_frozen_stop": 0,
            "singular_stop_count": 0,
            "max_center_step_px": np.nan,
            "max_eps_step": np.nan,
            "max_pa_step_rad": np.nan,
            "max_shape_step": np.nan,
            "max_intensity_rise_fraction": np.nan,
        }

    rows = []
    for item in items:
        rows.append({
            "sma": float(getattr(item, "sma", np.nan)),
            "x0": float(getattr(item, "x0", np.nan)),
            "y0": float(getattr(item, "y0", np.nan)),
            "eps": float(getattr(item, "eps", np.nan)),
            "pa": float(getattr(item, "pa", np.nan)),
            "intens": float(getattr(item, "intens", np.nan)),
            "stop_code": float(getattr(item, "stop_code", np.nan)),
        })
    frame = pd.DataFrame(rows).sort_values("sma").reset_index(drop=True)
    finite_sma = np.isfinite(frame["sma"].to_numpy(float))
    overall = frame.loc[finite_sma].copy()
    if overall.empty:
        return isophote_sequence_diagnostics([], center_x, center_y)

    center_shift = np.hypot(
        overall["x0"].to_numpy(float) - float(center_x),
        overall["y0"].to_numpy(float) - float(center_y),
    )
    quality = overall.copy()
    if sma_min is not None:
        quality = quality.loc[quality["sma"] >= float(sma_min)]
    if sma_max is not None:
        quality = quality.loc[quality["sma"] <= float(sma_max)]

    result = {
        "n_isophotes": int(len(overall)),
        "max_sma_px": float(overall["sma"].max()),
        "median_center_shift_px": float(np.nanmedian(center_shift)),
        "max_center_shift_px": np.nan,
        "quality_n_isophotes": int(len(quality)),
        "quality_sma_min_px": (
            float(quality["sma"].min()) if not quality.empty else np.nan
        ),
        "quality_sma_max_px": (
            float(quality["sma"].max()) if not quality.empty else np.nan
        ),
        "stop2_count": 0,
        "stop2_fraction": np.nan,
        "max_consecutive_stop2": 0,
        "frozen_stop_count": 0,
        "frozen_stop_fraction": np.nan,
        "max_consecutive_frozen_stop": 0,
        "singular_stop_count": 0,
        "max_center_step_px": np.nan,
        "max_eps_step": np.nan,
        "max_pa_step_rad": np.nan,
        "max_shape_step": np.nan,
        "max_intensity_rise_fraction": np.nan,
    }
    if quality.empty:
        return result

    stop2 = np.isclose(quality["stop_code"].to_numpy(float), 2.0)
    result["stop2_count"] = int(stop2.sum())
    result["stop2_fraction"] = float(stop2.mean())
    result["max_consecutive_stop2"] = _longest_true_run(stop2)
    # Codes 1, 4, and 5 can lead Photutils to propagate one fixed geometry
    # through many radii.  Such a sequence looks deceptively smooth.  Code 3
    # is a singular harmonic fit and is tracked separately as a hard failure.
    stop_codes = quality["stop_code"].to_numpy(float)
    frozen_stop = np.any(
        np.isclose(stop_codes[:, None], np.asarray([1.0, 4.0, 5.0])[None, :]),
        axis=1,
    )
    result["frozen_stop_count"] = int(frozen_stop.sum())
    result["frozen_stop_fraction"] = float(frozen_stop.mean())
    result["max_consecutive_frozen_stop"] = _longest_true_run(frozen_stop)
    result["singular_stop_count"] = int(np.isclose(stop_codes, 3.0).sum())
    if len(quality) < 2:
        return result

    x0 = quality["x0"].to_numpy(float)
    y0 = quality["y0"].to_numpy(float)
    eps = quality["eps"].to_numpy(float)
    pa = quality["pa"].to_numpy(float)
    intensity = quality["intens"].to_numpy(float)
    center_step = np.hypot(np.diff(x0), np.diff(y0))
    eps_step = np.abs(np.diff(eps))
    pa_delta = np.diff(pa)
    pa_step = 0.5 * np.abs(
        np.arctan2(np.sin(2.0 * pa_delta), np.cos(2.0 * pa_delta))
    )
    valid_intensity = (
        np.isfinite(intensity[:-1])
        & np.isfinite(intensity[1:])
        & (intensity[:-1] > 0.0)
    )
    rises = np.full(len(intensity) - 1, np.nan, dtype=float)
    rises[valid_intensity] = (
        (intensity[1:][valid_intensity] - intensity[:-1][valid_intensity])
        / np.abs(intensity[:-1][valid_intensity])
    )
    quality_center_shift = np.hypot(
        x0 - float(center_x), y0 - float(center_y)
    )
    complex_shape = eps * np.exp(2.0j * pa)
    shape_step = np.abs(np.diff(complex_shape))
    result["max_center_shift_px"] = float(np.nanmax(quality_center_shift))
    result["max_center_step_px"] = float(np.nanmax(center_step))
    result["max_eps_step"] = float(np.nanmax(eps_step))
    result["max_pa_step_rad"] = float(np.nanmax(pa_step))
    result["max_shape_step"] = float(np.nanmax(shape_step))
    if np.any(np.isfinite(rises)):
        result["max_intensity_rise_fraction"] = float(np.nanmax(rises))
    return result


def isophote_sequence_qc(
    details: dict[str, Any],
    *,
    required_sma_px: float,
    min_isophotes: int,
    max_median_center_shift_px: float,
    max_center_shift_px: float | None = None,
    max_stop2_fraction: float | None = None,
    max_consecutive_stop2: int | None = None,
    max_frozen_stop_fraction: float | None = None,
    max_consecutive_frozen_stop: int | None = None,
    max_singular_stop_count: int | None = None,
    max_center_step_px: float | None = None,
    max_eps_step: float | None = None,
    max_pa_step_rad: float | None = None,
    max_shape_step: float | None = None,
    max_intensity_rise_fraction: float | None = None,
) -> tuple[bool, str]:
    """Apply explicit, target-independent gates to isophote diagnostics."""

    problems: list[str] = []
    if int(details.get("n_isophotes", 0)) < int(min_isophotes):
        problems.append(
            f"only {int(details.get('n_isophotes', 0))} isophotes"
        )
    max_sma = float(details.get("max_sma_px", np.nan))
    if not np.isfinite(max_sma) or max_sma < float(required_sma_px):
        problems.append(f"maximum sma {max_sma:.1f} px")
    median_shift = float(details.get("median_center_shift_px", np.nan))
    if (
        not np.isfinite(median_shift)
        or median_shift > float(max_median_center_shift_px)
    ):
        problems.append(f"median center offset {median_shift:.2f} px")
    quality_n = int(details.get("quality_n_isophotes", 0))
    if quality_n < 2:
        problems.append(f"only {quality_n} isophotes in the QC interval")

    limits = (
        ("max_center_shift_px", max_center_shift_px, "center offset", "px"),
        ("stop2_fraction", max_stop2_fraction, "stop-code-2 fraction", ""),
        ("max_consecutive_stop2", max_consecutive_stop2, "consecutive stop-code-2 run", ""),
        (
            "frozen_stop_fraction", max_frozen_stop_fraction,
            "stop-code-1/4/5 fraction", "",
        ),
        (
            "max_consecutive_frozen_stop", max_consecutive_frozen_stop,
            "consecutive stop-code-1/4/5 run", "",
        ),
        (
            "singular_stop_count", max_singular_stop_count,
            "stop-code-3 count", "",
        ),
        ("max_center_step_px", max_center_step_px, "center step", "px"),
        ("max_eps_step", max_eps_step, "ellipticity step", ""),
        ("max_pa_step_rad", max_pa_step_rad, "position-angle step", "rad"),
        ("max_shape_step", max_shape_step, "complex-shape step", ""),
        (
            "max_intensity_rise_fraction",
            max_intensity_rise_fraction,
            "outward intensity rise",
            "",
        ),
    )
    for key, limit, label, unit in limits:
        if limit is None or quality_n < 2:
            continue
        value = float(details.get(key, np.nan))
        if not np.isfinite(value) or value > float(limit):
            suffix = f" {unit}" if unit else ""
            problems.append(f"{label} {value:.4g}{suffix}")
    return not problems, "; ".join(problems)


def isophote_bootstrap_rank(
    details: dict[str, Any], start_index: int
) -> tuple[float, ...]:
    """Rank viable bootstrap fits; a larger start radius is never preferred."""

    def finite_or_inf(value: Any) -> float:
        value = float(value)
        return value if np.isfinite(value) else float("inf")

    start_sma = finite_or_inf(details.get("start_sma_px", np.nan))
    rise = finite_or_inf(details.get("max_intensity_rise_fraction", np.nan))
    if np.isfinite(rise):
        rise = max(0.0, rise)
    return (
        finite_or_inf(details.get("stop2_fraction", np.nan)),
        finite_or_inf(details.get("max_consecutive_stop2", np.nan)),
        finite_or_inf(details.get("frozen_stop_fraction", np.nan)),
        finite_or_inf(details.get("max_consecutive_frozen_stop", np.nan)),
        finite_or_inf(details.get("singular_stop_count", np.nan)),
        finite_or_inf(details.get("max_eps_step", np.nan)) / 0.03,
        finite_or_inf(details.get("max_shape_step", np.nan)) / 0.05,
        rise / 0.05,
        finite_or_inf(details.get("median_center_shift_px", np.nan)) / 50.0,
        abs(start_sma - 50.0),
        float(start_index),
    )


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV completion artifact without exposing a partial file."""

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    atomic_write_text(Path(path), buffer.getvalue())
    return Path(path)


def load_f150w_reference_center(
    galaxy: str,
    auxiliary_f150_path: Path,
    signal_wcs: Any,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Transfer the accepted F150W model centre onto the F090W WCS.

    The accepted F150W centre is stored in the model FITS header.  This is a
    much stronger reference than another unrestricted peak search in F090W.
    """

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    slug = "_".join(str(galaxy).upper().split())
    result_path = root / "runs" / "sbf2_go3055" / "batch" / f"{slug}_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"accepted F150W result is absent: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model_path = Path(result.get("model_full_fits", ""))
    if not model_path.is_file():
        raise FileNotFoundError(f"accepted F150W model is absent: {model_path}")

    model_header = fits.getheader(model_path, 0)
    x_f150 = float(model_header["SBFXCEN"])
    y_f150 = float(model_header["SBFYCEN"])
    f150_header = fits.getheader(Path(auxiliary_f150_path), "SCI")
    from astropy.wcs import WCS

    f150_wcs = WCS(f150_header).celestial
    ra_deg, dec_deg = f150_wcs.pixel_to_world_values(x_f150, y_f150)
    x_f090, y_f090 = signal_wcs.celestial.world_to_pixel_values(ra_deg, dec_deg)
    values = np.asarray([x_f090, y_f090, ra_deg, dec_deg], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{galaxy}: non-finite F150W reference-centre transfer")
    return {
        "source": "accepted-F150W-model/WCS",
        "x_pixel": float(x_f090),
        "y_pixel": float(y_f090),
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "f150_x_pixel": x_f150,
        "f150_y_pixel": y_f150,
        "f150_model_fits": str(model_path.resolve()),
        "f150_result_json": str(result_path.resolve()),
    }


def _cell_source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def _set_cell_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _f090w_center_cell() -> str:
    return '''print("choosing F090W galaxy center from the accepted F150W reference...")

from sbf090_pipeline_support import load_f150w_reference_center, write_dataframe_atomic

center_reference_method = "accepted-F150W-model/WCS"
signal_primary_header = fits.getheader(f150w_path, 0)
reference = None
reference_error = ""
try:
    reference = load_f150w_reference_center(
        TARGET_GALAXY, f090w_path, wcs150,
    )
except Exception as exc:
    reference_error = f"{type(exc).__name__}: {exc}"
    print(f"[CENTER][WARN] F150W reference unavailable: {reference_error}")

if FIXED_CENTER is not None:
    anchor_x, anchor_y = map(float, FIXED_CENTER)
    anchor_source = "fixed"
elif reference is not None:
    anchor_x = float(reference["x_pixel"])
    anchor_y = float(reference["y_pixel"])
    anchor_source = str(reference["source"])
else:
    try:
        target_ra = float(signal_primary_header["TARG_RA"])
        target_dec = float(signal_primary_header["TARG_DEC"])
        anchor_x, anchor_y = map(
            float, wcs150.celestial.world_to_pixel_values(target_ra, target_dec)
        )
        anchor_source = "F090W TARG_RA/TARG_DEC fallback"
    except Exception as exc:
        raise RuntimeError(
            f"[CENTER] neither the accepted F150W centre nor a valid F090W "
            f"target coordinate is available: {exc}"
        ) from exc

in_frame = (
    np.isfinite(anchor_x) and np.isfinite(anchor_y)
    and 0.0 <= anchor_x < img.shape[1]
    and 0.0 <= anchor_y < img.shape[0]
)
if not in_frame:
    raise RuntimeError(
        f"[CENTER] reference centre is outside F090W: ({anchor_x}, {anchor_y})"
    )

yy_center, xx_center = np.ogrid[:img.shape[0], :img.shape[1]]
local_valid = (
    valid150 & (~premask)
    & ((xx_center - anchor_x) ** 2 + (yy_center - anchor_y) ** 2
       <= CENTER_LOCAL_RADIUS_PX ** 2)
)
local_x = local_y = local_shift_px = np.nan
center_src = anchor_source
x0_center, y0_center = anchor_x, anchor_y
if FIXED_CENTER is None and int(local_valid.sum()) >= CENTER_GUESS_MIN_PIXELS:
    local_x, local_y = guess_center_fast(
        img, local_valid,
        down=CENTER_GUESS_DOWN,
        sigma=CENTER_GUESS_SMOOTH_SIGMA,
        q=CENTER_GUESS_Q,
        min_sel_pixels=CENTER_GUESS_MIN_PIXELS,
        wcs=wcs150,
        log=True,
    )
    local_shift_px = float(np.hypot(local_x - anchor_x, local_y - anchor_y))
    if np.isfinite(local_shift_px) and local_shift_px <= CENTER_REFINE_MAX_SHIFT_PX:
        x0_center, y0_center = float(local_x), float(local_y)
        center_src = f"local F090W refinement of {anchor_source}"
    else:
        print(
            f"[CENTER][WARN] local shift={local_shift_px:.2f} px exceeds "
            f"{CENTER_REFINE_MAX_SHIFT_PX:.1f}; retaining the reference"
        )

center_ra_deg, center_dec_deg = wcs150.celestial.pixel_to_world_values(
    x0_center, y0_center
)
center_table_path = out_dir / f"{stem}_sbf_center.csv"
center_df = pd.DataFrame([{
    "galaxy": TARGET_GALAXY,
    "adopted_source": center_src,
    "adopted_x_f090": float(x0_center),
    "adopted_y_f090": float(y0_center),
    "adopted_ra_deg": float(center_ra_deg),
    "adopted_dec_deg": float(center_dec_deg),
    "anchor_source": anchor_source,
    "reference_method": center_reference_method,
    "anchor_x_f090": float(anchor_x),
    "anchor_y_f090": float(anchor_y),
    "local_x_f090": float(local_x),
    "local_y_f090": float(local_y),
    "local_shift_px": float(local_shift_px),
    "local_radius_px": float(CENTER_LOCAL_RADIUS_PX),
    "accepted_shift_limit_px": float(CENTER_REFINE_MAX_SHIFT_PX),
    "f150_model_fits": "" if reference is None else reference["f150_model_fits"],
    "f150_result_json": "" if reference is None else reference["f150_result_json"],
    "reference_error": reference_error,
}])
write_dataframe_atomic(center_df, center_table_path)
print(
    f"[CENTER] using ({x0_center:.2f}, {y0_center:.2f}) [{center_src}]; "
    f"diagnostics -> {center_table_path}"
)
'''


def _f090w_isophote_cell() -> str:
    return '''print("fitting F090W isophotes with quality-ranked multiple starts...")

from sbf090_pipeline_support import (
    F090W_ISOPHOTE_METHOD,
    isophote_bootstrap_rank,
    isophote_sequence_diagnostics,
    isophote_sequence_qc,
    write_dataframe_atomic,
)

minsma = ISO_MINSMA
fit_maxsma = ISO_MAXSMA_FIT
bootstrap_maxsma = min(ISO_BOOTSTRAP_MAXSMA, fit_maxsma)
science_sma_min = float(SBF_LIT_INNER_ARCSEC[0] / np.sqrt(pix_area))
science_sma_max = float(SBF_LIT_OUTER_ARCSEC[1] / np.sqrt(pix_area))
required_full_sma = science_sma_max
fit_attempt_rows = []
isophote_attempts_path = out_dir / f"{stem}_sbf_isophote_attempts.csv"
isophote_table_path = out_dir / f"{stem}_sbf_isophotes.csv"

# The Sersic model is used only for a better initial ellipticity and position
# angle.  Both parameters remain free in every isophotal fit.
isophote_seed_source = "configured fallback"
isophote_seed_eps = float(ISO_START_EPS)
isophote_seed_pa_rad = float(ISO_START_PA)
try:
    seed_values = np.asarray([
        float(sersic_fit.amplitude.value),
        float(sersic_fit.ellip.value),
        float(sersic_fit.theta.value),
        float(sersic_fit.x_0.value),
        float(sersic_fit.y_0.value),
    ])
    seed_center_shift = float(np.hypot(
        seed_values[3] - x0_center, seed_values[4] - y0_center
    ))
    if (
        np.all(np.isfinite(seed_values))
        and seed_values[0] > SERSIC_MIN_AMPLITUDE
        and ISO_SEED_MIN_EPS <= seed_values[1] <= 0.85
        and seed_center_shift <= ISO_SEED_MAX_CENTER_SHIFT_PX
    ):
        isophote_seed_eps = float(seed_values[1])
        isophote_seed_pa_rad = float(np.mod(seed_values[2], np.pi))
        isophote_seed_source = "F090W Sersic initial geometry"
except Exception as exc:
    print(f"[ISO][WARN] Sersic geometry unavailable: {type(exc).__name__}: {exc}")

print(
    f"[ISO] seed={isophote_seed_source}: eps={isophote_seed_eps:.3f}, "
    f"PA={np.degrees(isophote_seed_pa_rad):.2f} deg"
)

def _qc(items, phase):
    if phase == "full":
        details = isophote_sequence_diagnostics(
            items, x0c, y0c,
            sma_min=science_sma_min, sma_max=science_sma_max,
        )
        passed, reason = isophote_sequence_qc(
            details,
            required_sma_px=required_full_sma,
            min_isophotes=MIN_WORKING_ISOPHOTES,
            max_median_center_shift_px=ISO_MAX_MEDIAN_CENTER_SHIFT_PX,
            max_center_shift_px=ISO_QC_MAX_CENTER_OFFSET_PX,
            max_stop2_fraction=ISO_QC_MAX_STOP2_FRACTION,
            max_consecutive_stop2=ISO_QC_MAX_CONSECUTIVE_STOP2,
            max_frozen_stop_fraction=ISO_QC_MAX_FROZEN_STOP_FRACTION,
            max_consecutive_frozen_stop=ISO_QC_MAX_CONSECUTIVE_FROZEN_STOP,
            max_singular_stop_count=ISO_QC_MAX_SINGULAR_STOP_COUNT,
            max_center_step_px=ISO_QC_MAX_CENTER_STEP_PX,
            max_eps_step=ISO_QC_MAX_EPS_STEP,
            max_pa_step_rad=None,
            max_shape_step=ISO_QC_MAX_SHAPE_STEP,
            max_intensity_rise_fraction=ISO_QC_MAX_INTENSITY_RISE_FRACTION,
        )
    else:
        details = isophote_sequence_diagnostics(items, x0c, y0c)
        passed, reason = isophote_sequence_qc(
            details,
            required_sma_px=0.75 * bootstrap_maxsma,
            min_isophotes=MIN_WORKING_ISOPHOTES,
            max_median_center_shift_px=ISO_MAX_MEDIAN_CENTER_SHIFT_PX,
            max_stop2_fraction=ISO_QC_MAX_STOP2_FRACTION,
            max_consecutive_stop2=ISO_QC_MAX_CONSECUTIVE_STOP2,
            max_frozen_stop_fraction=ISO_QC_MAX_FROZEN_STOP_FRACTION,
            max_consecutive_frozen_stop=ISO_QC_MAX_CONSECUTIVE_FROZEN_STOP,
            max_singular_stop_count=ISO_QC_MAX_SINGULAR_STOP_COUNT,
        )
    return passed, reason, details

def _fit_once(data_try, dataset, n_finite, phase, start_sma, maxsma, step, fix_center):
    error = ""
    items = None
    try:
        geometry = EllipseGeometry(
            x0=x0c, y0=y0c, sma=float(start_sma),
            eps=isophote_seed_eps, pa=isophote_seed_pa_rad,
        )
        items = Ellipse(data_try, geometry).fit_image(
            minsma=minsma, maxsma=maxsma, step=step, linear=True,
            fix_center=fix_center, fix_pa=False, fix_eps=False,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    qc_ok, qc_reason, details = _qc(items, phase)
    status = "working" if qc_ok else ("error" if error else "rejected")
    row = {
        "phase": phase,
        "dataset": dataset,
        "n_finite": int(n_finite),
        "start_sma_px": float(start_sma),
        "step_px": float(step),
        "fix_center": bool(fix_center),
        "seed_source": isophote_seed_source,
        "seed_eps": float(isophote_seed_eps),
        "seed_pa_rad": float(isophote_seed_pa_rad),
        "requested_max_sma_px": float(maxsma),
        "required_sma_px": float(
            0.75 * bootstrap_maxsma if phase == "bootstrap" else required_full_sma
        ),
        "qc_sma_min_px": float(science_sma_min) if phase == "full" else np.nan,
        "qc_sma_max_px": float(science_sma_max) if phase == "full" else np.nan,
        "status": status,
        "selected": False,
        "error": error,
        "qc_reason": qc_reason,
        **details,
    }
    fit_attempt_rows.append(row)
    attempt_index = len(fit_attempt_rows) - 1
    write_dataframe_atomic(pd.DataFrame(fit_attempt_rows), isophote_attempts_path)
    print(
        f"[ISO:{phase}] {dataset}, start={start_sma:g}, step={step:g}, "
        f"fixed={fix_center}: N={details['n_isophotes']}, "
        f"max={details['max_sma_px']:.1f}, stop2={details['stop2_fraction']:.3f}, "
        f"frozen={details['frozen_stop_fraction']:.3f}, "
        f"singular={details['singular_stop_count']}, "
        f"shape-step={details['max_shape_step']:.3f}, {status}"
    )
    if qc_reason:
        print(f"[ISO:{phase}] rejection: {qc_reason}")
    return items, details, qc_ok, (error or qc_reason), attempt_index

def _fit_dataset(data_try, dataset, n_finite, fix_center):
    candidates = []
    for start_index, start_sma in enumerate(ISO_START_SMA_CANDIDATES):
        for step in (ISO_STEP_MAIN, ISO_STEP_COARSE):
            _, details, usable, _, attempt_index = _fit_once(
                data_try, dataset, n_finite, "bootstrap", start_sma,
                bootstrap_maxsma, step, fix_center,
            )
            if usable:
                ranked = dict(details)
                ranked["start_sma_px"] = float(start_sma)
                candidates.append((
                    isophote_bootstrap_rank(ranked, start_index),
                    float(start_sma), float(step), attempt_index,
                ))
                break
    candidates.sort(key=lambda item: item[0])
    if candidates:
        print(
            "[ISO:bootstrap] quality order: "
            + ", ".join(f"start={start:g}" for _, start, _, _ in candidates)
        )
    for rank, start_sma, preferred_step, _ in candidates:
        steps = (
            (preferred_step,)
            if preferred_step == ISO_STEP_COARSE
            else (ISO_STEP_MAIN, ISO_STEP_COARSE)
        )
        for step in steps:
            items, details, usable, error, attempt_index = _fit_once(
                data_try, dataset, n_finite, "full", start_sma,
                fit_maxsma, step, fix_center,
            )
            if usable:
                fit_attempt_rows[attempt_index]["selected"] = True
                fit_attempt_rows[attempt_index]["bootstrap_rank"] = repr(rank)
                write_dataframe_atomic(
                    pd.DataFrame(fit_attempt_rows), isophote_attempts_path
                )
                signature = (
                    f"method={F090W_ISOPHOTE_METHOD}, start={start_sma:g}, "
                    f"step={step:g}, linear=True, fix_center={fix_center}, "
                    f"seed_eps={isophote_seed_eps:.4f}, "
                    f"seed_pa={isophote_seed_pa_rad:.4f}"
                )
                return (
                    items, details, signature, error,
                    start_sma, step, attempt_index,
                )
    return None, None, None, "no full fit passed universal QC", None, None, None

fit_datasets = []
if ISO_FIT_USE_REAL_PIXELS:
    fit_datasets.append(("real_only", data_real_ma, int(ok_real.sum())))
    if ISO_FIT_ALLOW_FILLED_FALLBACK:
        fit_datasets.append(("filled_fallback", data_fill_ma, int(ok_fill.sum())))
else:
    fit_datasets.append(("filled_primary", data_fill_ma, int(ok_fill.sum())))

isolist = None
failure_notes = []
for mode_label, data_try, n_finite in fit_datasets:
    if n_finite < MIN_PIXELS_ISO_CROP:
        failure_notes.append(f"{mode_label}: only {n_finite} finite pixels")
        continue
    for fix_center_try in (False, True):
        result = _fit_dataset(data_try, mode_label, n_finite, fix_center_try)
        items, details, signature, error, start_sma, step, attempt_index = result
        if items is not None:
            isolist = items
            isophote_qc_details = details
            isophote_qc_passed = True
            isophote_qc_reason = ""
            iso_fit_mode_used = mode_label
            iso_fit_signature_used = signature
            iso_start_sma_used = float(start_sma)
            iso_step_used = float(step)
            iso_fix_center_used = bool(fix_center_try)
            iso_selected_attempt_index = int(attempt_index)
            break
        failure_notes.append(f"{mode_label}, fixed={fix_center_try}: {error}")
    if isolist is not None:
        break

if isolist is None:
    raise RuntimeError(
        "[ISO] no full-radius solution passed universal science-annulus QC: "
        + " | ".join(failure_notes)
        + f"; attempts={isophote_attempts_path}"
    )

rows = []
for index, iso in enumerate(isolist):
    sampled_real = sampled_total = 0
    real_fraction = np.nan
    try:
        xs, ys = iso.sampled_coordinates()
        xi = np.rint(xs).astype(int)
        yi = np.rint(ys).astype(int)
        inside = (
            (xi >= 0) & (xi < valid_c.shape[1])
            & (yi >= 0) & (yi < valid_c.shape[0])
        )
        sampled_total = int(inside.sum())
        if sampled_total:
            sampled_real = int(valid_c[yi[inside], xi[inside]].sum())
            real_fraction = float(sampled_real / sampled_total)
    except Exception:
        pass
    rows.append({
        "index": index,
        "sma_px": float(iso.sma),
        "intensity": float(getattr(iso, "intens", np.nan)),
        "intensity_error": float(getattr(iso, "int_err", np.nan)),
        "x_crop": float(iso.x0),
        "y_crop": float(iso.y0),
        "x_full": float(x1 + iso.x0),
        "y_full": float(y1 + iso.y0),
        "eps": float(iso.eps),
        "pa_rad": float(iso.pa),
        "gradient": float(getattr(iso, "grad", np.nan)),
        "gradient_error": float(getattr(iso, "grad_error", np.nan)),
        "rms": float(getattr(iso, "rms", np.nan)),
        "n_data": float(getattr(iso, "ndata", np.nan)),
        "n_flag": float(getattr(iso, "nflag", np.nan)),
        "stop_code": float(getattr(iso, "stop_code", np.nan)),
        "a3": float(getattr(iso, "a3", np.nan)),
        "b3": float(getattr(iso, "b3", np.nan)),
        "a4": float(getattr(iso, "a4", np.nan)),
        "b4": float(getattr(iso, "b4", np.nan)),
        "real_data_fraction": real_fraction,
        "sampled_real": sampled_real,
        "sampled_total": sampled_total,
    })

isophote_df = pd.DataFrame(rows).sort_values("sma_px").reset_index(drop=True)
isophote_df["center_offset_px"] = np.hypot(
    isophote_df["x_crop"] - x0c, isophote_df["y_crop"] - y0c
)
isophote_df["center_step_px"] = np.hypot(
    isophote_df["x_crop"].diff(), isophote_df["y_crop"].diff()
)
isophote_df["eps_step"] = isophote_df["eps"].diff().abs()
pa_delta = isophote_df["pa_rad"].diff().to_numpy(float)
isophote_df["pa_step_rad"] = 0.5 * np.abs(
    np.arctan2(np.sin(2.0 * pa_delta), np.cos(2.0 * pa_delta))
)
complex_shape = (
    isophote_df["eps"].to_numpy(float)
    * np.exp(2.0j * isophote_df["pa_rad"].to_numpy(float))
)
isophote_df["shape_step"] = np.r_[np.nan, np.abs(np.diff(complex_shape))]
intensity = isophote_df["intensity"].to_numpy(float)
intensity_rise = np.full(len(isophote_df), np.nan)
valid_pair = (
    np.isfinite(intensity[:-1]) & np.isfinite(intensity[1:])
    & (intensity[:-1] > 0.0)
)
intensity_rise[1:][valid_pair] = (
    (intensity[1:][valid_pair] - intensity[:-1][valid_pair])
    / np.abs(intensity[:-1][valid_pair])
)
isophote_df["intensity_rise_fraction"] = intensity_rise
write_dataframe_atomic(isophote_df, isophote_table_path)

isophote_count = int(len(isophote_df))
fitted_sma_max_px = float(isophote_df["sma_px"].max())
x0_fit = float(isophote_df["x_crop"].median())
y0_fit = float(isophote_df["y_crop"].median())
eps_fit = float(isophote_df["eps"].median())
pa_fit = float(isophote_df["pa_rad"].median())

isophote_qc_passed, isophote_qc_reason = isophote_sequence_qc(
    isophote_qc_details,
    required_sma_px=required_full_sma,
    min_isophotes=MIN_WORKING_ISOPHOTES,
    max_median_center_shift_px=ISO_MAX_MEDIAN_CENTER_SHIFT_PX,
    max_center_shift_px=ISO_QC_MAX_CENTER_OFFSET_PX,
    max_stop2_fraction=ISO_QC_MAX_STOP2_FRACTION,
    max_consecutive_stop2=ISO_QC_MAX_CONSECUTIVE_STOP2,
    max_frozen_stop_fraction=ISO_QC_MAX_FROZEN_STOP_FRACTION,
    max_consecutive_frozen_stop=ISO_QC_MAX_CONSECUTIVE_FROZEN_STOP,
    max_singular_stop_count=ISO_QC_MAX_SINGULAR_STOP_COUNT,
    max_center_step_px=ISO_QC_MAX_CENTER_STEP_PX,
    max_eps_step=ISO_QC_MAX_EPS_STEP,
    max_pa_step_rad=None,
    max_shape_step=ISO_QC_MAX_SHAPE_STEP,
    max_intensity_rise_fraction=ISO_QC_MAX_INTENSITY_RISE_FRACTION,
)
if not isophote_qc_passed:
    raise RuntimeError("[ISO-QC] selected isophotes failed: " + isophote_qc_reason)

print(
    f"[ISO] accepted {isophote_count} isophotes through {fitted_sma_max_px:.1f} px; "
    f"mode={iso_fit_mode_used}; {iso_fit_signature_used}"
)
print(
    "[ISO-QC] science annuli: "
    f"stop2={isophote_qc_details['stop2_fraction']:.3f}, "
    f"run={isophote_qc_details['max_consecutive_stop2']}, "
    f"frozen={isophote_qc_details['frozen_stop_fraction']:.3f}, "
    f"frozen-run={isophote_qc_details['max_consecutive_frozen_stop']}, "
    f"singular={isophote_qc_details['singular_stop_count']}, "
    f"shape-step={isophote_qc_details['max_shape_step']:.4f}, "
    f"rise={100.0 * isophote_qc_details['max_intensity_rise_fraction']:.2f}%"
)
print(f"[ISO] isophotes -> {isophote_table_path}")
print(f"[ISO] attempts  -> {isophote_attempts_path}")
'''


def build_f090w_template(base_notebook: Path, output_notebook: Path) -> Path:
    """Build the private F090W execution notebook from frozen ``sbf-2``.

    Only the execution copy is changed.  The base notebook is never written.
    The generated file is deterministic and lives under the F090W run tree.
    """

    base_notebook = Path(base_notebook).resolve()
    output_notebook = Path(output_notebook).resolve()
    notebook = json.loads(base_notebook.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    metadata["sbf_pipeline_family"] = "sbf2"
    metadata["sbf_f090w_generated"] = {
        "signal_filter": F090W_FILTER,
        "psf_size": F090W_PSF_SIZE,
        "source_schema": F090W_SOURCE_SCHEMA,
        "centre": "accepted F150W model transferred by WCS + local F090W refinement",
        "isophotes": F090W_ISOPHOTE_METHOD,
        "mask": F090W_MASK_METHOD,
        "isophote_inner_mask_guard": F090W_INNER_MASK_GUARD_METHOD,
        "isophote_inner_mask_guard_targets": sorted(
            F090W_INNER_MASK_GUARD_TARGETS
        ),
        "colour_measurement": "deferred",
        "base_notebook": base_notebook.name,
    }

    kept_cells: list[dict[str, Any]] = []
    colour_section_reached = False
    psf_cell_found = False
    extinction_cell_found = False
    config_cell_found = False
    premask_cell_found = False
    isophote_prep_cell_found = False
    center_cell_found = False
    isophote_cell_found = False
    final_mask_cell_found = False
    print_wrapper_found = False

    for cell in notebook.get("cells", []):
        source = _cell_source(cell)
        if (
            cell.get("cell_type") == "markdown"
            and source.lstrip().startswith("## Цвета в тех же annuli")
        ):
            colour_section_reached = True
        if colour_section_reached:
            continue

        if cell.get("cell_type") == "markdown":
            if source.lstrip().startswith("# SBF-2:"):
                source = (
                    "# SBF F090W: GO-3055\n\n"
                    "Частный execution-шаблон, автоматически собранный из "
                    "замороженного `sbf-2.ipynb`. Сигнал: F090W. "
                    "Цветовая калибровка в этом прогоне не выполняется.\n"
                )
            source = source.replace(
                "поправка $A_{150}$", "поправка $A_{090}$"
            )

        if cell.get("cell_type") == "code" and source.startswith(
            'print("loading/building PSF...")'
        ):
            psf_cell_found = True
            colour_marker = (
                "# Отдельная F090W PSF нужна только для "
                "согласования цветовой фотометрии."
            )
            tail_marker = "psf_library_arrays = [entry[\"array\"] for entry in psf_library]"
            if colour_marker not in source or tail_marker not in source:
                raise RuntimeError("PSF cell contract in sbf-2.ipynb has changed")
            prefix = source.split(colour_marker, 1)[0]
            tail = tail_marker + source.split(tail_marker, 1)[1]
            source = prefix + tail
            source = source.replace(
                "PSF_NEAREST_OPD_COUNT = min(3, len(eligible_wss))",
                "PSF_NEAREST_OPD_COUNT = min(1, len(eligible_wss))",
            )
            old_write = (
                "fits.HDUList(cache_hdus).writeto("
                "psf_library_fits_path, overwrite=True)"
            )
            new_write = (
                "write_psf_cache_atomic(\n"
                "    psf_library_fits_path, fits.HDUList(cache_hdus),\n"
                "    expected_filter=actual_signal_filter, "
                "expected_size=PSF_SIZE,\n"
                ")"
            )
            if old_write not in source:
                raise RuntimeError("PSF write contract in sbf-2.ipynb has changed")
            source = source.replace(old_write, new_write)
            cache_header_anchor = (
                'cache_hdus[0].header["OPDDT"] = float(opd_delta_days)\n'
                'cache_hdus[0].header["SCI_PXS"] = float(science_pixel_scale_arcsec)'
            )
            cache_header_replacement = (
                'cache_hdus[0].header["OPDDT"] = float(opd_delta_days)\n'
                'cache_hdus[0].header["OPDSIGN"] = float(opd_signed_delta_days)\n'
                'cache_hdus[0].header["SCIFILE"] = Path(science_psf_file).name\n'
                'cache_hdus[0].header["SCIMJD"] = float(science_time.mjd)\n'
                'cache_hdus[0].header["SCIDET"] = str(science_hdr.get("DETECTOR", ""))\n'
                'cache_hdus[0].header["SCI_PXS"] = float(science_pixel_scale_arcsec)'
            )
            if cache_header_anchor not in source:
                raise RuntimeError("PSF primary-header contract has changed")
            source = source.replace(
                cache_header_anchor, cache_header_replacement
            )
            extension_header_anchor = (
                'hdu.header["PSFKIND"] = entry["kind"]\n'
                '    cache_hdus.append(hdu)'
            )
            extension_header_replacement = (
                'hdu.header["PSFKIND"] = entry["kind"]\n'
                '    hdu.header["OPDPATH"] = Path(entry.get("opd_path", "")).name\n'
                '    hdu.header["PSFEXT"] = str(entry.get("selected_extension", ""))\n'
                '    hdu.header["DETPOS"] = str(entry.get("detector_position", ""))\n'
                '    cache_hdus.append(hdu)'
            )
            if extension_header_anchor not in source:
                raise RuntimeError("PSF extension-header contract has changed")
            source = source.replace(
                extension_header_anchor, extension_header_replacement
            )
            prelude = (
                "from sbf090_pipeline_support import (\n"
                "    load_f090w_psf_cache, write_psf_cache_atomic,\n"
                ")\n"
                "_f090_psf_path = out_dir / f\"{stem}_psf_{PSF_SIZE}.fits\"\n"
                "_f090_psf_cache = load_f090w_psf_cache(\n"
                "    _f090_psf_path, f150w_path, out_dir, stem,\n"
                "    expected_filter=SIGNAL_FILTER, expected_size=PSF_SIZE,\n"
                ")\n"
                "if _f090_psf_cache is not None:\n"
                "    globals().update(_f090_psf_cache)\n"
                "    print(f\"[PSF] valid F090W cache reused: {_f090_psf_path}\")\n"
                "else:\n"
            )
            source = prelude + textwrap.indent(source, "    ")

        if cell.get("cell_type") == "code" and "ISO_START_SMA = 50.0" in source:
            config_cell_found = True
            source = source.replace(
                "ISO_START_SMA = 50.0\n",
                "ISO_START_SMA_CANDIDATES = (40.0, 50.0, 60.0, 70.0, 100.0)\n"
                "ISO_BOOTSTRAP_MAXSMA = 200.0\n"
                "ISO_MAX_MEDIAN_CENTER_SHIFT_PX = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_median_center_shift_px']!r}\n"
                "ISO_SEED_MIN_EPS = 0.05\n"
                "ISO_SEED_MAX_CENTER_SHIFT_PX = 100.0\n"
                "ISO_QC_MAX_CENTER_OFFSET_PX = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_center_shift_px']!r}\n"
                "ISO_QC_MAX_STOP2_FRACTION = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_stop2_fraction']!r}\n"
                "ISO_QC_MAX_CONSECUTIVE_STOP2 = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_consecutive_stop2']!r}\n"
                "ISO_QC_MAX_FROZEN_STOP_FRACTION = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_frozen_stop_fraction']!r}\n"
                "ISO_QC_MAX_CONSECUTIVE_FROZEN_STOP = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_consecutive_frozen_stop']!r}\n"
                "ISO_QC_MAX_SINGULAR_STOP_COUNT = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_singular_stop_count']!r}\n"
                "ISO_QC_MAX_CENTER_STEP_PX = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_center_step_px']!r}\n"
                "ISO_QC_MAX_EPS_STEP = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_eps_step']!r}\n"
                "ISO_QC_MAX_SHAPE_STEP = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_shape_step']!r}\n"
                "ISO_QC_MAX_INTENSITY_RISE_FRACTION = "
                f"{F090W_ISOPHOTE_QC_LIMITS['max_intensity_rise_fraction']!r}\n"
                "CENTER_LOCAL_RADIUS_PX = 250.0\n"
                "CENTER_REFINE_MAX_SHIFT_PX = 50.0\n"
                "EXTERNAL_CONTAMINANT_CORE_GUARD_PX = 250.0\n"
                "EXTERNAL_CONTAMINANT_MIN_PEAK_SNR = 100.0\n"
                "F090W_ISOPHOTE_METHOD = "
                f"{F090W_ISOPHOTE_METHOD!r}\n"
                "F090W_MASK_METHOD = "
                f"{F090W_MASK_METHOD!r}\n",
            )

        if cell.get("cell_type") == "code" and source.startswith(
            'print("building Jensen-like primary compact-source mask...")'
        ):
            premask_cell_found = True
            source = (
                "from sbf090_pipeline_support import (\n"
                "    is_large_external_contaminant, write_dataframe_atomic,\n"
                ")\n\n"
                + source
            )
            old_none = """if segm is None:
    premask_src_raw = np.zeros_like(img, dtype=bool)
else:
"""
            new_none = """external_contaminant_mask_raw = np.zeros_like(img, dtype=bool)
external_contaminant_labels = []
large_contaminant_rows = []
if segm is None:
    premask_src_raw = np.zeros_like(img, dtype=bool)
else:
"""
            if old_none not in source:
                raise RuntimeError("premask empty-segmentation contract has changed")
            source = source.replace(old_none, new_none)
            old_labels = """    compact_labels = [
        int(label)
        for label in labels
        if 0 < counts[label] <= PREMASK_MAX_COMPACT_AREA
    ]

    label_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    if compact_labels:
        label_lookup[np.asarray(compact_labels, dtype=int)] = True
    premask_src_raw = label_lookup[segm.data]
"""
            new_labels = """    compact_labels = [
        int(label)
        for label in labels
        if 0 < counts[label] <= PREMASK_MAX_COMPACT_AREA
    ]

    # A fixed area ceiling fails when a bright star grows beyond the nominal
    # compact-source limit.  Inspect every larger component using one rule for
    # all galaxies: it must be bright and must not touch the central 250 px.
    label_slices = ndimage.find_objects(segm.data)
    for label in labels:
        label = int(label)
        area_pixels = int(counts[label])
        if area_pixels <= PREMASK_MAX_COMPACT_AREA:
            continue
        label_slice = label_slices[label - 1]
        if label_slice is None:
            continue
        component = segm.data[label_slice] == label
        y_local, x_local = np.nonzero(component)
        y_pixels = y_local + label_slice[0].start
        x_pixels = x_local + label_slice[1].start
        radius_pixels = np.hypot(
            x_pixels - x_premask_center, y_pixels - y_premask_center
        )
        component_snr = (
            premask_preliminary_residual[y_pixels, x_pixels]
            / np.maximum(
                premask_noise_sigma[y_pixels, x_pixels], ROBUST_SCALE_FLOOR
            )
        )
        min_radius_pixels = float(np.nanmin(radius_pixels))
        peak_snr = float(np.nanmax(component_snr))
        selected = is_large_external_contaminant(
            area_pixels=area_pixels,
            min_radius_pixels=min_radius_pixels,
            peak_snr=peak_snr,
            compact_max_area=PREMASK_MAX_COMPACT_AREA,
            core_guard_radius_pixels=EXTERNAL_CONTAMINANT_CORE_GUARD_PX,
            min_peak_snr=EXTERNAL_CONTAMINANT_MIN_PEAK_SNR,
        )
        large_contaminant_rows.append({
            "galaxy": TARGET_GALAXY,
            "label": label,
            "area_pixels": area_pixels,
            "centroid_x": float(np.mean(x_pixels)),
            "centroid_y": float(np.mean(y_pixels)),
            "centroid_radius_pixels": float(np.hypot(
                np.mean(x_pixels) - x_premask_center,
                np.mean(y_pixels) - y_premask_center,
            )),
            "min_radius_pixels": min_radius_pixels,
            "peak_snr": peak_snr,
            "selected": bool(selected),
            "method": F090W_MASK_METHOD,
        })
        if selected:
            external_contaminant_labels.append(label)
            external_contaminant_mask_raw[label_slice] |= component

    print(
        "[PREMASK] large external contaminants selected: "
        f"{[(label, int(counts[label])) for label in external_contaminant_labels]}"
    )

    label_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    masking_labels = compact_labels + external_contaminant_labels
    if masking_labels:
        label_lookup[np.asarray(masking_labels, dtype=int)] = True
    premask_src_raw = label_lookup[segm.data]

"""
            if old_labels not in source:
                raise RuntimeError("premask label-selection contract has changed")
            source = source.replace(old_labels, new_labels)
            dilation_anchor = """premask_src = ndimage.binary_dilation(
    premask_src_raw,
    iterations=PREMASK_DILATE_ITERATIONS,
) & premask_work_region
"""
            dilation_replacement = dilation_anchor + """external_contaminant_mask = ndimage.binary_dilation(
    external_contaminant_mask_raw,
    iterations=PREMASK_DILATE_ITERATIONS,
) & premask_work_region
"""
            if dilation_anchor not in source:
                raise RuntimeError("premask dilation contract has changed")
            source = source.replace(dilation_anchor, dilation_replacement)
            mask_path_anchor = (
                'premask_noise_table_path = out_dir / f"{stem}_sbf_premask_noise_model.csv"\n'
            )
            mask_path_replacement = mask_path_anchor + (
                'external_contaminant_mask_path = out_dir / '
                'f"{stem}_sbf_external_contaminant_mask.fits"\n'
                'external_contaminant_table_path = out_dir / '
                'f"{stem}_sbf_external_contaminants.csv"\n'
            )
            if mask_path_anchor not in source:
                raise RuntimeError("premask output-path contract has changed")
            source = source.replace(mask_path_anchor, mask_path_replacement)
            write_anchor = (
                "fits.writeto(premask_mask_path, np.asarray(premask_src, dtype=np.uint8), hdr150, overwrite=True)\n"
            )
            write_replacement = write_anchor + (
                "fits.writeto(external_contaminant_mask_path, "
                "np.asarray(external_contaminant_mask, dtype=np.uint8), "
                "hdr150, overwrite=True)\n"
            )
            if write_anchor not in source:
                raise RuntimeError("premask FITS-write contract has changed")
            source = source.replace(write_anchor, write_replacement)
            table_write_anchor = (
                "df_premask_noise_model.to_csv(premask_noise_table_path, index=False)\n"
            )
            table_write_replacement = table_write_anchor + (
                "large_contaminant_columns = [\n"
                "    'galaxy', 'label', 'area_pixels', 'centroid_x', 'centroid_y',\n"
                "    'centroid_radius_pixels', 'min_radius_pixels', 'peak_snr',\n"
                "    'selected', 'method',\n"
                "]\n"
                "write_dataframe_atomic(\n"
                "    pd.DataFrame(large_contaminant_rows, columns=large_contaminant_columns),\n"
                "    external_contaminant_table_path,\n"
                ")\n"
            )
            if table_write_anchor not in source:
                raise RuntimeError("premask table-write contract has changed")
            source = source.replace(table_write_anchor, table_write_replacement)
            print_anchor = 'print(f"[OUT] initial compact mask   -> {premask_mask_path}")\n'
            print_replacement = print_anchor + (
                'print(f"[OUT] external contaminant  -> '
                '{external_contaminant_mask_path}")\n'
                'print(f"[OUT] contaminant decisions -> '
                '{external_contaminant_table_path}")\n'
            )
            if print_anchor not in source:
                raise RuntimeError("premask output-log contract has changed")
            source = source.replace(print_anchor, print_replacement)

        if cell.get("cell_type") == "code" and source.startswith(
            'print("preparing data for isophote fit...")'
        ):
            isophote_prep_cell_found = True
            old_prep = """print("preparing data for isophote fit...")

data_real = np.asarray(img_real_c, dtype=float).copy()
data_real[(~valid_c) | mask_c] = np.nan
ok_real = np.isfinite(data_real)
data_real_ma = np.ma.array(data_real, mask=~ok_real)

data_fill = np.asarray(img_fill_c, dtype=float).copy()
data_fill[mask_c] = np.nan
ok_fill = np.isfinite(data_fill)
data_fill_ma = np.ma.array(data_fill, mask=~ok_fill)
"""
            new_prep = """print("preparing data for isophote fit...")

from sbf090_pipeline_support import (
    F090W_INNER_MASK_GUARD_METHOD,
    F090W_INNER_MASK_GUARD_TARGETS,
)

# NGC 4636 is a targeted exception for the isophote fit only.  Its F090W
# compact-source detector mistakes galaxy structure for a nearly closed mask
# ring at 120--200 px, preventing Ellipse from reaching the science annuli.
# The original mask_c is deliberately left untouched for the SBF measurement.
isophote_mask_c = np.array(mask_c, dtype=bool, copy=True)
f090_inner_mask_guard_enabled = TARGET_GALAXY in F090W_INNER_MASK_GUARD_TARGETS
f090_inner_mask_guard_radius_px = float(
    SBF_LIT_INNER_ARCSEC[0] / np.sqrt(pix_area)
)
f090_inner_mask_guard_cleared_pixels = 0
if f090_inner_mask_guard_enabled:
    yy_iso_mask, xx_iso_mask = np.indices(mask_c.shape, dtype=float)
    inside_inner_sbf_radius = np.hypot(
        xx_iso_mask - x0c, yy_iso_mask - y0c
    ) < f090_inner_mask_guard_radius_px
    f090_inner_mask_guard_cleared_pixels = int(
        np.count_nonzero(isophote_mask_c & inside_inner_sbf_radius)
    )
    isophote_mask_c[inside_inner_sbf_radius] = False
    print(
        "[ISO-MASK] targeted inner guard: "
        f"method={F090W_INNER_MASK_GUARD_METHOD}, "
        f"r<{f090_inner_mask_guard_radius_px:.1f} px, "
        f"cleared={f090_inner_mask_guard_cleared_pixels} pixels; "
        "science mask unchanged"
    )

data_real = np.asarray(img_real_c, dtype=float).copy()
data_real[(~valid_c) | isophote_mask_c] = np.nan
ok_real = np.isfinite(data_real)
data_real_ma = np.ma.array(data_real, mask=~ok_real)

data_fill = np.asarray(img_fill_c, dtype=float).copy()
data_fill[isophote_mask_c] = np.nan
ok_fill = np.isfinite(data_fill)
data_fill_ma = np.ma.array(data_fill, mask=~ok_fill)
"""
            if old_prep not in source:
                raise RuntimeError("isophote-input mask contract has changed")
            source = source.replace(old_prep, new_prep)
            source = source.replace(
                'print(f"[ISO] masked by source mask = {mask_c.sum()}, old invalid in crop = {(~valid_c).sum()}")',
                'print(f"[ISO] masked for isophote fit = {isophote_mask_c.sum()}, original source mask = {mask_c.sum()}, old invalid in crop = {(~valid_c).sum()}")',
            )

        if cell.get("cell_type") == "code" and source.startswith(
            'print("choosing galaxy center...")'
        ):
            center_cell_found = True
            source = _f090w_center_cell()

        if cell.get("cell_type") == "code" and source.startswith(
            'print("fitting isophotes...")'
        ):
            isophote_cell_found = True
            source = _f090w_isophote_cell()

        final_mask_anchor = (
            "premask = np.array((~valid150) | final_catalog_source_mask, dtype=bool)\n"
            "premask_src = np.array(final_catalog_source_mask, dtype=bool, copy=True)"
        )
        if cell.get("cell_type") == "code" and final_mask_anchor in source:
            final_mask_cell_found = True
            source = source.replace(
                final_mask_anchor,
                "premask = np.array(\n"
                "    (~valid150) | final_catalog_source_mask | external_contaminant_mask,\n"
                "    dtype=bool,\n"
                ")\n"
                "premask_src = np.array(\n"
                "    final_catalog_source_mask | external_contaminant_mask,\n"
                "    dtype=bool, copy=True,\n"
                ")",
            )

        if cell.get("cell_type") == "code" and source.startswith(
            'print("loading i2d files...")'
        ):
            auxiliary_marker = "img_f090 = None"
            if auxiliary_marker not in source:
                raise RuntimeError("image-loading contract in sbf-2.ipynb has changed")
            source = source.split(auxiliary_marker, 1)[0] + (
                "img_f090 = None\n"
                "hdr090 = None\n"
                "valid090 = None\n"
                "wcs090 = None\n\n"
                "print(f\"F090W signal shape = {img_f150.shape}, "
                "valid = {valid150.sum()}\")\n"
                "print(\"F150W auxiliary image is present but is not loaded; "
                "colour calibration is deferred\")\n"
            )

        if cell.get("cell_type") == "code" and "A_F150W_SBF" in source:
            extinction_cell_found = True
            source = source.replace("A_F150W", "A_F090W")
            source = source.replace("F150W", "F090W")

        if cell.get("cell_type") == "code":
            old_print_wrapper = (
                "_orig_print = builtins.print\n\n"
                "def print(*args, **kwargs):\n"
                "    _orig_print(f\"[{time.strftime('%H:%M:%S')}]\", "
                "*args, **kwargs)\n"
            )
            if old_print_wrapper in source:
                print_wrapper_found = True
                source = source.replace(
                    old_print_wrapper,
                    "# Console prefixes are supplied by run_sbf_f090w.py.\n",
                )
            source = source.replace(
                'print(f"F150W shape = {img_f150.shape}, valid = {valid150.sum()}")',
                'print(f"F090W signal shape = {img_f150.shape}, valid = {valid150.sum()}")',
            )
            source = source.replace(
                'print(f"F090W shape = {img_f090.shape}, valid = {valid090.sum()}")',
                'print(f"F150W auxiliary shape = {img_f090.shape}, valid = {valid090.sum()}")',
            )

        _set_cell_source(cell, source)
        kept_cells.append(cell)

    if not colour_section_reached:
        raise RuntimeError("Deferred colour section was not found in sbf-2.ipynb")
    if not psf_cell_found:
        raise RuntimeError("PSF cell was not found in sbf-2.ipynb")
    if not extinction_cell_found:
        raise RuntimeError("F150W extinction cell was not found in sbf-2.ipynb")
    if not config_cell_found:
        raise RuntimeError("F090W robustness configuration anchor was not found")
    if not premask_cell_found:
        raise RuntimeError("F090W pre-isophote mask cell was not found")
    if not isophote_prep_cell_found:
        raise RuntimeError("F090W isophote-input mask cell was not found")
    if not center_cell_found:
        raise RuntimeError("F090W centre cell was not found")
    if not isophote_cell_found:
        raise RuntimeError("F090W isophote cell was not found")
    if not final_mask_cell_found:
        raise RuntimeError("F090W final catalogue-mask contract was not found")
    if not print_wrapper_found:
        raise RuntimeError("Legacy notebook print wrapper was not found")

    notebook["cells"] = kept_cells
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    atomic_write_text(output_notebook, text)
    return output_notebook


def _valid_psf_arrays(
    hdul: fits.HDUList, expected_size: int
) -> tuple[list[np.ndarray], list[fits.Header]] | None:
    arrays: list[np.ndarray] = []
    headers: list[fits.Header] = []
    for hdu in hdul[1:]:
        if hdu.data is None:
            continue
        array = np.asarray(hdu.data, dtype=float)
        if array.shape != (expected_size, expected_size):
            return None
        if not np.all(np.isfinite(array)):
            return None
        total = float(array.sum())
        if not np.isfinite(total) or abs(total - 1.0) > 1e-5:
            return None
        arrays.append(array / total)
        headers.append(hdu.header.copy())
    if len(arrays) != F090W_PSF_COUNT:
        return None
    return arrays, headers


def _checksums_valid(hdul: fits.HDUList) -> bool:
    for hdu in hdul:
        if "CHECKSUM" not in hdu.header or "DATASUM" not in hdu.header:
            return False
        if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
            return False
    return True


def _header_time_mjd(*headers: fits.Header) -> float:
    for header in headers:
        for key in ("MJD-AVG", "MJD-BEG", "MJD-OBS"):
            value = header.get(key)
            if value is not None:
                try:
                    return float(Time(float(value), format="mjd", scale="utc").mjd)
                except Exception:
                    pass
        for key in ("DATE-AVG", "DATE-BEG", "DATE-OBS", "DATE"):
            value = header.get(key)
            if not value:
                continue
            text = str(value)
            if key == "DATE-OBS" and "T" not in text and header.get("TIME-OBS"):
                text = f"{text}T{header['TIME-OBS']}"
            try:
                return float(Time(text, scale="utc").mjd)
            except Exception:
                pass
    return np.nan


def load_f090w_psf_cache(
    cache_path: Path,
    science_path: Path,
    out_dir: Path,
    stem: str,
    *,
    expected_filter: str = F090W_FILTER,
    expected_size: int = F090W_PSF_SIZE,
) -> dict[str, Any] | None:
    """Return notebook variables only for a complete compatible PSF cache."""

    cache_path = Path(cache_path)
    science_path = Path(science_path)
    if not cache_path.is_file() or not science_path.is_file():
        return None
    try:
        science_header = fits.getheader(science_path, 0)
        science_sci_header = fits.getheader(science_path, "SCI")
        with fits.open(cache_path, memmap=True, checksum=True) as hdul:
            hdul.verify("exception")
            if not _checksums_valid(hdul):
                return None
            primary = hdul[0].header.copy()
            if str(primary.get("FILTER", "")).strip().upper() != str(
                expected_filter
            ).strip().upper():
                return None
            if int(primary.get("PSFSIZE", -1)) != int(expected_size):
                return None
            if int(primary.get("NPSF", -1)) != F090W_PSF_COUNT:
                return None
            if str(primary.get("APERNAME", "")) != str(
                science_header.get("APERNAME", "")
            ):
                return None
            if str(primary.get("SCIFILE", "")) != science_path.name:
                return None
            if str(primary.get("SCIDET", "")) != str(
                science_header.get("DETECTOR", "")
            ):
                return None
            science_mjd = _header_time_mjd(science_sci_header, science_header)
            cached_mjd = float(primary.get("SCIMJD", np.nan))
            if (
                not np.isfinite(science_mjd)
                or not np.isfinite(cached_mjd)
                or abs(science_mjd - cached_mjd) > 1e-6
            ):
                return None
            if float(primary.get("OPDDT", np.inf)) > 7.0:
                return None
            checked = _valid_psf_arrays(hdul, expected_size)
            if checked is None:
                return None
            arrays, headers = checked
    except Exception:
        return None

    science_pixel_scale = float(
        np.sqrt(float(science_sci_header["PIXAR_SR"]) / 2.350443e-11)
    )
    psf_pixel_scale = float(primary.get("PSF_PXS", np.nan))
    scale_error = (
        abs(psf_pixel_scale / science_pixel_scale - 1.0)
        if np.isfinite(psf_pixel_scale) and science_pixel_scale > 0
        else np.inf
    )
    if scale_error > 0.01:
        return None

    library = []
    for index, (array, header) in enumerate(zip(arrays, headers)):
        library.append({
            "id": str(header.get("PSFID", f"psf_{index}")),
            "kind": str(header.get("PSFKIND", "model")),
            "opd_path": str(header.get("OPDPATH", "")),
            "opd_corr_id": str(primary.get("OPDCORR", "")),
            "opd_delta_days": float(primary.get("OPDDT", np.nan)),
            "selected_extension": str(header.get("PSFEXT", "cached")),
            "filter": str(primary.get("FILTER", expected_filter)),
            "aperture": str(primary.get("APERNAME", "")),
            "detector_position": str(header.get("DETPOS", "")),
            "array": array,
        })

    table = pd.DataFrame([
        {
            "psf_id": entry["id"],
            "kind": entry["kind"],
            "opd_path": entry["opd_path"],
            "opd_corr_id": entry["opd_corr_id"],
            "opd_delta_days": entry["opd_delta_days"],
            "selected_extension": entry["selected_extension"],
            "filter": entry["filter"],
            "aperture": entry["aperture"],
            "detector_position": entry["detector_position"],
            "science_pixel_scale_arcsec": science_pixel_scale,
            "psf_pixel_scale_arcsec": psf_pixel_scale,
            "shape": str(entry["array"].shape),
            "sum": float(entry["array"].sum()),
        }
        for entry in library
    ])
    csv_path = Path(out_dir) / f"{stem}_psf_library.csv"
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    atomic_write_text(csv_path, buffer.getvalue())

    return {
        "science_psf_file": science_path,
        "psf_library_fits_path": cache_path,
        "psf_library_csv_path": csv_path,
        "psf_library": library,
        "psf_library_arrays": [entry["array"] for entry in library],
        "psf_library_table": table,
        "psf": library[0]["array"],
        "psf_method_id": str(primary.get("PSFMETH", "cached_stpsf")),
        "psf_method_limitations": (
            "cached local STPSF F090W model ensemble; nearest WSS OPD and "
            "four detector-position offsets"
        ),
        "psf_detector_set": str(science_header.get("DETECTOR", "")),
        "psf_input_count": len(library),
        "psf_selected_ext": str(headers[0].get("PSFEXT", "cached")),
        "science_pixel_scale_arcsec": science_pixel_scale,
        "psf_pixel_scale_arcsec": psf_pixel_scale,
        "psf_scale_rel_error": scale_error,
        "psf_native_scale_rel_error": scale_error,
        "opd_corr_id": str(primary.get("OPDCORR", "")),
        "opd_delta_days": float(primary.get("OPDDT", np.nan)),
        "opd_signed_delta_days": float(primary.get("OPDSIGN", np.nan)),
    }


def write_psf_cache_atomic(
    cache_path: Path,
    hdul: fits.HDUList,
    *,
    expected_filter: str,
    expected_size: int = F090W_PSF_SIZE,
) -> Path:
    """Validate and atomically publish one F090W PSF ensemble."""

    cache_path = Path(cache_path)
    checked = _valid_psf_arrays(hdul, expected_size)
    if checked is None:
        raise RuntimeError(
            f"PSF cache must contain exactly {F090W_PSF_COUNT} normalized "
            f"{expected_size}x{expected_size} image extensions"
        )
    hdul[0].header["FILTER"] = str(expected_filter).strip().upper()
    hdul[0].header["PSFSIZE"] = int(expected_size)
    hdul[0].header["NPSF"] = F090W_PSF_COUNT
    hdul[0].header["CACHEOK"] = True
    hdul[0].header.add_history(
        "Published atomically after shape, finiteness and unit-sum checks."
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.stem}.", suffix=".fits", dir=cache_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        hdul.writeto(temporary, overwrite=True, checksum=True)
        with fits.open(temporary, memmap=True, checksum=True) as check_hdul:
            check_hdul.verify("exception")
            if (
                not _checksums_valid(check_hdul)
                or _valid_psf_arrays(check_hdul, expected_size) is None
            ):
                raise RuntimeError("written PSF cache failed validation")
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return cache_path
