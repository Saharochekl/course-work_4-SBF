#!/usr/bin/env python3
"""
Build a fast Jensen-like SBF error budget from completed sbf-2 batch outputs.

No isophotes, galaxy model, PSF image, or SBF-region spectra are rerun here.
The script only uses saved batch CSV files and logs:

    sigma^2 = sigma_Pk^2 + sigma_bkg^2 + sigma_PSF^2 + sigma_Pr^2 + sigma_ext^2

Fast components:
- sigma_Pk: Jensen-style annulus combination from saved sbf-2 formal errors.
- sigma_bkg: propagated from saved [BKG-CHECK] residual sky offset and saved Imean.
- sigma_PSF: from saved systematics branch CSVs, or DET_DIST-vs-DET_SAMP fast rerun.
- sigma_Pr: proxy from Pr/P0 with an assumed fractional Pr uncertainty.
- sigma_ext: direct CLI term, CSV, or IRSA/SFD E(B-V) uncertainty propagated to F150W.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path


MAG_FROM_FRAC = 2.5 / math.log(10.0)
SCIENCE_REGIONS = ("circular_inner_lit", "circular_outer_lit")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fast Jensen-like mbar error budgets from sbf-2 batch results."
    )
    parser.add_argument(
        "--batch-results",
        type=Path,
        default=Path("sbf2_batch_outputs/sbf2_batch_results.csv"),
        help="Batch summary CSV written by run_sbf2_batch.py.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("sbf2_batch_outputs"),
        help="Directory with per-galaxy worker logs.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("sbf2_batch_outputs/sbf2_jensen_like_error_budget.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("sbf2_batch_outputs/sbf2_jensen_like_error_budget.md"),
        help="Output markdown summary path.",
    )
    parser.add_argument(
        "--pr-fractional-error",
        type=float,
        default=0.25,
        help="Fractional uncertainty assigned to Pr for the fast proxy.",
    )

    ext = parser.add_argument_group("extinction uncertainty")
    ext.add_argument(
        "--sigma-ext-mag",
        type=float,
        default=None,
        help="Direct extinction/photometry uncertainty to include, in mag.",
    )
    ext.add_argument(
        "--sigma-ebv",
        type=float,
        default=None,
        help="Common sigma_E(B-V). Used with --af150w-per-ebv if --sigma-ext-mag is omitted.",
    )
    ext.add_argument(
        "--af150w-per-ebv",
        type=float,
        default=None,
        help="A_F150W / E(B-V) coefficient for analytic sigma_ext.",
    )
    ext.add_argument(
        "--ebv-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with galaxy-specific extinction uncertainty. Supported columns: "
            "galaxy plus sigma_ext_mag, or sigma_ebv/ebv_sigma/ebv_err with --af150w-per-ebv."
        ),
    )
    ext.add_argument(
        "--disable-auto-ext",
        action="store_true",
        help="Do not query/estimate extinction uncertainty automatically.",
    )
    ext.add_argument(
        "--ext-cache-csv",
        type=Path,
        default=Path("sbf2_batch_outputs/sbf2_extinction_sigma_cache.csv"),
        help="Cache for automatically estimated extinction uncertainties.",
    )
    ext.add_argument(
        "--ext-lambda-micron",
        type=float,
        default=1.501,
        help="Effective wavelength used for automatic F150W extinction coefficient.",
    )
    ext.add_argument(
        "--ext-rv",
        type=float,
        default=3.1,
        help="R_V used for automatic near-IR extinction coefficient.",
    )
    ext.add_argument(
        "--ext-fractional-ebv-floor",
        type=float,
        default=0.10,
        help="Fallback sigma_E(B-V)=this*E(B-V) when IRSA does not provide an uncertainty.",
    )
    ext.add_argument(
        "--ext-query-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for automatic IRSA extinction queries.",
    )

    bkg = parser.add_argument_group("background uncertainty")
    bkg.add_argument(
        "--sigma-bkg-mag",
        type=float,
        default=None,
        help=(
            "Direct common background uncertainty in mag. If omitted, use fast auto estimate "
            "from [BKG-CHECK] and saved Imean."
        ),
    )
    bkg.add_argument(
        "--bkg-stat",
        choices=("median", "mean", "max_abs_mean_median", "sem"),
        default="median",
        help="Which [BKG-CHECK] statistic to propagate when --sigma-bkg-mag is omitted.",
    )
    bkg.add_argument(
        "--disable-auto-bkg",
        action="store_true",
        help="Do not estimate sigma_bkg automatically from logs.",
    )

    psf = parser.add_argument_group("PSF uncertainty")
    psf.add_argument(
        "--sigma-psf-mag",
        type=float,
        default=None,
        help=(
            "Direct common PSF uncertainty in mag. If omitted, use saved systematics branch "
            "CSV when available."
        ),
    )
    psf.add_argument(
        "--disable-psf-branch-auto",
        action="store_true",
        help="Do not estimate sigma_PSF from saved systematics branch CSVs.",
    )
    psf.add_argument(
        "--disable-psf-fast-rerun",
        action="store_true",
        help="Do not run DET_DIST-vs-DET_SAMP fast PSF systematic when branch CSV is missing.",
    )
    psf.add_argument(
        "--psf-size",
        type=int,
        default=129,
        help="Detector-sampled PSF size for fast PSF systematic.",
    )
    psf.add_argument(
        "--psf-nlambda",
        type=int,
        default=7,
        help="Number of wavelengths for STPSF fast PSF systematic.",
    )
    psf.add_argument(
        "--psf-e-realizations",
        type=int,
        default=64,
        help="Monte Carlo realizations for E(k) in fast PSF systematic.",
    )
    psf.add_argument(
        "--psf-kbins-n",
        type=int,
        default=80,
        help="Number of k-bin edges for fast PSF systematic.",
    )
    psf.add_argument(
        "--fft-rng-seed",
        type=int,
        default=1489,
        help="Random seed for fast PSF E(k) simulations.",
    )
    psf.add_argument(
        "--psf-crop-pad",
        type=int,
        default=512,
        help="Padding around finite annulus residual before fast PSF FFT.",
    )
    psf.add_argument(
        "--stpsf-path",
        type=Path,
        default=Path.home() / "data" / "stpsf-data",
        help="Local STPSF data directory.",
    )
    psf.add_argument(
        "--reuse-fast-psf-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-galaxy fast PSF systematic CSV cache when compatible.",
    )
    return parser.parse_args()


def as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def same_float(a: object, b: object, tol: float = 1.0e-9) -> bool:
    af = as_float(a)
    bf = as_float(b)
    return math.isfinite(af) and math.isfinite(bf) and abs(af - bf) <= tol


def inv_variance_weights(sigmas: list[float]) -> list[float]:
    return [1.0 / (sigma * sigma) if finite_positive(sigma) else math.nan for sigma in sigmas]


def weighted_average(values: list[float], weights: list[float]) -> float:
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights)
        if math.isfinite(value) and math.isfinite(weight) and weight > 0.0
    ]
    if not pairs:
        return math.nan
    return sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)


def quadrature(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value) and value >= 0.0]
    if not vals:
        return math.nan
    return math.sqrt(sum(value * value for value in vals))


def sigma_pr_from_ratio(pr_over_p0: float, fractional_error: float) -> float:
    if not (math.isfinite(pr_over_p0) and 0.0 <= pr_over_p0 < 1.0):
        return math.nan
    if not finite_positive(fractional_error):
        return 0.0
    return MAG_FROM_FRAC * fractional_error * pr_over_p0 / (1.0 - pr_over_p0)


def galaxy_log_path(batch_root: Path, galaxy: str) -> Path:
    return batch_root / f"{galaxy.replace(' ', '_')}.log"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_log_metadata(log_path: Path) -> dict[str, object]:
    if not log_path.exists():
        return {
            "log_exists": False,
            "iso_dataset": "",
            "psf_extension": "",
            "bkg_check_med": math.nan,
            "bkg_check_mean": math.nan,
            "bkg_check_std": math.nan,
            "bkg_check_n": math.nan,
            "warnings": "log missing",
        }

    text = log_path.read_text(errors="replace")
    warnings = []
    if "File may have been truncated" in text:
        warnings.append("truncated_file_warning")
    if "No meaningful fit was possible" in text:
        warnings.append("isophote_real_only_failed")
    if "fit may be unsuccessful" in text:
        warnings.append("sersic_fit_warning")

    iso_dataset = ""
    m = re.search(r"\[ISO\] fit dataset used = ([^,\n]+)", text)
    if m:
        iso_dataset = m.group(1).strip()

    psf_extension = ""
    m = re.search(r"\[PSF\] selected extension = ([^\n]+)", text)
    if m:
        psf_extension = m.group(1).strip()

    bkg = {"med": math.nan, "mean": math.nan, "std": math.nan, "n": math.nan}
    m = re.search(
        r"\[BKG-CHECK\] corners: med=([+\-0-9.eE]+), mean=([+\-0-9.eE]+), std=([+\-0-9.eE]+), N=([0-9]+)",
        text,
    )
    if m:
        bkg = {
            "med": as_float(m.group(1)),
            "mean": as_float(m.group(2)),
            "std": as_float(m.group(3)),
            "n": as_float(m.group(4)),
        }

    return {
        "log_exists": True,
        "iso_dataset": iso_dataset,
        "psf_extension": psf_extension,
        "bkg_check_med": bkg["med"],
        "bkg_check_mean": bkg["mean"],
        "bkg_check_std": bkg["std"],
        "bkg_check_n": bkg["n"],
        "warnings": ";".join(warnings),
    }


def bkg_delta_from_meta(meta: dict[str, object], method: str) -> tuple[float, str]:
    med = as_float(meta.get("bkg_check_med"))
    mean = as_float(meta.get("bkg_check_mean"))
    std = as_float(meta.get("bkg_check_std"))
    n = as_float(meta.get("bkg_check_n"))

    if method == "median":
        return abs(med), "abs(BKG-CHECK median)"
    if method == "mean":
        return abs(mean), "abs(BKG-CHECK mean)"
    if method == "max_abs_mean_median":
        vals = [abs(v) for v in (med, mean) if math.isfinite(v)]
        return (max(vals) if vals else math.nan), "max(abs(mean), abs(median))"
    if method == "sem":
        if finite_positive(std) and finite_positive(n):
            return std / math.sqrt(n), "BKG-CHECK std/sqrt(N)"
        return math.nan, "BKG-CHECK std/sqrt(N)"
    return math.nan, "unknown background statistic"


def science_rows_by_region(row: dict[str, str], kmin: float, kmax: float) -> dict[str, dict[str, str]]:
    path_text = row.get("df_sbf_csv", "")
    path = Path(path_text) if path_text else Path()
    out: dict[str, dict[str, str]] = {}
    for rec in read_csv_rows(path):
        if rec.get("role") != "science" or rec.get("status") != "ok":
            continue
        if rec.get("region") not in SCIENCE_REGIONS:
            continue
        if same_float(rec.get("kmin"), kmin) and same_float(rec.get("kmax"), kmax):
            out[rec["region"]] = rec
    return out


def bkg_sigma_from_science_rows(
    science_rows: dict[str, dict[str, str]],
    weights: list[float],
    delta_bkg: float,
) -> tuple[float, float, float]:
    if not finite_nonnegative(delta_bkg):
        return math.nan, math.nan, math.nan

    sigmas = []
    for region in SCIENCE_REGIONS:
        rec = science_rows.get(region, {})
        imean = as_float(rec.get("Imean"))
        if finite_positive(imean):
            sigmas.append(MAG_FROM_FRAC * delta_bkg / imean)
        else:
            sigmas.append(math.nan)
    return sigmas[0], sigmas[1], weighted_average(sigmas, weights)


def load_ebv_uncertainty(args: argparse.Namespace) -> dict[str, float]:
    if args.ebv_csv is None:
        return {}
    rows = read_csv_rows(args.ebv_csv)
    out: dict[str, float] = {}
    for row in rows:
        galaxy = (row.get("galaxy") or row.get("name") or "").strip()
        if not galaxy:
            continue
        direct = as_float(row.get("sigma_ext_mag"))
        if math.isfinite(direct) and direct >= 0.0:
            out[galaxy] = direct
            continue

        sigma_ebv = math.nan
        for key in ("sigma_ebv", "ebv_sigma", "ebv_err", "E_BV_err", "sigma_E_BV"):
            sigma_ebv = as_float(row.get(key))
            if math.isfinite(sigma_ebv):
                break
        if math.isfinite(sigma_ebv) and args.af150w_per_ebv is not None:
            out[galaxy] = abs(float(args.af150w_per_ebv) * sigma_ebv)
    return out


def load_ext_cache(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("galaxy", ""): row for row in read_csv_rows(path) if row.get("galaxy")}


def save_ext_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "galaxy",
        "sigma_ext_mag",
        "status",
        "ebv",
        "sigma_ebv",
        "af150w_per_ebv",
        "ra_deg",
        "dec_deg",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for galaxy in sorted(cache):
            row = {key: cache[galaxy].get(key, "") for key in fieldnames}
            writer.writerow(row)


def f150w_extinction_coeff_ccm(lambda_micron: float, rv: float) -> float:
    """Return A_lambda/E(B-V) from the near-IR CCM law."""
    if not (finite_positive(lambda_micron) and finite_positive(rv)):
        return math.nan
    x = 1.0 / lambda_micron
    x_pow = x ** 1.61
    a = 0.574 * x_pow
    b = -0.527 * x_pow
    return float(rv * a + b)


def header_target_coord(row: dict[str, str]) -> tuple[float, float, str]:
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception:
        return math.nan, math.nan, "astropy_unavailable"

    path_text = row.get("f150w_path", "")
    if not path_text:
        return math.nan, math.nan, "no_f150w_path"
    path = Path(path_text)
    if not path.exists():
        return math.nan, math.nan, "f150w_missing"

    try:
        hdr0 = fits.getheader(path, 0)
        hdr_sci = fits.getheader(path, "SCI")
    except Exception as err:
        return math.nan, math.nan, f"header_read_failed:{err}"

    for hdr in (hdr0, hdr_sci):
        for ra_key, dec_key in (
            ("TARG_RA", "TARG_DEC"),
            ("RA_TARG", "DEC_TARG"),
            ("CRVAL1", "CRVAL2"),
        ):
            ra = as_float(hdr.get(ra_key))
            dec = as_float(hdr.get(dec_key))
            if math.isfinite(ra) and math.isfinite(dec):
                return ra, dec, f"{ra_key}/{dec_key}"

    try:
        wcs = WCS(hdr_sci)
        nx = as_float(hdr_sci.get("NAXIS1"))
        ny = as_float(hdr_sci.get("NAXIS2"))
        if finite_positive(nx) and finite_positive(ny):
            ra, dec = wcs.wcs_pix2world([[0.5 * nx, 0.5 * ny]], 0)[0]
            return float(ra), float(dec), "SCI_WCS_center"
    except Exception as err:
        return math.nan, math.nan, f"wcs_failed:{err}"

    return math.nan, math.nan, "no_coordinate_keys"


def numeric_table_values(table) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for col in getattr(table, "colnames", []):
        values = table[col]
        for idx, value in enumerate(values):
            try:
                if hasattr(value, "value"):
                    value = value.value
                value_f = float(value)
            except Exception:
                continue
            if math.isfinite(value_f):
                out.append((str(col).lower(), value_f))
    return out


def pick_irsa_ebv_stats(table) -> tuple[float, float, str]:
    values = numeric_table_values(table)
    if not values:
        return math.nan, math.nan, "no_numeric_irsa_values"

    def score(name: str, want_std: bool) -> int:
        s = 0
        if "sandf" in name or "schlafly" in name:
            s += 8
        if "sfd" in name:
            s += 5
        if "ebv" in name or "e(b-v)" in name or "reddening" in name or "ext" in name:
            s += 3
        if want_std and ("std" in name or "sigma" in name or "err" in name):
            s += 4
        if (not want_std) and ("mean" in name or "value" in name):
            s += 4
        return s

    mean_candidates = [(score(name, False), name, value) for name, value in values if value >= 0.0]
    std_candidates = [(score(name, True), name, value) for name, value in values if value >= 0.0]
    mean_candidates = [item for item in mean_candidates if item[0] > 0]
    std_candidates = [item for item in std_candidates if item[0] > 0 and ("std" in item[1] or "sigma" in item[1] or "err" in item[1])]

    ebv = math.nan
    sigma_ebv = math.nan
    notes = []
    if mean_candidates:
        mean_candidates.sort(reverse=True)
        _, name, ebv = mean_candidates[0]
        notes.append(f"ebv_col={name}")
    if std_candidates:
        std_candidates.sort(reverse=True)
        _, name, sigma_ebv = std_candidates[0]
        notes.append(f"sigma_col={name}")
    return ebv, sigma_ebv, ";".join(notes)


def query_irsa_extinction_sigma(
    row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[float, dict[str, str]]:
    galaxy = row["galaxy"]
    coeff = f150w_extinction_coeff_ccm(args.ext_lambda_micron, args.ext_rv)
    if not finite_positive(coeff):
        return math.nan, {
            "galaxy": galaxy,
            "sigma_ext_mag": "",
            "status": "not_available_bad_extinction_coefficient",
            "note": "bad CCM coefficient",
        }

    ra, dec, coord_note = header_target_coord(row)
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return math.nan, {
            "galaxy": galaxy,
            "sigma_ext_mag": "",
            "status": "not_available_no_coordinates",
            "ra_deg": "",
            "dec_deg": "",
            "af150w_per_ebv": f"{coeff:.12g}",
            "note": coord_note,
        }

    try:
        log(f"[EXT] {galaxy}: querying IRSA Dust at RA={ra:.6f}, Dec={dec:.6f}")
        try:
            from astroquery.ipac.irsa.irsa_dust import IrsaDust
        except Exception:
            from astroquery.irsa_dust import IrsaDust
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        IrsaDust.TIMEOUT = float(args.ext_query_timeout)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        table = IrsaDust.get_query_table(coord, section="ebv")
        ebv, sigma_ebv, note = pick_irsa_ebv_stats(table)
    except Exception as err:
        log(f"[EXT] {galaxy}: IRSA query failed: {err}")
        return math.nan, {
            "galaxy": galaxy,
            "sigma_ext_mag": "",
            "status": "not_available_irsa_query_failed",
            "ra_deg": f"{ra:.12g}",
            "dec_deg": f"{dec:.12g}",
            "af150w_per_ebv": f"{coeff:.12g}",
            "note": f"{coord_note};{err}",
        }

    status = "auto_irsa_sigma_ebv_included"
    if not math.isfinite(sigma_ebv):
        if math.isfinite(ebv) and args.ext_fractional_ebv_floor >= 0.0:
            sigma_ebv = abs(float(args.ext_fractional_ebv_floor) * ebv)
            status = "auto_irsa_ebv_fractional_floor_included"
            note = f"{note};sigma_ebv={args.ext_fractional_ebv_floor:g}*ebv"
        else:
            return math.nan, {
                "galaxy": galaxy,
                "sigma_ext_mag": "",
                "status": "not_available_no_irsa_sigma_ebv",
                "ebv": f"{ebv:.12g}" if math.isfinite(ebv) else "",
                "sigma_ebv": "",
                "af150w_per_ebv": f"{coeff:.12g}",
                "ra_deg": f"{ra:.12g}",
                "dec_deg": f"{dec:.12g}",
                "note": f"{coord_note};{note}",
            }

    sigma_ext = abs(coeff * sigma_ebv)
    log(
        f"[EXT] {galaxy}: sigma_ext={sigma_ext:.5f} mag "
        f"(ebv={ebv:.5g}, sigma_ebv={sigma_ebv:.5g}, coeff={coeff:.5g})"
    )
    cache_row = {
        "galaxy": galaxy,
        "sigma_ext_mag": f"{sigma_ext:.12g}",
        "status": status,
        "ebv": f"{ebv:.12g}" if math.isfinite(ebv) else "",
        "sigma_ebv": f"{sigma_ebv:.12g}",
        "af150w_per_ebv": f"{coeff:.12g}",
        "ra_deg": f"{ra:.12g}",
        "dec_deg": f"{dec:.12g}",
        "note": f"{coord_note};{note}",
    }
    return sigma_ext, cache_row


def extinction_sigma_for_galaxy(
    row: dict[str, str],
    galaxy: str,
    args: argparse.Namespace,
    ebv_uncertainties: dict[str, float],
    ext_cache: dict[str, dict[str, str]],
) -> tuple[float, str, str | None]:
    if args.sigma_ext_mag is not None:
        return max(args.sigma_ext_mag, 0.0), "supplied_cli_included", None

    if galaxy in ebv_uncertainties:
        return max(ebv_uncertainties[galaxy], 0.0), "auto_from_ebv_csv_included", None

    cached = ext_cache.get(galaxy)
    if cached:
        sigma_ext_cached = as_float(cached.get("sigma_ext_mag"))
        if math.isfinite(sigma_ext_cached) and sigma_ext_cached >= 0.0:
            return sigma_ext_cached, cached.get("status", "auto_ext_cache_included"), None

    if args.sigma_ebv is not None and args.af150w_per_ebv is not None:
        sigma_ext = abs(float(args.sigma_ebv) * float(args.af150w_per_ebv))
        return sigma_ext, "analytic_common_sigma_ebv_included", None

    if not args.disable_auto_ext:
        sigma_ext, cache_row = query_irsa_extinction_sigma(row, args)
        if cache_row:
            ext_cache[galaxy] = cache_row
        if math.isfinite(sigma_ext) and sigma_ext >= 0.0:
            return sigma_ext, cache_row.get("status", "auto_irsa_included"), None
        return 0.0, cache_row.get("status", "not_available_auto_ext_failed"), "sigma_ext"

    return 0.0, "not_measured_not_included", "sigma_ext"


def branch_recommended_path(row: dict[str, str]) -> Path:
    out_dir = Path(row.get("out_dir", ""))
    stem = row.get("stem", "")
    if not stem:
        return Path()
    return out_dir / f"{stem}_systematics_branch_recommended.csv"


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def estimate_psf_sigma_from_branch(row: dict[str, str], kmin: float, kmax: float) -> dict[str, object]:
    path = branch_recommended_path(row)
    rows = [
        rec
        for rec in read_csv_rows(path)
        if same_float(rec.get("kmin"), kmin) and same_float(rec.get("kmax"), kmax)
    ]
    if not rows:
        return {
            "sigma": math.nan,
            "status": "not_available_no_branch_csv",
            "branch_file": str(path),
            "branch_ref": "",
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }

    ref = None
    for rec in rows:
        if "detdist_ref" in rec.get("branch", ""):
            ref = rec
            break
    if ref is None:
        for rec in rows:
            if truthy(rec.get("uses_two_annuli")):
                ref = rec
                break
    if ref is None:
        ref = rows[0]

    ref_mbar = as_float(ref.get("mbar_weighted"))
    ref_resid = ref.get("resid_source", "")
    ref_model = ref.get("model_source", "")
    ref_psf = ref.get("psf_ext", "")
    ref_opd = ref.get("opd_path", "")

    diffs: list[tuple[str, float]] = []
    for rec in rows:
        same_science_inputs = (
            rec.get("resid_source", "") == ref_resid
            and rec.get("model_source", "") == ref_model
        )
        changed_psf = rec.get("psf_ext", "") != ref_psf or rec.get("opd_path", "") != ref_opd
        if not same_science_inputs or not changed_psf:
            continue
        mbar = as_float(rec.get("mbar_weighted"))
        if math.isfinite(ref_mbar) and math.isfinite(mbar):
            diffs.append((rec.get("branch", ""), abs(mbar - ref_mbar)))

    if not diffs:
        return {
            "sigma": math.nan,
            "status": "not_available_no_psf_alt_branch",
            "branch_file": str(path),
            "branch_ref": ref.get("branch", ""),
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }

    sigma = max(diff for _, diff in diffs)
    return {
        "sigma": sigma,
        "status": "auto_from_saved_psf_branch_included",
        "branch_file": str(path),
        "branch_ref": ref.get("branch", ""),
        "branch_alt_count": len(diffs),
        "branch_alt_diffs": ";".join(f"{name}:{diff:.6g}" for name, diff in diffs),
    }


def fast_psf_cache_path(row: dict[str, str]) -> Path:
    out_dir = Path(row.get("out_dir", ""))
    stem = row.get("stem", "")
    if not stem:
        return Path()
    return out_dir / f"{stem}_jensen_fast_psf_systematic.csv"


def read_compatible_fast_psf_cache(path: Path, kmin: float, kmax: float, args: argparse.Namespace) -> dict[str, object] | None:
    if not (args.reuse_fast_psf_cache and path.exists()):
        return None
    rows = read_csv_rows(path)
    for row in rows:
        if not (same_float(row.get("kmin"), kmin) and same_float(row.get("kmax"), kmax)):
            continue
        if int(as_float(row.get("psf_e_realizations"), -1)) != int(args.psf_e_realizations):
            continue
        if int(as_float(row.get("psf_crop_pad"), -1)) != int(args.psf_crop_pad):
            continue
        sigma = as_float(row.get("sigma_PSF_mag"))
        if math.isfinite(sigma) and sigma >= 0.0:
            return {
                "sigma": sigma,
                "status": "auto_from_fast_psf_cache_included",
                "branch_file": str(path),
                "branch_ref": "fast_detdist",
                "branch_alt_count": 1,
                "branch_alt_diffs": f"fast_detsamp:{sigma:.6g}",
            }
    return None


def write_fast_psf_cache(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def setup_local_stpsf(args: argparse.Namespace) -> tuple[object, Path, str | None]:
    try:
        import os
        import stpsf
        from astropy.io import fits
    except Exception as err:
        return None, Path(), f"imports_failed:{err}"

    if not args.stpsf_path.exists():
        return None, Path(), f"stpsf_path_missing:{args.stpsf_path}"
    os.environ["STPSF_PATH"] = str(args.stpsf_path)

    local_wss_dir = args.stpsf_path / "MAST_JWST_WSS_OPDs"
    local_wss_files = sorted(local_wss_dir.glob("*.fits"))
    if not local_wss_files:
        return None, Path(), f"no_local_wss_opd:{local_wss_dir}"
    local_wss_opd = max(local_wss_files, key=lambda p: p.stat().st_mtime)
    return (stpsf, fits), local_wss_opd, None


def psf_array_from_hdu(hdu) -> object:
    import numpy as np

    arr = np.array(hdu.data, dtype=float)
    if arr.ndim == 3:
        arr = arr.sum(axis=0)
    arr = np.nan_to_num(arr, nan=0.0)
    arr_sum = float(arr.sum())
    if (not np.isfinite(arr_sum)) or arr_sum <= 0.0:
        raise RuntimeError(f"invalid PSF sum={arr_sum}")
    return arr / arr_sum


def build_detdist_detsamp_psfs(row: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object], str | None]:
    libs, local_wss_opd, err = setup_local_stpsf(args)
    if err:
        return {}, {}, err
    stpsf, fits = libs

    psf_file = Path(row.get("f150w_path", ""))
    if not psf_file.exists():
        return {}, {}, f"f150w_missing:{psf_file}"

    try:
        log(f"[PSF] {row['galaxy']}: building DET_DIST/DET_SAMP with STPSF")
        science_hdr = fits.getheader(str(psf_file), 0)
        sim = stpsf.instrument(science_hdr["INSTRUME"])
        if (sim.name == "NIRCam") and (science_hdr["PUPIL"][0] == "F") and (science_hdr["PUPIL"][-1] in ["N", "M"]):
            sim.filter = science_hdr["PUPIL"]
        else:
            sim.filter = science_hdr["FILTER"]
        sim.set_position_from_aperture_name(science_hdr["APERNAME"])
        if (
            (sim.name == "NIRCam")
            and (science_hdr.get("PUPIL", "CLEAR") != "CLEAR")
            and (not science_hdr["PUPIL"].startswith("F"))
            and (not science_hdr["PUPIL"].startswith("MASK"))
        ):
            sim.pupil_mask = science_hdr["PUPIL"]
        sim.load_wss_opd(str(local_wss_opd), verbose=False, plot=False)
        sim.options["output_mode"] = "both"
        psf_hdul = sim.calc_psf(
            nlambda=int(args.psf_nlambda),
            fov_pixels=int(args.psf_size),
            fft_oversample=4,
            detector_oversample=1,
            add_distortion=True,
        )
        psfs = {
            "DET_DIST": psf_array_from_hdu(psf_hdul["DET_DIST"]),
            "DET_SAMP": psf_array_from_hdu(psf_hdul["DET_SAMP"]),
        }
        log(f"[PSF] {row['galaxy']}: STPSF PSFs ready")
    except Exception as err:
        return {}, {}, f"psf_build_failed:{err}"

    meta = {
        "opd_path": str(local_wss_opd),
        "psf_size": int(args.psf_size),
        "psf_nlambda": int(args.psf_nlambda),
    }
    return psfs, meta, None


def radial_median_power(data, kbins_n: int) -> tuple[object, object, int]:
    import numpy as np
    from scipy.fft import fft2, fftfreq, fftshift

    data = np.asarray(data, dtype=float)
    window = np.isfinite(data)
    n_use = int(np.count_nonzero(window))
    if n_use <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0

    data_fft = np.zeros_like(data, dtype=float)
    data_fft[window] = data[window]
    data_fft[window] -= float(np.nanmean(data_fft[window]))
    F = fft2(data_fft)
    P2d = (np.abs(F) ** 2) / float(n_use)

    ny, nx = data.shape
    ky = fftshift(fftfreq(ny))
    kx = fftshift(fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    P2d_s = fftshift(P2d)
    kbins = np.linspace(0.0, float(kr.max()), int(kbins_n))

    vals_out = np.full(len(kbins) - 1, np.nan, dtype=float)
    k_out = np.full_like(vals_out, np.nan)
    for i in range(len(vals_out)):
        sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
        vals = P2d_s[sel]
        if vals.size >= 10:
            vals_out[i] = float(np.nanmedian(vals))
            k_out[i] = 0.5 * (kbins[i] + kbins[i + 1])

    good = np.isfinite(vals_out) & np.isfinite(k_out) & (k_out > 0.0)
    return k_out[good], vals_out[good], n_use


def radial_median_ek(data_shape, window, psf, args: argparse.Namespace) -> tuple[object, object]:
    import numpy as np
    from scipy.fft import fft2, fftfreq, fftshift

    ny, nx = data_shape
    window = np.asarray(window, dtype=bool)
    n_use = int(np.count_nonzero(window))
    if n_use <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    big_psf = np.zeros((ny, nx), dtype=float)
    py, px = psf.shape
    y0_psf = ny // 2 - py // 2
    x0_psf = nx // 2 - px // 2
    big_psf[y0_psf:y0_psf + py, x0_psf:x0_psf + px] = psf
    F_psf = fft2(big_psf)

    ky = fftshift(fftfreq(ny))
    kx = fftshift(fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    kbins = np.linspace(0.0, float(kr.max()), int(args.psf_kbins_n))
    rng = np.random.default_rng(int(args.fft_rng_seed))

    stack = []
    n_real = int(args.psf_e_realizations)
    progress_step = max(1, n_real // 8)
    for i_real in range(n_real):
        if i_real == 0 or (i_real + 1) % progress_step == 0 or (i_real + 1) == n_real:
            log(f"[PSF-FFT] E(k) realization {i_real + 1}/{n_real} for crop {ny}x{nx}")
        noise = rng.normal(loc=0.0, scale=1.0, size=(ny, nx))
        sim = np.real(np.fft.ifft2(fft2(noise) * F_psf))
        sim_masked = np.zeros_like(sim, dtype=float)
        sim_masked[window] = sim[window]
        sim_masked[window] -= float(np.nanmean(sim_masked[window]))
        E2d = (np.abs(fft2(sim_masked)) ** 2) / float(n_use)
        E2d_s = fftshift(E2d)

        vals_out = np.full(len(kbins) - 1, np.nan, dtype=float)
        for i in range(len(vals_out)):
            sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
            vals = E2d_s[sel]
            if vals.size >= 10:
                vals_out[i] = float(np.nanmedian(vals))
        stack.append(vals_out)

    stack_arr = np.asarray(stack, dtype=float)
    e_vals = np.nanmedian(stack_arr, axis=0)
    e_k = 0.5 * (kbins[:-1] + kbins[1:])
    good = np.isfinite(e_vals) & np.isfinite(e_k) & (e_k > 0.0)
    return e_k[good], e_vals[good]


def fit_p0_from_pk_ek(kp, pk, ke, ek, kmin: float, kmax: float) -> tuple[float, float]:
    import numpy as np

    kp = np.asarray(kp, dtype=float)
    pk = np.asarray(pk, dtype=float)
    ke = np.asarray(ke, dtype=float)
    ek = np.asarray(ek, dtype=float)
    sel = (kp >= kmin) & (kp <= kmax)
    if int(np.count_nonzero(sel)) < 10:
        return math.nan, math.nan
    e_int = np.interp(kp[sel], ke, ek, left=np.nan, right=np.nan)
    x = e_int
    y = pk[sel]
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[good]
    y = y[good]
    if x.size < 10:
        return math.nan, math.nan
    A = np.vstack([x, np.ones_like(x)]).T
    p0, p1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(p0), float(p1)


def mbar_from_p0(p0: float, pr: float, imean: float, pix_area: float) -> float:
    if not (finite_positive(p0) and math.isfinite(pr) and finite_positive(imean) and finite_positive(pix_area)):
        return math.nan
    p_fluc = p0 - pr
    if not finite_positive(p_fluc):
        return math.nan
    mjy_sr_to_jy_per_arcsec2 = 2.350443e-5
    ab_zeropoint_jy = 3631.0
    zp = -2.5 * math.log10((mjy_sr_to_jy_per_arcsec2 * pix_area) / ab_zeropoint_jy)
    return float(-2.5 * math.log10(p_fluc / imean) + zp)


def crop_to_finite_annulus(resid, pad: int):
    import numpy as np

    finite = np.isfinite(resid)
    yy, xx = np.nonzero(finite)
    if yy.size == 0:
        return resid, (0, resid.shape[0], 0, resid.shape[1])

    pad = max(int(pad), 0)
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(resid.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(resid.shape[1], int(xx.max()) + pad + 1)
    return resid[y0:y1, x0:x1], (y0, y1, x0, x1)


def fast_psf_annulus_mbar(resid_path: Path, rec: dict[str, str], psf, row: dict[str, str], args: argparse.Namespace) -> float:
    import numpy as np
    from astropy.io import fits

    if not resid_path.exists():
        return math.nan
    with fits.open(resid_path, memmap=False) as hdul:
        resid = np.asarray(hdul[0].data, dtype=float)
    original_shape = resid.shape
    resid, crop = crop_to_finite_annulus(resid, int(args.psf_crop_pad))
    log(
        f"[PSF-FFT] {row['galaxy']}: {Path(resid_path).name} "
        f"crop y[{crop[0]}:{crop[1]}] x[{crop[2]}:{crop[3]}], "
        f"{original_shape[0]}x{original_shape[1]} -> {resid.shape[0]}x{resid.shape[1]}"
    )
    window = np.isfinite(resid)
    kp, pk, n_use = radial_median_power(resid, int(args.psf_kbins_n))
    if n_use <= 0:
        return math.nan
    ke, ek = radial_median_ek(resid.shape, window, psf, args)
    p0, _ = fit_p0_from_pk_ek(kp, pk, ke, ek, as_float(row.get("recommended_kmin")), as_float(row.get("recommended_kmax")))
    return mbar_from_p0(
        p0=p0,
        pr=as_float(rec.get("Pr")),
        imean=as_float(rec.get("Imean")),
        pix_area=pix_area_from_fits(row),
    )


def pix_area_from_fits(row: dict[str, str]) -> float:
    try:
        from astropy.io import fits
    except Exception:
        return math.nan
    path = Path(row.get("f150w_path", ""))
    if not path.exists():
        return math.nan
    try:
        hdr = fits.getheader(path, "SCI")
        pixar_sr = as_float(hdr.get("PIXAR_SR"))
        return pixar_sr / 2.350443e-11
    except Exception:
        return math.nan


def estimate_psf_sigma_fast_rerun(row: dict[str, str], kmin: float, kmax: float, args: argparse.Namespace) -> dict[str, object]:
    cache_path = fast_psf_cache_path(row)
    cached = read_compatible_fast_psf_cache(cache_path, kmin, kmax, args)
    if cached is not None:
        return cached

    science_rows = science_rows_by_region(row, kmin, kmax)
    if set(science_rows) != set(SCIENCE_REGIONS):
        return {
            "sigma": math.nan,
            "status": "not_available_missing_saved_science_rows",
            "branch_file": str(cache_path),
            "branch_ref": "",
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }

    psfs, psf_meta, err = build_detdist_detsamp_psfs(row, args)
    if err:
        return {
            "sigma": math.nan,
            "status": f"not_available_{err}",
            "branch_file": str(cache_path),
            "branch_ref": "",
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }

    sigma_inner = as_float(row.get("recommended_sigma_inner"))
    sigma_outer = as_float(row.get("recommended_sigma_outer"))
    weights = inv_variance_weights([sigma_inner, sigma_outer])
    paths = {
        "circular_inner_lit": Path(row.get("inner_usable_residual_fits", "")),
        "circular_outer_lit": Path(row.get("outer_usable_residual_fits", "")),
    }

    mbar_by_ext: dict[str, list[float]] = {}
    for extname in ("DET_DIST", "DET_SAMP"):
        log(f"[PSF-SYS] {row['galaxy']}: measuring {extname} on saved annulus residuals")
        values = []
        for region in SCIENCE_REGIONS:
            log(f"[PSF-SYS] {row['galaxy']}: {extname} {region}")
            values.append(fast_psf_annulus_mbar(paths[region], science_rows[region], psfs[extname], row, args))
        mbar_by_ext[extname] = values

    mbar_detdist = weighted_average(mbar_by_ext["DET_DIST"], weights)
    mbar_detsamp = weighted_average(mbar_by_ext["DET_SAMP"], weights)
    if not (math.isfinite(mbar_detdist) and math.isfinite(mbar_detsamp)):
        return {
            "sigma": math.nan,
            "status": "not_available_fast_psf_measurement_failed",
            "branch_file": str(cache_path),
            "branch_ref": "fast_detdist",
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }

    sigma = abs(mbar_detsamp - mbar_detdist)
    cache_row = {
        "galaxy": row["galaxy"],
        "kmin": kmin,
        "kmax": kmax,
        "sigma_PSF_mag": sigma,
        "mbar_detdist": mbar_detdist,
        "mbar_detsamp": mbar_detsamp,
        "mbar_detdist_inner": mbar_by_ext["DET_DIST"][0],
        "mbar_detdist_outer": mbar_by_ext["DET_DIST"][1],
        "mbar_detsamp_inner": mbar_by_ext["DET_SAMP"][0],
        "mbar_detsamp_outer": mbar_by_ext["DET_SAMP"][1],
        "psf_e_realizations": int(args.psf_e_realizations),
        "psf_kbins_n": int(args.psf_kbins_n),
        "psf_crop_pad": int(args.psf_crop_pad),
        "psf_size": int(args.psf_size),
        "psf_nlambda": int(args.psf_nlambda),
        "opd_path": psf_meta.get("opd_path", ""),
        "status": "ok",
    }
    write_fast_psf_cache(cache_path, cache_row)
    return {
        "sigma": sigma,
        "status": "auto_from_fast_detdist_detsamp_rerun_included",
        "branch_file": str(cache_path),
        "branch_ref": "fast_detdist",
        "branch_alt_count": 1,
        "branch_alt_diffs": f"fast_detsamp:{sigma:.6g}",
    }


def psf_sigma_for_galaxy(
    row: dict[str, str],
    args: argparse.Namespace,
    kmin: float,
    kmax: float,
) -> tuple[float, dict[str, object], str | None]:
    if args.sigma_psf_mag is not None:
        return max(args.sigma_psf_mag, 0.0), {
            "status": "supplied_cli_included",
            "branch_file": "",
            "branch_ref": "",
            "branch_alt_count": 0,
            "branch_alt_diffs": "",
        }, None

    if not args.disable_psf_branch_auto:
        branch = estimate_psf_sigma_from_branch(row, kmin, kmax)
        sigma = as_float(branch.get("sigma"))
        if math.isfinite(sigma) and sigma >= 0.0:
            return sigma, branch, None

    if not args.disable_psf_fast_rerun:
        fast = estimate_psf_sigma_fast_rerun(row, kmin, kmax, args)
        sigma = as_float(fast.get("sigma"))
        if math.isfinite(sigma) and sigma >= 0.0:
            return sigma, fast, None
        return 0.0, fast, "sigma_PSF"

    return 0.0, {
        "status": "not_measured_not_included",
        "branch_file": "",
        "branch_ref": "",
        "branch_alt_count": 0,
        "branch_alt_diffs": "",
    }, "sigma_PSF"


def compute_row(
    row: dict[str, str],
    args: argparse.Namespace,
    ebv_uncertainties: dict[str, float],
    ext_cache: dict[str, dict[str, str]],
) -> dict[str, object]:
    galaxy = row["galaxy"]
    log(f"[GALAXY] {galaxy}: start")
    kmin = as_float(row.get("recommended_kmin"))
    kmax = as_float(row.get("recommended_kmax"))

    sigma_inner = as_float(row.get("recommended_sigma_inner"))
    sigma_outer = as_float(row.get("recommended_sigma_outer"))
    weights = inv_variance_weights([sigma_inner, sigma_outer])

    sigma_pk_current = as_float(row.get("recommended_sigma_weighted_formal"))
    sigma_pk_jensen = weighted_average([sigma_inner, sigma_outer], weights)

    pr_ratio_inner = as_float(row.get("recommended_Pr_over_P0_inner"))
    pr_ratio_outer = as_float(row.get("recommended_Pr_over_P0_outer"))
    sigma_pr_inner = sigma_pr_from_ratio(pr_ratio_inner, args.pr_fractional_error)
    sigma_pr_outer = sigma_pr_from_ratio(pr_ratio_outer, args.pr_fractional_error)
    sigma_pr = weighted_average([sigma_pr_inner, sigma_pr_outer], weights)

    meta = parse_log_metadata(galaxy_log_path(args.batch_root, galaxy))
    science_rows = science_rows_by_region(row, kmin, kmax)

    sigma_ext, ext_status, ext_missing = extinction_sigma_for_galaxy(row, galaxy, args, ebv_uncertainties, ext_cache)
    log(f"[GALAXY] {galaxy}: sigma_ext={sigma_ext:.5f} ({ext_status})")

    bkg_delta = math.nan
    bkg_delta_note = ""
    if args.sigma_bkg_mag is not None:
        sigma_bkg_inner = math.nan
        sigma_bkg_outer = math.nan
        sigma_bkg = max(args.sigma_bkg_mag, 0.0)
        bkg_status = "supplied_cli_included"
    elif args.disable_auto_bkg:
        sigma_bkg_inner = math.nan
        sigma_bkg_outer = math.nan
        sigma_bkg = 0.0
        bkg_status = "not_measured_not_included"
    else:
        bkg_delta, bkg_delta_note = bkg_delta_from_meta(meta, args.bkg_stat)
        sigma_bkg_inner, sigma_bkg_outer, sigma_bkg = bkg_sigma_from_science_rows(
            science_rows,
            weights,
            bkg_delta,
        )
        if math.isfinite(sigma_bkg):
            bkg_status = f"auto_from_{args.bkg_stat}_bkg_check_and_Imean_included"
        else:
            sigma_bkg = 0.0
            bkg_status = "not_available_bkg_check_or_Imean_missing"
    log(f"[GALAXY] {galaxy}: sigma_bkg={sigma_bkg:.5f} ({bkg_status})")

    sigma_psf, psf_info, psf_missing = psf_sigma_for_galaxy(row, args, kmin, kmax)
    log(f"[GALAXY] {galaxy}: sigma_PSF={sigma_psf:.5f} ({psf_info.get('status', '')})")

    missing = []
    if ext_missing:
        missing.append(ext_missing)
    if bkg_status.startswith("not_"):
        missing.append("sigma_bkg")
    if psf_missing:
        missing.append(psf_missing)

    sigma_fast = quadrature([sigma_pk_jensen, sigma_bkg, sigma_psf, sigma_pr, sigma_ext])
    sigma_fast_current_pk = quadrature([sigma_pk_current, sigma_bkg, sigma_psf, sigma_pr, sigma_ext])
    log(f"[GALAXY] {galaxy}: sigma_fast={sigma_fast:.5f}; missing={','.join(missing) if missing else 'none'}")

    formula_terms = [
        f"sigma_Pk_jensen_weighted^2={sigma_pk_jensen:.6g}^2",
        f"sigma_bkg^2={sigma_bkg:.6g}^2",
        f"sigma_PSF^2={sigma_psf:.6g}^2",
        f"sigma_Pr_proxy^2={sigma_pr:.6g}^2",
        f"sigma_ext^2={sigma_ext:.6g}^2",
    ]

    return {
        "galaxy": galaxy,
        "mbar_150": as_float(row.get("recommended_mbar_weighted")),
        "kmin": kmin,
        "kmax": kmax,
        "sigma_Pk_current_sqrt_1_over_sumw": sigma_pk_current,
        "sigma_Pk_jensen_weighted_annuli": sigma_pk_jensen,
        "sigma_inner_used_for_weights": sigma_inner,
        "sigma_outer_used_for_weights": sigma_outer,
        "weight_inner": weights[0],
        "weight_outer": weights[1],
        "Pr_over_P0_inner": pr_ratio_inner,
        "Pr_over_P0_outer": pr_ratio_outer,
        "sigma_Pr_inner_proxy": sigma_pr_inner,
        "sigma_Pr_outer_proxy": sigma_pr_outer,
        "sigma_Pr_proxy_weighted": sigma_pr,
        "Pr_fractional_error_assumed": args.pr_fractional_error,
        "sigma_bkg_inner": sigma_bkg_inner,
        "sigma_bkg_outer": sigma_bkg_outer,
        "sigma_bkg_mag": sigma_bkg,
        "sigma_bkg_status": bkg_status,
        "bkg_delta_image_units": bkg_delta,
        "bkg_delta_note": bkg_delta_note,
        "Imean_inner": as_float(science_rows.get("circular_inner_lit", {}).get("Imean")),
        "Imean_outer": as_float(science_rows.get("circular_outer_lit", {}).get("Imean")),
        "sigma_PSF_mag": sigma_psf,
        "sigma_PSF_status": psf_info.get("status", ""),
        "psf_branch_file": psf_info.get("branch_file", ""),
        "psf_branch_ref": psf_info.get("branch_ref", ""),
        "psf_branch_alt_count": psf_info.get("branch_alt_count", 0),
        "psf_branch_alt_diffs": psf_info.get("branch_alt_diffs", ""),
        "sigma_ext_mag": sigma_ext,
        "sigma_ext_status": ext_status,
        "sigma_jensen_like_fast": sigma_fast,
        "sigma_jensen_like_fast_with_current_pk_combine": sigma_fast_current_pk,
        "missing_components": ",".join(missing),
        "formula_used": "sqrt(" + " + ".join(formula_terms) + ")",
        "sigma_rad_diagnostic_half_inner_outer": as_float(row.get("recommended_annulus_scatter")),
        "sigma_adopted_old_conservative": as_float(row.get("recommended_sigma_adopted")),
        **meta,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, ndigits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{ndigits}f}"


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fast Jensen-like SBF Error Budget",
        "",
        "No isophote/model/full SBF-region rerun was performed. A fast PSF-only rerun on saved annulus residual FITS may be used when no saved PSF branch CSV exists.",
        "",
        "Formula:",
        "",
        "```text",
        "sigma_fast^2 = sigma_Pk^2 + sigma_bkg^2 + sigma_PSF^2 + sigma_Pr^2 + sigma_ext^2",
        "```",
        "",
        "- `sigma_Pk` uses Jensen-style weighted average of the inner/outer annulus errors already measured by sbf-2.",
        "- `sigma_bkg` is auto-propagated from saved `[BKG-CHECK]` sky residual and annulus `Imean`, unless supplied by CLI.",
        "- `sigma_PSF` is read from saved systematics branch CSVs when available; otherwise DET_DIST-vs-DET_SAMP is measured on saved residual FITS.",
        f"- `sigma_Pr` uses proxy `1.0857 * {args.pr_fractional_error:.3g} * (Pr/P0) / (1 - Pr/P0)`.",
        "- `sigma_ext` is direct CLI, CSV, or automatic IRSA/SFD `sigma_E(B-V)` propagated with a near-IR CCM F150W coefficient.",
        "",
        "| galaxy | mbar_150 | sigma_Pk | sigma_bkg | sigma_PSF | sigma_Pr | sigma_ext | sigma_fast | missing | bkg_status | psf_status | ext_status | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {galaxy} | {mbar} | {pk} | {bkg} | {psf} | {pr} | {ext} | {fast} | {missing} | {bkg_status} | {psf_status} | {ext_status} | {flags} |".format(
                galaxy=row["galaxy"],
                mbar=fmt(row["mbar_150"]),
                pk=fmt(row["sigma_Pk_jensen_weighted_annuli"]),
                bkg=fmt(row["sigma_bkg_mag"]),
                psf=fmt(row["sigma_PSF_mag"]),
                pr=fmt(row["sigma_Pr_proxy_weighted"]),
                ext=fmt(row["sigma_ext_mag"]),
                fast=fmt(row["sigma_jensen_like_fast"]),
                missing=row["missing_components"],
                bkg_status=row["sigma_bkg_status"],
                psf_status=row["sigma_PSF_status"],
                ext_status=row["sigma_ext_status"],
                flags=row["warnings"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    log("[START] computing Jensen-like error budget")
    with args.batch_results.open(newline="") as f:
        input_rows = list(csv.DictReader(f))

    ebv_uncertainties = load_ebv_uncertainty(args)
    ext_cache = load_ext_cache(args.ext_cache_csv)
    ok_rows = [row for row in input_rows if row.get("status") == "ok"]
    log(f"[START] galaxies to process: {len(ok_rows)}")
    rows = []
    for idx, row in enumerate(ok_rows, start=1):
        log(f"[PROGRESS] {idx}/{len(ok_rows)} {row.get('galaxy', '')}")
        rows.append(compute_row(row, args, ebv_uncertainties, ext_cache))
    save_ext_cache(args.ext_cache_csv, ext_cache)

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows, args)

    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote MD : {args.out_md}")
    print()
    print(
        f"{'galaxy':<9} {'mbar':>8} {'sig_Pk':>8} {'sig_bkg':>8} "
        f"{'sig_PSF':>8} {'sig_Pr':>8} {'sig_ext':>8} {'sig_fast':>9} {'missing':>28}"
    )
    for row in rows:
        print(
            f"{row['galaxy']:<9} "
            f"{fmt(row['mbar_150']):>8} "
            f"{fmt(row['sigma_Pk_jensen_weighted_annuli']):>8} "
            f"{fmt(row['sigma_bkg_mag']):>8} "
            f"{fmt(row['sigma_PSF_mag']):>8} "
            f"{fmt(row['sigma_Pr_proxy_weighted']):>8} "
            f"{fmt(row['sigma_ext_mag']):>8} "
            f"{fmt(row['sigma_jensen_like_fast']):>9} "
            f"{row['missing_components']:>28}"
        )


if __name__ == "__main__":
    main()
