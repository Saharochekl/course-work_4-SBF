from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def _require(ns, names):
    missing = [name for name in names if name not in ns]
    if missing:
        raise RuntimeError("missing notebook globals: " + ", ".join(missing))


def _finite_positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0.0


def _robust_mag_scatter(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size >= 3:
        med = float(np.nanmedian(arr))
        mad_sigma = float(1.4826 * np.nanmedian(np.abs(arr - med)))
        std_sigma = float(np.nanstd(arr, ddof=1))
        if _finite_positive(mad_sigma):
            return mad_sigma, "k-window MAD"
        if _finite_positive(std_sigma):
            return std_sigma, "k-window std"

    if arr.size >= 2:
        std_sigma = float(np.nanstd(arr, ddof=1))
        half_range = float(0.5 * (np.nanmax(arr) - np.nanmin(arr)))
        if _finite_positive(std_sigma):
            return std_sigma, "k-window std"
        if _finite_positive(half_range):
            return half_range, "k-window half-range"

    return np.nan, "no k-window scatter"


def _fit_resid_proxy_mag(row):
    fit_resid_std = row.get("fit_resid_std", np.nan)
    power_ref = row.get("P_fluc", np.nan)
    if not _finite_positive(power_ref):
        power_ref = row.get("P0", np.nan)
    if _finite_positive(fit_resid_std) and _finite_positive(abs(power_ref)):
        return float((2.5 / np.log(10.0)) * fit_resid_std / abs(power_ref))
    return np.nan


def _measurement_sigma_for_row(row, region_sigma_info):
    candidates = []
    info = region_sigma_info.get(row["region"], {})

    sigma_k = info.get("sigma_kwindow", np.nan)
    if _finite_positive(sigma_k):
        candidates.append((float(sigma_k), info.get("sigma_method", "k-window scatter")))

    sigma_fit = row.get("mbar_fit_sigma", np.nan)
    if _finite_positive(sigma_fit):
        candidates.append((float(sigma_fit), "fit covariance"))

    if candidates:
        sigma, method = max(candidates, key=lambda item: item[0])
        if len(candidates) > 1:
            method = "max(" + ", ".join(m for _, m in candidates) + ")"
        return sigma, method

    sigma_proxy = _fit_resid_proxy_mag(row)
    if _finite_positive(sigma_proxy):
        return sigma_proxy, "fit_resid_std/P_fluc proxy"

    return np.nan, "sigma unavailable"


def _configure_sim_from_header(ns, science_hdr):
    stpsf = ns["stpsf"]
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
    return sim


def _psf_array_from_hdu(hdu):
    arr = np.array(hdu.data, dtype=float)
    if arr.ndim == 3:
        arr = arr.sum(axis=0)
    return arr


def build_branch_psf(ns, opd_path, prefer_ext):
    _require(ns, ["PSFREF", "f150w_path", "PSF_SIZE", "PSF_NLAMBDA", "stpsf"])

    psf_file = ns["PSFREF"] if ns["PSFREF"] is not None else ns["f150w_path"]
    science_hdr = fits.getheader(str(psf_file), 0)
    sim = _configure_sim_from_header(ns, science_hdr)
    sim.load_wss_opd(str(opd_path), verbose=False, plot=False)
    sim.options["output_mode"] = "both"
    psf_hdul = sim.calc_psf(
        nlambda=ns["PSF_NLAMBDA"],
        fov_pixels=ns["PSF_SIZE"],
        fft_oversample=4,
        detector_oversample=1,
        add_distortion=True,
    )

    psf = None
    selected_ext = None
    ext_rows = []
    if isinstance(psf_hdul, fits.HDUList):
        for i, hdu in enumerate(psf_hdul):
            if getattr(hdu, "data", None) is None:
                continue
            arr = np.array(hdu.data)
            extname = hdu.header.get("EXTNAME", "PRIMARY" if i == 0 else f"EXT{i}")
            ext_rows.append(
                {
                    "idx": i,
                    "extname": extname,
                    "shape": tuple(arr.shape),
                    "oversamp": hdu.header.get("OVERSAMP", "NA"),
                    "det_samp": hdu.header.get("DET_SAMP", "NA"),
                    "pixelscl": hdu.header.get("PIXELSCL", "NA"),
                }
            )
        for extname in (prefer_ext, "DET_DIST", "DET_SAMP"):
            try:
                psf = _psf_array_from_hdu(psf_hdul[extname])
                selected_ext = extname
                break
            except Exception:
                continue
        if psf is None:
            detector_like = []
            for i, hdu in enumerate(psf_hdul):
                if getattr(hdu, "data", None) is None:
                    continue
                arr = _psf_array_from_hdu(hdu)
                if arr.shape == (ns["PSF_SIZE"], ns["PSF_SIZE"]):
                    detector_like.append((hdu.header.get("EXTNAME", f"EXT{i}"), arr))
            if detector_like:
                selected_ext, psf = detector_like[0]
            else:
                selected_ext = "PRIMARY_FALLBACK"
                psf = _psf_array_from_hdu(psf_hdul[0])
    else:
        selected_ext = "PRIMARY"
        psf = np.array(psf_hdul.data, dtype=float)

    psf = np.nan_to_num(psf, nan=0.0)
    psf_sum = float(psf.sum())
    if (not np.isfinite(psf_sum)) or psf_sum <= 0.0:
        raise RuntimeError(f"[SYS-PSF] invalid PSF sum={psf_sum} for branch {prefer_ext}")
    psf /= psf_sum
    return psf, {
        "selected_ext": selected_ext,
        "shape": tuple(psf.shape),
        "sum": float(psf.sum()),
        "opd_path": str(opd_path),
        "preferred_ext": prefer_ext,
        "ext_rows": ext_rows,
    }


def rebuild_science_residual_for_premask(ns, premask_in, branch_suffix):
    _require(
        ns,
        [
            "img",
            "model_full",
            "CLIP_SIGMA_QC",
            "CLIP_MAXIT_QC",
            "CLIP_TAG_QC",
            "MIN_PIXELS_SBF",
        ],
    )
    premask_local = np.array(premask_in, dtype=bool, copy=True)
    resid_full_local = np.array(ns["img"] - ns["model_full"], dtype=np.float32, copy=True)
    bad = premask_local | (~np.isfinite(ns["model_full"])) | (~np.isfinite(ns["img"]))
    resid_full_local[bad] = np.nan

    science_clip_mask = (~premask_local) & np.isfinite(resid_full_local) & np.isfinite(ns["model_full"]) & (ns["model_full"] > 0.0)
    n_science = int(science_clip_mask.sum())
    if n_science < ns["MIN_PIXELS_SBF"]:
        raise RuntimeError(f"[SYS-RESID] too few valid science pixels after rebuild: N={n_science}")

    vals = np.asarray(resid_full_local[science_clip_mask], dtype=float)
    _, med_clip, std_clip = sigma_clipped_stats(
        vals,
        sigma=ns["CLIP_SIGMA_QC"],
        maxiters=ns["CLIP_MAXIT_QC"],
    )
    if (not np.isfinite(std_clip)) or std_clip <= 0.0:
        raise RuntimeError(f"[SYS-RESID] bad clipped std={std_clip}")

    clip_lo = float(med_clip - ns["CLIP_SIGMA_QC"] * std_clip)
    clip_hi = float(med_clip + ns["CLIP_SIGMA_QC"] * std_clip)
    resid_full_clip_local = np.array(resid_full_local, dtype=np.float32, copy=True)
    resid_full_clip_local[science_clip_mask] = np.clip(resid_full_clip_local[science_clip_mask], clip_lo, clip_hi)
    resid_full_clip_local[~science_clip_mask] = np.nan

    return {
        "premask": premask_local,
        "science_resid": resid_full_clip_local,
        "science_model": ns["model_full"],
        "science_resid_name": f"resid_full_clip_{ns['CLIP_TAG_QC']}sigma_{branch_suffix}",
        "science_model_name": "model_full",
        "n_science_clip": n_science,
        "clip_med": float(med_clip),
        "clip_std": float(std_clip),
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
    }


def _build_science_regions(ns, shape):
    pix_scale = float(np.sqrt(ns["pix_area"]))
    r1_in = float(ns["SBF_LIT_INNER_ARCSEC"][0] / pix_scale)
    r1_out = float(ns["SBF_LIT_INNER_ARCSEC"][1] / pix_scale)
    r2_in = float(ns["SBF_LIT_OUTER_ARCSEC"][0] / pix_scale)
    r2_out = float(ns["SBF_LIT_OUTER_ARCSEC"][1] / pix_scale)
    return [
        {
            "region": "circular_inner_lit",
            "role": "science",
            "shape": "circle",
            "mask": ns["build_region_mask_circle"](shape, ns["x0_sbf_circ"], ns["y0_sbf_circ"], r1_in, r1_out),
            "rin_px": r1_in,
            "rout_px": r1_out,
            "rin_arcsec": ns["SBF_LIT_INNER_ARCSEC"][0],
            "rout_arcsec": ns["SBF_LIT_INNER_ARCSEC"][1],
        },
        {
            "region": "circular_outer_lit",
            "role": "science",
            "shape": "circle",
            "mask": ns["build_region_mask_circle"](shape, ns["x0_sbf_circ"], ns["y0_sbf_circ"], r2_in, r2_out),
            "rin_px": r2_in,
            "rout_px": r2_out,
            "rin_arcsec": ns["SBF_LIT_OUTER_ARCSEC"][0],
            "rout_arcsec": ns["SBF_LIT_OUTER_ARCSEC"][1],
        },
    ]


def _build_kdiag_region_local(ns, shape, region_name):
    pix_scale = float(np.sqrt(ns["pix_area"]))

    if region_name == "elliptical_chosen":
        return {
            "region": region_name,
            "role": "diagnostic",
            "mask": ns["build_region_mask_ellipse"](
                shape, ns["x0_ann"], ns["y0_ann"], ns["q_ann"], ns["pa_ann"], ns["sma_in"], ns["sma_out"]
            ),
            "rin_arcsec": float(ns["sma_in"] * pix_scale),
            "rout_arcsec": float(ns["sma_out"] * pix_scale),
        }

    if region_name == "circular_inner_lit":
        rin_arcsec, rout_arcsec = ns["SBF_LIT_INNER_ARCSEC"]
        rin_px = float(rin_arcsec / pix_scale)
        rout_px = float(rout_arcsec / pix_scale)
        return {
            "region": region_name,
            "role": "science",
            "mask": ns["build_region_mask_circle"](shape, ns["x0_sbf_circ"], ns["y0_sbf_circ"], rin_px, rout_px),
            "rin_arcsec": rin_arcsec,
            "rout_arcsec": rout_arcsec,
        }

    if region_name == "circular_outer_lit":
        rin_arcsec, rout_arcsec = ns["SBF_LIT_OUTER_ARCSEC"]
        rin_px = float(rin_arcsec / pix_scale)
        rout_px = float(rout_arcsec / pix_scale)
        return {
            "region": region_name,
            "role": "science",
            "mask": ns["build_region_mask_circle"](shape, ns["x0_sbf_circ"], ns["y0_sbf_circ"], rin_px, rout_px),
            "rin_arcsec": rin_arcsec,
            "rout_arcsec": rout_arcsec,
        }

    raise ValueError(f"unknown KDIAG region: {region_name}")


def _compute_pk_ek_diagnostic_local(ns, region_mask, resid, model, premask, psf, n_e_realizations, kbins_n):
    mask_local = premask | (~region_mask)
    window = (~mask_local) & np.isfinite(resid) & np.isfinite(model) & (model > 0.0)

    n_use = int(window.sum())
    if n_use < ns["MIN_PIXELS_SBF"]:
        raise RuntimeError(f"not enough usable pixels for FFT diagnostic: {n_use}")

    Imean = float(np.nanmean(model[window]))
    if (not np.isfinite(Imean)) or (Imean <= 0.0):
        raise RuntimeError("mean model intensity is not usable for FFT diagnostic")

    data_fft = np.zeros_like(resid, dtype=float)
    data_fft[window] = resid[window]
    data_fft[window] -= float(np.nanmean(data_fft[window]))

    with ns["set_workers"](ns["FFT_WORKERS"]):
        F = ns["fft2"](data_fft)

    P2d = (np.abs(F) ** 2) / float(n_use)

    ny, nx = resid.shape
    ky = ns["fftshift"](ns["fftfreq"](ny))
    kx = ns["fftshift"](ns["fftfreq"](nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    P2d_s = ns["fftshift"](P2d)

    kbins = np.linspace(0.0, float(kr.max()), kbins_n)
    Pk_vals = np.full(len(kbins) - 1, np.nan, dtype=float)
    Pk_k = np.full_like(Pk_vals, np.nan)

    for i in range(len(Pk_vals)):
        sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
        vals = P2d_s[sel]
        if vals.size >= ns["MIN_POINTS_PK_BIN"]:
            Pk_vals[i] = float(np.nanmedian(vals))
            Pk_k[i] = 0.5 * (kbins[i] + kbins[i + 1])

    mP = np.isfinite(Pk_vals) & np.isfinite(Pk_k) & (Pk_k > 0.0)
    if int(mP.sum()) < ns["MIN_POINTS_FIT"]:
        raise RuntimeError("not enough valid P(k) bins for FFT diagnostic")

    big_psf = np.zeros((ny, nx), dtype=float)
    py, px = psf.shape
    y0_psf = ny // 2 - py // 2
    x0_psf = nx // 2 - px // 2
    big_psf[y0_psf:y0_psf + py, x0_psf:x0_psf + px] = psf

    with ns["set_workers"](ns["FFT_WORKERS"]):
        F_psf = ns["fft2"](big_psf)

    rng = np.random.default_rng(ns["FFT_RNG_SEED"])
    Ek_stack = []
    for _ in range(int(n_e_realizations)):
        noise = rng.normal(loc=0.0, scale=1.0, size=(ny, nx))
        with ns["set_workers"](ns["FFT_WORKERS"]):
            F_noise = ns["fft2"](noise)
            sim = np.real(np.fft.ifft2(F_noise * F_psf))

        sim_masked = np.zeros_like(sim, dtype=float)
        sim_masked[window] = sim[window]
        sim_masked[window] -= float(np.nanmean(sim_masked[window]))

        with ns["set_workers"](ns["FFT_WORKERS"]):
            F_sim = ns["fft2"](sim_masked)

        E2d_sim = (np.abs(F_sim) ** 2) / float(n_use)
        E2d_sim_s = ns["fftshift"](E2d_sim)

        Ek_vals_i = np.full(len(kbins) - 1, np.nan, dtype=float)
        for j in range(len(Ek_vals_i)):
            sel = (kr >= kbins[j]) & (kr < kbins[j + 1])
            vals = E2d_sim_s[sel]
            if vals.size >= ns["MIN_POINTS_PK_BIN"]:
                Ek_vals_i[j] = float(np.nanmedian(vals))
        Ek_stack.append(Ek_vals_i)

    Ek_stack = np.array(Ek_stack, dtype=float)
    Ek_vals = np.nanmedian(Ek_stack, axis=0)
    Ek_k = 0.5 * (kbins[:-1] + kbins[1:])
    mE = np.isfinite(Ek_vals) & np.isfinite(Ek_k) & (Ek_k > 0.0)
    if int(mE.sum()) < ns["MIN_POINTS_FIT"]:
        raise RuntimeError("not enough valid E(k) bins for FFT diagnostic")

    return {
        "window": window,
        "n_use": n_use,
        "Imean": Imean,
        "kP": Pk_k[mP],
        "P": Pk_vals[mP],
        "kE": Ek_k[mE],
        "E": Ek_vals[mE],
    }


def _fit_pk_window_local(ns, spec, region_mask, region_name, kmin, kmax):
    kP = spec["kP"]
    P = spec["P"]
    kE = spec["kE"]
    E = spec["E"]
    Imean = spec["Imean"]

    fit_sel = (kP >= kmin) & (kP <= kmax)
    if int(fit_sel.sum()) < ns["MIN_POINTS_FIT"]:
        raise RuntimeError(f"not enough bins for k-window {kmin:.3f}..{kmax:.3f}")

    x_fit = np.interp(kP[fit_sel], kE, E, left=np.nan, right=np.nan)
    y_fit = P[fit_sel]
    good = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0.0) & (y_fit > 0.0)
    x = x_fit[good]
    y = y_fit[good]
    if x.size < ns["MIN_POINTS_FIT"]:
        raise RuntimeError(f"bad interpolated bins for k-window {kmin:.3f}..{kmax:.3f}")

    A = np.vstack([x, np.ones_like(x)]).T
    P0, P1 = np.linalg.lstsq(A, y, rcond=None)[0]
    if (not np.isfinite(P0)) or (P0 <= 0.0):
        raise RuntimeError(f"invalid P0 for k-window {kmin:.3f}..{kmax:.3f}")

    E_all = np.interp(kP, kE, E, left=np.nan, right=np.nan)
    curve_all = P0 * E_all + P1
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan

    zpt = float(ns["sbf_ab_zeropoint_from_pix_area"](ns["pix_area"]))
    Pf_spec_raw = float(P0 / Imean)
    mbar_spec_raw = float(-2.5 * np.log10(Pf_spec_raw) + zpt) if Pf_spec_raw > 0.0 else np.nan

    pr_info = ns["estimate_pr_for_region"](
        region_mask=region_mask,
        model=ns["science_model"],
        region_name=region_name,
        region_origin_x=0,
        region_origin_y=0,
    )
    power_info = ns["solve_sbf_power_budget"](
        P0=P0,
        Imean=Imean,
        Pr=pr_info.get("Pr", np.nan),
        pix_area=ns["pix_area"],
    )

    low_sel = (kP > 0.0) & (kP < kmin) & np.isfinite(curve_all) & np.isfinite(P)
    low_mean_delta = float(np.nanmean(P[low_sel] - curve_all[low_sel])) if int(low_sel.sum()) else np.nan
    low_median_delta = float(np.nanmedian(P[low_sel] - curve_all[low_sel])) if int(low_sel.sum()) else np.nan

    return {
        "kmin": float(kmin),
        "kmax": float(kmax),
        "P0": float(P0),
        "P1": float(P1),
        "corr": corr,
        "n_fit": int(x.size),
        "Pf_spec_raw": Pf_spec_raw,
        "mbar_spec_raw": mbar_spec_raw,
        "Pr": pr_info.get("Pr", np.nan),
        "mbar_spec": power_info.get("mbar_spec", np.nan),
        "curve_all": curve_all,
        "E_all": E_all,
        "low_mean_delta": low_mean_delta,
        "low_median_delta": low_median_delta,
        "n_low_bins": int(low_sel.sum()),
    }


def _scalarize_out(out):
    keep = [
        "measurement_ok",
        "failure_reason",
        "n_use",
        "Imean",
        "P0",
        "P1",
        "P_fluc",
        "Pr",
        "Pr_over_P0",
        "frac",
        "corr",
        "n_fit",
        "fit_resid_std",
        "P0_fit_sigma",
        "mbar_fit_sigma_raw",
        "mbar_fit_sigma",
        "Pf_spec_raw",
        "Pf_spec",
        "Pf_spec_sigma_formal_raw",
        "Pf_spec_sigma_formal",
        "mbar_spec_raw",
        "mbar_spec",
        "n_detected_sources",
        "n_detected_sources_total",
        "m_lim",
        "m_lim_method",
        "lf_gamma",
        "lf_fit_method",
    ]
    return {key: out.get(key, np.nan) for key in keep}


def run_branch_measurements(ns, branch_name, resid, resid_name, premask, psf, psf_meta):
    _require(
        ns,
        [
            "measure_sbf_for_mask",
            "pix_area",
            "FFT_WORKERS",
            "FFT_E_REALIZATIONS_MAIN",
            "FFT_KBINS_N",
            "SBF_REGION_K_WINDOWS",
        ],
    )
    regions = _build_science_regions(ns, resid.shape)
    rows_df = []
    print(
        f"[SYS-BRANCH] {branch_name}: residual={resid_name}, psf_ext={psf_meta.get('selected_ext')}, "
        f"opd={Path(psf_meta.get('opd_path', 'NA')).name}"
    )

    for reg in regions:
        region_mask = reg["mask"]
        region_pixels = int(region_mask.sum())
        usable_region = region_mask & (~premask) & np.isfinite(resid) & np.isfinite(ns["model_full"]) & (ns["model_full"] > 0.0)
        usable_pixels = int(usable_region.sum())
        usable_fraction = float(usable_pixels / region_pixels) if region_pixels > 0 else np.nan
        print(
            f"[SYS-BRANCH]   {branch_name}/{reg['region']}: N_region={region_pixels}, "
            f"N_usable={usable_pixels}, usable_fraction={usable_fraction:.3f}"
        )
        for kmin, kmax in list(ns["SBF_REGION_K_WINDOWS"]):
            out = ns["measure_sbf_for_mask"](
                region_mask=region_mask,
                resid=resid,
                model=ns["model_full"],
                premask=premask,
                psf=psf,
                pix_area=ns["pix_area"],
                kmin=kmin,
                kmax=kmax,
                fft_workers=ns["FFT_WORKERS"],
                n_e_realizations=ns["FFT_E_REALIZATIONS_MAIN"],
                kbins_n=ns["FFT_KBINS_N"],
                region_name=reg["region"],
                region_role=reg["role"],
                region_origin_x=0,
                region_origin_y=0,
            )

            row = {
                "branch": branch_name,
                "psf_ext": psf_meta.get("selected_ext"),
                "opd_path": psf_meta.get("opd_path"),
                "region": reg["region"],
                "role": reg["role"],
                "shape": reg["shape"],
                "rin_px": reg["rin_px"],
                "rout_px": reg["rout_px"],
                "rin_arcsec": reg["rin_arcsec"],
                "rout_arcsec": reg["rout_arcsec"],
                "resid_source": resid_name,
                "model_source": "model_full",
                "region_pixels": region_pixels,
                "usable_pixels": usable_pixels,
                "usable_fraction": usable_fraction,
                "kmin": float(kmin),
                "kmax": float(kmax),
            }
            if out is None:
                row["status"] = "failed_raw"
                rows_df.append(row)
                print(f"[SYS-BRANCH]     k={kmin:.3f}..{kmax:.3f} -> raw spectral fit failed")
                continue

            row["status"] = "ok" if out.get("measurement_ok", False) else "invalid_corrected"
            row.update(_scalarize_out(out))
            rows_df.append(row)
            if row["status"] == "ok":
                print(
                    f"[SYS-BRANCH]     k={kmin:.3f}..{kmax:.3f}: corrected mbar={row['mbar_spec']:.4f}, "
                    f"P0={row['P0']:.3e}, Pr/P0={row['Pr_over_P0']:.3%}"
                )
            else:
                print(
                    f"[SYS-BRANCH]     k={kmin:.3f}..{kmax:.3f}: invalid corrected "
                    f"({row.get('failure_reason', '')})"
                )

    return pd.DataFrame(rows_df)


def summarize_branch_annuli(df_sbf, sigma_pipeline=np.nan):
    required_annuli = ["circular_inner_lit", "circular_outer_lit"]
    df_sbf_ok = df_sbf[df_sbf["status"].eq("ok")].copy()

    region_sigma_info = {}
    for region_name, grp in df_sbf_ok.groupby("region"):
        sigma_k, sigma_method = _robust_mag_scatter(grp["mbar_spec"].values)
        region_sigma_info[region_name] = {
            "sigma_kwindow": sigma_k,
            "sigma_method": sigma_method,
        }

    def _row_for(region_name, kmin_value, kmax_value):
        match = df_sbf_ok[
            df_sbf_ok["region"].eq(region_name)
            & np.isclose(df_sbf_ok["kmin"].astype(float), float(kmin_value))
            & np.isclose(df_sbf_ok["kmax"].astype(float), float(kmax_value))
        ]
        if match.empty:
            return None
        return match.iloc[0]

    def _annulus_measurement(region_name, kmin_value, kmax_value):
        row = _row_for(region_name, kmin_value, kmax_value)
        if row is None:
            return np.nan, np.nan, "not ok", None

        mbar = float(row["mbar_spec"]) if np.isfinite(row.get("mbar_spec", np.nan)) else np.nan
        sigma, sigma_method = _measurement_sigma_for_row(row, region_sigma_info)
        return mbar, sigma, sigma_method, row

    pairs_from_df = df_sbf[df_sbf["region"].isin(required_annuli)][["kmin", "kmax"]].dropna().drop_duplicates()
    k_pairs = sorted({(float(r.kmin), float(r.kmax)) for r in pairs_from_df.itertuples(index=False)})
    summary_rows = []
    recommended = None

    for kmin_value, kmax_value in k_pairs:
        mbar_inner, sigma_inner, method_inner, row_inner = _annulus_measurement(
            "circular_inner_lit", kmin_value, kmax_value
        )
        mbar_outer, sigma_outer, method_outer, row_outer = _annulus_measurement(
            "circular_outer_lit", kmin_value, kmax_value
        )

        mbar_inner_raw = float(row_inner["mbar_spec_raw"]) if row_inner is not None and np.isfinite(row_inner.get("mbar_spec_raw", np.nan)) else np.nan
        mbar_outer_raw = float(row_outer["mbar_spec_raw"]) if row_outer is not None and np.isfinite(row_outer.get("mbar_spec_raw", np.nan)) else np.nan
        inner_ok = np.isfinite(mbar_inner) and _finite_positive(sigma_inner)
        outer_ok = np.isfinite(mbar_outer) and _finite_positive(sigma_outer)
        both_measured = np.isfinite(mbar_inner) and np.isfinite(mbar_outer)

        mbar_weighted = np.nan
        mbar_weighted_raw = np.nan
        sigma_weighted_formal = np.nan
        annulus_scatter = float(abs(mbar_inner - mbar_outer) / 2.0) if both_measured else np.nan
        sigma_adopted = np.nan

        if inner_ok and outer_ok:
            mvals = np.array([mbar_inner, mbar_outer], dtype=float)
            sigmas = np.array([sigma_inner, sigma_outer], dtype=float)
            weights = 1.0 / (sigmas * sigmas)
            mbar_weighted = float(np.sum(weights * mvals) / np.sum(weights))
            sigma_weighted_formal = float(np.sqrt(1.0 / np.sum(weights)))
            if np.isfinite(mbar_inner_raw) and np.isfinite(mbar_outer_raw):
                mbar_weighted_raw = float(np.sum(weights * np.array([mbar_inner_raw, mbar_outer_raw])) / np.sum(weights))

            if _finite_positive(sigma_pipeline):
                sigma_adopted = float(
                    np.sqrt(sigma_weighted_formal**2 + annulus_scatter**2 + float(sigma_pipeline)**2)
                )
                notes = "inner+outer weighted; adopted includes pipeline systematic"
            else:
                sigma_adopted = float(max(sigma_weighted_formal, annulus_scatter))
                notes = "inner+outer weighted; no pipeline systematic yet"
        elif inner_ok or outer_ok:
            if inner_ok:
                mbar_weighted = float(mbar_inner)
                mbar_weighted_raw = float(mbar_inner_raw) if np.isfinite(mbar_inner_raw) else np.nan
                sigma_weighted_formal = float(sigma_inner)
                sigma_adopted = float(sigma_inner)
                notes = "only circular_inner_lit ok; annulus scatter unavailable"
            else:
                mbar_weighted = float(mbar_outer)
                mbar_weighted_raw = float(mbar_outer_raw) if np.isfinite(mbar_outer_raw) else np.nan
                sigma_weighted_formal = float(sigma_outer)
                sigma_adopted = float(sigma_outer)
                notes = "only circular_outer_lit ok; annulus scatter unavailable"
        else:
            notes = "no circular annulus has both corrected mbar and sigma_meas"

        if method_inner != "not ok" or method_outer != "not ok":
            notes += f"; sigma_inner={method_inner}; sigma_outer={method_outer}"

        summary_rows.append(
            {
                "kmin": float(kmin_value),
                "kmax": float(kmax_value),
                "mbar_inner_raw": mbar_inner_raw,
                "mbar_inner": mbar_inner,
                "sigma_inner": sigma_inner,
                "Pr_inner": float(row_inner.get("Pr", np.nan)) if row_inner is not None else np.nan,
                "Pr_over_P0_inner": float(row_inner.get("Pr_over_P0", np.nan)) if row_inner is not None else np.nan,
                "mbar_outer_raw": mbar_outer_raw,
                "mbar_outer": mbar_outer,
                "sigma_outer": sigma_outer,
                "Pr_outer": float(row_outer.get("Pr", np.nan)) if row_outer is not None else np.nan,
                "Pr_over_P0_outer": float(row_outer.get("Pr_over_P0", np.nan)) if row_outer is not None else np.nan,
                "mbar_weighted_raw": mbar_weighted_raw,
                "mbar_weighted": mbar_weighted,
                "sigma_weighted_formal": sigma_weighted_formal,
                "annulus_scatter": annulus_scatter,
                "sigma_adopted": sigma_adopted,
                "notes": notes,
            }
        )

    df_annulus_summary = pd.DataFrame(summary_rows)
    recommendable = df_annulus_summary[
        np.isfinite(df_annulus_summary["mbar_weighted"]) & np.isfinite(df_annulus_summary["sigma_adopted"])
    ].copy()
    if not recommendable.empty:
        recommendable["uses_two_annuli"] = recommendable["notes"].str.contains(r"inner\\+outer weighted", regex=True)
        recommendable = recommendable.sort_values(
            ["uses_two_annuli", "sigma_adopted", "sigma_weighted_formal"],
            ascending=[False, True, True],
        )
        recommended = recommendable.iloc[0].to_dict()
    return df_annulus_summary, recommended


def _current_reference_rows(ns):
    _require(ns, ["df_sbf", "df_annulus_summary", "recommended_sbf", "science_resid_name", "science_model_name"])
    current_opd = str(ns.get("local_wss_opd", ""))
    current_ext = ns.get("psf_selected_ext", "UNKNOWN")

    df_rows = ns["df_sbf"].copy()
    df_rows["branch"] = "current_mask_currentopd_detdist_ref"
    df_rows["psf_ext"] = current_ext
    df_rows["opd_path"] = current_opd
    df_rows["resid_source"] = ns["science_resid_name"]
    df_rows["model_source"] = ns["science_model_name"]

    df_summary = ns["df_annulus_summary"].copy()
    df_summary["branch"] = "current_mask_currentopd_detdist_ref"
    df_summary["psf_ext"] = current_ext
    df_summary["opd_path"] = current_opd
    df_summary["resid_source"] = ns["science_resid_name"]
    df_summary["model_source"] = ns["science_model_name"]

    recommended = dict(ns["recommended_sbf"]) if ns.get("recommended_sbf") is not None else None
    if recommended is not None:
        recommended["branch"] = "current_mask_currentopd_detdist_ref"
        recommended["psf_ext"] = current_ext
        recommended["opd_path"] = current_opd
        recommended["resid_source"] = ns["science_resid_name"]
        recommended["model_source"] = ns["science_model_name"]

    return df_rows, df_summary, recommended


def prepare_systematic_assets(ns):
    _require(
        ns,
        [
            "science_resid",
            "science_model",
            "science_resid_name",
            "science_model_name",
            "premask",
            "premask_base",
            "psf",
            "psf_selected_ext",
            "local_wss_opd",
            "local_wss_files",
        ],
    )
    current_opd = Path(ns["local_wss_opd"])
    current_meta = {
        "selected_ext": ns.get("psf_selected_ext", "UNKNOWN"),
        "opd_path": str(current_opd),
        "shape": tuple(ns["psf"].shape),
        "sum": float(np.sum(ns["psf"])),
    }

    no_extra = rebuild_science_residual_for_premask(ns, ns["premask_base"], "nomaskbranch")
    psf_det_samp, meta_det_samp = build_branch_psf(ns, current_opd, "DET_SAMP")

    alt_opd = None
    alt_psf = None
    alt_meta = None
    other_opds = [Path(p) for p in ns["local_wss_files"] if Path(p).resolve() != current_opd.resolve()]
    if other_opds:
        other_opds = sorted(other_opds, key=lambda p: abs(p.stat().st_mtime - current_opd.stat().st_mtime))
        alt_opd = other_opds[0]
        alt_psf, alt_meta = build_branch_psf(ns, alt_opd, "DET_DIST")

    assets = {
        "current_reference": {
            "branch": "current_mask_currentopd_detdist_ref",
            "resid": ns["science_resid"],
            "resid_name": ns["science_resid_name"],
            "model": ns["science_model"],
            "model_name": ns["science_model_name"],
            "premask": np.array(ns["premask"], dtype=bool, copy=True),
            "psf": ns["psf"],
            "psf_meta": current_meta,
            "source_kind": "reference",
        },
        "no_extra_mask_detdist": {
            "branch": "no_extra_mask_currentopd_detdist",
            "resid": no_extra["science_resid"],
            "resid_name": no_extra["science_resid_name"],
            "model": no_extra["science_model"],
            "model_name": no_extra["science_model_name"],
            "premask": no_extra["premask"],
            "psf": ns["psf"],
            "psf_meta": current_meta,
            "source_kind": "recomputed",
        },
        "current_mask_detsamp": {
            "branch": "current_mask_currentopd_detsamp",
            "resid": ns["science_resid"],
            "resid_name": ns["science_resid_name"],
            "model": ns["science_model"],
            "model_name": ns["science_model_name"],
            "premask": np.array(ns["premask"], dtype=bool, copy=True),
            "psf": psf_det_samp,
            "psf_meta": meta_det_samp,
            "source_kind": "recomputed",
        },
    }
    if alt_psf is not None:
        assets["current_mask_altopd_detdist"] = {
            "branch": "current_mask_altopd_detdist",
            "resid": ns["science_resid"],
            "resid_name": ns["science_resid_name"],
            "model": ns["science_model"],
            "model_name": ns["science_model_name"],
            "premask": np.array(ns["premask"], dtype=bool, copy=True),
            "psf": alt_psf,
            "psf_meta": alt_meta,
            "source_kind": "recomputed",
        }

    plan_rows = []
    for key, asset in assets.items():
        plan_rows.append(
            {
                "asset_key": key,
                "branch": asset["branch"],
                "source_kind": asset["source_kind"],
                "residual": asset["resid_name"],
                "psf_ext": asset["psf_meta"].get("selected_ext"),
                "psf_shape": asset["psf_meta"].get("shape"),
                "opd": Path(asset["psf_meta"].get("opd_path", "")).name,
                "premask_fraction": float(np.mean(asset["premask"])),
            }
        )
    return assets, pd.DataFrame(plan_rows)


def run_systematic_branch_suite(ns, assets):
    df_rows_all = []
    df_summary_all = []
    recommended_rows = []

    ref_rows, ref_summary, ref_rec = _current_reference_rows(ns)
    df_rows_all.append(ref_rows)
    df_summary_all.append(ref_summary)
    if ref_rec is not None:
        recommended_rows.append(ref_rec)

    for key in ["no_extra_mask_detdist", "current_mask_detsamp", "current_mask_altopd_detdist"]:
        asset = assets.get(key)
        if asset is None:
            continue
        df_branch = run_branch_measurements(
            ns,
            branch_name=asset["branch"],
            resid=asset["resid"],
            resid_name=asset["resid_name"],
            premask=asset["premask"],
            psf=asset["psf"],
            psf_meta=asset["psf_meta"],
        )
        df_branch_summary, recommended = summarize_branch_annuli(df_branch, sigma_pipeline=ns.get("sigma_pipeline", np.nan))
        df_branch_summary["branch"] = asset["branch"]
        df_branch_summary["psf_ext"] = asset["psf_meta"].get("selected_ext")
        df_branch_summary["opd_path"] = asset["psf_meta"].get("opd_path")
        df_branch_summary["resid_source"] = asset["resid_name"]
        df_branch_summary["model_source"] = asset["model_name"]

        df_rows_all.append(df_branch)
        df_summary_all.append(df_branch_summary)
        if recommended is not None:
            recommended["branch"] = asset["branch"]
            recommended["psf_ext"] = asset["psf_meta"].get("selected_ext")
            recommended["opd_path"] = asset["psf_meta"].get("opd_path")
            recommended["resid_source"] = asset["resid_name"]
            recommended["model_source"] = asset["model_name"]
            recommended_rows.append(recommended)

    return {
        "df_rows": pd.concat(df_rows_all, ignore_index=True, sort=False),
        "df_summary": pd.concat(df_summary_all, ignore_index=True, sort=False),
        "df_recommended": pd.DataFrame(recommended_rows),
    }


def run_systematic_kdiag_suite(ns, assets):
    _require(
        ns,
        [
            "build_region_mask_circle",
            "build_region_mask_ellipse",
            "estimate_pr_for_region",
            "solve_sbf_power_budget",
            "sbf_ab_zeropoint_from_pix_area",
            "set_workers",
            "fft2",
            "fftfreq",
            "fftshift",
            "science_model",
        ],
    )
    branch_rows = []
    lowbin_tables = []
    plot_payload = {}
    kdiag_region_name = ns.get("KDIAG_REGION", "circular_inner_lit")
    kdiag_k_windows = ns.get(
        "KDIAG_K_WINDOWS",
        [(0.01, 0.25), (0.03, 0.25), (0.04, 0.25), (0.01, 0.40), (0.01, 0.55), (0.01, 0.65)],
    )

    def _compute_branch_kdiag(branch_name, resid, premask, psf, psf_meta):
        region_info = _build_kdiag_region_local(ns, resid.shape, kdiag_region_name)
        region_mask = region_info["mask"]
        spec = _compute_pk_ek_diagnostic_local(
            ns,
            region_mask=region_mask,
            resid=resid,
            model=ns["model_full"],
            premask=premask,
            psf=psf,
            n_e_realizations=ns.get("KDIAG_E_REALIZATIONS", ns["FFT_E_REALIZATIONS_MAIN"]),
            kbins_n=ns.get("KDIAG_KBINS_N", ns["FFT_KBINS_N"]),
        )
        fit_results = []
        fit_curves = {}
        first_n_bins = int(ns.get("KDIAG_FIRST_N_BINS", 15))
        df_low = pd.DataFrame({"k": spec["kP"][:first_n_bins], "Pk": spec["P"][:first_n_bins]})
        for kmin, kmax in list(kdiag_k_windows):
            fit = _fit_pk_window_local(ns, spec=spec, region_mask=region_mask, region_name=region_info["region"], kmin=kmin, kmax=kmax)
            fit_results.append(
                {
                    "branch": branch_name,
                    "psf_ext": psf_meta.get("selected_ext"),
                    "opd_path": psf_meta.get("opd_path"),
                    "kmin": fit["kmin"],
                    "kmax": fit["kmax"],
                    "P0": fit["P0"],
                    "P1": fit["P1"],
                    "corr": fit["corr"],
                    "n_fit": fit["n_fit"],
                    "mbar_spec_raw": fit["mbar_spec_raw"],
                    "Pr": fit["Pr"],
                    "mbar_spec": fit["mbar_spec"],
                    "n_low_bins": fit["n_low_bins"],
                    "low_mean_delta": fit["low_mean_delta"],
                    "low_median_delta": fit["low_median_delta"],
                }
            )
            fit_key = f"{fit['kmin']:.2f}".replace(".", "p")
            fit_curves[fit_key] = fit
            df_low[f"fit_{fit_key}"] = fit["curve_all"][:first_n_bins]
            df_low[f"delta_{fit_key}"] = df_low["Pk"] - df_low[f"fit_{fit_key}"]
        plot_payload[branch_name] = {
            "spec": spec,
            "fit_results": fit_results,
            "fit_curves": fit_curves,
            "region_info": region_info,
            "psf_meta": psf_meta,
        }
        return pd.DataFrame(fit_results), df_low

    if "df_pk_kdiag_summary" in ns and "df_pk_kdiag_lowbins" in ns and "spec" in ns and "fit_results" in ns and "fit_curves" in ns and "region_info" in ns:
        df_current = ns["df_pk_kdiag_summary"].copy()
        df_current["branch"] = "current_mask_currentopd_detdist_ref"
        df_current["psf_ext"] = ns.get("psf_selected_ext", "UNKNOWN")
        df_current["opd_path"] = str(ns.get("local_wss_opd", ""))
        branch_rows.append(df_current)
        df_low = ns["df_pk_kdiag_lowbins"].copy()
        df_low["branch"] = "current_mask_currentopd_detdist_ref"
        lowbin_tables.append(df_low)
        plot_payload["current_mask_currentopd_detdist_ref"] = {
            "spec": ns["spec"],
            "fit_results": list(ns["fit_results"]),
            "fit_curves": dict(ns["fit_curves"]),
            "region_info": ns["region_info"],
            "psf_meta": {
                "selected_ext": ns.get("psf_selected_ext", "UNKNOWN"),
                "opd_path": str(ns.get("local_wss_opd", "")),
            },
        }
    else:
        ref = assets["current_reference"]
        df_ref, df_low_ref = _compute_branch_kdiag(
            ref["branch"],
            resid=ref["resid"],
            premask=ref["premask"],
            psf=ref["psf"],
            psf_meta=ref["psf_meta"],
        )
        branch_rows.append(df_ref)
        df_low_ref["branch"] = ref["branch"]
        lowbin_tables.append(df_low_ref)

    detsamp = assets.get("current_mask_detsamp")
    if detsamp is not None:
        df_det, df_low_det = _compute_branch_kdiag(
            detsamp["branch"],
            resid=detsamp["resid"],
            premask=detsamp["premask"],
            psf=detsamp["psf"],
            psf_meta=detsamp["psf_meta"],
        )
        branch_rows.append(df_det)
        df_low_det["branch"] = detsamp["branch"]
        lowbin_tables.append(df_low_det)

    return {
        "df_summary": pd.concat(branch_rows, ignore_index=True, sort=False),
        "df_lowbins": pd.concat(lowbin_tables, ignore_index=True, sort=False),
        "plot_payload": plot_payload,
    }


def plot_systematic_kdiag(bundle, zoom_max=None):
    import matplotlib.pyplot as plt

    payload = bundle["plot_payload"]
    if not payload:
        return

    zoom_max = 0.08 if zoom_max is None else float(zoom_max)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    styles = {
        "current_mask_currentopd_detdist_ref": "-",
        "current_mask_currentopd_detsamp": "--",
    }

    ref_key = "current_mask_currentopd_detdist_ref"
    ref_payload = payload.get(ref_key)
    if ref_payload is None:
        ref_key = next(iter(payload))
        ref_payload = payload[ref_key]

    spec_ref = ref_payload["spec"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(spec_ref["kP"], spec_ref["P"], "o", ms=4, color="k", alpha=0.75, label="P(k)")
    axes[1].plot(spec_ref["kP"], spec_ref["P"], "o", ms=4, color="k", alpha=0.75, label="P(k)")

    for branch_name, branch_payload in payload.items():
        line_style = styles.get(branch_name, "-.")
        for i, fit_row in enumerate(branch_payload["fit_results"]):
            fit_key = f"{fit_row['kmin']:.2f}".replace(".", "p")
            fit = branch_payload["fit_curves"][fit_key]
            color = colors[i % len(colors)]
            label = (
                f"{branch_name}: k=[{fit_row['kmin']:.2f}, {fit_row['kmax']:.2f}] "
                f"P0={fit_row['P0']:.3e}, mbar={fit_row['mbar_spec']:.3f}"
            )
            axes[0].plot(spec_ref["kP"], fit["curve_all"], color=color, lw=2.0, ls=line_style, label=label)
            pk_minus_p1 = spec_ref["P"] - fit_row["P1"]
            p0e = fit_row["P0"] * fit["E_all"]
            axes[1].plot(spec_ref["kP"], pk_minus_p1, "o", ms=3, color=color, alpha=0.35)
            axes[1].plot(spec_ref["kP"], p0e, color=color, lw=2.0, ls=line_style, label=label)

    for v in sorted({float(row["kmin"]) for branch_payload in payload.values() for row in branch_payload["fit_results"]}):
        axes[0].axvline(v, color="0.6", lw=0.8, ls="--", alpha=0.7)
        axes[1].axvline(v, color="0.6", lw=0.8, ls="--", alpha=0.7)

    axes[0].set_title("Systematic KDIAG: P(k) fits")
    axes[0].set_xlabel("k [cycles/pixel]")
    axes[0].set_ylabel("power")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Systematic KDIAG: P(k)-P1 vs P0E(k)")
    axes[1].set_xlabel("k [cycles/pixel]")
    axes[1].set_ylabel("power")
    axes[1].set_xlim(0.0, zoom_max)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.show()
