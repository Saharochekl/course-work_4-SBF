#!/usr/bin/env python3
# sbf_quickstart_ngc1380.py
# Вход: пути к i2d для F150W (SBF) и F090W (цвет). Выход: m_bar(F150W), цвет, окно k.

import argparse, numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from photutils.segmentation import detect_sources, deblend_sources
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
import stpsf                                    # PSF под JWST файл
from numpy.fft import rfft2, rfftfreq, fftshift

def load_i2d(path):
    hdul = fits.open(path, memmap=False)
    sci = hdul['SCI'].data.astype(float)
    hdr = hdul['SCI'].header
    wcs = WCS(hdul['SCI'].header)

    # valid pixels: finite SCI and, if present, positive WHT
    valid = np.isfinite(sci)
    if 'WHT' in hdul:
        try:
            wht = hdul['WHT'].data
            valid &= np.isfinite(wht) & (wht > 0)
        except Exception:
            pass

    # площадь пикселя в arcsec^2
    cd = np.abs(wcs.pixel_scale_matrix)
    pix_scale = (cd[0,0]*3600, cd[1,1]*3600)
    pix_area = np.abs(np.linalg.det(wcs.pixel_scale_matrix))*(3600**2)  # arcsec^2
    hdul.close()
    return sci, hdr, pix_area, valid

def sigma_clipped_bkg(img, box=128):
    h, w = img.shape
    bs = box
    tiles = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            tile = img[y:y+bs, x:x+bs]
            if np.any(np.isfinite(tile)):
                tiles.append(np.nanmedian(tile))
    return float(np.nanmedian(tiles)) if tiles else float(np.nanmedian(img))

def mask_sources(img, nsigma=2.0, npixels=20, do_deblend=True):
    """Build a boolean source mask. Nan-safe. Deblend is optional (needs scikit-image)."""
    # If everything is NaN, return empty mask
    if not np.any(np.isfinite(img)):
        return np.zeros_like(img, bool)

    # Nan-safe smoothing for thresholding
    ker = Gaussian2DKernel(7.0)  # a bit wider kernel to bridge big NaN holes
    sm = interpolate_replace_nans(img, ker)

    # Robust stats; sigma_clipped_stats tolerates NaNs
    bkg, med, rms = sigma_clipped_stats(sm, sigma=3.0)
    thr = bkg + nsigma * max(rms, 1e-12)  # guard against zero/NaN rms

    segm = detect_sources(sm, threshold=thr, npixels=npixels)
    if segm is None:
        return np.zeros_like(img, bool)

    if do_deblend:
        try:
            # deblend_sources imports skimage internally; skip if missing
            segm = deblend_sources(sm, segm, npixels=npixels, nlevels=16, contrast=0.001)
        except Exception:
            pass

    # drop the largest connected regions (likely the target galaxy), keep compact sources
    labels = segm.labels
    counts = np.bincount(segm.data.ravel(), minlength=labels.max()+1)
    # ignore background label 0
    counts[0] = 0
    if counts.max() > 0:
        # zero out the single largest label
        drop = counts.argmax()
        segm.remove_labels(drop)
        # if second-largest is still huge (covers >1% of image), drop it too
        second = counts.argsort()[-2] if counts.size > 2 else None
        if second is not None and counts[second] > 0.01 * img.size:
            segm.remove_labels(second)

    return segm.make_source_mask(size=5)

def pick_start_radius(mask, valid, x0, y0, max_probe=80, min_frac=0.20, min_pix=200):
    """Pick the smallest SMA whose 1px ring has enough usable pixels."""
    ny, nx = mask.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)
    blocked = mask | (~valid)  # blocked pixels are either masked or invalid by WHT/NaN

    for r in range(3, max_probe):
        ring = (rr >= r - 0.5) & (rr < r + 0.5)
        total = int(ring.sum())
        if total < 1:
            continue
        good = int((ring & (~blocked)).sum())
        if good >= min_pix and good/total >= min_frac:
            return r

    # fallback: pick ring with maximum good pixels
    best_r, best_good = None, -1
    for r in range(3, max_probe):
        ring = (rr >= r - 0.5) & (rr < r + 0.5)
        good = int((ring & (~blocked)).sum())
        if good > best_good:
            best_good, best_r = good, r
    return max(3, best_r or 6)

def guess_center(img, valid):
    """Rough center guess from the brightest smoothed pixel (NaN/valid aware)."""
    sm = interpolate_replace_nans(img, Gaussian2DKernel(7.0))
    # ignore invalid pixels when choosing the maximum
    sm = np.where(valid, sm, np.nan)
    # if everything is NaN, fall back to image center
    if not np.isfinite(sm).any():
        ny, nx = img.shape
        return nx/2.0, ny/2.0
    y, x = np.unravel_index(np.nanargmax(sm), sm.shape)
    return float(x), float(y)

def cutout_connected_valid(img, valid, x0, y0, pad=24):
    """Return a cutout around the connected valid region that contains (x0,y0).
    Returns: (img_cut, valid_cut, (x0_cut, y0_cut), (x1, x2, y1, y2))
    where [y1:y2, x1:x2] are slice bounds in the original image.
    """
    ny, nx = valid.shape
    y0i = int(round(y0)); x0i = int(round(x0))
    y0i = np.clip(y0i, 0, ny - 1); x0i = np.clip(x0i, 0, nx - 1)

    # If the guessed center is invalid, jump to nearest valid pixel
    if not valid[y0i, x0i]:
        vv = np.argwhere(valid)
        if vv.size == 0:
            # hopeless, return the whole image
            return img, valid, (x0, y0), (0, nx, 0, ny)
        d2 = (vv[:, 0] - y0i) ** 2 + (vv[:, 1] - x0i) ** 2
        k = int(np.argmin(d2))
        y0i, x0i = int(vv[k, 0]), int(vv[k, 1])
        y0, x0 = float(y0i), float(x0i)

    try:
        from scipy.ndimage import label
        lab, _ = label(valid)
        lab_id = int(lab[y0i, x0i])
        comp = (lab == lab_id)
        ys, xs = np.nonzero(comp)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
    except Exception:
        # Fallback: grow to nearest seam in four directions
        y1 = y0i
        while y1 > 0 and valid[y1 - 1, x0i]:
            y1 -= 1
        y2 = y0i
        while y2 < ny - 1 and valid[y2 + 1, x0i]:
            y2 += 1
        x1 = x0i
        while x1 > 0 and valid[y0i, x1 - 1]:
            x1 -= 1
        x2 = x0i
        while x2 < nx - 1 and valid[y0i, x2 + 1]:
            x2 += 1

    # Pad but stay inside
    y1 = max(0, y1 - pad); y2 = min(ny, y2 + pad)
    x1 = max(0, x1 - pad); x2 = min(nx, x2 + pad)

    img_c = img[y1:y2, x1:x2]
    valid_c = valid[y1:y2, x1:x2]
    x0_c = x0 - x1
    y0_c = y0 - y1
    return img_c, valid_c, (x0_c, y0_c), (x1, x2, y1, y2)


def seam_limited_maxsma(valid, x0, y0):
    """Max SMA before hitting invalid pixels (seams)."""
    ny, nx = valid.shape
    y0i = int(round(y0)); x0i = int(round(x0))
    y0i = np.clip(y0i, 0, ny - 1); x0i = np.clip(x0i, 0, nx - 1)
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(valid)
        return max(5, int(dist[y0i, x0i]) - 2)
    except Exception:
        # crude fallback: grow until ring touches invalid
        yy, xx = np.ogrid[:ny, :nx]
        rr = np.hypot(yy - y0, xx - x0)
        for r in range(5, min(ny, nx)):
            ring = (rr >= r - 0.5) & (rr < r + 0.5)
            if np.any(ring & (~valid)):
                return max(5, r - 2)
        return max(5, int(min(ny, nx) / 2) - 2)

def subtract_galaxy(img, mask, valid):
    """Fit elliptical isophotes and build a smooth model (NaN/mask aware).
    Подбираем масштаб яркости, чтобы избежать 'No meaningful fit was possible'.
    """
    # 1) Центр по данным (не по центру кадра)
    #x0, y0 = guess_center(img, valid)
    x0, y0 = 6306.0, 2730.0


    img_c, valid_c, (x0, y0), (x1, x2, y1, y2) = cutout_connected_valid(img, valid, x0, y0, pad=24)
    mask_c = mask[y1:y2, x1:x2]

    finite_c = np.isfinite(img_c)
    print(f"[ISO-DBG] img_c shape={img_c.shape}")
    print(f"[ISO-DBG] finite={finite_c.sum()}, valid={valid_c.sum()}, mask_c={mask_c.sum()}")
    print(f"[ISO-DBG] min/max img_c (finite): {np.nanmin(img_c):.3g}/{np.nanmax(img_c):.3g}")

    # 2) Стартовый радиус по реально доступным пикселям
    r0 = int(pick_start_radius(mask_c, valid_c, x0, y0, max_probe=200, min_frac=0.15, min_pix=150))
    print(f"[ISO-DBG] start center=({x0:.1f},{y0:.1f}), r0={r0}")

    # Немного размаскируем центр, чтобы было от чего стартовать
    # yy, xx = np.ogrid[:img_c.shape[0], :img_c.shape[1]]
    # center = (yy - y0)**2 + (xx - x0)**2 <= (r0 * r0)
    # mask2 = mask_c.copy()
    # mask2[center] = False

    ny, nx = img_c.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)

    # маска, которая реально блокирует пиксели для фиттинга
    mask2 = mask_c.copy()
    mask2[(yy - y0) ** 2 + (xx - x0) ** 2 <= (r0 * r0)] = False  # как у тебя



    # 3) Перебор масштабов яркости
    finite = np.isfinite(img_c)
    s = float(np.nanmedian(np.abs(img_c[finite]))) if finite.any() else 0.0
    if not np.isfinite(s) or s <= 0:
        s = 1e-12
    base_exp = int(np.clip(np.ceil(-np.log10(s)), 0, 18))
    exps = [e for e in sorted(set([base_exp-2, base_exp-1, base_exp, base_exp+1, base_exp+2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])) if 0 <= e <= 12]

    best = None
    last_err = None
    maxsma = seam_limited_maxsma(valid_c, x0, y0)
    print(f"[ISO-DBG] seam_limited_maxsma={maxsma}")


    # --- Fallback на случай бредового maxsma ---
    # допустимый максимум по размеру вырезки (чтобы не вылезать за края)
    max_possible = int(0.45 * min(img_c.shape))  # 45% от меньшего размера

    # если seam_limited_maxsma слишком маленький или кривой – переопределяем
    if (not np.isfinite(maxsma)) or (maxsma <= r0 + 5):
        print(f"[ISO-DBG] seam_limited_maxsma={maxsma} <= r0+5, "
              f"используем fallback")
        maxsma = max(r0 + 10, max_possible)
    else:
        # на всякий ограничим сверху разумным пределом
        maxsma = min(maxsma, max_possible)

    print(f"[ISO-DBG] final maxsma={maxsma}")


    for test_r in [r0 - 10, r0 - 5, r0, r0 + 5, r0 + 10]:
        if test_r <= 0 or test_r >= maxsma:
            continue
        ring = (rr >= test_r - 0.5) & (rr < test_r + 0.5)
        total = int(ring.sum())
        good = int((ring & valid_c & (~mask2)).sum())
        frac = good / max(total, 1)
        print(f"[ISO-DBG] r={test_r}: total={total}, good={good}, frac={frac:.3f}")

    for exp in exps:
        scale = 10.0 ** exp
        img_scaled = img_c * scale

        img_ma = np.ma.masked_invalid(img_scaled, copy=False)
        img_ma.mask |= mask2

        geom = EllipseGeometry(x0=x0, y0=y0, sma=r0, eps=0.2, pa=0.0)
        print(f"[ISO-DBG] try scale=1e{exp}")
        print(f"[ISO-DBG] img_scaled median={np.median(img_scaled[finite]):.3g}, "
              f"min={np.min(img_scaled[finite]):.3g}, max={np.max(img_scaled[finite]):.3g}")
        print(f"[ISO-DBG] masked fraction={img_ma.mask.mean():.3f}")

        e = Ellipse(img_ma, geometry=geom, threshold=0.05)  # включаем центрировщик

        try:
            isolist = e.fit_image(
                sma0=r0,
                minsma=max(2, r0 - 3),
                maxsma=maxsma,
                step=0.18,
                sclip=3.0,
                nclip=2,
                fflag=0.7,
                fix_center=False,
                integrmode="median",
            )
        except Exception as e_fit:
            print(f"[ISO] scale=1e{exp}: {e_fit.__class__.__name__}: {e_fit}")
            last_err = e_fit
            continue

        if len(isolist) == 0:
            print(f"[ISO] scale=1e{exp}: empty isolist")
            continue

        print(f"[ISO] scale=1e{exp}: OK, n_isophotes={len(isolist)}")
        best = (isolist, scale)
        break

    if best is None:
        print(f"[ISO] Ellipse fit failed for all scales; base_exp={base_exp}, r0={r0}. "
              f"Last error: {last_err}")
        print("[ISO] FALLBACK: using pure Gaussian smoothing model (no isophotes).")

        from astropy.convolution import convolve, Gaussian2DKernel

        # сглаживаем вырезку
        img_c_smooth = convolve(
            img_c,
            Gaussian2DKernel(10.0),  # можно чуть крупнее, чем в <5 изофот
            normalize_kernel=True,
        )

        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()

        model_full[y1:y2, x1:x2] = img_c_smooth
        resid_full[y1:y2, x1:x2] = img_c - img_c_smooth
        resid_full[mask] = np.nan
        return resid_full, model_full

    print(f"[ISO] maxsma={maxsma}, n_isophotes={len(isolist)}")

    isolist, scale = best
    if len(isolist) < 5:
        print(f"[ISO] too few isophotes ({len(isolist)}), "
              "fallback to simple smoothing model")

        # грубый fallback: сгладить галактику фильтром
        from astropy.convolution import convolve, Gaussian2DKernel

        img_c_smooth = convolve(img_c, Gaussian2DKernel(5.0), normalize_kernel=True)

        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()

        model_full[y1:y2, x1:x2] = img_c_smooth
        resid_full[y1:y2, x1:x2] = img_c - img_c_smooth
        resid_full[mask] = np.nan
        return resid_full, model_full

    model_scaled = build_ellipse_model(img_c.shape, isolist, fill=0.0)
    model_sub = model_scaled / scale

    model_full = np.full_like(img, np.nan)
    resid_full = img.copy()
    model_full[y1:y2, x1:x2] = model_sub
    resid_full[y1:y2, x1:x2] = img_c - model_sub
    resid_full[mask] = np.nan
    return resid_full, model_full

def build_psf_for_file(fits_path, size=129):
    # stpsf сам читает header и настраивает инструмент
    sim = stpsf.setup_sim_to_match_file(fits_path)
    psf = sim.calc_psf(nlambda=7, fov_pixels=size)

    # на всякий лог, если хочется убедиться:
    # print("DEBUG psf type:", type(psf))

    # 1) Разруливаем тип
    if isinstance(psf, fits.HDUList):
        # классический случай для webbpsf/stpsf – вернули HDUList
        data = psf[0].data
    elif isinstance(psf, fits.PrimaryHDU):
        data = psf.data
    else:
        # вдруг calc_psf уже вернул чистый numpy-массив
        data = psf

    arr = np.array(data, dtype=float)

    # 2) Если это куб (λ, y, x) – схлопываем по λ
    if arr.ndim == 3:
        arr = arr.sum(axis=0)

    # 3) Чистим и нормируем
    arr = np.nan_to_num(arr, nan=0.0)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"PSF из {fits_path} имеет нулевую/невалидную сумму, shape={arr.shape}")

    arr /= s
    return arr

def radial_power(img, mask):
    # k-спектр по остаткам с маской: FFT(mask*img); азим. усреднение
    data = np.zeros_like(img)
    tmp = np.nan_to_num(img, nan=0.0)
    data[~mask] = tmp[~mask]
    F = np.abs(rfft2(data))**2
    # радиальная выкладка (по частоте пик^-1)
    ny, nx = data.shape
    ky = np.fft.fftfreq(ny)
    kx = rfftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    kbins = np.linspace(0, kr.max(), 80)
    P = np.zeros(len(kbins)-1); kc = np.zeros_like(P)
    for i in range(len(P)):
        sel = (kr>=kbins[i]) & (kr<kbins[i+1])
        vals = F[sel]
        if vals.size>10:
            P[i] = np.median(vals)
            kc[i] = 0.5*(kbins[i]+kbins[i+1])
        else:
            P[i] = np.nan; kc[i]=np.nan
    m = np.isfinite(P) & (kc>0)
    return kc[m], P[m]

def fit_P0(Pk, Ek, kmin=0.02, kmax=0.25):
    sel = (Pk[0]>=kmin) & (Pk[0]<=kmax)
    x = Ek[1][sel]   # E(k)
    y = Pk[1][sel]   # P(k)
    A = np.vstack([x, np.ones_like(x)]).T
    P0, P1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(P0), float(P1), (kmin, kmax)

def mjysr_to_ab_zp(pix_area_arcsec2):
    # m_AB для 1 (MJy/sr) на 1 пиксель
    jy_per_pix = 2.350443e-5 * pix_area_arcsec2  # 1 MJy/sr -> Jy/arcsec^2
    m_ab = -2.5*np.log10(jy_per_pix/3631.0)
    return float(m_ab)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f150w", required=True, help="i2d F150W")
    ap.add_argument("--f090w", help="i2d F090W (для цвета), optional")
    ap.add_argument("--psfref", help="i2d с полным JWST header для расчёта PSF") # Костыль для кропа
    ap.add_argument("--dry", action="store_true", help="только оценить фон/маску")
    args = ap.parse_args()

    img_f150, hdr150, area, valid150 = load_i2d(args.f150w)
    # второй кадр грузим только если он передан
    if args.f090w is not None:
        img_f090, hdr090, _, valid090 = load_i2d(args.f090w)
    else:
        img_f090 = None

    bkg = sigma_clipped_bkg(img_f150)
    img = img_f150 - bkg

    mask_src = mask_sources(img, nsigma=2.5, npixels=25, do_deblend=not args.dry)
    mask = (~valid150) | mask_src
    if args.dry:
        finite = int(valid150.sum())
        total = int(img.size)
        masked = int(mask.sum())
        print(f"[DRY] shape={img.shape}, finite={finite}/{total} ({100*finite/total:.1f}%)")
        print(f"[DRY] bkg≈{bkg:.6g} (MJy/sr units), mask_coverage={100*masked/total:.2f}%")
        return

    resid, model = subtract_galaxy(img, mask, valid150)

    # PSF и его E(k)
    psf_file = args.psfref if args.psfref is not None else args.f150w
    psf = build_psf_for_file(psf_file, size=129)
    Pk = radial_power(resid, mask)
    Ek = radial_power(psf, np.zeros_like(psf, bool))

    P0, P1, kwin = fit_P0(Pk, Ek, kmin=0.03, kmax=0.20)  # окно по стабильности подберешь позже

    # первый проход: Pr ~ 0
    Pf = max(P0, 0.0)

    # величина колебаний в абс. системе единиц изображения
    mbar = -2.5*np.log10(Pf) + mjysr_to_ab_zp(area)

    # цвет (в AB маг/arcsec^2 на тех же масках и изофотах): возьмем медиану по модели
    # только если F090W есть
    if img_f090 is not None:
        color = -2.5*np.log10(
            np.nanmedian(img_f090[~mask]) / np.nanmedian(img_f150[~mask])
        )
        print(f"m̄(F150W) = {mbar:.3f} mag   (окно k={kwin[0]:.02f}..{kwin[1]:.02f} pix^-1)")
        print(f"(F090W − F150W) ≈ {color:.3f} mag (грубая оценка)")
    else:
        print(f"m̄(F150W) = {mbar:.3f} mag   (окно k={kwin[0]:.02f}..{kwin[1]:.02f} pix^-1)")
        print("[INFO] Второй фильтр не задан, цвет не считается.")

if __name__ == "__main__":
    main()