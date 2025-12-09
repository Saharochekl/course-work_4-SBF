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
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans



def build_radial_profile(img_c, valid_c, mask_c, x0, y0, dr=1.0, min_pix=200):
    ny, nx = img_c.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)

    ok = valid_c & (~mask_c) & np.isfinite(img_c)

    if ok.sum() < 1000:
        raise RuntimeError("слишком мало валидных пикселей")

    r_max = float(rr[ok].max())
    nbins = int(r_max // dr)

    radii, intens = [], []
    for i in range(nbins):
        r_in, r_out = i*dr, (i+1)*dr
        ring = (rr>=r_in) & (rr<r_out) & ok
        n = int(ring.sum())
        if n < min_pix:
            continue
        vals = img_c[ring]
        if not np.isfinite(vals).any():
            continue
        radii.append(0.5*(r_in+r_out))
        intens.append(float(np.nanmedian(vals)))

    return np.array(radii), np.array(intens)


def inpaint_for_isophote(img_c, valid_c):
    """
    Заполнение больших NaN / дыр для задачи аппроксимации изофот.
    ВАЖНО: это используется ТОЛЬКО для подгонки гладкой модели галактики.
    Для SBF-статистики по остаткам продолжаем использовать исходный img_c.
    """


    arr = img_c.copy()
    # всё, что заведомо невалидно, считаем NaN
    arr[~valid_c] = np.nan

    nan0 = int(np.isnan(arr).sum())
    total = arr.size
    print(f"[INPAINT] start: {nan0} NaNs из {total} пикселей ({100*nan0/total:.2f}%) в вырезке")

    if nan0 == 0:
        return arr

    # последовательно увеличиваем размер ядра, чтобы перешить большие дыры
    for fwhm in (7.0, 15.0, 25.0, 40.0, 60.0, 100.0, 150.0, 200.0):
        ker = Gaussian2DKernel(fwhm)
        arr = interpolate_replace_nans(arr, ker)
        nan_now = int(np.isnan(arr).sum())
        print(f"[INPAINT] после FWHM={fwhm}: {nan_now} NaNs "
              f"({100*nan_now/total:.2f}%)")
        if nan_now == 0:
            break

    return arr

def fill_nan_with_radial(img_c, valid_c, mask_c, x0, y0, radii, intens):
    ny, nx = img_c.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)

    ok = valid_c & (~mask_c) & np.isfinite(img_c)
    img_iso = img_c.copy()

    # где нет данных – заливаем профилем
    bad = ~ok
    img_iso[bad] = np.interp(rr[bad], radii, intens,
                             left=intens[0], right=intens[-1])
    return img_iso



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

def cutout_box(img, valid, x0, y0, half_size):
    ny, nx = img.shape
    x1 = max(0, int(x0 - half_size))
    x2 = min(nx, int(x0 + half_size))
    y1 = max(0, int(y0 - half_size))
    y2 = min(ny, int(y0 + half_size))
    img_c   = img[y1:y2, x1:x2]
    valid_c = valid[y1:y2, x1:x2]
    return img_c, valid_c, (x0 - x1, y0 - y1), (x1, x2, y1, y2)

def subtract_galaxy(img, mask, valid):
    """
    Строим гладкую модель галактики через радиальный профиль вокруг центра
    и вычитаем её. Без photutils.Ellipse, без изофот.

    На вход:
      img   – кадр F150W (фон уже вычтен)
      mask  – bool-маска (True = пиксель не использовать)
      valid – bool-маска валидных пикселей (WHT>0, не NaN)
    """
    # 1) Центр. Можно вернуть guess_center, можно оставить руками:
    x0, y0 = guess_center(img, valid)
    #x0, y0 = 6306.0, 2730.0
    print(f"[RADIAL] use center=({x0:.1f}, {y0:.1f})")

    ny, nx = img.shape

    # # если центр вообще не попадает в кадр (кроп), переопределяем его по данным
    # if not (0 <= x0 < nx and 0 <= y0 < ny):
    #     x0, y0 = guess_center(img, valid)
    #     print(f"[CENTER] global center out of frame, "
    #           f"use guess_center → ({x0:.1f}, {y0:.1f})")

    # 2) Вырезка по связной валидной области вокруг центра
    img_c, valid_c, (x0, y0), (x1, x2, y1, y2) = cutout_box(
        img, valid, x0, y0, half_size=3000  # подобрать под размер галактики
    )
    mask_c = mask[y1:y2, x1:x2]

    finite_c = np.isfinite(img_c)
    print(f"[RADIAL] img_c shape={img_c.shape}, "
          f"finite={finite_c.sum()}, valid={valid_c.sum()}, "
          f"mask_c={mask_c.sum()}")

    # 3) Радиальный профиль по сырому img_c
    radii0, intens0 = build_radial_profile(img_c, valid_c, mask_c, x0, y0)
    print(f"[RADIAL] primary radial bins: {len(radii0)}")

    # ny, nx = img_c.shape
    # yy, xx = np.ogrid[:ny, :nx]
    # rr = np.hypot(yy - y0, xx - x0)
    # prof_1d = np.interp(rr.ravel(), radii, intens, left=intens[0], right=intens[-1]).reshape(rr.shape)
    #
    # model_full = np.full_like(img, np.nan)
    # resid_full = img.copy()
    # model_full[y1:y2, x1:x2] = prof_1d
    # resid_full[y1:y2, x1:x2] = img_c - prof_1d
    # resid_full[mask] = np.nan

    # print(f"[RADIAL] img_c shape={img_c.shape}")
    # print(f"[RADIAL] finite={finite_c.sum()}, valid={valid_c.sum()}, "
    #       f"mask_c={mask_c.sum()}")

    # 3) inpaint только по valid_c, как и раньше

    # 4) Залечиваем NaN по этому профилю
    img_iso = fill_nan_with_radial(img_c, valid_c, mask_c, x0, y0, radii0, intens0)

    # 5) Строим ОКОНЧАТЕЛЬНЫЙ радиальный профиль по img_iso
    ny, nx = img_iso.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)

    bad = (~valid_c) | mask_c | (~np.isfinite(img_iso))
    ok = ~bad
    if ok.sum() < 1000:
        raise RuntimeError("[RADIAL] слишком мало валидных пикселей в вырезке")


    r_max = float(rr[ok].max())
    dr = 1.0  # шаг по радиусу в пикселях
    nbins = int(r_max // dr)
    min_pix = 200

    radii, intens = [], []
    print(f"[RADIAL] r_max≈{r_max:.1f}, nbins≈{nbins}")

    for i in range(nbins):
        r_in = i * dr
        r_out = r_in + dr
        ring = (rr >= r_in) & (rr < r_out) & ok
        n = int(ring.sum())
        if n < min_pix:
            continue
        vals = img_iso[ring]
        if not np.isfinite(vals).any():
            continue
        radii.append(0.5 * (r_in + r_out))
        intens.append(float(np.nanmedian(vals)))

    radii = np.array(radii)
    intens = np.array(intens)
    print(f"[RADIAL] good radial bins: {len(radii)}")

    if len(radii) < 5:
        # fallback: гауссовое сглаживание
        from astropy.convolution import convolve, Gaussian2DKernel
        img_c_smooth = convolve(img_c, Gaussian2DKernel(10.0),
                                normalize_kernel=True)

        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()
        model_full[y1:y2, x1:x2] = img_c_smooth
        resid_full[y1:y2, x1:x2] = img_c - img_c_smooth
        resid_full[mask] = np.nan
        return resid_full, model_full

    # 6) ОДИН раз строим модель и вычитаем
    prof_1d = np.interp(rr.ravel(), radii, intens,
                        left=intens[0], right=intens[-1]).reshape(rr.shape)

    model_full = np.full_like(img, np.nan)
    resid_full = img.copy()
    model_full[y1:y2, x1:x2] = prof_1d
    resid_full[y1:y2, x1:x2] = img_c - prof_1d
    resid_full[mask] = np.nan

    return resid_full, model_full

def build_psf_for_file(fits_path, size=129):
    # stpsf сам читает header и настраивает инструмент
    sim = stpsf.setup_sim_to_match_file(fits_path)
    psf = sim.calc_psf(nlambda=7, fov_pixels=size)

    # на всякий лог, если хочется убедиться:
    print("DEBUG psf type:", type(psf))

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
    data = np.zeros_like(img)
    tmp = np.nan_to_num(img, nan=0.0)
    data[~mask] = tmp[~mask]

    use = (~mask) & np.isfinite(data)
    n_pix = int(use.sum())
    if n_pix == 0:
        raise RuntimeError("radial_power: no unmasked pixels")

    # вычитаем среднее, чтобы не тянуть DC-компонент
    mean = float(np.nanmean(data[use]))
    data[use] -= mean

    # FFT
    F = rfft2(data)
    power2d = np.abs(F)**2

    # дисперсия по пикселям
    var_direct = float(np.nanvar(data[use]))

    # "по-Parseval'у" из FFT
    var_fft = power2d.sum() / (n_pix**2)
    if var_fft > 0 and np.isfinite(var_fft):
        scale = var_direct / var_fft
    else:
        scale = 1.0

    # окончательная 2D-мощность
    P2d = power2d * scale / (n_pix**2)

    ny, nx = data.shape
    ky = np.fft.fftfreq(ny)
    kx = rfftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)

    kbins = np.linspace(0, kr.max(), 80)
    P = np.zeros(len(kbins) - 1, float)
    kc = np.zeros_like(P)

    for i in range(len(P)):
        sel = (kr >= kbins[i]) & (kr < kbins[i+1])
        vals = P2d[sel]
        if vals.size > 10:
            P[i] = np.nanmedian(vals)
            kc[i] = 0.5 * (kbins[i] + kbins[i+1])
        else:
            P[i] = np.nan
            kc[i] = np.nan

    m = np.isfinite(P) & (kc > 0)
    return kc[m], P[m]


def radial_power_psf(psf_small, img_shape):
    ny, nx = img_shape
    big = np.zeros((ny, nx), float)

    py, px = psf_small.shape
    y0 = ny // 2 - py // 2
    x0 = nx // 2 - px // 2
    big[y0:y0+py, x0:x0+px] = psf_small

    mask_big = np.zeros_like(big, bool)
    return radial_power(big, mask_big)

def fit_P0(Pk, Ek, kmin=0.03, kmax=0.40):
    """
    Линейный фит P(k) = P0 * E(k) + P1 с проверками адекватности.

    Pk, Ek: кортежи (k, P(k)) и (k, E(k))
    Возвращает (P0, P1, (kmin, kmax)) или (nan, nan, (kmin, kmax)), если фит не доверяем.
    """
    kP, P = Pk
    kE, E = Ek

    kP = np.asarray(kP, float)
    P = np.asarray(P, float)
    kE = np.asarray(kE, float)
    E = np.asarray(E, float)

    sel = (kP >= kmin) & (kP <= kmax)
    n_sel = int(sel.sum())
    if n_sel < 10:
        print(f"[FIT] мало точек в окне k={kmin:.3f}..{kmax:.3f}, N={n_sel}")
        return np.nan, np.nan, (kmin, kmax)

    # интерполируем E(k) в те же k, где есть P(k)
    E_int = np.interp(kP[sel], kE, E, left=np.nan, right=np.nan)

    x = E_int
    y = P[sel]

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]

    if x.size < 10:
        print(f"[FIT] после маскировки валидных точек < 10, N={x.size}")
        return np.nan, np.nan, (kmin, kmax)

    # корреляция между E и P
    try:
        corr = np.corrcoef(x, y)[0, 1]
    except Exception:
        corr = np.nan

    print(f"[FIT] window k={kmin:.3f}..{kmax:.3f}, N={x.size}, corr(E,P)≈{corr:.3f}")
    if not np.isfinite(corr) or abs(corr) < 0.3:
        print("[FIT] слабая корреляция E(k) и P(k) → фит недостоверен")
        return np.nan, np.nan, (kmin, kmax)

    A = np.vstack([x, np.ones_like(x)]).T
    try:
        P0, P1 = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception as e:
        print(f"[FIT] lstsq failed: {e}")
        return np.nan, np.nan, (kmin, kmax)

    if (not np.isfinite(P0)) or (not np.isfinite(P1)):
        print(f"[FIT] невалидные коэффициенты: P0={P0}, P1={P1}")
        return np.nan, np.nan, (kmin, kmax)

    if P0 <= 0 or (P0 + P1) <= 0:
        print(f"[FIT] нефизичные значения: P0={P0}, P0+P1={P0+P1}")
        return np.nan, np.nan, (kmin, kmax)

    frac = P0 / (P0 + P1)
    print(f"[FIT] P0={P0:.3e}, P1={P1:.3e}, frac={frac:.3f}")
    if abs(P1) > 5.0 * P0:
        print(f"[FIT] |P1|={abs(P1):.3e} >> P0={P0:.3e} → фит ненадёжен")
        return np.nan, np.nan, (kmin, kmax)

    return float(P0), float(P1), (kmin, kmax)
def check_sbf_region(resid, mask_sbf, label="[SBF]"):
    """
    Проверка адекватности области, в которой меряем SBF:
    - достаточно пикселей
    - разумный динамический диапазон и дисперсия.
    """
    use = (~mask_sbf) & np.isfinite(resid)
    n = int(use.sum())
    if n < 5000:
        print(f"{label} слишком мало пикселей в аннулусе: N={n}")
        return False

    vals = resid[use]
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    med = float(np.nanmedian(vals))
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    dyn = vmax - vmin

    print(
        f"{label} N={n}, min={vmin:.3e}, med={med:.3e}, "
        f"max={vmax:.3e}, mean={mean:.3e}, std={std:.3e}, dyn={dyn:.3e}"
    )

    # Порог по динамическому диапазону: на полном кадре у тебя ~1e3,
    # на нормальном кропе ~1e-2. Здесь оставляем запас.
    if dyn > 0.2:
        print(f"{label} динамический диапазон {dyn:.3e} слишком велик → область грязная")
    #     return False

    # Слишком маленький std означает, что сигнал ниже шума/квантизации
    if std < 1e-7:
        print(f"{label} std={std:.3e} слишком мал → сигнал неотличим от нуля")
    #     return False

    return True


# === Диагностический прогон по радиусам SBF-аннулуса ===
def scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf, label_prefix="[SCAN]"):
    """
    Диагностический прогон по радиусам SBF-аннулуса.
    Для каждой пары (rin, rout) печатает Imean, std, dyn, var_sbf, Pf, m̄.

    Радиусы подбираются автоматически под размер кадра:
    - для больших полей (полный кадр): rin~100..800, ширина ~100 пикс;
    - для меньших (кроп): rin~30..(Rmax-20), ширина ~60 пикс.
    """
    ny, nx = resid.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0_sbf, xx - x0_sbf)

    # максимальный радиус, который ещё помещается внутри кадра
    r_max_img = min(
        np.hypot(y0_sbf, x0_sbf),
        np.hypot(y0_sbf, nx - 1 - x0_sbf),
        np.hypot(ny - 1 - y0_sbf, x0_sbf),
        np.hypot(ny - 1 - y0_sbf, nx - 1 - x0_sbf),
    )

    if not np.isfinite(r_max_img) or r_max_img <= 60.0:
        print(f"{label_prefix} r_max_img≈{r_max_img:.1f} px, сканировать радиусы бессмысленно")
        return

    # крупное поле (полный кадр JWST) vs кроп
    if r_max_img > 800.0:
        r_min, r_max, step, width = 100.0, min(800.0, r_max_img - 50.0), 50.0, 100.0
    else:
        r_min, r_max, step, width = 30.0, max(60.0, r_max_img - 20.0), 20.0, 60.0

    if r_max <= r_min:
        print(f"{label_prefix} r_max_img≈{r_max_img:.1f} px, рабочий диапазон радиусов не нашёлся")
        return

    print(f"{label_prefix} scan radii: rin∈[{r_min:.0f},{r_max:.0f}], width≈{width:.0f} px")

    for rin in np.arange(r_min, r_max, step):
        rout = rin + width
        if rout > r_max_img:
            rout = r_max_img

        annulus = (rr >= rin) & (rr <= rout)
        if not annulus.any():
            continue

        mask_sbf = mask | (~annulus)

        # пиксели для флуктуаций
        use = (~mask_sbf) & np.isfinite(resid)
        n = int(use.sum())
        if n < 5000:
            continue

        vals = resid[use]
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        std = float(np.nanstd(vals))
        dyn = vmax - vmin

        # дисперсия по усечённому распределению (обрезаем хвосты)
        lo, hi = np.nanpercentile(vals, [5, 95])
        core_mask = (vals >= lo) & (vals <= hi)
        if core_mask.sum() < 100:
            continue
        var_sbf = float(np.nanvar(vals[core_mask]))

        # средняя яркость галактики в той же области (по модели)
        use_I = (~mask_sbf) & np.isfinite(model)
        n_I = int(use_I.sum())
        if n_I < 5000:
            continue
        Imean = float(np.nanmean(model[use_I]))

        if (not np.isfinite(Imean)) or (Imean <= 0.0):
            continue
        if (not np.isfinite(var_sbf)) or (var_sbf <= 0.0):
            continue

        Pf = var_sbf / Imean
        if (not np.isfinite(Pf)) or (Pf <= 0.0):
            continue

        mbar = -2.5 * np.log10(Pf) + mjysr_to_ab_zp(area)

        print(
            f"{label_prefix} rin={rin:5.1f}, rout={rout:5.1f}, "
            f"N={n:7d}, Imean={Imean:.3e}, std={std:.3e}, dyn={dyn:.3e}, "
            f"var_sbf={var_sbf:.3e}, Pf={Pf:.3e}, m̄={mbar:.3f}"
        )

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

    # === Сохраняем модель и остатки в FITS, чтобы результат был НЕ "виртуальный" ===
    base = Path(args.f150w)
    stem = base.stem  # jw03055-o001_t001_nircam_clear-f150w_i2d или i2d_crop1
    out_dir = base.parent

    model_path = out_dir / f"{stem}_sbf_model.fits"
    resid_path = out_dir / f"{stem}_sbf_resid.fits"

    # модель и остатки в тех же единицах, что img (MJy/sr после вычитания фона)
    fits.writeto(model_path, model, hdr150, overwrite=True)
    fits.writeto(resid_path, resid, hdr150, overwrite=True)

    print(f"[OUT] model → {model_path}")
    print(f"[OUT] resid → {resid_path}")

    finite_resid = np.isfinite(resid)
    if finite_resid.sum() == 0:
        print("[CHK] resid: вообще нет валидных пикселей, что-то сломали")
    else:
        print("[CHK] resid: finite={:d}, min={:.3e}, med={:.3e}, max={:.3e}".format(
            int(finite_resid.sum()),
            float(np.nanmin(resid)),
            float(np.nanmedian(resid)),
            float(np.nanmax(resid))
        ))

    # DEBUG: локальный mean / RMS флуктуаций в кольце вокруг центра
    x0_dbg, y0_dbg = 6306.0, 2730.0  # центр галактики в полных координатах кадра
    yy_dbg, xx_dbg = np.ogrid[:resid.shape[0], :resid.shape[1]]
    rr_dbg = np.hypot(yy_dbg - y0_dbg, xx_dbg - x0_dbg)

    R1, R2 = 30.0, 40.0  # внутренний/внешний радиусы кольца, в пикселях
    ring = (rr_dbg >= R1) & (rr_dbg < R2) & (~mask)

    vals = resid[ring]
    if vals.size > 0:
        mean_I = float(np.nanmean(vals))
        rms_I  = float(np.nanstd(vals))
        print(f"[RMS-DBG] ring {R1:.1f}-{R2:.1f} px: "
              f"mean={mean_I:.4e}, rms={rms_I:.4e}, N={vals.size}")
    else:
        print("[RMS-DBG] ring empty, no pixels.")


    # SBF-аннулус: считаем флуктуации только в кольце вокруг центра галактики
    # x0_sbf, y0_sbf = 6306.0, 2730.0  # грубый центр галактики в полных координатах
    ny_r, nx_r = resid.shape
    # if not (0 <= x0_sbf < nx_r and 0 <= y0_sbf < ny_r):
    #     # на случай кропа используем автоматический центр
    x0_sbf, y0_sbf = guess_center(resid, valid150)
    yy_r, xx_r = np.ogrid[:ny_r, :nx_r]
    rr_r = np.hypot(yy_r - y0_sbf, xx_r - x0_sbf)

    rin, rout = 50.0, 200.0  # радиусы кольца в пикселях, можно потом подкрутить
    annulus = (rr_r >= rin) & (rr_r <= rout)
    mask_sbf = mask | (~annulus)

    # Проверка адекватности области для измерения флуктуаций
    if not check_sbf_region(resid, mask_sbf, label="[SBF-REGION]"):
        print("[SBF] область для измерения флуктуаций неадекватна, m̄ не считаем")
        # даже если базовый аннулус плохой, всё равно прогоняем скан по радиусам
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)
        return

    # PSF и его E(k)
    psf_file = args.psfref if args.psfref is not None else args.f150w
    psf = build_psf_for_file(psf_file, size=129)
    Pk = radial_power(resid, mask_sbf)
    Ek = radial_power_psf(psf, resid.shape)

    # нормируем PSF power к сумме 1, чтобы P0 был примерно дисперсией
    Ek_k, Ek_P = Ek
    norm = np.trapezoid(Ek_P, Ek_k)
    if norm <= 0 or not np.isfinite(norm):
        Ek_P_norm = Ek_P
    else:
        Ek_P_norm = Ek_P / norm
    Ek = (Ek_k, Ek_P_norm)

    P0, P1, kwin = fit_P0(Pk, Ek, kmin=0.03, kmax=0.40)  # окно по стабильности подберёшь позже

    # Оценка дисперсии SBF-сигнала:
    # берём реальную дисперсию по остаткам в том же аннулусе
    use_sbf_vals = (~mask_sbf) & np.isfinite(resid)
    vals_sbf = resid[use_sbf_vals]
    if vals_sbf.size == 0:
        print("[SBF] в аннулусе нет валидных пикселей для оценки дисперсии")
        return
    lo, hi = np.nanpercentile(vals_sbf, [5, 95])  # или ещё жёстче, 10–90
    core = (vals_sbf >= lo) & (vals_sbf <= hi)
    var_sbf = float(np.nanvar(vals_sbf[core]))

    if (not np.isfinite(var_sbf)) or (var_sbf <= 0.0):
        print(f"[SBF] некорректная дисперсия в аннулусе: var={var_sbf}")
        return

    # Для отладки смотрим, насколько P0 занижен относительно реальной дисперсии
    if np.isfinite(P0) and (P0 > 0.0):
        print(f"[SBF-DBG] P0={P0:.3e}, var_sbf={var_sbf:.3e}, ratio=var_sbf/P0={var_sbf/P0:.3e}")
    else:
        print(f"[SBF-DBG] P0 некорректен (P0={P0}) → используем только var_sbf")

    sigma2 = var_sbf

    # Средняя поверхностная яркость галактики в том же аннулусе (по гладкой модели)
    use_I = (~mask_sbf) & np.isfinite(model)
    n_I = int(use_I.sum())
    if n_I < 5000:
        print(f"[SBF] слишком мало пикселей для оценки средней яркости галактики: N={n_I}")
        return

    Imean = float(np.nanmean(model[use_I]))
    if (not np.isfinite(Imean)) or (Imean <= 0.0):
        print(f"[SBF] некорректная средняя яркость галактики в аннулусе: Imean={Imean}")
        return

    # Флуктуационная яркость в тех же единицах, что и изображение (MJy/sr)
    Pf = sigma2 / Imean

    if (not np.isfinite(Pf)) or (Pf <= 0.0):
        print(f"[SBF] некорректный Pf после нормировки: Pf={Pf}")
        return

    print(f"[SBF] sigma^2={sigma2:.3e}, Imean={Imean:.3e}, Pf={Pf:.3e}")

    # Величина флуктуаций в AB-магнитудах
    mbar = -2.5 * np.log10(Pf) + mjysr_to_ab_zp(area)

    # очень грубый sanity-check по диапазону SBF-магнитуд
    if not (10.0 <= mbar <= 40.0):
        print(
            f"[SBF] WARNING: m̄(F150W) = {mbar:.3f} mag выглядит нефизично, "
            f"проверь маску/аннулус/PSF"
        )

    # цвет (в AB маг/arcsec^2 на тех же масках и изофотах): возьмем медиану по модели
    # только если F090W есть
    # цвет (в AB маг/arcsec^2 на тех же масках): берём медианы по совпадающей области
    if img_f090 is not None:
        ny = min(img_f090.shape[0], img_f150.shape[0], mask.shape[0])
        nx = min(img_f090.shape[1], img_f150.shape[1], mask.shape[1])

        if (ny, nx) != img_f090.shape or (ny, nx) != img_f150.shape or (ny, nx) != mask.shape:
            print(f"[COLOR] shape mismatch: F090W={img_f090.shape}, "
                  f"F150W={img_f150.shape}, mask={mask.shape} → crop to ({ny},{nx})")

        m_sub = mask[:ny, :nx]
        f090_sub = img_f090[:ny, :nx]
        f150_sub = img_f150[:ny, :nx]

        num = np.nanmedian(f090_sub[~m_sub])
        den = np.nanmedian(f150_sub[~m_sub])

        if not np.isfinite(num) or not np.isfinite(den) or den <= 0 or num <= 0:
            print(f"[COLOR] WARNING: bad medians for color: num={num}, den={den}")
            color = np.nan
        else:
            color = -2.5 * np.log10(num / den)

        print(f"m̄(F150W) = {mbar:.3f} mag   (окно k={kwin[0]:.02f}..{kwin[1]:.02f} pix^-1)")
        print(f"(F090W − F150W) ≈ {color:.3f} mag (грубая оценка)")
        # Диагностический прогон по радиусам SBF-аннулуса для оценки плато m̄
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)
    else:
        print(f"m̄(F150W) = {mbar:.3f} mag   (окно k={kwin[0]:.02f}..{kwin[1]:.02f} pix^-1)")
        print("[INFO] Второй фильтр не задан, цвет не считается.")
        # Диагностический прогон по радиусам SBF-аннулуса для оценки плато m̄
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)

if __name__ == "__main__":
    main()