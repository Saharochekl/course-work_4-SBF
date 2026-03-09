"""
SBF-пайплайн для JWST/NIRCam i2d (быстрый прототип).

Что делает скрипт
  1) Загружает i2d (SCI) и строит маску валидных пикселей (finite SCI и, если есть, WHT>0).
  2) Оценивает фон (robust tile-median) и вычитает его.
  3) Строит маску компактных источников (звёзды/ГК/фоновые) через photutils.segmentation.
  4) Строит гладкую модель галактики по радиальному профилю вокруг центра и вычитает её.
  5) В SBF-аннулусе оценивает дисперсию флуктуаций и среднюю яркость галактики.
  6) Переводит Pf = sigma^2 / Imean в m̄ по формуле AB-системы для MJy/sr.
  7) Дополнительно: считает P(k) и E(k) (PSF), фитит P(k)=P0*E(k)+P1 как диагностику.

Вход/выход
  --f150w  : i2d FITS для F150W (SBF).
  --f090w  : i2d FITS для F090W (цвет), опционально.
  --psfref : i2d FITS с корректным header для stpsf (если PSF считаем не из того же файла).

  Вывод в stdout:
    - m̄(F150W) и использованное окно k (если был валидный P0-фит).
    - (F090W − F150W) грубо по медианам (если задан второй фильтр).
    - диагностический скан m̄ по кольцам (поиск «плато»).

  Пишет FITS рядом с входным файлом:
    *_sbf_model.fits  : гладкая модель галактики
    *_sbf_resid.fits  : остатки (img - model) с NaN на маске

Единицы и смысл величин
  - Входной SCI для JWST i2d обычно в MJy/sr (поверхностная яркость).
  - Imean: средняя поверхностная яркость галактики в аннулусе (MJy/sr).
  - sigma^2: дисперсия остатков в аннулусе (MJy/sr)^2.
  - Pf = sigma^2 / Imean имеет размерность MJy/sr и конвертится в AB mag через ZP.

Тюнинг (параметры, которые реально меняют поведение)
  - mask_sources: nsigma, npixels, (и kernel=7.0 для сглаживания при пороге)
  - subtract_galaxy: half_size вырезки, dr/min_pix для радиального профиля
  - SBF-аннулус: rin/rout (или скан scan_sbf_annuli)
  - окно частот k: kmin/kmax в fit_P0 (диагностика, не основное m̄ в текущей версии)

Ограничения прототипа
  - Модель галактики осесимметрична (радиальный профиль), без изофот/эллиптичности.
  - NaN-разрывы мозаики не «восстанавливаются»: для модели они заполняются профилем.
  - Плотная маска, неверный центр или «грязный» аннулус легко уводят m̄ в мусор.
"""
import argparse, numpy as np
import time
import builtins
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
from photutils.segmentation import detect_sources, deblend_sources
import stpsf                                    # PSF под JWST файл
from scipy.fft import rfft2, rfftfreq, set_workers # fftshift сейчас не используется (оставлено на будущее)
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
# Все print в файле автоматически получают префикс с текущим временем устройства
_orig_print = builtins.print

def print(*args, **kwargs):
    _orig_print(f"[{time.strftime('%H:%M:%S')}]", *args, **kwargs)


# -----------------------------------------------------------------------------
# Галактика: радиальный профиль и заливка дыр (модель для вычитания)
# -----------------------------------------------------------------------------
def build_radial_profile(img_c, valid_c, mask_c, x0, y0, dr=1.0, min_pix=200):
    print("working function build_radial_profile")
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
    print("end working function build_radial_profile")
    return np.array(radii), np.array(intens)

# Заливаем невалидные/замаскированные области радиальным профилем (только для построения модели)
def fill_nan_with_radial(img_c, valid_c, mask_c, x0, y0, radii, intens):
    print("working function fill_nan_with_radial")
    ny, nx = img_c.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - y0, xx - x0)

    ok = valid_c & (~mask_c) & np.isfinite(img_c)
    img_iso = img_c.copy()

    # bad: невалидно/замаскировано/NaN → подставляем профиль (только для модели)
    bad = ~ok
    img_iso[bad] = np.interp(rr[bad], radii, intens,
                             left=intens[0], right=intens[-1])
    print("end working function fill_nan_with_radial")
    return img_iso


# -----------------------------------------------------------------------------
# I/O: чтение i2d и базовые маски валидности
# -----------------------------------------------------------------------------
def load_i2d(path):
    print("working function load_i2d")
    hdul = fits.open(path, memmap=False)
    sci = hdul['SCI'].data.astype(float)
    hdr = hdul['SCI'].header

    # valid pixels: finite SCI and, if present, positive WHT
    valid = np.isfinite(sci)
    if 'WHT' in hdul:
        try:
            wht = hdul['WHT'].data
            valid &= np.isfinite(wht) & (wht > 0)
        except Exception:
            pass

    # площадь пикселя: используем PIXAR_SR из заголовка (sr) как источник истины
    pixar_sr = float(hdr['PIXAR_SR'])
    pix_area = pixar_sr / 2.350443e-11  # arcsec^2 (1 arcsec^2 = 2.350443e-11 sr)
    print(f"[WCS] PIXAR_SR={pixar_sr:.8e} sr → pix_area={pix_area:.8e} arcsec^2")
    hdul.close()
    print("end working function load_i2d")
    return sci, hdr, pix_area, valid

# -----------------------------------------------------------------------------
# Фон и маскирование источников
# -----------------------------------------------------------------------------
def sigma_clipped_bkg(img, box=256, return_2d=False, mask=None):
    print("working function sigma_clipped_bkg")
    # 2D фон (SBF-friendly): крупная сетка + sigma clipping.
    # По умолчанию mask=None, чтобы не усложнять main. Маску можно подать во 2-й проход позже.
    try:
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkg2d = Background2D(
            img,
            box_size=(int(box), int(box)),
            filter_size=(3, 3),
            mask=mask,
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
            exclude_percentile=10,
        )
        bkg_map = np.array(bkg2d.background, dtype=float)
        if return_2d:
            print("end working function sigma_clipped_bkg")
            return bkg_map
        # скаляр по умолчанию, чтобы не ломать старые вызовы
        bkg0 = float(np.nanmedian(bkg_map))
        print("end working function sigma_clipped_bkg")
        return bkg0

    except Exception:
        # Fallback: старый tile-median (скаляр)
        h, w = img.shape
        bs = int(box)
        tiles = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                tile = img[y:y+bs, x:x+bs]
                if np.any(np.isfinite(tile)):
                    tiles.append(np.nanmedian(tile))
        bkg0 = float(np.nanmedian(tiles)) if tiles else float(np.nanmedian(img))
        print("end working function sigma_clipped_bkg")
        return bkg0



def mask_sources(img, nsigma=2.0, npixels=20, do_deblend=True):
    print("working function mask_sources")
    """Build a boolean source mask. Nan-safe. Deblend is optional (needs scikit-image)."""
    # If everything is NaN, return empty mask
    if not np.any(np.isfinite(img)):
        return np.zeros_like(img, bool)

    # Сглаживание для стабильного порога (interpolate_replace_nans мостит NaN-дыры) smoothing for thresholding
    ker = Gaussian2DKernel(7.0)  # a bit wider kernel to bridge big NaN holes
    sm = interpolate_replace_nans(img, ker)

    # Робастные статистики (sigma_clipped_stats терпит NaN)
    bkg, med, rms = sigma_clipped_stats(sm, sigma=2.5, maxiters=5)
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

    # Выкидываем все «слишком большие» области: это почти всегда галактика, иногда разбитая швами/NaN на несколько лейблов.
    labels = segm.labels
    counts = np.bincount(segm.data.ravel(), minlength=labels.max() + 1)
    counts[0] = 0  # ignore background

    # Если галактика развалилась на несколько сегментов, прежняя логика могла замаскировать её куски как «источники».
    # Поэтому убираем ВСЕ лейблы, которые занимают заметную долю изображения.
    big_frac = 0.005  # 0.5% кадра; подстрой при необходимости
    big_labels = [lab for lab in labels if lab > 0 and counts[lab] > big_frac * img.size]
    if big_labels:
        segm.remove_labels(big_labels)
    print("end working function mask_sources")
    return segm.make_source_mask(size=5)


def prepare_image_and_mask(img_f150, valid150, nsigma=2.5, npixels=25, do_deblend=True,
                           bkg_box0=256, bkg_box2d=256):
    print("working function prepare_image_and_mask")

    # Грубый скалярный фон: только чтобы не строить маску на сильно смещённом кадре
    bkg0 = sigma_clipped_bkg(img_f150, box=bkg_box0, return_2d=False, mask=None)
    img0 = img_f150 - bkg0

    # 2) Маска источников на грубо выровненном кадре
    mask_src = mask_sources(img0, nsigma=nsigma, npixels=npixels, do_deblend=do_deblend)
    mask = (~valid150) | mask_src

    # 3) Финальный 2D-фон уже с маской источников/невалидных пикселей
    bkg2d = sigma_clipped_bkg(img_f150, box=bkg_box2d, return_2d=True, mask=mask)
    img = img_f150 - bkg2d

    print("end working function prepare_image_and_mask")
    return img, mask_src, mask, bkg0, bkg2d


def cutout_box(img, valid, x0, y0, half_size):
    print("working function cutout_box")
    ny, nx = img.shape
    x1 = max(0, int(x0 - half_size))
    x2 = min(nx, int(x0 + half_size))
    y1 = max(0, int(y0 - half_size))
    y2 = min(ny, int(y0 + half_size))
    img_c   = img[y1:y2, x1:x2]
    valid_c = valid[y1:y2, x1:x2]
    print("end working function cutout_box")
    return img_c, valid_c, (x0 - x1, y0 - y1), (x1, x2, y1, y2)

def subtract_galaxy(img, mask, valid, center=None):
    print("working function subtract_galaxy")
    """
    Строим гладкую модель галактики через радиальный профиль вокруг центра
    и вычитаем её. Без photutils.Ellipse, без изофот.

    На вход:
      img   – кадр F150W (фон уже вычтен)
      mask  – bool-маска (True = пиксель не использовать)
      valid – bool-маска валидных пикселей (WHT>0, не NaN)
    """
    # Центр галактики: либо фиксированный (--center), либо авто-оценка
    if center is None:
        x0, y0 = guess_center_fast(img, valid & (~mask))
        src = "auto"
    else:
        x0, y0 = center
        src = "fixed"
    print(f"[RADIAL] use center=({x0:.1f}, {y0:.1f}) [{src}]")

    ny, nx = img.shape


    # Вырезаем квадрат вокруг центра, чтобы не гонять вычисления по всему кадру
    img_c, valid_c, (x0, y0), (x1, x2, y1, y2) = cutout_box(
        img, valid, x0, y0, half_size=3000  # подобрать под размер галактики
    )
    mask_c = mask[y1:y2, x1:x2]

    finite_c = np.isfinite(img_c)
    print(f"[RADIAL] img_c shape={img_c.shape}, "
          f"finite={finite_c.sum()}, valid={valid_c.sum()}, "
          f"mask_c={mask_c.sum()}")

    # Радиальный профиль по данным в вырезке (игнорируем маску и невалидные)
    radii0, intens0 = build_radial_profile(img_c, valid_c, mask_c, x0, y0)
    print(f"[RADIAL] primary radial bins: {len(radii0)}")


    # Прототип: разрывы мозаики заполняем профилем (это заполнение для модели, не восстановление данных)

    # Строим img_iso: подставляем профиль во все bad-пиксели (seams/маска/NaN)
    img_iso = fill_nan_with_radial(img_c, valid_c, mask_c, x0, y0, radii0, intens0)

    # Финальный профиль считаем по img_iso, чтобы кольца не разваливались на NaN-разрывах
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

    
    # Медиана по тонким кольцам толщиной dr; кольца с малым числом пикселей пропускаем
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
        # Fallback: если профиля почти нет, используем гауссово сглаживание как грубую модель
        from astropy.convolution import convolve, Gaussian2DKernel
        img_c_smooth = convolve(img_c, Gaussian2DKernel(10.0),
                                normalize_kernel=True)

        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()
        model_full[y1:y2, x1:x2] = img_c_smooth
        resid_full[y1:y2, x1:x2] = img_c - img_c_smooth
        resid_full[mask] = np.nan
        print("end working function subtract_galaxy")
        return resid_full, model_full

    # Строим 2D модель из 1D профиля и вычитаем её из вырезки
    prof_1d = np.interp(rr.ravel(), radii, intens,
                        left=intens[0], right=intens[-1]).reshape(rr.shape)

    model_full = np.full_like(img, np.nan)
    resid_full = img.copy()
    model_full[y1:y2, x1:x2] = prof_1d
    resid_full[y1:y2, x1:x2] = img_c - prof_1d
    resid_full[mask] = np.nan

    print("end working function subtract_galaxy")
    return resid_full, model_full


def subtract_galaxy_isophote(img, mask, valid, center=None, return_debug=False, wcs=None):
    print("working function subtract_galaxy_isophote")
    """
    Заглушка/первая рабочая версия: строим гладкую модель галактики через изофоты
    (photutils.isophote.Ellipse) и вычитаем её.

    Возвращает (resid_full, model_full) в тех же соглашениях, что и subtract_galaxy:
      - model_full: 2D модель (MJy/sr)
      - resid_full: img - model_full, с NaN на mask

    Если изофоты не сошлись/упали — делаем fallback на гауссово сглаживание в кропе.
    """
    if center is None:
        print("guessing center of galaxy")
        x0, y0 = guess_center_fast(
            img,
            valid & (~mask),
            down=4,
            sigma=3.0,
            q=99.5,
            wcs=wcs,
            log=True,
        )
        src = "auto"
    else:
        x0, y0 = center
        src = "fixed"
    print(f"[ISO] use center=({x0:.1f}, {y0:.1f}) [{src}]")
    # Нашли центр автоматически. При желании x0,y0 можно задать вручную

    img_c, valid_c, (x0c, y0c), (x1, x2, y1, y2) = cutout_box(img, valid, x0, y0, half_size=3000)
    mask_c = mask[y1:y2, x1:x2]

    # Подготовка данных для Ellipse: невалид/маска -> NaN
    data = img_c.astype(float).copy()
    data[~valid_c] = np.nan
    data[mask_c] = np.nan

    # Быстрый sanity-check
    ok = np.isfinite(data)
    data_ma = np.ma.array(data, mask=~ok)
    print("sanity-check done.")
    if ok.sum() < 5000:
        print(f"[ISO] слишком мало валидных пикселей в кропе: N={int(ok.sum())} → fallback")
        from astropy.convolution import convolve, Gaussian2DKernel
        smooth = convolve(np.nan_to_num(img_c, nan=0.0), Gaussian2DKernel(10.0), normalize_kernel=True)
        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()
        model_full[y1:y2, x1:x2] = smooth
        resid_full[y1:y2, x1:x2] = img_c - smooth
        resid_full[mask] = np.nan
        if return_debug:
            dbg = {
                "iso_map_crop": np.full_like(img_c, np.nan, dtype=np.float32),
                "bounds": (x1, x2, y1, y2),
            }
            print("end working function subtract_galaxy_isophote")
            return resid_full, model_full, dbg
        print("end working function subtract_galaxy_isophote")
        return resid_full, model_full

    # Параметры начальной геометрии (eps и pa грубо, дальше Ellipse должен подстроиться)
    # NOTE: eps = 1 - b/a, pa в радианах
    # Slightly larger starting sma helps the first fit in big galaxies
    geom = EllipseGeometry(x0=x0c, y0=y0c, sma=50.0, eps=0.2, pa=0.0)


    try:
        ell = Ellipse(data_ma, geom)
        # Максимальная полуось: до края кропа от стартового центра (с запасом)
        ny0, nx0 = data.shape
        maxsma = 0.90 * float(min(x0c, y0c, (nx0 - 1 - x0c), (ny0 - 1 - y0c)))
        try:
            isolist = ell.fit_image(maxsma=maxsma, step=10.0, linear=True)
        except TypeError:
            isolist = ell.fit_image(maxsma=maxsma)

        if isolist is None or len(isolist) < 10:
            raise RuntimeError(f"isolist слишком короткий: {0 if isolist is None else len(isolist)}")

        model_c = build_ellipse_model(data.shape, isolist)

        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()
        model_full[y1:y2, x1:x2] = model_c
        resid_full[y1:y2, x1:x2] = img_c - model_c
        resid_full[mask] = np.nan

        print(f"[ISO] isolist N={len(isolist)}, maxsma≈{float(isolist[-1].sma):.1f} px")
        # DEBUG: фактический центр/геометрия по изофотам (Ellipse может подстроить центр даже если старт был мимо)
        try:
            x0_fit = float(np.nanmedian([iso.x0 for iso in isolist]))
            y0_fit = float(np.nanmedian([iso.y0 for iso in isolist]))
            eps_fit = float(np.nanmedian([iso.eps for iso in isolist]))
            pa_fit = float(np.nanmedian([iso.pa for iso in isolist]))
            print(f"[ISO] fitted center≈({x0_fit:.1f},{y0_fit:.1f}) in crop, eps≈{eps_fit:.3f}, pa≈{pa_fit:.3f} rad")
        except Exception:
            pass
        # Диагностическая карта изофот в координатах кропа
        nyc, nxc = data.shape
        yy, xx = np.indices((nyc, nxc), dtype=float)
        iso_map = np.zeros((nyc, nxc), dtype=np.float32)

        for iso in isolist:
            x0i = float(getattr(iso, "x0", np.nan))
            y0i = float(getattr(iso, "y0", np.nan))
            sma = float(getattr(iso, "sma", np.nan))
            eps = float(getattr(iso, "eps", np.nan))
            pa  = float(getattr(iso, "pa", np.nan))

            if not (np.isfinite(x0i) and np.isfinite(y0i) and np.isfinite(sma)
                    and np.isfinite(eps) and np.isfinite(pa)):
                continue
            if sma <= 0:
                continue

            q = max(1e-3, 1.0 - eps)  # b/a
            cosp = np.cos(pa)
            sinp = np.sin(pa)
            dx = xx - x0i
            dy = yy - y0i

            xp =  dx * cosp + dy * sinp
            yp = -dx * sinp + dy * cosp

            rell = np.sqrt(xp * xp + (yp / q) * (yp / q))
            line = np.abs(rell - sma) <= 0.6  # толщина линии ~1 px
            iso_map[line] = 1.0

        iso_map[~ok] = np.nan


        if return_debug:
            dbg = {
                "iso_map_crop": iso_map,
                "bounds": (x1, x2, y1, y2),
            }
            print("end working function subtract_galaxy_isophote")
            return resid_full, model_full, dbg

        print("end working function subtract_galaxy_isophote")
        return resid_full, model_full
    

    except Exception as e:
        print(f"[ISO] isophote fit failed: {e} → fallback")
        from astropy.convolution import convolve, Gaussian2DKernel
        smooth = convolve(np.nan_to_num(img_c, nan=0.0), Gaussian2DKernel(10.0), normalize_kernel=True)
        model_full = np.full_like(img, np.nan)
        resid_full = img.copy()
        model_full[y1:y2, x1:x2] = smooth
        resid_full[y1:y2, x1:x2] = img_c - smooth
        resid_full[mask] = np.nan
        if return_debug:
            dbg = {
                "iso_map_crop": np.full_like(img_c, np.nan, dtype=np.float32),
                "bounds": (x1, x2, y1, y2),
            }
            print("end working function subtract_galaxy_isophote")
            return resid_full, model_full, dbg
        print("end working function subtract_galaxy_isophote")
        return resid_full, model_full   





# -----------------------------------------------------------------------------
# PSF: получение и нормировка PSF через stpsf
# -----------------------------------------------------------------------------
def build_psf_for_file(fits_path, size=129, nlambda=7):
    print("working function build_psf_for_file")
    # stpsf настраивает симуляцию, читая JWST header (инструмент/фильтр/детектор)
    sim = stpsf.setup_sim_to_match_file(fits_path)
    psf = sim.calc_psf(nlambda=nlambda, fov_pixels=size)

    # DEBUG: тип PSF-объекта (на случай смены API)
    print("DEBUG psf type:", type(psf))

    # Разруливаем тип результата calc_psf (HDUList / HDU / ndarray)
    if isinstance(psf, fits.HDUList):
        # классический случай для webbpsf/stpsf – вернули HDUList
        data = psf[0].data
    elif isinstance(psf, fits.PrimaryHDU):
        data = psf.data
    else:
        # вдруг calc_psf уже вернул чистый numpy-массив
        data = psf

    arr = np.array(data, dtype=float)

    # Если PSF пришёл как куб (λ, y, x), интегрируем по λ
    if arr.ndim == 3:
        arr = arr.sum(axis=0)

    # Чистим NaN и нормируем сумму PSF к 1
    arr = np.nan_to_num(arr, nan=0.0)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"PSF из {fits_path} имеет нулевую/невалидную сумму, shape={arr.shape}")

    arr /= s
    print("end working function build_psf_for_file")
    return arr


# -----------------------------------------------------------------------------
# Спектр: оценка P(k) по остаткам и E(k) по PSF
# -----------------------------------------------------------------------------
def radial_power(img, mask, fft_workers=-1):
    print("working function radial_power")
    # Готовим массив для FFT: замаскированные пиксели = 0, остальное = resid
    data = np.zeros_like(img)
    tmp = np.nan_to_num(img, nan=0.0)
    data[~mask] = tmp[~mask]

    use = (~mask) & np.isfinite(data)
    n_pix = int(use.sum())
    if n_pix == 0:
        raise RuntimeError("radial_power: no unmasked pixels")

    # Вычитаем среднее по используемым пикселям, чтобы не тащить DC-компонент
    mean = float(np.nanmean(data[use]))
    data[use] -= mean

    # FFT и 2D power
    with set_workers(fft_workers):
        F = rfft2(data)
    power2d = np.abs(F)**2

    # Прямая дисперсия в пространстве (для нормировки по Parseval)
    var_direct = float(np.nanvar(data[use]))

    # Дисперсия из FFT (Parseval) и поправка масштаба под прямую дисперсию
    # Parseval for numpy FFT (unnormalized forward): sum|x|^2 ~= (1/N) sum|X|^2
    N = data.size
    var_fft = power2d.sum() / (N**2)
    if var_fft > 0 and np.isfinite(var_fft):
        scale = var_direct / var_fft
    else:
        scale = 1.0

    # Финальная 2D мощность (нормирована на число пикселей области)
    P2d = power2d * scale / (N**2)

    ny, nx = data.shape
    ky = np.fft.fftfreq(ny)
    kx = rfftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)

    # Азимутальное усреднение: медиана мощности по кольцам в k-пространстве
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
    print("end working function radial_power")
    return kc[m], P[m]


def radial_power_psf(psf_small, img_shape, fft_workers= -1):
    print("working function radial_power_psf")
    ny, nx = img_shape
    big = np.zeros((ny, nx), float)

    py, px = psf_small.shape
    y0 = ny // 2 - py // 2
    x0 = nx // 2 - px // 2
    big[y0:y0+py, x0:x0+px] = psf_small

    mask_big = np.zeros_like(big, bool) 
    print("end working function radial_power_psf")
    return radial_power(big, mask_big, fft_workers=fft_workers)



def guess_center_fast(img, valid, down=4, sigma=3.0, q=99.5, wcs=None, log=True):
    print("working function guess_center_fast")
    ny, nx = img.shape

    def _log_center(xc, yc, note=""):
        if not log:
            return
        msg = f"[CENTER-FAST] x={xc:.2f}, y={yc:.2f}"
        if note:
            msg += f" ({note})"
        if wcs is not None:
            try:
                ra_deg, dec_deg = wcs.pixel_to_world_values(xc, yc)
                msg += f" | RA={ra_deg:.8f} deg, Dec={dec_deg:.8f} deg"
            except Exception as e:
                msg += f" | RA/Dec fail: {e}"
        print(msg)

    # downsample
    img_d = img[::down, ::down]
    val_d = valid[::down, ::down] & np.isfinite(img_d)
    if not np.any(val_d):
        xc, yc = nx / 2.0, ny / 2.0
        _log_center(xc, yc, note="fallback: no valid downsampled pixels")
        return xc, yc

    data = np.where(val_d, img_d, 0.0).astype(np.float32)
    w = val_d.astype(np.float32)

    # nan-aware blur via normalized convolution
    num = gaussian_filter(data, sigma=sigma)
    den = gaussian_filter(w, sigma=sigma)
    sm = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)

    if not np.isfinite(sm).any():
        xc, yc = nx / 2.0, ny / 2.0
        _log_center(xc, yc, note="fallback: sm is all-NaN")
        return xc, yc

    thr = np.nanpercentile(sm, q)
    sel = np.isfinite(sm) & (sm >= thr)
    if sel.sum() < 50:
        y, x = np.unravel_index(np.nanargmax(sm), sm.shape)
        xc, yc = float(x * down), float(y * down)
        _log_center(xc, yc, note="fallback: argmax")
        return xc, yc

    ys, xs = np.nonzero(sel)
    ws = sm[sel] - np.nanmin(sm[sel])
    ws = np.nan_to_num(ws, nan=0.0) + 1e-12

    x0 = float((xs * ws).sum() / ws.sum()) * down
    y0 = float((ys * ws).sum() / ws.sum()) * down
    _log_center(x0, y0, note=f"down={down}, sigma={sigma}, q={q}")
    print("end working function guess_center_fast")
    return x0, y0

# -----------------------------------------------------------------------------
# Фит P(k) = P0 * E(k) + P1 (диагностика качества/PSF, не основное m̄)
# -----------------------------------------------------------------------------
def fit_P0(Pk, Ek, kmin=0.03, kmax=0.40):
    print("working function fit_P0")
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
    print("end working function fit_P0")
    return float(P0), float(P1), (kmin, kmax)


# -----------------------------------------------------------------------------
# SBF: проверки адекватности области и поиск «плато» по радиусам
# -----------------------------------------------------------------------------
def check_sbf_region(resid, mask_sbf, label="[SBF]"):
    print("working function check_sbf_region")
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
    # NOTE: при желании можно сделать это строгим фейлом (return False)

    # Слишком маленький std означает, что сигнал ниже шума/квантизации
    if std < 1e-7:
        print(f"{label} std={std:.3e} слишком мал → сигнал неотличим от нуля")
    # NOTE: при желании можно сделать это строгим фейлом (return False)
    print("end working function check_sbf_region")
    return True


# Поиск «плато» m̄ по результатам скана радиусов
def pick_mbar_plateau(rows, label_prefix="[PLATEAU]", dm_max=0.4, min_len=3):
    print("working function pick_mbar_plateau")
    """
    По итогам скана по радиусам выбирает 'плато' m̄:
    - rows: список словарей с полями rin, rout, mbar.
    - dm_max: макс. разброс m̄ внутри плато.
    - min_len: минимальное число колец в плато.

    Возвращает (rin_min, rout_max, mbar_mean, mbar_min, mbar_max, n_rings)
    или (None,)*6, если плато не найдено.
    """
    if not rows:
        print(f"{label_prefix} нет данных для поиска плато")
        return None, None, None, None, None, None

    # сортируем по внутреннему радиусу
    rows_sorted = sorted(rows, key=lambda r: r["rin"])
    best = None  # (score, i_start, i_end)

    mvals = [r["mbar"] for r in rows_sorted]

    n = len(rows_sorted)
    for i in range(n):
        cur_min = mvals[i]
        cur_max = mvals[i]
        for j in range(i + 1, n):
            v = mvals[j]
            if v < cur_min:
                cur_min = v
            if v > cur_max:
                cur_max = v
            if cur_max - cur_min > dm_max:
                break
            length = j - i + 1
            if length >= min_len:
                # чем длиннее плато и ближе к середине радиуса, тем лучше
                rin = rows_sorted[i]["rin"]
                rout = rows_sorted[j]["rout"]
                rmid = 0.5 * (rin + rout)
                score = (length, -abs(rmid))  # сначала макс. длина, потом минимальный |rmid|
                if (best is None) or (score > best[0]):
                    best = (score, i, j)

    if best is None:
        print(f"{label_prefix} плато m̄ с dm<={dm_max:.2f} mag не найдено")
        print("end working function pick_mbar_plateau")
        return None, None, None, None, None, None

    _, i0, j0 = best
    subset = rows_sorted[i0:j0 + 1]
    rin_min = subset[0]["rin"]
    rout_max = subset[-1]["rout"]
    m_list = [r["mbar"] for r in subset]
    mbar_mean = float(np.mean(m_list))
    mbar_min = float(min(m_list))
    mbar_max = float(max(m_list))
    n_rings = len(subset)

    print(
        f"{label_prefix} rin≈{rin_min:.1f}..{rout_max:.1f} px, "
        f"m̄≈{mbar_mean:.3f} mag, Δm̄={mbar_max - mbar_min:.3f} mag, "
        f"N_rings={n_rings}"
    )
    print("end working function pick_mbar_plateau")
    return rin_min, rout_max, mbar_mean, mbar_min, mbar_max, n_rings

# Диагностический прогон по радиусам SBF-аннулуса (ищем диапазон, где m̄ стабилен)
def scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf, label_prefix="[SCAN]"):
    print("working function scan_sbf_annuli")
    """
    Диагностический прогон по радиусам SBF-аннулуса.
    Для каждой пары (rin, rout) печатает Imean, std, dyn, var_sbf, Pf, m̄.

    Радиусы подбираются автоматически под размер кадра:
    - для больших полей (полный кадр): rin~100..800, ширина ~100 пикс;
    - для меньших (кроп): rin~30..(Rmax-20), ширина ~60 пикс.
    """
    rows = []
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
        rows.append(
            {
                "rin": float(rin),
                "rout": float(rout),
                "N": n,
                "Imean": Imean,
                "std": std,
                "dyn": dyn,
                "var_sbf": var_sbf,
                "Pf": Pf,
                "mbar": mbar,
            }
        )
    pick_mbar_plateau(rows, label_prefix=f"{label_prefix} PLATEAU")
    print("end working function scan_sbf_annuli")


# Перевод 1 (MJy/sr) * pixel_area в AB magnitude (нулевая точка для Pf)
def mjysr_to_ab_zp(pix_area_arcsec2):
    print("working function mjysr_to_ab_zp")

    # m_AB для 1 (MJy/sr) на 1 пиксель
    jy_per_pix = 2.350443e-5 * pix_area_arcsec2  # 1 MJy/sr -> Jy/arcsec^2
    m_ab = -2.5*np.log10(jy_per_pix/3631.0)
    print("end working function mjysr_to_ab_zp")
    return float(m_ab)

# ----------------------------------------------------------------------------
# PSF cache: load/save PSF to avoid slow OPD/initialization work in stpsf
# ----------------------------------------------------------------------------
def load_or_build_psf(psf_file, cache_path=None, size=129, nlambda=7):
    print("working function load_or_build_psf")
    """
    cache_path:
      - None: no caching, always compute
      - path exists: load
      - path does not exist: compute and save
    """
    print("working function load_or_build_psf")
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with fits.open(cache_path, memmap=False) as hd:
                arr = np.array(hd[0].data, dtype=float)
            if arr.ndim != 2 or arr.size == 0:
                raise ValueError(f"PSF cache {cache_path} has bad shape {arr.shape}")
            arr = np.nan_to_num(arr, nan=0.0)
            s = float(arr.sum())
            if not np.isfinite(s) or s <= 0:
                raise ValueError(f"PSF cache {cache_path} has non-positive sum")
            arr /= s
            print(f"[PSF] loaded cache → {cache_path} (size={arr.shape[0]}×{arr.shape[1]})")
            print("end working function load_or_build_psf")
            return arr

    # compute
    arr = build_psf_for_file(psf_file, size=size, nlambda=nlambda)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(cache_path, arr.astype(np.float32), overwrite=True)
        print(f"[PSF] saved cache → {cache_path}")
    print("end working function load_or_build_psf")
    return arr


def parse_center(s):
    print("working function parse_center")
    """Parse center from 'x,y' (floats)."""
    print("working function parse_center")
    if s is None:
        return None
    try:
        a, b = s.split(',')
        print("end working function parse_center")
        return float(a), float(b)
    except Exception:
        raise argparse.ArgumentTypeError("--center must be 'x,y' (floats)")

def main():
    print("working function main")
    ap = argparse.ArgumentParser()
    ap.add_argument("--f150w", required=True, help="i2d F150W")
    ap.add_argument("--f090w", help="i2d F090W (для цвета), optional")
    ap.add_argument("--psfref", help="i2d с полным JWST header для расчёта PSF (если PSF считаем не из того же файла)")
    ap.add_argument("--psf-cache", default=None, help="путь к кэшу PSF (FITS). Если существует — грузим. Если нет — считаем и сохраняем.")
    ap.add_argument("--psf-size", type=int, default=129, help="размер PSF в пикселях (квадрат), по умолчанию 129")
    ap.add_argument("--psf-nlambda", type=int, default=7, help="nlambda для stpsf.calc_psf, по умолчанию 7")
    ap.add_argument("--center", type=parse_center, default=None, help="фиксированный центр галактики 'x,y' в пикселях (для отладки, если auto-center мажет)")
    ap.add_argument("--model", help="0 - radial (простая), 1 - isophote (Ellipse), 2 - both (сравнение)", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="только оценить фон/маску")
    ap.add_argument("--dump-isophotes", action="store_true", help="сохранить диагностические FITS с изофотами (только для --model 1/2)")
    ap.add_argument("--fft-workers", type=int, default=-1, help="число workers для scipy.fft, где 1 = без параллелизма, -1 = все ядра(default)")
    ap.add_argument("--no-bkg", action="store_true", help="экспериментальный режим: не оценивать и не вычитать фон вообще")
    args = ap.parse_args()
    print("begin working...")
    fft_workers = int(args.fft_workers)
    if fft_workers == 0: # Защита от ввода нуля. Чатик предложил, я добавил, в целом, похуй
        fft_workers = 1
    img_f150, hdr150, area, valid150 = load_i2d(args.f150w)
    wcs150 = WCS(hdr150)
    # Второй фильтр (для цвета) загружаем только если он задан
    if args.f090w is not None:
        img_f090, hdr090, _, valid090 = load_i2d(args.f090w)
    else:
        img_f090 = None
    if args.no_bkg:
        print("i2d's loaded, NO-BKG mode: skip any background estimation/subtraction")
        img = np.array(img_f150, copy=True)
        mask_src = mask_sources(img, nsigma=2.5, npixels=25, do_deblend=not args.dry)
        mask = (~valid150) | mask_src
        bkg0 = 0.0
        bkg = np.zeros_like(img_f150, dtype=float)
        print("no-bkg preparation done.")
    else:
        print("i2d's loaded, preparing background and source mask...")
        img, mask_src, mask, bkg0, bkg = prepare_image_and_mask(
            img_f150,
            valid150,
            nsigma=2.5,
            npixels=25,
            do_deblend=not args.dry,
            bkg_box0=256,
            bkg_box2d=256,
        )
        print(f"[BKG] rough scalar background={bkg0:.6g}")
        print("background+mask preparation done.")

    # DEBUG: сохраняем диагностические маски (valid / mask_src / mask) отдельными FITS
    base_dbg = Path(args.f150w)
    stem_dbg = base_dbg.stem
    out_dir_dbg = base_dbg.parent

    valid_path = out_dir_dbg / f"{stem_dbg}_dbg_valid.fits"
    masksrc_path = out_dir_dbg / f"{stem_dbg}_dbg_mask_src.fits"
    mask_path = out_dir_dbg / f"{stem_dbg}_dbg_mask.fits"

    fits.writeto(valid_path, valid150.astype(np.uint8), hdr150, overwrite=True)
    fits.writeto(masksrc_path, mask_src.astype(np.uint8), hdr150, overwrite=True)
    fits.writeto(mask_path, mask.astype(np.uint8), hdr150, overwrite=True)

    print(f"[DBG] valid150  → {valid_path}")
    print(f"[DBG] mask_src → {masksrc_path}")
    print(f"[DBG] mask     → {mask_path}")
    if args.dry:
        finite = int(valid150.sum())
        total = int(img.size)
        masked = int(mask.sum())
        print(f"[DRY] shape={img.shape}, finite={finite}/{total} ({100*finite/total:.1f}%)")
        print(f"[DRY] bkg≈{bkg:.6g} (MJy/sr units), mask_coverage={100*masked/total:.2f}%")
        print("end working function main")
        return

    # resid, model = subtract_galaxy(img, mask, valid150)
    # 0 - radial, 1 - isophote, 2 - both (для сравнения)
    extra_model_r = None
    extra_resid_r = None
    if args.model == 0:
        iso_dbg_main = None
        resid, model = subtract_galaxy(img, mask, valid150, center=args.center)
    elif args.model == 1:
        if args.dump_isophotes:
            resid, model, iso_dbg = subtract_galaxy_isophote(
                img, mask, valid150, center=args.center, return_debug=True, wcs=wcs150
            )
        else:
            resid, model = subtract_galaxy_isophote(img, mask, valid150, center=args.center, wcs=wcs150)
            iso_dbg = None
        iso_dbg_main = iso_dbg
    elif args.model == 2:
        # Считаем обе модели: isophote как основную, radial как дополнительную для сравнения
        if args.dump_isophotes:
            resid_i, model_i, iso_dbg = subtract_galaxy_isophote(
                img, mask, valid150, center=args.center, return_debug=True, wcs=wcs150
            )
        else:
            resid_i, model_i = subtract_galaxy_isophote(img, mask, valid150, center=args.center, wcs=wcs150)
            iso_dbg = None

        resid_r, model_r = subtract_galaxy(img, mask, valid150, center=args.center)

        # Основной поток анализа оставляем на isophote (как и раньше)
        resid, model = resid_i, model_i
        iso_dbg_main = iso_dbg

        # Сохраним radial отдельно ниже
        extra_model_r = model_r
        extra_resid_r = resid_r
    else:
        raise ValueError(f"bad --model={args.model}, expected 0/1/2")


    # Сохраняем модель и остатки в FITS, чтобы проверить глазами в DS9/Siril
    base = Path(args.f150w)
    stem = base.stem  # jw03055-o001_t001_nircam_clear-f150w_i2d или i2d_crop1
    out_dir = base.parent

    model_path = out_dir / f"{stem}_sbf_model.fits"
    resid_path = out_dir / f"{stem}_sbf_resid.fits"
    if args.dump_isophotes and (iso_dbg_main is not None):
        try:
            iso_map_crop = iso_dbg_main.get("iso_map_crop")
            x1, x2, y1, y2 = iso_dbg_main.get("bounds")

            iso_crop_path = out_dir / f"{stem}_sbf_isophotes_crop.fits"
            fits.writeto(iso_crop_path, iso_map_crop.astype(np.float32), hdr150, overwrite=True)
            print(f"[OUT] isophotes(crop) → {iso_crop_path}")

            iso_full = np.full_like(img, np.nan, dtype=np.float32)
            iso_full[y1:y2, x1:x2] = iso_map_crop.astype(np.float32)
            iso_full_path = out_dir / f"{stem}_sbf_isophotes_full.fits"
            fits.writeto(iso_full_path, iso_full, hdr150, overwrite=True)
            print(f"[OUT] isophotes(full) → {iso_full_path}")

            overlay = np.array(img, dtype=np.float32, copy=True)
            good = np.isfinite(overlay)
            if np.any(good):
                p99 = np.nanpercentile(overlay[good], 99.0)
                amp = float(0.2 * p99) if np.isfinite(p99) else 1.0
            else:
                amp = 1.0
            lines = np.isfinite(iso_full) & (iso_full > 0.5)
            overlay[lines] = np.nan_to_num(overlay[lines], nan=0.0) + amp

            iso_ovl_path = out_dir / f"{stem}_sbf_isophotes_overlay.fits"
            fits.writeto(iso_ovl_path, overlay.astype(np.float32), hdr150, overwrite=True)
            print(f"[OUT] isophotes(overlay) → {iso_ovl_path}")
        except Exception as e:
            print(f"[ISO-DBG] failed to write isophote debug FITS: {e}")

    if args.model == 2 and (extra_model_r is not None) and (extra_resid_r is not None):
        model_r_path = out_dir / f"{stem}_sbf_model_radial.fits"
        resid_r_path = out_dir / f"{stem}_sbf_resid_radial.fits"
        fits.writeto(model_r_path, extra_model_r, hdr150, overwrite=True)
        fits.writeto(resid_r_path, extra_resid_r, hdr150, overwrite=True)
        print(f"[OUT] model(radial) → {model_r_path}")
        print(f"[OUT] resid(radial) → {resid_r_path}")

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

    # DEBUG: локальная проверка RMS остатков в маленьком кольце (sanity-check)
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
    # на случай кропа используем автоматический центр, либо фиксированный
    print("before x0_sbf, y0_sbf = args.center if  ....")
    x0_sbf, y0_sbf = args.center if args.center is not None else guess_center_fast(img, valid150 & (~mask), down=4, sigma=3.0, q=99.5, wcs=wcs150, log=True)
    print("After x0_sbf, y0_sbf = args.center if ...")
    yy_r, xx_r = np.ogrid[:ny_r, :nx_r]
    rr_r = np.hypot(yy_r - y0_sbf, xx_r - x0_sbf)

    rin, rout = 50.0, 200.0  # радиусы кольца в пикселях, можно потом подкрутить
    annulus = (rr_r >= rin) & (rr_r <= rout)
    mask_sbf = mask | (~annulus)

    # Быстрые проверки базового аннулуса; при провале всё равно прогоняем скан радиусов
    if not check_sbf_region(resid, mask_sbf, label="[SBF-REGION]"):
        print("[SBF] область для измерения флуктуаций неадекватна, m̄ не считаем")
        # даже если базовый аннулус плохой, всё равно прогоняем скан по радиусам
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)
        print("end working function main")
        return

    # PSF → E(k) и остатки → P(k)
    print(f"[PSF] psfref={args.psfref if args.psfref is not None else '(same as f150w)'}")
    psf_file = args.psfref if args.psfref is not None else args.f150w

    # PSF cache path: default рядом с F150W, если --psf-cache не задан
    if args.psf_cache is None:
        psf_cache_path = Path(args.f150w).with_name(f"{Path(args.f150w).stem}_psf_{args.psf_size}.fits")
    else:
        psf_cache_path = args.psf_cache

    psf = load_or_build_psf(psf_file, cache_path=psf_cache_path,
                            size=args.psf_size, nlambda=args.psf_nlambda)
    print(f"[FFT] workers={fft_workers}")
    Pk = radial_power(resid, mask_sbf, fft_workers=fft_workers)
    Ek = radial_power_psf(psf, resid.shape, fft_workers=fft_workers)

    # NOTE: нормировка E(k) экспериментальная; m̄ ниже считается через Pf=sigma^2/Imean
    Ek_k, Ek_P = Ek
    norm = np.trapezoid(Ek_P, Ek_k)
    if norm <= 0 or not np.isfinite(norm):
        Ek_P_norm = Ek_P
    else:
        Ek_P_norm = Ek_P / norm
    Ek = (Ek_k, Ek_P_norm)

    P0, P1, kwin = fit_P0(Pk, Ek, kmin=0.03, kmax=0.40)  # окно по стабильности подберёшь позже

    # Дисперсия флуктуаций: var по остаткам в аннулусе (с обрезкой хвостов)
    # берём реальную дисперсию по остаткам в том же аннулусе
    use_sbf_vals = (~mask_sbf) & np.isfinite(resid)
    vals_sbf = resid[use_sbf_vals]
    if vals_sbf.size == 0:
        print("[SBF] в аннулусе нет валидных пикселей для оценки дисперсии")
        print("end working function main")
        return
    lo, hi = np.nanpercentile(vals_sbf, [5, 95])  # или ещё жёстче, 10–90
    core = (vals_sbf >= lo) & (vals_sbf <= hi)
    var_sbf = float(np.nanvar(vals_sbf[core]))

    if (not np.isfinite(var_sbf)) or (var_sbf <= 0.0):
        print(f"[SBF] некорректная дисперсия в аннулусе: var={var_sbf}")
        print("end working function main")
        return

    # DEBUG: сравнение P0 из фита и var_sbf по пикселям (ожидаемо одного порядка)
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
        print("end working function main")
        return

    Imean = float(np.nanmean(model[use_I]))
    if (not np.isfinite(Imean)) or (Imean <= 0.0):
        print(f"[SBF] некорректная средняя яркость галактики в аннулусе: Imean={Imean}")
        print("end working function main")
        return

    # Pf в единицах SCI (MJy/sr): Pf = sigma^2 / Imean
    Pf = sigma2 / Imean

    if (not np.isfinite(Pf)) or (Pf <= 0.0):
        print(f"[SBF] некорректный Pf после нормировки: Pf={Pf}")
        print("end working function main")
        return

    print(f"[SBF] sigma^2={sigma2:.3e}, Imean={Imean:.3e}, Pf={Pf:.3e}")

    # Перевод Pf → m̄ в AB-системе
    mbar = -2.5 * np.log10(Pf) + mjysr_to_ab_zp(area)

    # Грубый sanity-check по диапазону m̄ (порог можно настроить)
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
        # Диагностика: сканируем радиусы и ищем «плато» m̄
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)
    else:
        print(f"m̄(F150W) = {mbar:.3f} mag   (окно k={kwin[0]:.02f}..{kwin[1]:.02f} pix^-1)")
        print("[INFO] Второй фильтр не задан, цвет не считается.")
        # Диагностика: сканируем радиусы и ищем «плато» m̄
        scan_sbf_annuli(resid, model, mask, area, x0_sbf, y0_sbf)
    print("end working function main")
    return

if __name__ == "__main__":
    main()