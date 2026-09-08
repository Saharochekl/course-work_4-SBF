Параметры sbf-2.ipynb / sbf-2.ipynb parameters
===========================================

Граница применимости / Scope
---------------------------

Это справочник оставленных настроек исходного ``code/sbf-2.ipynb``. Обработчик
может переопределять их при создании notebook для конкретной галактики/фильтра;
фактическая конфигурация сохраняется вместе с результатом. Ниже указаны значения
шаблона, а не обещание одинаковых настроек всех исторических запусков.

**Не путать исходный и финальный спектральный этапы.** Шаблон сохраняет старую
ветвь винзорирования сырых остатков при 3.5σ, до деления на √M. Его ячейка 61
также содержит старый выбор максимума между оценками ошибки и полуразностью
колец. Это не определение финальных ошибок статьи. Принятый повторный
спектральный расчёт выполняется в ``sbf2_normalized_winsor_core.py``: сначала
нормировка, затем винзорирование всей валидной области. Итоговые оценки берутся
из его продуктов и последующего анализа, а не автоматически из старой ячейки 61.

These are source-template defaults, not all historical or runner-injected values.
The source notebook retains raw-residual 3.5σ winsorization and the older error
combination in cell 61. The adopted final spectral stage instead normalizes first
and winsorizes the full valid region in ``sbf2_normalized_winsor_core.py``. Do not
quote the source notebook's old uncertainty columns as final article errors.

Числа ниже разделены на определения единиц, принятый протокол и эвристики.
Начальные приближения, нижние границы и минимальные числа точек не являются
физическими законами, оптимальными значениями или доказательством сходимости.
Для коррелированных пикселей их количество не равно числу независимых измерений.
Numerical seeds, floors and sample-size gates are heuristics, not physical laws,
optimality/convergence proofs or counts of independent resolution elements.

Единицы и метаданные / Units and metadata
----------------------------------------

* ``ARCSEC2_PER_SR=2.350443e-11`` — несмотря на имя, **sr в 1 arcsec²**:
  (π/648000)². Поэтому ``PIXAR_SR / ARCSEC2_PER_SR`` даёт arcsec²/pixel.
  Despite the historical name, this is steradians per square arcsecond, not its inverse.
* ``MJY_SR_TO_JY_PER_ARCSEC2=2.350443e-5`` — предыдущий коэффициент ×10⁶ для
  MJy/sr → Jy/arcsec² / the same solid-angle factor times 10⁶.
* ``AB_ZEROPOINT_JY=3631.0`` — принятая нулевая плотность потока AB в Jy,
  не подобранный SBF-нуль-пункт / adopted AB flux-density reference, not a fitted SBF zero point.
* ``TARGET_GALAXY`` — значение обработчика либо имя каталога входа; выбирает ровно
  одну строку метаданных / runner value or input-directory name; selects one metadata row.
* ``PAPER_IV_METADATA_PATH`` — найденный локальный ``go3055_paper_iv_metadata.csv``;
  отсутствие — ошибка / local reference table; a missing table is an error.
* ``PAPER_IV_HIGH_QUALITY``, ``PAPER_IV_NAME``, ``EXTINCTION_SOURCE`` — флаг,
  обозначение и источник из выбранной строки, не параметры настройки / row metadata, not tuning constants.
* ``A_F090W``, ``A_F150W``, ``E_BV``, ``sigma_E_BV``, ``sigma_A_F090W``,
  ``sigma_A_F150W``, ``sigma_color_extinction`` загружаются из той же таблицы,
  а не задаются общими числами для 14 галактик / target-specific table inputs.
* ``A_F150W_SBF=A_F150W``, ``SIGMA_A_F150W_SBF=sigma_A_F150W`` и
  ``EXTINCTION_AVAILABLE=isfinite(A_F150W_SBF)`` — проверенные значения для
  коррекции SBF / checked extinction values and availability flag.
* ``A_F090W_COLOR=A_F090W``, ``A_F150W_COLOR=A_F150W``,
  ``SIGMA_COLOR_EXTINCTION=sigma_color_extinction`` — такие же входы цветовой
  ячейки; не добавочная независимая ошибка / color-cell aliases, not extra independent errors.

Центр и фон / Centre and sky
---------------------------

* ``USE_BKG=True`` — вычитать принятый скалярный фон, не 2D-карту галактики /
  subtract the adopted scalar sky, not an extended-galaxy background map.
* ``FIXED_CENTER=None`` — разрешить поиск центра; заданная пара заменяет его /
  determine the centre unless an explicit coordinate pair is supplied.
* ``CENTER_GUESS_DOWN=4`` — ускоряющее разрежение изображения /
  subsampling for a cheap initial centroid.
* ``CENTER_GUESS_SMOOTH_SIGMA=3.0`` — сглаживание в пикселях разреженной копии,
  чтобы старт не определялся одним пиком / smoothing in subsampled pixels suppresses isolated peaks.
* ``CENTER_GUESS_Q=99.5``, ``CENTER_GUESS_MIN_PIXELS=50`` — яркая вершина для
  центроида и минимальный размер её выборки / bright-tail percentile and minimum centroid sample.
* ``CENTER_GUESS_VALID_FLOOR=1e-6`` — не делить на почти нулевой вес маски /
  avoid dividing by vanishing smoothed-mask support.
* ``CENTER_GUESS_WEIGHT_FLOOR=1e-12`` — положительные веса при плоской вершине /
  positive centroid weights when the selected peak is flat.
* ``BKG_BOX2D=256`` — базовый размер сетки вспомогательной гладкой фоновой оценки /
  mesh scale for the auxiliary smooth sky estimate; adjusted on the sparse copy.
* ``SIGMA_STAT=3.0``, ``SIGMA_MAXIT=5`` — устойчивые статистики: порог и конечное
  число итераций / robust-statistics threshold and finite iteration budget.
* ``BKG_CHECK_CORNER_FRAC=0.10``, ``BKG_CHECK_MIN_PIXELS=100`` — линейная доля
  углового участка и минимум для статистики / corner side fraction and minimum statistics sample.
* ``BKG_JENSEN_DOWNSAMPLE=8`` — фон оценивается на разреженной копии, научный
  кадр не прореживается / cheap sky estimation without subsampling the science image.
* ``BKG_JENSEN_CORNER_FRAC=0.10`` — та же доля сторон для четырёх углов F150W /
  side fraction for the four F150W corner samples.
* ``BKG_JENSEN_OUTER_QUANTILE=0.80`` — внешняя по радиусу часть для фоновой
  опоры / radial quantile selecting the outer sky-reference pixels.
* ``BKG_JENSEN_PROFILE_BINS=64``, ``BKG_JENSEN_SKY_GRID=500`` — дискретизация
  радиального профиля и перебора фона, компромисс разрешения/времени /
  profile and sky-search resolution; a computational choice, not proven convergence.
* ``BKG_JENSEN_MODEL_ITERS=3`` — ограничить число поправок грубой модели /
  cap the coarse residual-sky iterations, with earlier stopping when small.

Маски источников / Source masks
------------------------------

* ``SIGMA_DET=2.5``, ``MASK_NPIXELS=4`` — базовый порог и минимальная связная
  площадь каталожного кандидата / baseline detection threshold and connected area.
* ``SOURCE_DETECT_CONNECTIVITY=8`` — диагональное соседство тоже соединяет
  пиксели источника / diagonal neighbours belong to the same connected component.
* ``DO_DEBLEND=True`` — разделять перекрывающиеся кандидаты /
  separate overlapping source candidates.
* ``DEBLEND_NLEVELS=8``, ``DEBLEND_CONTRAST=0.001`` — число уровней разделения и
  минимальная относительная яркость ветви / deblending levels and minimum branch contrast.
* ``DEBLEND_NPROC=8`` — степень параллелизма, не научный параметр /
  deblending parallelism, not a measurement parameter.
* ``PREMASK_MAX_COMPACT_AREA=5000`` — отсечь слишком протяжённые сегменты,
  которые нельзя автоматически принять за компактные источники /
  reject extended segments from the compact-source premask.
* ``PREMASK_MODEL_BOX=128``, ``PREMASK_MODEL_FILTER=5`` — сетка предварительной
  модели и медианный фильтр 5×5 ячеек / detection-only background mesh and mesh filtering.
* ``PREMASK_NOISE_SAMPLE_STEP=8``, ``PREMASK_NOISE_BINS=20`` — разрежение и
  диапазоны яркости для эмпирической зависимости шума от модели /
  subsampling and model-brightness bins for the empirical noise curve.
* ``PREMASK_DET_SIGMA=max(3.5,SIGMA_DET)=3.5``,
  ``PREMASK_DET_NPIXELS=max(9,MASK_NPIXELS)=9`` — первичная маска намеренно
  строже каталога, чтобы меньше задевать структуру галактики /
  a more restrictive premask detector than the later source catalog.
* ``PREMASK_DILATE_ITERATIONS=2`` — расширение сегментов для краёв источника /
  dilate detected segments to cover their immediate outskirts.
* ``PREMASK_RADIUS_MARGIN=1.15`` — запас к большему из радиусов изофот и SBF /
  margin around the larger isophotal/SBF working radius.
* ``RESID_EXTRA_MASK_SIGMA=3``, ``RESID_EXTRA_MASK_NPIX=9`` — порог и площадь
  **диагностических** остаточных кандидатов / diagnostic residual-candidate threshold and area.
* ``RESID_EXTRA_MASK_KERNEL_SIGMA=1.0`` — сглаживание кандидатов на масштабе
  одного пикселя / one-pixel Gaussian smoothing for candidate detection.
* ``RESID_EXTRA_MASK_DILATE=2``, ``RESID_EXTRA_MASK_MAX_AREA=3000`` — расширение
  и верхний размер этих кандидатов / candidate dilation and maximum segment area.
* ``RESID_EXTRA_MASK_POSITIVE_ONLY=True`` — искать положительные пики, не
  отрицательные провалы модели / look for positive peaks, not negative model residuals.
* ``RESID_EXTRA_MASK_BRANCH="compactresidmask"`` — метка выходных файлов, не
  формула обработки / output label only. Эта ячейка сама не добавляет кандидатов
  в финальную science mask; её определяет следующий каталог / the following catalog defines the final mask.

Грубая модель и изофоты / Coarse model and isophotes
-------------------------------------------------

Sersic здесь нужен для вспомогательного заполнения пропусков; финальная модель
строится по измеренным изофотам. Sersic supports gap filling, not the final SBF galaxy model.

* ``SERSIC_FIT_SAMPLE_STEP=16`` — каждый 16-й валидный пиксель плоской выборки,
  не уменьшение обеих координат в 16 раз / stride through the flattened valid-pixel sample.
* ``SERSIC_INIT_PERCENTILE=95``, ``SERSIC_MIN_AMPLITUDE=1e-6`` — яркий, но не
  максимальный старт амплитуды и положительный минимум /
  bright-percentile amplitude seed and positive seed floor, in image units.
* ``SERSIC_INIT_REFF_DIV=8.0``, ``SERSIC_MIN_REFF=10.0`` — стартовый радиус
  min(image shape)/8, не меньше 10 px / image-size-based radius seed, at least 10 pixels.
* ``SERSIC_INIT_N=4.0`` — традиционный старт для гладкой ранней галактики, не
  фиксированное измеренное n / conventional early-type seed, not a fixed measured index.
* ``SERSIC_INIT_ELLIP=0.2``, ``SERSIC_INIT_THETA=0.0`` — невырожденные
  геометрические старты; угол в радианах / nondegenerate ellipticity/orientation seeds.
* ``SERSIC_REFF_BOUND_MIN=5.0``, ``SERSIC_N_BOUNDS=(0.5,8.0)``,
  ``SERSIC_ELLIP_BOUND_MAX=0.9`` — ограничить нефизично узкие/вытянутые решения
  вспомогательного фита / regularizing bounds on the auxiliary fit, not measured limits.
* ``HALF_SIZE=3000`` — максимальная полуширина crop вокруг центра, с обрезкой
  по краям кадра / maximum centre-crop half-width in pixels, clipped at image edges.
* ``ISO_START_SMA=50.0``, ``ISO_START_EPS=0.2``, ``ISO_START_PA=0.0`` — старт
  изофоты: пиксели, эллиптичность, радианы / initial semimajor axis, ellipticity and angle.
* ``ISO_MAXSMA_FIT=2000.0``, ``ISO_MINSMA=15.0`` — границы радиусов изофот;
  покрытие научных колец проверяется отдельно / fitted-radius bounds; ring coverage has a separate QC gate.
* ``ISO_STEP_MAIN=10.0`` — шаг в пикселях при ``linear=True`` /
  linear-mode semimajor-axis spacing in pixels.
* ``ISO_STEP_COARSE=20.0`` — параметр старой запасной попытки ``fit_image``;
  в ней ``linear=True`` не передаётся. Не трактовать автоматически как 20 px /
  old alternate-call step; without explicit linear mode its interpretation follows Photutils, not a guaranteed 20-pixel step.
* ``ISO_FIT_USE_REAL_PIXELS=True``, ``ISO_FIT_ALLOW_FILLED_FALLBACK=True`` —
  сначала реальные пиксели, затем разрешённая попытка с заполнением; выбранная
  ветвь протоколируется / real-pixel fit first, documented filled-data retry if needed.
* ``MODEL_FULL_WARN_ANNULUS_COVERAGE=0.95`` — предупреждение о неполном
  покрытии моделью, не повод выдумывать свет вне модели или исключать галактику /
  coverage warning, not permission to extrapolate or automatically reject a galaxy.

Области, численная защита и винзорирование / Regions, guards, winsorization
-------------------------------------------------------------------------

* ``SBF_LIT_INNER_ARCSEC=(8.2,16.4)``, ``SBF_LIT_OUTER_ARCSEC=(16.4,32.8)`` —
  принятые круговые угловые кольца для сопоставления с Jensen, не равные физические
  радиусы всех галактик / adopted comparison geometry, not matched physical radii.
* ``ROBUST_SCALE_FLOOR=1e-12`` — защита весов/масштабов от деления на ноль;
  не добавленная погрешность / numerical positive floor, not an uncertainty term.
* ``GEOM_Q_FLOOR=1e-3`` — не допускать нулевую малую ось в геометрии /
  prevent a singular zero axis ratio.
* ``MIN_PIXELS_ISO_CROP=5000``, ``MIN_PIXELS_SBF=5000`` — минимальная поддержка
  изофотного фита и спектра; счётчик валидных пикселей, не независимых звёзд /
  minimum valid-pixel support, not independent stars or Fourier modes.
* ``MIN_PIXELS_SIGMA_CLIP=100``, ``MIN_COLOR_PIXELS=100`` — не оценивать
  устойчивую шкалу/цвет на почти пустой выборке / reject nearly empty robust-scale/color samples.
* ``MIN_POINTS_PK_BIN=10`` — минимум Fourier-ячеек в радиальном интервале /
  minimum Fourier-grid samples per radial bin.
* ``MIN_POINTS_FIT=10`` — минимум пригодных радиальных точек для спектрального
  фита / minimum usable radial spectrum points; not a theorem about fit precision.
* ``MIN_WORKING_ISOPHOTES=10``, ``MIN_ISOPHOTES_MODEL_PROFILE=8`` — нижние
  пороги для принятия списка и построения профиля модели /
  minimum working isophote list and model-profile support.
* ``CLIP_SIGMA_QC=3.5``, ``CLIP_MAXIT_QC=5`` — шкала устойчивых статистик и
  предел итераций; затем значения заменяются границами median±3.5σ /
  robust-statistics settings followed by capping, not removal of those pixels.
* ``CLIP_TAG_QC="3p5"`` — имя продукта, автоматически получаемое из порога /
  filename tag derived from the threshold. Порядок raw/final различается — см. начало /
  source versus adopted ordering differs; see Scope.

PSF и спектр / PSF and spectrum
-------------------------------

* ``FFT_WORKERS=-1`` — все доступные SciPy FFT workers, только производительность /
  use available FFT workers; a performance setting.
* ``PSF_SIZE=129`` — нечётный stamp с центральным пикселем; принятый размер,
  сравнивавшийся с 257 в диагностике / adopted odd-sized stamp; comparison with 257 is a separate diagnostic, not absolute PSF calibration.
* ``PSF_NLAMBDA=7`` — дискретизация пропускания фильтра в STPSF /
  wavelength sampling through the bandpass; no convergence proof is implied.
* ``PSFREF=None`` — по умолчанию без пользовательской эмпирической PSF /
  no user-supplied empirical PSF by default.
* ``PSF_MAX_OPD_DELTA_DAYS=7.0`` — максимум для science-stage выбора OPD,
  переопределяется ``SBF_PSF_MAX_OPD_DELTA_DAYS``. Не путать с 30-дневным
  инвентарным допуском загрузчика / science-stage age cap, distinct from downloader inventory coverage.
* ``PSF_NEAREST_OPD_COUNT=min(3,Neligible)`` — до трёх ближайших допустимых
  состояний фронта / bounded local time-variation sample, not three guaranteed files.
* ``PSF_FIELD_OFFSETS_PIX=((256,0),(-256,0),(0,256),(0,-256))`` — четыре
  пространственных пробы, ограниченные границами детектора /
  four bounded detector-position probes, not a complete field calibration.
* ``FFT_E_REALIZATIONS_MAIN=64``, ``FFT_E_REALIZATIONS_DIAG=64`` — число
  усредняемых реализаций E(k); уменьшение шума Монте-Карло при конечном времени /
  Monte Carlo averaging budget; accuracy must be assessed separately.
* ``FFT_RNG_SEED=1489`` — воспроизводимый случайный поток, число физического
  смысла не имеет; индекс PSF добавляется к seed / reproducible random stream, offset per PSF.
* ``FFT_KBINS_N=80`` — **80 границ, 79 радиальных интервалов** в ``linspace`` /
  80 bin edges, not 80 populated bins.
* ``SBF_FFT_CROP_PAD_PX=PSF_SIZE=129`` — запас вокруг рабочей маски для
  ограниченного FFT-crop / crop padding tied to the PSF support size.
* ``FFT_K_RANGE_MAIN=(0.04,0.25)`` — принятый диапазон в cycles/pixel;
  подавление влияния крупных структур при сохранении PSF-формы спектра /
  adopted fit window, limiting large-scale residual contamination.
* ``SBF_REGION_K_WINDOWS=[(0.01,0.25),(0.03,0.25),FFT_K_RANGE_MAIN]`` —
  согласованная проверка нижней границы, не независимые измерения /
  lower-cut sensitivity windows, not independent measurements.

Каталог, P_r и цвет / Catalog, P_r and color
------------------------------------------

* ``SBF_PR_ENABLE=True`` — учитывать остаточную мощность неразрешённых шаровых
  скоплений и фоновых галактик / subtract unresolved-cluster/background-galaxy power.
* ``SBF_PR_SOURCE_IMAGE="img_minus_model_full_unmasked"`` — каталог строится
  на остатке до маски каталога, чтобы не искать уже удалённые источники /
  catalog residual before application of its own source mask.
* ``SBF_PR_MAG_BIN=0.25`` — ширина LF-интервала в mag: компромисс числа
  источников и разрешения / source-count versus magnitude-resolution choice.
* ``SBF_PR_MLIM_OVERRIDE=None``, ``SBF_PR_MLIM_OFFSET=0.0`` — автоматический
  предел без ручного сдвига / automatic threshold with no arbitrary offset.
* ``SOURCE_CATALOG_RELIABLE_SNR=5.0``, ``SOURCE_CATALOG_AUTO_CUT_QUANTILE=0.95`` —
  предел по 95-му перцентилю каталожных величин с SNR≥5. Это приближение, не
  измеренная искусственными источниками полнота / approximate catalog threshold, not injection-measured completeness.
* ``SBF_PR_MIN_SOURCES_GLOBAL=20`` — минимум для этого каталожного правила /
  minimum source count for the preferred catalog threshold.
* ``SBF_PR_FALLBACK_QUANTILE=0.80`` — явно маркируемый запасной предел при малом
  каталоге / recorded approximate threshold for sparse catalogs.
* ``SBF_PR_MIN_FIT_BINS=3`` — нижний запрос; текущий четырёхпараметрический
  LF-фит фактически требует ``max(4,3)=4`` непустых интервала /
  requested lower bound; the current LF fit enforces at least four nonempty bins.
* ``SBF_PR_GAMMA_BOUNDS=(0.10,0.70)`` — диапазон наклона фоновых counts;
  верхняя граница ниже 0.8, чтобы вклад counts×flux² убывал к слабым источникам /
  slope bounds keep the variance-weighted background tail decreasing.
* ``SBF_PR_DEFAULT_GAMMA=0.30`` — начальный наклон и значение фиксированной
  запасной формы LF, не измеренный наклон каждой галактики /
  initial/fixed-shape slope, not an observed value for every target.
* ``SBF_PR_CATALOG_MIN_FLUX_JY=0.0`` — только положительный поток имеет
  конечную логарифмическую величину / require positive flux for magnitudes.
* ``SBF_PR_DET_SIGMA=SIGMA_DET=2.5``, ``SBF_PR_DET_NPIXELS=MASK_NPIXELS=4``,
  ``SBF_PR_MAX_COMPACT_AREA=PREMASK_MAX_COMPACT_AREA=5000``,
  ``SBF_PR_DO_DEBLEND=DO_DEBLEND=True`` — наследуемые каталожные настройки,
  не четыре дополнительные независимо подобранные константы / inherited detector settings.
* ``SBF_PR_MASK_DILATE_ITERATIONS=3`` — конечная маска расширяет каталожные
  сегменты для подавления крыльев источников / source-mask dilation to reduce unmasked wings.
* ``GCLF_SIGMA_MAG=1.2`` — принятая ширина гауссовой GCLF в mag; отдельные
  ширины являются тестом чувствительности / adopted GCLF width, with alternatives tested separately.
* ``PR_FAINT_INTEGRATION_SPAN_MAG=12.0`` и **4097 узлов** — конечный интеграл
  P_r до m_cut+12 mag, 4096 равных шагов ≈0.00293 mag. Это сетка интегрирования,
  **не размер PSF/kernel**; сама плотная сетка не доказывает малость хвоста за
  пределом / finite trapezoidal quadrature; grid density alone does not prove tail convergence.
* ``COLOR_CLIP_SIGMA=3.0``, ``COLOR_CLIP_MAXIT=5`` — общий устойчивый отбор
  пикселей двух фильтров для цвета, отдельно от SBF-винзорирования /
  common two-band color-pixel selection, distinct from SBF winsorization.

Важные коэффициенты без отдельных имён / Other significant coefficients
----------------------------------------------------------------------

* **1.4826** ≈1/Φ⁻¹(0.75) переводит MAD в σ для нормального распределения.
  При ненормальном распределении это условная устойчивая шкала, не доказанная
  Gaussian σ и не ошибка среднего / Gaussian-consistent MAD scale, not automatic
  normality or a standard error of the mean.
* ``SplitCosineBellWindow(alpha=0.35,beta=0.30)`` — регуляризация Fourier-отношения
  PSF при согласовании F090W с F150W для цвета. ``beta`` задаёт радиус единичного
  ядра окна, ``alpha`` — ширину косинусного спада, в долях полуширины сетки.
  Они подавляют неустойчивые высокочастотные отношения; оптимальность этих
  значений не установлена / taper-core radius and taper width regularize the
  PSF-matching Fourier ratio; the chosen values are not proven optimal.
* **2.5**, **0.4**, **ln(10)** в величинах и распространении ошибок — алгебра
  определения magnitude / magnitude-definition algebra, not fitted coefficients.

Изменение эвристик моделирования/маски меняет измерительный эксперимент и требует
отдельной проверки; изменение seed/сетки требует численной проверки. Этот файл
объясняет существующие значения, но не заменяет такие эксперименты.
Changing model/mask heuristics changes the measurement experiment; changing seeds
or grids requires numerical checks. This inventory documents choices, not their universal optimality.
