Измерение SBF / SBF measurement
==============================

Назначение / Scope
------------------

RU. Текущий код измеряет SBF в 14 галактиках GO-3055. Обработчики не
скачивают и не удаляют исходные FITS. Скачивание выполняет отдельно
``download_go3055_go7763.py``. Команды ниже запускаются из ``code/`` с
активированным окружением. Научный notebook не исполняется при импорте.

EN. The current pipeline measures the 14 GO-3055 galaxies. Processing never
downloads or deletes input FITS; ``download_go3055_go7763.py`` owns downloads.
Run these commands from ``code/`` in the active environment. Importing modules
does not execute the scientific notebook.

Порядок / Order
--------------

1. ``py run_sbf_2_batch.py`` — F150W: модель, маска, PSF и исходные спектры /
   F150W model, mask, PSF and source spectra.
2. ``py run_sbf_2_normalized_winsor.py`` — принятый F150W после нормировки /
   adopted F150W measurement after normalization, using saved source products.
3. ``py run_sbf_f090w.py`` — F090W: модель/маска/PSF и нормированные спектры /
   F090W source products and normalized spectra. The accepted F150W centre
   provides the WCS reference.
4. Анализ и расстояния выполняются отдельно / Calibration and distances are
   computed separately; a spectral ``sigma_adopted_internal`` is not the full
   uncertainty of the distance printed in the article.

``--galaxies "NGC 4636"`` ограничивает очередь / selects a target subset.
Обычный повтор команды использует проверенные продукты / An ordinary restart
reuses verified products. ``--force-reprocess`` (F150W source), ``--force``
(normalized F150W) and ``--force-source`` / ``--force-spectra`` (F090W) explicitly
request recomputation; do not add them to an ordinary restart.

``run_sbf_2_batch.py`` больше не поддерживает ``--allow-download``,
``--download-worker``, ``--allow-input-cleanup`` или ``--prefetch-targets``.
EN. Those retired switches belonged to the duplicated downloader. The explicit
``--no-download`` and ``--no-cleanup-inputs`` remain harmless aliases of the
always-offline, input-preserving behaviour for old commands.

Модули и контракты / Modules and contracts
----------------------------------------

``run_sbf_2_batch.py``
  RU: очередь, изолированные процессы, исполнение ячеек ``sbf-2.ipynb``,
  проверка пяти FITS и двух CSV, журналы и восстановление после прерывания.
  EN: queue, subprocess supervision, notebook executor, five-FITS/two-CSV
  completion gate and restart state. SBF3 output handling is not supported.

``sbf2_normalized_winsor_core.py``
  RU: компактные входные кэши, E(k), нормировка, винзорирование и FFT/МНК.
  EN: compact inputs, Monte Carlo E(k), normalization, winsorization and
  Fourier-space weighted least squares. The adopted spectrum reads the saved
  normalized full-frame FITS back; it is not measured from an unsaved precursor.

``run_sbf_2_normalized_winsor.py``
  RU: последовательный запуск этого ядра для F150W. EN: sequential F150W
  wrapper, CSV progress and aggregate sensitivity tables.

``sbf090_pipeline_support.py``
  RU: сборка F090W execution-копии notebook, диагностика изофот, PSF-кэш.
  EN: deterministic F090W notebook adapter, isophote QC and PSF cache.
  Exact source anchors deliberately fail loudly if the base notebook changes.

``run_sbf_f090w.py``
  RU: два возобновляемых этапа и публикация стабильных ссылок на продукты.
  EN: two resumable stages and stable product links. ``science_config`` is the
  single settings constructor used by both parent and worker. Readiness checks
  load PSFs with ``write_table=False`` and do not rewrite their CSV catalogue.

Научные параметры / Scientific settings
--------------------------------------

Значения зафиксированы для сопоставимости с принятой обработкой; это не
универсальные оптимумы. / Values preserve the adopted analysis, not universal
optima. Changing them is a new scientific experiment.

* ``normalized_full_3p5``: общий порог 3.5 sigma по всей валидной положительной
  области модели после деления на sqrt(model). / One threshold over the full
  valid positive-model support after division by sqrt(model). Winsorization
  caps tails; it does not delete pixels or redefine the source mask.
* ``RAW_PRODUCTION_SIGMA=3.5``, ``RAW_PRODUCTION_MAXITERS=5``: прежний порог и
  предел итераций оценки центра/масштаба для воспроизводимости. / Frozen
  threshold and robust-statistic iteration budget. The same full-region
  estimation settings are used for the adopted normalized branch.
* ``normalized_sigma=3.5``: меняет только контроль по объединению колец, не
  принятую ветвь. / Only the union-of-annuli sensitivity control is adjustable.
* ``kmins=(0.01,0.03,0.04)``, ``kmax=0.25``: три проверки низкочастотной
  чувствительности; основной нижний предел 0.04. / Three low-frequency windows;
  adopted lower edge 0.04, fixed upper edge 0.25 cycles/pixel.
* ``MIN_FIELD_WAVES=10``: фактический kmin не ниже 10/min(crop.shape). /
  Finite-crop low-frequency guard inherited unchanged from the notebook.
* ``k_bins=80``: 80 границ, то есть 79 радиальных бинов. / Eighty edges give
  79 radial bins, matching saved spectra. ``min_modes_per_bin=10`` and
  ``MIN_FIT_BINS=10`` reject undersupported bins/fits; they are engineering
  guards, not fitted population parameters.
* ``e_realizations=64``, ``random_seed=1489``: фиксированный бюджет и
  произвольное зерно Монте-Карло. / Fixed expectation budget and arbitrary
  reproducible seed; neither is a guarantee of physical accuracy.
* Кольца 8.2–16.4 и 16.4–32.8 arcsec повторяют принятую геометрию сравнения. /
  Angular annuli 8.2–16.4 and 16.4–32.8 arcsec preserve the comparison geometry;
  they are not equal physical radii across galaxies.
* ``F090W_PSF_SIZE=129``, ``F090W_PSF_COUNT=5``: принятый проверенный размер;
  центральная модель и четыре смещения на детекторе. / Adopted tested stamp;
  one central model plus four detector offsets. Odd size provides a centre pixel.
* WSS OPD не дальше 7 дней: фиксированный предел близости эпохи. / Fixed
  seven-day maximum time separation; an operational match criterion, not an
  uncertainty estimate. PSF cache checks require unit sum within 1e-5,
  pixel-scale agreement within 0.01 and matching observation MJD within 1e-6 d;
  these numerical tolerances reject incompatible caches.
* ``ARCSEC2_TO_SR=2.350443e-11``, ``MJY_SR_TO_JY_ARCSEC2=2.350443e-5``:
  перевод телесного угла и MJy/sr. / Rounded unit conversions deliberately
  identical to the frozen notebook. ``AB_ZERO_JY=3631`` is the AB reference.
* ``MAD_TO_SIGMA=1.4826``: Gaussian-equivalent MAD. ``POSITIVE_FLOOR=1e-12``:
  защита деления, не добавочная sigma / arithmetic floor, not additional error.
* ``PIXEL_CLOSURE_TOL=1e-5``: абсолютный допуск реконструкции float32-пикселей.
  ``SPECTRAL_CLOSURE_TOL=0.005``: старый контроль в mag и относительных P0/P1. /
  Pixel and spectral closure tolerances check reproduction of the old branch;
  they are not reported measurement uncertainties.

Ошибки и контрольные ветви / Errors and retained controls
-------------------------------------------------------

RU. Модель спектра — P(k)=P0 E(k)+P1. МНК взвешен обратной дисперсией
радиального среднего; ковариация умножается на max(chi2/dof,1). Поправка Pr
вычитается из P0 до перевода в звёздную величину. MAD ансамбля PSF и разброс
по k-окнам остаются отдельными диагностическими величинами. В spectral CSV
есть консервативная полуразность колец; это не автоматически окончательная
ошибка калибровки или расстояния. Финальный бюджет рассчитывает анализатор.

EN. Fit P(k)=P0 E(k)+P1 by inverse-variance weighted least squares using the
radial mean's SEM. Scale covariance by max(reduced chi-square,1). Subtract Pr
from P0 before magnitude conversion. PSF-ensemble MAD and k-window scatter
remain separately recorded. Spectral CSV files retain a conservative annular
half-difference diagnostic; they are not the final calibration/distance budget.

RU. ``no_winsor``, ``raw_global_3p5`` и ``normalized_union_*`` не мёртвый код:
их используют notebook сравнения, тест искусственного сигнала и таблицы
чувствительности. Старые значения sigma/результаты не подменяются принятыми.
EN. These three controls are still consumed by comparison notebooks, synthetic
recovery and sensitivity tables. Removing them would erase reproducibility of
an explicitly used validation, not merely remove an obsolete branch.

Изофоты F090W / F090W isophotes
------------------------------

RU. Сохранены ограниченное уточнение центра, несколько стартов изофот и
переход к фиксированному центру/заполненным пикселям: это работающая защита
для конкретных кадров, не молчаливая подстановка результата. Все попытки
записываются, принимается только решение с полной областью и пройденным QC.
EN. Bounded centre refinement, ranked initial radii and controlled fixed-centre /
filled-data attempts remain because actual frames need them. Every attempt is
recorded; failed science-annulus QC is never accepted as success.

* Стартовые радиусы / start radii: 40, 50, 60, 70, 100 pixels; bootstrap до /
  through 200 pixels and at least 0.75 of that range. These are search budgets.
* Поиск центра / centre search: 250-pixel neighbourhood, maximum 50-pixel
  refinement. Sersic seed requires ellipticity 0.05–0.85 and centre within
  100 pixels. These bounds reject implausible starting guesses.
* Крупный внешний загрязнитель / large external contaminant: outside the
  central 250 pixels, peak S/N above 100, area above the existing compact-source
  ceiling. Conservative heuristics prevent the galaxy core being masked as a star.
* ``F090W_MIN_WORKING_ISOPHOTES=10``: минимальная поддержка профиля /
  minimum profile support.
* ``F090W_ISOPHOTE_QC_LIMITS``: median centre shift <=50 px; science-annulus
  centre offset <=15 px; stop-code-2 fraction <=0.30 and run <=8; frozen-code
  fraction <=0.25 and run <=20; singular-code count=0; centre step <=10 px;
  ellipticity step <=0.03; complex-shape step <=0.05; outward intensity rise
  <=0.05. RU: фиксированные эвристики против застывшей/скачущей геометрии;
  не вероятности и не научные отсечения выборки. EN: fixed guardrails against
  frozen/jumping solutions, not confidence levels or sample selection cuts.
* ``NGC 4636``: только для изофот снимается ложное замкнутое кольцо premask
  внутри внутренней SBF-границы. / Only the isophote-fitting premask is relaxed
  inside the inner SBF boundary; the final science mask is unchanged.
* ``*_METHOD`` and ``F090W_SOURCE_SCHEMA=3`` name the accepted algorithm and
  product contract. They prevent incompatible old results from passing restart
  validation. They are labels, not tunable numerical coefficients.

Хранение и выполнение / Storage and execution
--------------------------------------------

RU. ``EXPERIMENT_VERSION=v3`` описывает контракт результатов; отдельные
``INPUT_CACHE_VERSION`` и ``EXPECTATION_CACHE_VERSION=v2`` оставлены, потому
что алгоритмы этих кэшей не менялись. Пути и схемы полей централизованы для
согласованности writer/reader. Исторические SHA — provenance, не команда
пересчитать всё после косметического изменения notebook.

EN. Result version v3 is distinct from unchanged input/expectation cache v2.
Path/default and CSV field constants define shared writer/reader contracts.
Fingerprints protect numerical inputs; notebook SHA records provenance and
does not alone force recalculation. Small-file hashing is capped at 16 MiB,
read in 1-MiB blocks to avoid hashing multi-GiB images on every restart.

``fft_workers=-1`` uses available SciPy workers. Full-frame output is written
in 256-row chunks to limit temporary memory without changing pixel values.
``save_ring_fft_fits`` and ``save_all_branch_fits`` control diagnostic storage.
Plots use quarter-resolution previews and the 99.5th absolute-value percentile
only for display; neither affects FFT measurements.

RU. Пределы ресурсов — настройки машины, не астрофизика. / Resource defaults
are machine safeguards, not science: F150W 48 h campaign, 12 h worker,
60 min soft-stop reserve, up to 5 attempts, 30 s telemetry, 60 s polling,
300/10 s termination grace, 40 GiB free-space floor plus 6 GiB next-product
estimate. Zero RAM/RSS limits disable optional ceilings. F090W reserves
20 GiB before a new target and reports an approximate 50 GiB campaign size.
All are explicit CLI choices. A fitting failure is recorded, not replaced
by an invented measurement.

Проверка / Verification
----------------------

RU. В ``sbf-2-systematics.ipynb`` оставлен контроль альтернативного N(k).
``TILE_SIZE=512`` и ``TILE_MARGIN=96`` задают размер и отступ пробных площадок;
доли валидных пикселей 0.80/0.55 ограничивают дырявые окна, максимум 4 площадки
ограничивает время. Отношение WHT 0.5–2 и относительный MAD <=0.25 отбирают
сопоставимую экспозицию; маска 5 sigma с расширением на 3 пикселя подавляет
источники; минимум 1000 пикселей защищает статистику от пустой выборки.
Это эвристики диагностического опыта, не параметры принятой калибровки.

EN. The retained N(k) control uses 512-pixel tiles with a 96-pixel margin,
valid fractions 0.80/0.55 and at most four tiles to bound support and runtime.
WHT ratios 0.5–2 and relative MAD <=0.25 select comparable exposure; a 5-sigma
mask dilated by three pixels suppresses sources. At least 1000 valid pixels
are needed for the tile statistics. These are fixed diagnostic heuristics,
not calibrated confidence levels or production sample cuts.

RU. Старый флаг ``physical_solution`` не гарантирует физичность каждого
коэффициента: в constant-фите P1 не ограничен снизу, тогда как у N(k) проверяется
Pn>=0. Этот флаг нельзя использовать как симметричное сравнение моделей.
EN. The legacy ``physical_solution`` flag allows unconstrained P1 in the
constant-noise fit but requires nonnegative Pn for N(k); it is not a symmetric
physical-model comparison or proof that all fitted coefficients are physical.

RU. В диагностике 129/257 повторно используется только PSF с совпадающими
записанными настройками. Cache hit не переписывает исходный FITS/header;
неизвестные версии генерации не заменяются версиями текущего окружения.
EN. The size test reuses only PSFs with matching recorded settings. Reuse
does not rewrite FITS provenance; unknown generator versions stay unknown.

``py -m unittest test_run_sbf_2_batch test_run_sbf_f090w test_run_sbf_2_normalized_winsor``

RU. Тесты используют маленькие искусственные массивы/временные FITS, не
галактики. Они проверяют контракты, FITS-кэш и МНК; не заменяют полный
научный прогон. EN. Tests use tiny synthetic arrays/FITS, verify contracts,
cache behaviour and weighted fitting, and do not claim end-to-end validation
of all galaxy modelling systematics.
