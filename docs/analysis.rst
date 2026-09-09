Анализ и рисунки / Analysis and figures
=====================================

Назначение / Scope
------------------

RU: Этот слой читает принятые результаты двух обработчиков, калибрует SBF по
индивидуальным расстояниям TRGB Paper IV, проверяет ошибки и строит рисунки.
Он не заменяет обработку изображений. Основные модели: F150W — постоянная,
F090W — линейная; все 14 галактик остаются в основной выборке.

EN: This layer reads the accepted products, calibrates SBF against individual
Paper IV TRGB distances, checks uncertainties and renders figures. It does not
replace image processing. The adopted models are constant F150W and linear
F090W, both using all 14 calibrators.

Точки входа / Entry points
-------------------------

Все команды из ``code/`` с активным окружением. / Run from ``code/`` with the
environment activated. Importing a builder never runs its analysis.

* ``sbf-2-graph.ipynb``: F150W calibration, tables, diagnostics and plots.
  RU: запускать последовательно вручную; старые проверки сохранены как
  обоснование метода. EN: run sequentially; diagnostic cells document why
  alternatives were rejected, not alternative adopted distances.
* ``sbf-f090w-graph.ipynb``: the corresponding F090W analysis; stable
  ``products.json`` files identify each accepted source and spectrum.
* ``py -m figures.build_sbf_f090w_graph_notebook``: recreate the F090W notebook source,
  **not execute it**. RU: заменяет файл и его сохранённые выводы; обычно не нужен.
  EN: replaces the saved notebook, including outputs; normally unnecessary.
* ``py -m figures.build_go3055_article_figures``: draw both bands from completed CSVs.
  RU: текущие публикационные графики. EN: the current publication plotter.
* ``py -m figures.build_f090w_residual_montage``: montage of saved normalized F090W
  residuals. Image subsampling/stretch are display-only.
* ``py -m figures.build_f090w_appendix_diagnostics``: the seven selected appendix
  figures: three paired histograms and two spectrum/PSF comparisons per band.
  It reads FITS pixels for histograms, but does not repeat source extraction or
  spectral measurement. Superseded duplicate figures are no longer generated.
* ``py -m figures.build_sbf2_article_tables``: joins F150W/F090W/TRGB/Jensen results,
  writes CSV/TeX tables and provenance. No fitting or figure production.
* ``py -m figures.publish_article_assets``: refresh selected copies beside the TeX;
  ``--check`` validates without writing. The manifest is
  ``texts/paper_work/materials/figure_sources.json``. Producer copies stay in
  ``runs/`` so notebooks can redraw them independently.

RU: Рабочие notebook остаются в ``code/``. Служебные сборщики находятся в
``code/figures/``, общие пути — в ``code/sbf/sbf_paths.py``. Литературные CSV
лежат в ``code/reference/``. Продукты F150W разделены на
``runs/F150W/source/``, ``spectra/`` и ``analysis/``; F090W — в ``runs/F090W/``.

EN: Notebook entry points remain in ``code/``. Builders live in ``figures/``;
``sbf.sbf_paths`` anchors paths independently of the current directory.
Reference CSVs live in ``code/reference/``. F150W source products, spectra and
analysis are separated under ``runs/F150W/``; F090W uses ``runs/F090W/``.

Контрольные опыты / Validation experiments
----------------------------------------

RU: ``sbf-2-systematics.ipynb``, ``sbf-2-normalized-winsor.ipynb``, recovery и
сравнение четырёх ветвей удалены из рабочей копии вместе со старым карантином.
Это история проверок шума, PSF и порядка операций, не часть релизного запуска.
Основной код её не импортирует. Принятые таблицы чувствительности сохранены в
рабочих результатах. Для возвращения к архивному опыту потребуется отдельно
восстановить его код, зависимости и входы. История Git содержит только ранее
отслеживаемые файлы и не восстанавливает удалённые игнорируемые данные.

EN: These validation notebooks and their helper scripts have been removed
from the working tree; they are not release dependencies. Resuming an experiment requires explicit
restoration of its code and inputs; accepted sensitivity tables remain in the
working results. Git history can restore only previously tracked files, not
deleted ignored data. Rejected branches remain useful controls;
only ``normalized_full_3p5`` is
adopted. ``sbf2_normalized_winsor_recovery.run_recovery_test`` generates paired
synthetic realizations to isolate operation order. It does not validate sky,
isophotal models, catalogue completeness, or correlated detector noise.
``summarize_normalized_winsor_distances.py`` recalibrates the four saved branches
with the same constant model and LOO protocol; it does not remeasure images.

Числа и их смысл / Numerical choices
-----------------------------------

* ``8.2, 16.4, 32.8 arcsec``: boundaries of the adopted circular annuli, retained
  for Jensen-like geometry. They are angular, not matched physical radii.
* ``3.5 sigma``: adopted full-support winsorization **after** division by the
  square root of the galaxy model. Alternatives at 3 and 4 sigma color the
  histogram for comparison; they do not change the saved measurements.
* ``k=0.04..0.25 pixel^-1``: the published window. Other lower cutoffs are
  diagnostics. Plotters select stored fits and never re-optimize this window.
* ``0.047 mag``: shared Paper IV absolute TRGB scale. ``0.063 mag``: Paper III
  common scale, already including its zero-point term. Shared errors do not
  shrink with the number of galaxies. Jensen F110W coefficients and errors in
  the table builder are literature inputs from Paper III Table 1 / Eq. (2),
  not fitted JWST F090W coefficients. F160W is a separate five-object comparison.
* ``A090/E(B-V)=1.4156``, ``A150/E(B-V)=0.6021``: the adopted bandpass coefficients
  used by the saved extinction metadata. Their difference propagates reddening
  into color; they are not independent errors added twice.
* ``sigma(P_r)/P_r=0.25``: an adopted uncertainty assumption, not measured
  completeness. The saved zero/adopted/doubled-P_r tests quantify sensitivity.
* F090W bootstrap budgets ``400`` (curves/parameters), ``350`` (each LOO target),
  seed ``3090``; four-branch comparison budget ``300``, seed ``3055``. These
  reproduce the recorded Monte Carlo experiment and bound runtime. They are
  computational choices, not proof of convergence or physical constants.
* A starting scatter of ``0.05 mag`` initializes likelihood optimization;
  bounds ``1e-5..1 mag`` maintain a positive scatter and approximate zero.
  ``1e-12`` protects positive variances numerically. These are not extra errors.
* Exploratory exponential scale ``0.05 mag`` fixes its shape and avoids an
  extra poorly constrained parameter. A ``0.01 mag`` color floor and half the
  radial color difference are sensitivity tests, not primary photometric errors.
* Synthetic recovery uses ``512`` pixels, amplitudes ``P0=0.90, P1=0.08``,
  ``64`` trials, ``96`` expectation realizations and ``80`` radial edges (79 bins).
  Its radial model, annuli and 35 mask holes are a reproducible toy fixture,
  not measured galaxy properties. Separate seed offsets avoid reusing the
  same random realization for the template and measured signal.
* PSF-size diagnostic: ``129`` versus ``257`` pixels, FFT grid ``512``;
  STPSF ``nlambda=7``, oversampling ``4/1`` reproduce the saved comparison.
  Do not interpret its magnitude shift as an independently measured PSF sigma.
* Figure sizes, colors, label offsets, histogram bins and display percentiles
  are presentation choices only. Masked pixels stay black; display subsampling
  never modifies the arrays supplied to the scientific FFT.

RU: Физические/статистические настройки выше отделены от численных предохранителей
и оформления. Менять первые без нового теста чувствительности нельзя. Для
bootstrap допускается пропуск только явно неудачных оптимизаций; если успешно
меньше половины попыток, расчёт останавливается, а не выдаёт случайную sigma.

EN: Physical/statistical settings are distinct from numerical safeguards and
styling. Failed bootstrap optimizations may be skipped, but fewer than half
successful draws stops the calculation rather than producing a spurious sigma.
That threshold is a failure guard, not a claim of Monte Carlo precision.
Missing fit inputs or uncertainties raise an error; they are neither silently
dropped nor replaced with zero. / Пропуски во входах фита и sigma не удаляются
молча и не заменяются нулями.

RU: В старых диагностических кадрах F150W сохранены используемые псевдонимы
столбцов. Полная вычитка выявила отдельные несогласованности ковариации при
смене TRGB-якоря и в аннулярном фите; это не исправлено под видом уборки кода.
Сохранённые выводы ноутбуков не пересчитывались. ``fit_basis_model`` — прежний
исследовательский фит только с вертикальными ошибками, не основной EIV-фит.

EN: F150W diagnostic column aliases remain supported. The source audit found
separate covariance inconsistencies in cluster-anchor and annular diagnostics;
repository cleanup does not silently revise those scientific assumptions.
Saved outputs were not recomputed. The exploratory ``fit_basis_model`` uses
vertical errors only, unlike the primary errors-in-variables fit.

Ошибки и интерфейсы / Errors and interfaces
-----------------------------------------

RU: Для двух колец независимая спектральная часть складывается квадратично с
принятыми весами. Общие PSF, фон и модель P_r объединяются с теми же весами как
коррелированные между кольцами члены. Разность колец — отдельная диагностика.
Для расстояния добавляются внутренний разброс калибровки и конечность выборки;
общая шкала TRGB показывается отдельно. LOO — внутренняя проверка, не внешняя.

EN: Independent annular spectral terms combine in quadrature with the adopted
weights; shared PSF, background and P_r model terms combine as correlated
annular contributions. The annular difference remains a separate diagnostic.
Distance errors additionally contain intrinsic calibration scatter and finite
calibration uncertainty. LOO is internal validation, not an external test.

* ``model_basis / model_derivative``: basis and analytic derivative of each
  law. ``fit_model`` minimizes the same Gaussian likelihood with color errors
  and covariance; ``loo_predictions`` excludes the target from each training set.
* ``bootstrap_band / bootstrap_parameters``: curve and parameter resampling.
  Notebook bands and the article's uniform ``±sigma_int`` strip have different
  meanings; do not substitute one for the other by relabeling.
* ``distance_from_modulus / distance_uncertainty``: conversion to Mpc and
  first-order propagation; ``common_part`` extracts a known quadrature term.
* ``save_figure / save_show``: persist figures; notebook helper also displays a
  PNG inline before closing it. ``publish_figure`` copies only manifest entries.
  The removed normalized-winsor notebook also used explicit PNG output after
  its recovery module selected the non-GUI Agg backend.
  RU: это свойство исторического опыта, не зависимость рабочего вывода от recovery.

Безопасные проверки / Safe checks
--------------------------------

Границы старых диагностик / Limits of older diagnostics:

RU: Аннулярные фиты и две проверки со средними расстояниями скоплений ещё
используют не полностью согласованные бюджеты поглощения/ковариации. Ошибки
нелинейных расстояний в исследовательской сводке также короче финального
бюджета constant/linear. Не использовать эти дополнительные таблицы для
утверждения об улучшении точности без отдельного согласования и пересчёта.
Основной индивидуальный Paper IV LOO-бюджет — другой путь расчёта.

EN: Annular and cluster-mean-anchor diagnostic error/covariance budgets are not
yet fully harmonized. Exploratory nonlinear-distance errors omit terms included
in the final constant/linear budget. These optional tables cannot establish a
precision improvement without a separate budget review and recomputation.
The primary individual-Paper-IV LOO calculation is a distinct path.

::

    py -m unittest tests.test_analysis_tools tests.test_article_assets
    py -m sbf.check_project_layout --with-products
    py -m figures.publish_article_assets --check

RU: Эти проверки не запускают научные ячейки и не пересчитывают галактики.
Тест производных использует только определения чистых функций и искусственные
массивы. Полный независимый запуск с нуля остаётся отдельной проверкой.

EN: These checks neither execute scientific cells nor remeasure galaxies.
Derivative tests use only pure function definitions and toy arrays. A complete
independent run from raw inputs is a separate validation step.
