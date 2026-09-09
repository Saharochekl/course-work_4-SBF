Репозиторий / Repository
=======================

Русский
-------

Рабочая структура
~~~~~~~~~~~~~~~~~

Из корня ``code/`` запускаются два скрипта: ``download.py`` и ``process.py``.
Рядом остаются три notebook: общий измерительный ``sbf-2.ipynb``,
анализ F150W ``sbf-2-graph.ipynb`` и анализ F090W ``sbf-f090w-graph.ipynb``.

* ``code/sbf/`` — загрузка, обработка, FFT, состояние кампании и пути.
* ``code/figures/`` — построение таблиц, рисунков и генератор F090W notebook.
* ``code/tests/`` — активные модульные тесты.
* ``code/config/`` — манифесты целей; ``code/reference/`` — литературные таблицы.
* ``data/`` — исходные кадры, OPD/reference data и небольшие метаданные.
* ``runs/F150W/source/``, ``spectra/``, ``analysis/`` — исходный этап,
  принятый нормированный FFT-пересчёт и анализ F150W.
* ``runs/F090W/`` — исходный этап, спектры, результаты и анализ F090W.
* ``.cache/runtime/`` и ``.cache/matplotlib/`` — служебные кэши вне ``runs/``.
* ``texts/paper_work/materials/`` — материалы статьи; TeX-сборка остаётся в ``build/``.
* ``trash/`` — необязательный временный карантин новых находок, не резервная копия.
  Прежний карантин и исторические эксперименты удалены владельцем.

В ``runs/`` только две научные кампании. Активные пути не требуют совместимых
symlink на прежнюю структуру. Ссылки в метаданных приведены к текущим каталогам;
наличие старого имени в исторической записи само по себе не означает наличие файла.

Пути и проверки
~~~~~~~~~~~~~~~

``sbf.sbf_paths`` определяет ``PROJECT_ROOT`` от расположения пакета,
``CODE_DIR`` — от корня проекта. Текущий каталог оболочки не меняет входные данные.
``_LEGACY_ROOT`` распознаёт прежнее имя checkout; ``_PROJECT_DIRS`` ограничивает
каталоги, которые можно считать путями. Неизвестный внешний адрес не заменяется
случайно найденным файлом. ``load_project_json`` преобразует распознанные пути
в памяти; само чтение не переписывает provenance. Перенос каталогов — отдельная
служебная операция с проверкой ссылок и идентичности продуктов.

STPSF использует явно заданный ``STPSF_PATH``, затем локальный
``data/stpsf-data``; существующая домашняя установка поддерживается для текущей
рабочей станции. Для воспроизводимого запуска задайте путь явно и сохраните версию
reference data. Это не автоматическая замена научной модели.

Из ``code/``, в уже активном окружении:

* ``py download.py images --program 3055`` и ``py download.py opd --program 3055``
  проверяют входы; только ``--download`` разрешает загрузку.
* ``py process.py --filter both --check`` проверяет подготовку, без моделирования/FFT.
* ``py -m sbf.check_project_layout --with-products`` проверяет синтаксис и ссылки.
* ``py -m unittest discover -s tests -t .`` запускает модульные тесты.

Эти проверки не доказывают научную точность или полную побитовую воспроизводимость.
Для последней нужны независимый полный запуск, внешние reference data и сравнение
численных результатов. Пропавший вход нельзя скрывать подстановкой другого файла.

Ручная очистка
~~~~~~~~~~~~~

После завершения процессов можно удалить воспроизводимые служебные кэши,
``__pycache__/``, временный визуальный контроль и TeX ``build/``.
Сначала сохраните нужные PDF и notebook checkpoints. ``.cache/runtime/`` нельзя
удалять во время работающего обработчика. Рабочий запуск не обращается в
``trash/``; этот каталог можно использовать только как временный карантин.

После аудита 2026-09-09 владелец удалил карантин: 224 промежуточных FITS (45.248 GiB),
14 старых нормированных F150W FITS, 50 прежних F090W-кэшей, четыре F277W/F356W
кадра вне текущей статьи и архивный код. Принятые файлы двух рабочих фильтров
остались на месте. Это описание завершённой очистки, не список существующих
кандидатов удаления. Ранее отслеживаемые файлы можно найти в истории Git;
игнорируемые данные Git не восстановит. Прежние legacy/systematics-запуски
пользователь удалил раньше; они не учитываются повторно.

Сохранить для перерасчёта/перерисовки: F090W/F150W SCI, рабочие модели и остатки,
маски, кольца, PSF/OPD, текущие спектральные кэши, CSV/JSON, литературные входы,
журналы расчёта и материалы статьи. ``worker.log`` — не просто мусор:
из него извлекается в том числе информация об ошибке фона. Не применять
``git clean -fdx`` или массовое удаление ``*model*``/``*resid*``/``*clip*``.

Git и индекс кода
~~~~~~~~~~~~~~~~

Личные Markdown исключены из Git; исключения — корневые ``Readme.md`` (EN)
и ``Readme_RUS.md`` (RU). Документация выпуска — ``docs/*.rst``.
Локальная память проекта и отчёты аудита в выпуск не входят. Игнорирование файла
не удаляет его с диска и не вычищает старые коммиты. Git-история здесь не переписывается.

``.cbmignore`` исключает локальный архив, окружение, служебные кэши и двоичные
массивы/рисунки. Активные модули, notebook, config/reference и текстовые результаты
остаются доступны индексатору. Это не запрет агенту читать FITS напрямую:
граф нужен для структуры кода, а не для хранения пиксельных массивов.

English
-------

Layout and entry points
~~~~~~~~~~~~~~~~~~~~~~~

Run from ``code/`` with the project environment active. The root contains
``download.py``, ``process.py`` and three notebooks: shared source processing,
F150W analysis and F090W analysis. Internals live in ``sbf/``; plotting/table
builders in ``figures/``; tests in ``tests/``; manifests and literature inputs
in ``config/`` and ``reference/``. Historical experiments have been removed
from the working tree; active code neither imports them nor requires
filesystem aliases to them.

There are two science campaigns: ``runs/F150W/`` with ``source/``,
``spectra/``, ``analysis/``, and ``runs/F090W/``. Runtime and plotting caches
live in project-root ``.cache/``, not ``runs/``. Article assets remain beside
the TeX source in ``materials/``; compilation output belongs in ``build/``.
Active paths do not depend on compatibility symlinks.

Checks and reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

``sbf.sbf_paths`` anchors paths to the checkout, not cwd. Root/name/directory
constants define path recognition, not scientific parameters. JSON reading
rebases recognized paths in memory; migration is a separate checked operation.
Unrelated external paths are not guessed. Set ``STPSF_PATH`` explicitly and
record the reference-data version for reproducible runs.

* ``py download.py images --program 3055`` and ``py download.py opd --program 3055``
  inventory inputs; add ``--download`` to authorize downloads.
* ``py process.py --filter both --check`` validates preparation without modelling/FFT.
* ``py -m sbf.check_project_layout --with-products`` checks syntax and saved paths.
* ``py -m unittest discover -s tests -t .`` runs the active unit suite.

Syntax, metadata and unit checks are not a cold scientific reproduction.
External reference data, a full independent run and numerical comparison are
still needed to establish that claim.

Cleanup and Git
~~~~~~~~~~~~~~~

Rebuildable caches/build output may be removed after processes stop; preserve
wanted PDFs and notebook checkpoints. Do not delete runtime caches under an
active worker. The owner has deleted the audited redundant exports, superseded
caches, unused-band images and historical-code quarantine. Previously tracked
files remain in Git history; Git cannot restore deleted ignored data.
``trash/`` is only an optional temporary quarantine for future findings,
not an existing backup. Previously deleted legacy/systematics runs are not
counted twice.

Keep source images, working models/masks/residuals, rings, PSF/OPD, current FFT
caches, result tables/metadata, background logs, literature inputs and article
assets for reprocessing/replotting. Avoid ``git clean -fdx`` and broad filename
wildcards: ignored files are often required scientific inputs.

Personal Markdown is ignored except the two root EN/RU READMEs; public
documentation uses ``docs/*.rst``. Untracking does not reclaim disk space or
rewrite Git history. ``.cbmignore`` excludes archives/caches/binaries, not current
source or lightweight result tables. Direct FITS inspection remains available
when needed; pixel arrays do not belong in a code-structure graph.
