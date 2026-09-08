Репозиторий / Repository
=======================

Русский
-------

Пути и проверка
~~~~~~~~~~~~~~~

``sbf_paths.py`` разрешает входные пути относительно расположения кода.
``project_path`` принимает относительный путь или исторический абсолютный адрес
в ``course_work-SBF``. ``load_project_json`` читает JSON и преобразует распознанные
пути только в памяти. Файлы результатов и fingerprints на диске не меняются.
Неизвестный внешний путь не заменяется на произвольный найденный файл.

Константы этого модуля:

* ``PROJECT_ROOT`` — расположение самого проекта; cwd не должен менять входы.
* ``_LEGACY_ROOT`` — ровно прежнее имя checkout, встречающееся в сохранённых JSON.
  Совместимость необходима, пока эти результаты используются.
* ``_PROJECT_DIRS`` — допустимые начала относительных путей; это защита от
  превращения обычного текста в имя файла.

STPSF ищется в явно заданном ``STPSF_PATH``, затем в ``data/stpsf-data``.
Существующая домашняя установка допускается ради текущих локальных прогонов.
Для воспроизводимого запуска задавайте путь явно. Это выбор reference data,
не автоматическая замена одной научной модели другой.

``check_project_layout.py`` проверяет синтаксис активных Python/notebook,
публикационные рисунки и, с ``--with-products``, состав выборки, статусы и ссылки
продуктов. Список галактик читается из манифеста, а не задаётся вторым числом 14.
Это проверка структуры, не проверка научной точности. Ошибка о недостающих
данных должна быть исправлена подготовкой входов, а не подавлена fallback.

Что можно удалить вручную
~~~~~~~~~~~~~~~~~~~~~~~~~

Без потери научных входов, после закрытия соответствующих процессов:

* ``tmp/`` — одноразовые извлечения PDF и визуальный контроль (около 25 MiB).
* ``texts/**/build/`` — продукты компиляции, восстановимые из TeX/рисунков;
  перед удалением сохраните нужный вам итоговый PDF.
* ``__pycache__/``, ``.matplotlib/``, ``.ipynb_checkpoints/`` — кэши Python,
  Matplotlib и notebook; checkpoints могут содержать вашу нужную копию правок.
* ``trash/review-tools/`` — одноразовые инструменты прежнего переноса.

После отказа от исторических экспериментов:

* ``code/legacy/`` — старый код и старые продукты, около 3.2 GiB.
* ``runs/legacy/`` — прежние запуски, около 7.5 GiB.

Это НЕ копии всех действующих измерений. Удаление архивов лишит возможности
повторить старые эксперименты. Старые совместимые symlink будут указывать в никуда;
их следует убирать вместе с выбранным архивом. Не запускайте ``git clean -fdx``:
он также уничтожит игнорируемые исходные FITS, действующие результаты и окружение.

Сейчас не удалять: ``data/`` целиком, текущие ``runs/sbf2_go3055/``,
``runs/sbf2_normalized_winsor/``, ``runs/sbf_f090w_go3055/``, ``runs/sbf2_systematics/``,
worker.log (фон используется при анализе), PSF/OPD/reference data, литературные
входы ``code/sbf2_batch_outputs/`` и рисунки статьи. Исключение из Git не отменяет
потребность локального анализа в этих файлах.

Git
~~~

Личные Markdown исключены, кроме корневого ``Readme.md``; документация выпуска
находится в ``docs/*.rst``. Уже отслеживавшиеся игнорируемые файлы убраны только
из индекса. Они доступны локально. Старые коммиты не переписаны и размер истории
от этого сам по себе не уменьшается. Для истории нужна отдельная согласованная
операция с резервной копией и перепубликацией веток.

English
-------

``sbf_paths.py`` resolves inputs relative to the source checkout. ``project_path``
supports the original checkout name; ``load_project_json`` rebases saved paths
in memory only. On-disk provenance and fingerprints are retained. ``PROJECT_ROOT``
makes paths independent of cwd; ``_LEGACY_ROOT`` recognizes the one old recorded
format; ``_PROJECT_DIRS`` prevents ordinary text from being interpreted as paths.
Unrelated external paths are not guessed. Set ``STPSF_PATH`` explicitly for
reproducible runs; local and existing home reference-data locations remain supported.

``check_project_layout.py`` compiles source without executing notebook cells.
With ``--with-products`` it checks manifest membership, success status and file
references, not scientific accuracy. Missing inputs are errors, not an invitation
to substitute different data.

Temporary PDF inspections, document build outputs and Python/font caches are
rebuildable. Preserve any final PDF or notebook checkpoint you still need.
``code/legacy`` (about 3.2 GiB) and ``runs/legacy`` (about 7.5 GiB) can be removed
only if historical experiments are no longer wanted; remove their compatibility
symlinks accordingly. Current source images, model/PSF products, spectral caches,
background logs, literature inputs and article figures must remain available
for analysis and replotting. Never use an indiscriminate ``git clean -fdx`` here.

Personal Markdown except the root README, archives and generated products are
untracked but kept on disk. Git history is unchanged. Untracking files neither
frees their disk space nor erases them from past commits. Exact cold-run
reproducibility also requires external reference data and numerical comparison;
syntax checks and small tests alone cannot establish it.
