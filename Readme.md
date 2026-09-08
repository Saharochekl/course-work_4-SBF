# JWST surface-brightness fluctuations — GO-3055

Измерение флуктуаций поверхностной яркости (SBF) в **F150W и F090W** для
14 галактик, калибровка по индивидуальным TRGB-расстояниям и сравнение расстояний.
Основная калибровка F150W — постоянная; F090W — линейная по цвету.
Цветовая зависимость проверяется, а не задаётся при измерении амплитуды SBF.

## Структура

| Путь | Назначение |
|---|---|
| `code/` | Загрузка, измерения, анализ, рисунки, тесты |
| `docs/` | Краткая документация к коду на русском и английском |
| `data/` | Локальные JWST i2d, OPD и небольшие входные таблицы |
| `runs/` | Локальные модели, маски, PSF, кэши и результаты; лёгкие таблицы анализа сохранены в Git |
| `texts/paper_work/` | TeX статьи; изображения в `materials/figures/` |
| `materials/` | Литература, использованная в работе |

Личные заметки, память ассистента, окружение, сборки и архивы не публикуются.
Git-ignore не означает, что локальный файл можно безусловно удалить.

## Запуск

Все команды ниже выполняются **из `code/`**, в активированном окружении.
`py` — пользовательское имя команды Python; без такого alias используйте `python3`.
Код рассчитан на macOS/Linux (POSIX). Проверенное окружение — macOS, Python 3.13;
прямые зависимости в `../requirements.txt`:

```bash
py -m pip install -r ../requirements.txt
py check_project_layout.py
```

### 1. Исходные изображения и OPD

```bash
py download_go3055_go7763.py --program 3055
py download_go3055_go7763.py --program 3055 --download
py download_wss_opds.py --program 3055 --download
```

Первая команда проверяет наличие; только `--download` разрешает скачивание.
Загрузчик также поддерживает GO-7763, но эта программа не входит в текущую выборку.
Имена точных i2d-продуктов заданы в `targets_go3055_manifest.csv`.
Отдельно нужны reference data STPSF: укажите `STPSF_PATH` либо разместите их
в `data/stpsf-data/`. OPD и reference data — разные входы.
Код использует готовые i2d и не запускает калибровочный JWST pipeline с detector-level кадров.

### 2. Измерения

```bash
py run_sbf_2_batch.py
py run_sbf_2_normalized_winsor.py
py run_sbf_f090w.py
```

Первый этап строит модель и маски F150W из `sbf-2.ipynb`; второй измеряет
нормированные остатки с принятым винзорированием; третий обрабатывает F090W.
Это длительные вычисления. Повторный запуск использует подходящие результаты и
кэши; `--force` и родственные флаги нужны только для намеренного пересчёта.
Готовые данные можно проверить без измерений:

```bash
py check_project_layout.py --with-products
```

### 3. Калибровки и рисунки

Откройте `sbf-2-graph.ipynb` для F150W и `sbf-f090w-graph.ipynb` для F090W.
Ячейки исполняются сверху вниз; графики отображаются в notebook.
Они используют готовые измерения, а не скачивают и не моделируют галактики заново.

Публикационные рисунки из уже подготовленных таблиц:

```bash
py build_go3055_article_figures.py
py publish_article_assets.py --check
```

Дополнительные построители, порядок их входов и параметры описаны в
[документации анализа](docs/analysis.rst).
`build_sbf_f090w_graph_notebook.py` пересоздаёт notebook: **не запускайте его
поверх ручных изменений, которые хотите сохранить**.

## Метод и проверка

Остаток `(SCI − sky − model) / sqrt(model)` винзорируется при `3.5σ` по всей
валидной области. Для двух круговых колец спектр описывается `P(k)=P0 E(k)+P1`.
Из `P0` вычитается вклад неразрешённых источников, затем амплитуда переводится
в видимую SBF-величину. Расстояние требует отдельной абсолютной калибровки.

- [Измерения и обоснование параметров](docs/measurement.rst)
- [Параметры исходного notebook](docs/notebook_parameters.rst)
- [Загрузка, возобновление и ресурсы](docs/infrastructure.rst)
- [Анализ и графики](docs/analysis.rst)
- [Пути, проверка и очистка](docs/repository.rst)

```bash
py -m unittest discover -s . -p 'test_*.py'
```

Тесты не заменяют полного численного прогона. Тестам загрузчика нужен локальный
HTTP-сервер; сетевые ограничения песочницы могут запрещать его запуск.
Научные notebook автоматически этим набором не выполняются.

Это пока **не обещание побитового воспроизведения на любом компьютере**:
нужны исходные i2d, OPD, reference data и согласованные версии библиотек.
`requirements.txt` фиксирует прямые зависимости, но не все транзитивные пакеты
и внешние данные. Сохранённые выходы notebook относятся к прежним выполненным
прогонам, а не доказывают выполнение текущей ревизии.

## English

This repository measures JWST/NIRCam F150W and F090W SBF for 14 GO-3055 galaxies
and calibrates distances against individual TRGB anchors. F150W uses a constant
calibration; F090W uses a linear color calibration. Leave-one-out evaluation is
internal cross-validation, not an independent external distance test.

Run the commands above from `code/` with the environment activated; `py` denotes
Python (`python3` without the local alias). Install `../requirements.txt`, prepare
the exact manifest i2d files, WSS OPDs and STPSF reference data, then run the three
measurement stages in order. Existing valid products are reused. Open the two
graph notebooks sequentially for calibration and plotting. Figure-only changes
do not require repeating the measurement stages.

Each linked `.rst` document contains Russian and English instructions, input/output
contracts and parameter rationale. `check_project_layout.py --with-products`
checks source syntax, target membership and saved file references without running
notebook cells or fitting galaxies. Tests use small fixtures; downloader tests
require a local HTTP server. Scientific equivalence still needs numerical
validation, not just passing unit tests.

Personal Markdown notes (except this README), local archives, intermediate images,
caches and document builds are excluded from Git. They are not deleted locally.
The source tree is not a self-contained dataset: exact external inputs are required
for a fresh scientific run. Existing notebook outputs are historical, not evidence
that the current revision has been rerun.
