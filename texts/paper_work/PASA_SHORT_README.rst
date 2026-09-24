Краткая версия статьи PASA
=========================

Состояние на 11 сентября 2026 года
--------------------------------

* ``go3055_jwst_sbf_article_draft.tex``: исходная полная версия, 35 страниц.
  Её текст, рисунки, таблицы и собранный PDF оставлены без изменений.
* ``go3055_jwst_sbf_article_short.tex``: отдельная краткая версия, 9 страниц,
  5 рисунков, 4 таблицы. Одна методология и одна система ошибок для двух
  фильтров. Последняя таблица расстояний непосредственно перед Discussion.
* ``go3055_jwst_sbf_technical_notes.tex``: самостоятельный технический
  документ, 23 страницы, 16 рисунков, 6 таблиц. Нумерация S1 и далее
  независима от основной статьи. Это пока документ внутри репозитория,
  а не утверждённый журналом supplementary material.

В этой редакции не пересчитывались измерения, калибровки или расстояния.
Графики построены из сохранённых результатов; исходные численные таблицы
используются напрямую. Уточнены библиографические данные Paper III,
Paper IV и Jensen et al. (2021), без изменения научных результатов.

Что осталось в статье
--------------------

1. Постановка задачи и общая выборка.
2. Обработка изображения, нормировка, винзорирование, спектральный фит,
   перевод амплитуды в SBF-величину и калибровка по TRGB.
3. Измерительные, калибровочные, популяционные и общие систематические
   ошибки; различие ошибок mbar, Mbar и расстояния.
4. Два фильтра, постоянная/линейная модели, проверка на общих восьми
   галактиках, внутренняя перекрёстная проверка и ограничения.
5. Итоговые расстояния, Discussion, Conclusions, доступность данных.

Карта рисунков краткой версии:

* Figure 1: две калибровки рядом; альтернативная модель пунктиром.
* Figure 2: два бюджета измерительной дисперсии рядом.
* Figure 3: общие восемь галактик, три SBF-полосы и два цвета.
* Figure 4: восстановление расстояний в двух фильтрах.
* Figure 5: два полных бюджета ошибки расстояния рядом.

В Technical notes перенесены подробные выводы формул, дополнительные
таблицы ошибок, проверки якорей/колец/качества, PSF, остаточных источников,
винзорирования, гистограммы, спектры и вспомогательные сравнения F160W.
Монтажи всех 14 остатков F150W и F090W сохранены раздельно в Figures S1 и S2:
уменьшать их до одного монтажа из 28 изображений было бы нечитаемо.
Ограничения коррелированного шума и отсутствие независимой внешней
валидации не спрятаны: они также оставлены в основном тексте.

Сборка и перерисовка
-------------------

Команды из ``code/``, при уже активированном окружении::

    latexmk -cd -pdf -outdir=build ../texts/paper_work/go3055_jwst_sbf_article_short.tex
    latexmk -cd -pdf -outdir=build ../texts/paper_work/go3055_jwst_sbf_technical_notes.tex

PDF и весь мусор сборки остаются в ``texts/paper_work/build/``.
Обе версии можно открыть одновременно с исходным драфтом.

Перерисовать только новые парные графики, без обработки FITS::

    py -m figures.build_short_article_figures --cmyk

Для ``--cmyk`` нужен установленный Ghostscript (``gs``). Без этого флага
получатся обычные RGB PDF для локального просмотра. Новый генератор пишет
только в ``materials/pasa/short/`` и ``build/short-figure-export/``.
Figure 3 копируется из готового рисунка полной версии, а не фитируется заново.

Проверка оформления
------------------

Проверены официальные `инструкции PASA`_ и `руководство по рисункам`_.
Обычный ориентир Research Paper: до 10 двухколоночных страниц с таблицами
и литературой, но более длинные работы также рассматриваются. Для
дополнительных материалов указан предел 10 MB на файл; текущие notes
занимают около 3.4 MB. Перед подачей отдельно согласовать их состав.

В новых документах сохранены класс ``pas.cls``, поля и базовые размеры
шрифта шаблона. Используются британское написание, ссылки автор-год,
полные слова Figure/Equation/Table и три ключевых термина `UAT`_. Все
рисунки, таблицы и библиографические записи процитированы в тексте.

Пять рисунков короткой статьи имеют белый фон и векторный PDF с
встроенными шрифтами; цветной экспорт переведён в CMYK. Группы галактик
различаются цветом и формой маркера, линии - стилем. На калибровках есть
горизонтальные и вертикальные ошибки. Серая область в Figure 1 означает
только плюс/минус sigma_int, в Figure 3 - плюс/минус наблюдаемый разброс
остатков: это не два доверительных коридора и не полная ошибка расстояния.

Объединённые рисунки проверены на конечной ширине страницы около 181 mm.
Подписи имеют размер 8-10 pt. Общее руководство Cambridge рекомендует
9 pt и определённые гарнитуры; здесь сохранена встроенная DejaVu Sans
для согласованности с исходными рисунками. Это отмеченное отличие от
типографской рекомендации, а не заявление о полном предпечатном приёмочном
контроле издателя. Технические notes используют исходные диагностические
рисунки; их полная предпечатная унификация не выполнялась.

Просмотрены рендеры обеих версий. Ссылки разрешены, таблицы и рисунки
не обрезаны. У старого издательского класса остаются служебные предупреждения
TeX о шрифтах и выходных блоках; они не равнозначны ошибке сборки.

До отправки остаётся авторская часть: окончательный список авторов,
контакт/ORCID, благодарности, финансирование, конфликт интересов, описание
использования ИИ и публичный адрес/DOI данных и кода. Такие сведения не
заполнялись предположениями. Ниже даны черновики описаний доступности для
рисунков; их следует проверить и перенести в форму издателя.

.. _инструкции PASA: https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/information/author-instructions/preparing-your-materials
.. _руководство по рисункам: https://www.cambridge.org/core/services/authors/journals-artwork-guide
.. _UAT: https://www.ivoa.net/rdf/uat/

Draft accessibility descriptions
--------------------------------

Figure 1
  Two panels plot extinction-corrected absolute SBF magnitude against
  F090W minus F150W colour for all 14 galaxies. Squares, circles and
  triangles distinguish Fornax, the Virgo region and other environments.
  Labels identify NGC objects. The adopted F150W calibration is horizontal;
  F090W uses a sloping line. Dashed lines show the alternative model.
  Grey strips show one fitted intrinsic-scatter unit around adopted lines.

Figure 2
  Paired horizontal stacked bars show measurement-variance contributions
  for all galaxies in identical order. Components are spectrum, sky, PSF,
  unresolved sources and extinction. NGC 4486 has the largest variance
  in both bands, chiefly from the background term. These are not full
  distance errors.

Figure 3
  Six panels compare the same eight galaxies. Rows show F090W, F150W and
  published F110W SBF; columns use infrared and optical colours. Every
  point has a galaxy label and two-axis error bars. F090W and F110W show
  clearer slopes than F150W. Grey strips describe residual scatter.

Figure 4
  Two panels compare internally cross-validated SBF distance moduli with
  individual TRGB moduli. Dashed diagonal lines indicate equality.
  Galaxy labels and environment markers match Figure 1. F090W has less
  residual scatter. Error bars omit the common absolute distance scale.

Figure 5
  Paired horizontal stacked bars show the full distance-modulus variance.
  Each band includes measurement, extinction, population scatter and
  finite-calibration terms; F090W additionally includes colour uncertainty.
  Hatched segments identify the common TRGB-scale variance, which is
  correlated between galaxies. Population scatter dominates most F150W
  bars; finite calibration is a substantial additional term in F090W.

Table descriptions
  Tables 1 and 2 give individual galaxy colours, apparent and absolute
  SBF magnitudes, their distinct uncertainties, TRGB information and
  radial/quality diagnostics. Table 3 compares constant and linear fits
  on the same sample. Table 4 compares five distance estimates in Mpc,
  with explicit full absolute one-sigma errors and missing literature
  entries left blank; common scale errors remain correlated.
