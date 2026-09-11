PASA manuscript / Вёрстка PASA
=============================

Scope / Что изменено
--------------------

The active English manuscript uses the official Cambridge ``pas`` class,
author--year citations, British spelling, native table typography, and
separately rendered journal-size figures. Scientific results, notebook cells,
input FITS and numerical tables were not recalculated or edited.

Изменено только оформление английской статьи. Научные замечания аудита не
исправлялись. Старые рисунки сохранены; новые PDF лежат в
``materials/pasa/figures/``. Серые полосы сохраняют исходный смысл разброса,
а не превращаются в доверительные интервалы из-за смены стиля.

Build from code/ / Сборка из code/
---------------------------------

The virtual environment is assumed to be active. ``py`` is the project's
Python command. To compile the existing figures and tables::

    latexmk -cd -pdf -outdir=build ../texts/paper_work/go3055_jwst_sbf_article_draft.tex

The current PDF and all auxiliary files are in ``texts/paper_work/build/``.
The older PDF beside the TeX is not overwritten by this command.

Текущий PDF и весь мусор сборки остаются в ``build/``. Старый PDF рядом с
исходником эта команда не обновляет. Открывать нужно результат из ``build/``.

Optional figure redraw / Перерисовка без нового измерения SBF
-----------------------------------------------------------

These commands use the completed analysis tables and, for image diagnostics,
the saved model, mask, PSF and residual FITS. They do not run scientific
notebook cells or change the original publication products::

    py -m figures.build_go3055_article_figures --pasa
    py -m figures.build_f090w_appendix_diagnostics --pasa
    py -m figures.build_f090w_residual_montage --pasa --band F150W
    py -m figures.build_f090w_residual_montage --pasa --band F090W

All new vector figures are authored at 515 TeX points (the class text width)
and included at ``\textwidth``. Final-size labels are 8--10 pt; PDF fonts are
embedded. Grey bands, galaxy membership, group symbols and both error-bar
directions retain their original scientific definitions. Raster residual
panels remain raster data inside the PDF, with vector labels and circles.

Рисунки нужно включать в полную ``\textwidth``: повторное уменьшение сделает
подписи мельче. Команды не запускают повторную обработку галактик.

Template provenance / Источник шаблона
--------------------------------------

``pas.cls`` (class header: 2021/04/29 v1.0) and ``cup_logo.pdf`` are unmodified
files from Cambridge's ``PAS LaTeX template 2022`` archive, downloaded on
2026-09-10. No publisher DOI, acceptance date or final copyright was invented.

* Overleaf template:
  https://www.overleaf.com/latex/templates/publications-of-the-astronomical-society-of-australia-pasa/pccfdbqhjbrv
* Official archive linked by Cambridge:
  https://www.cambridge.org/core/services/aop-file-manager/file/632c6a1b7139c70011677184
* Author instructions:
  https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/information/author-instructions/preparing-your-materials

The preamble retains the current LaTeX document-start routine because the
vendor class embeds a pre-2020 version. It loads ``amsmath`` before the
class's ``cleveref``, mirrors the class trim size into modern paper-size
lengths, closes the class's title group, and suppresses unassigned publication
metadata. Publisher geometry, fonts and class source are not patched.
Appendix floats receive distinct A/B/C prefixes. Small full-width section
blocks keep headings and their result tables/figures together.

Перед подачей автору ещё нужно утвердить соавторов и контакт, библиографию,
объём статьи, доступность данных/кода и обязательные редакционные декларации
(в том числе описание использования ИИ). Эти сведения не заполнялись
выдуманными значениями и не считаются закрытыми переносом в шаблон.
