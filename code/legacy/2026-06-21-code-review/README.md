# 2026-06-21 code review archive

This folder contains files moved out of active `code/` after a dependency review.

Kept active:
- current pipeline scripts: `run_sbf2_batch.py`, `compute_jensen_like_error_budget.py`, `make_jensen_f160w_comparison.py`, `find_jwst_article_galaxies.py`, `find_extended_sbf_jwst_candidates.py`
- current notebooks: `sbf-2.ipynb`, `sbf-paper-plots.ipynb`
- batch/result CSV and JSON products still used by the scripts and notebooks
- coursework PDFs directly referenced from `texts/course_work/*.tex`

Moved here:
- `scripts/`: old `sbf-1` prototype and tail/night/systematics experiment drivers
- `outputs/`: obsolete experiment output directories and batch residual FITS symlink cache
- `figures/`: PNG duplicates and non-coursework PDF figures not referenced by the active TeX files
- `logs/`: terminal/night/batch logs
- `caches/`: Python bytecode and PDF text cache
- `workspace/`: local Finder/VS Code workspace files

The active TeX files only require the remaining `code/paper_figures/coursework_*.pdf`
files and `code/sbf2_batch_outputs/coursework_distance_comparison_table.tex`.

Note: several archived file types are ignored by `.gitignore` (`*.fits`, `*.png`,
`*.log`, `*.pyc`, `*.code-workspace`), so `git status` only shows the tracked part
of this physical move unless ignored files are staged explicitly.
