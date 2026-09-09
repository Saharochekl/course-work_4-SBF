# JWST surface-brightness fluctuations — GO-3055

SBF measurements in NIRCam F150W and F090W for 14 galaxies, TRGB calibration,
distance comparisons and publication figures. [Russian version](Readme_RUS.md).

## Layout

```text
code/
  download.py                 image and OPD download/check entry point
  process.py                  F150W/F090W processing entry point
  sbf-2.ipynb                 shared image-processing template; interactive inspection
  sbf-2-graph.ipynb            F150W analysis and figures
  sbf-f090w-graph.ipynb        F090W analysis and figures
  sbf/                       production modules and product validator
  figures/                   figure, table and notebook builders
  config/                    exact target/product manifests
  reference/                 curated literature inputs
  tests/                     small offline/synthetic tests
runs/
  F150W/
    source/                  galaxy models, masks, PSFs and source measurements
    spectra/                 adopted normalized-residual measurements and cache
    analysis/                calibration tables and plotting inputs
  F090W/                     source, spectra, final products and analysis
data/                        downloaded images, OPDs and reference data
docs/                        concise Russian/English module documentation
texts/paper_work/materials/   article figures and their source manifest
.cache/                      regenerable runtime/font caches
trash/2026-09-09/removed/     quarantined exports and historical code; local only
```

There are two command-line entry scripts and three working notebooks: one shared
processing template and a separate analysis notebook for each band. Helpers are
regular Python packages, not filesystem aliases. Local notes, environments,
legacy experiments and large generated data are excluded from Git.
Historical code is now under `trash/2026-09-09/removed/code/legacy/`, not `code/`.
Production does not import it or require aliases to it.

## Run

All commands below run **from `code/`**, with the environment already activated.
`py` denotes Python; use `python3` if that command is not configured locally.
The processing code targets POSIX systems; the checked environment is macOS,
Python 3.13.

```bash
py -m pip install -r ../requirements.txt
py -m sbf.check_project_layout
```

### 1. Prepare inputs

```bash
py download.py images --program 3055
py download.py images --program 3055 --download
py download.py opd --program 3055 --download
```

Without `--download`, the downloader only checks local files. Product names are
in `config/targets_go3055_manifest.csv`. The optional GO-7763 manifest is not part
of the current article sample. STPSF reference data are needed separately: set
`STPSF_PATH` or install them in `../data/stpsf-data/`. OPDs are not a substitute
for those reference data. Processing starts from calibrated i2d images; this
repository does not recalibrate detector-level exposures.

### 2. Process

```bash
py process.py --dry-run
py process.py --filter F150W
py process.py --filter F090W
```

The first command only prints the planned commands. The next two are long
scientific runs; valid completed products and caches are reused. Run F150W before
a first F090W campaign because F090W uses its source products. To run both in that
order, use `py process.py`.

For an explicit F150W stage use `--stage source` or `--stage spectra`.
For a target subset use `--galaxies "NGC 4636"`.
F090W uses `--stage all` and resumes its individual stages from cache.
Advanced worker options remain available through `py -m sbf.run_sbf_f090w --help`
and the other modules in `sbf/`.

Validate saved inputs and products without remeasuring galaxies:

```bash
py -m sbf.check_project_layout --with-products
```

### 3. Analyze or redraw figures

Open `sbf-2-graph.ipynb` for F150W or `sbf-f090w-graph.ipynb` for F090W.
Their cells run top to bottom and display figures inline. Changing labels,
colors or backgrounds does not require refitting images.

Publication builders operate on the existing analysis products:

```bash
py -m figures.build_go3055_article_figures
py -m figures.publish_article_assets --check
```

See [analysis documentation](docs/analysis.rst) for the remaining builders and
their inputs. `figures.build_sbf_f090w_graph_notebook` **regenerates the notebook**:
do not run it over manual notebook edits you want to preserve.

## Documentation and checks

- [Measurement stages and parameter rationale](docs/measurement.rst)
- [Image-processing notebook parameters](docs/notebook_parameters.rst)
- [Downloads, resume and resources](docs/infrastructure.rst)
- [Analysis and figures](docs/analysis.rst)
- [Paths, repository layout and cleanup](docs/repository.rst)

```bash
py -m unittest discover -s tests -t .
```

Tests use small fixtures and do not execute scientific notebook cells. Downloader
tests need a local HTTP server. Syntax, metadata and unit checks are not a
substitute for numerical validation of a new scientific run.

## Reproducibility and local data

The source release consists of the active scripts/notebooks, their packages,
tests, manifests, literature inputs, documentation and publication assets.
Personal Markdown is excluded except these two READMEs. Downloaded FITS, caches
and local experiments are not release source code. Ignored does **not** mean
safe to delete: models, masks, normalized residuals, PSFs, tables and background
logs are still required for some figures or for resuming processing.

Audited redundant exports, superseded caches and historical code have been moved
to `trash/2026-09-09/removed/`, retaining their former relative paths. They are
outside the active workflow; permanent deletion is left to the owner. Moving
them within the same disk does not itself free disk space. Current scientific
tables and accepted processing inputs remain in `runs/` and `data/`.

This is not a promise of bit-identical output on any machine. A fresh run also
needs the exact i2d/OPD inputs, STPSF reference files and compatible dependencies.
`requirements.txt` pins direct dependencies, not every transitive package or
external dataset. Stored notebook outputs belong to earlier runs, not to an
automatic rerun of the current checkout.
