# Target manifest contract

`targets_go3055_manifest.csv` is the first input manifest using roles instead
of hard-coded F150W/F090W column names. The current 14 GO-3055 rows point to
the same products as the legacy `article_galaxies_jwst_f150w_selected.csv`.

Required columns:

- `program`, `target`, `obsid`;
- `signal_filter`, `color_filter`;
- `signal_product`, `color_product`;
- `signal_product_uri`, `color_product_uri`.

Optional provenance and sample-selection columns include exposure times, file
sizes, morphology, redshift, external distance and method, HST overlap, the
dust/spiral/resolved/field-contamination flags, and `calibration_family`.

The batch runner still accepts the old manifest format and the old worker flags
`--f150w`/`--f090w`. They are compatibility paths, not the contract for new
samples.

## Current safety boundary

`sbf-2.ipynb` is frozen and remains limited to F150W/F090W. `sbf-3.ipynb`
accepts manifest filter roles, validates them against the FITS primary headers,
requires calibrated JWST NIRCam/NIRISS i2d `SCI` data in MJy/sr, and reports a
stable `color_index` with an explicit `color_name`.

For different pixel grids, `sbf-3` maps only the SBF-annulus coordinates through
the two WCS solutions and bilinearly samples the color image. It does not
silently crop arrays by index.

Every `sbf-3` run requires separate product and batch roots:

```bash
./astro_env/bin/python code/run_sbf2_batch.py \
  --template code/sbf-3.ipynb \
  --batch-root code/sbf3_smoke_outputs/batch \
  --products-root code/sbf3_smoke_outputs/products \
  --no-download \
  --galaxies "NGC 1380"
```

The result identity includes the notebook SHA256, input paths, filters and
output directory. A result is reused only when the full identity matches.
Different filter/input pairs receive separate product directories and result
JSON files.

This is not yet an HST/Legacy image pipeline. Those instruments need separate
PSF and photometric adapters. Unlike the inherited `sbf-2` color code, `sbf-3`
applies the same background-subtraction policy to signal and color frames.
Before a new wide-baseline color calibration is adopted, the remaining color
systematics are PSF matching and construction of a union compact-source mask
for both filters.
