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

Only the input layer is generic at this stage. `sbf-2.ipynb` still contains
F150W/F090W-specific labels and calibration output names. The runner therefore
refuses other filter pairs instead of producing a numerically plausible but
mislabeled result. The next refactor must make the notebook configuration and
reported color name filter-aware; after that the guard can be widened.
