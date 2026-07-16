# Target manifest contract

There are two operational manifests in `code/`:

- `targets_go3055_manifest.csv`: the original 14 GO-3055 calibration targets;
- `targets_additional_manifest.csv`: the 2026-07-16 archive snapshot for
  GO-5989, GO-7763 and the additional demonstration/future targets.

The detailed archive inventories and their source notes live in
`code/catalog_search/`. They are evidence tables, not direct batch inputs.

## Additional-manifest snapshot

`targets_additional_manifest.csv` has 125 rows. Of these, 114 rows are enabled
and have a complete public two-filter product pair. Their authoritative MAST
`Content-Length` total is 246.573 decimal GB.

| Sample | Rows | Enabled | Signal + color | Enabled volume | Interpretation |
|---|---:|---:|---|---:|---|
| GO-5989 Coma | 39 | 38 | F150W + F356W | 60.686 GB | cluster calibration; NGC 4926 lacks the NIRCam pair |
| GO-7763 J-Virgo | 79 | 74 | F150W + F115W | 179.992 GB | current archive is larger than the old 73-target estimate |
| VV 191 / GTO-1176 | 1 | 1 | F150W + F090W | 2.424 GB | method/reach demonstration |
| M104 / GO-6565 | 1 | 1 | F200W + F090W | 3.471 GB | filter-extension test; not an F150W calibration |
| Cen A / GO-12496 | 1 | 0 | F200W + F090W | disabled | public, but resolved stars and dust require manual regions/masks |
| GO-8277 | 2 | 0 | NIRISS F200W + F115W | disabled | exclusive access; current release dates are 2027-05-09 and 2027-06-09 |
| COSMOS-Web + CEERS | 2 | 0 | screening placeholders | disabled | no concrete low-redshift object/tile has been established yet |

For GO-5989, `external_distance_modulus=35.0` is a common Coma-cluster prior
(approximately 100 Mpc), not 39 independent galaxy distances. It is useful for
relative/common-distance calibration, but cluster depth and population terms
must still enter the scientific error model.

## Downloader controls

`download_enabled=true` means that both exact products are public and the row
is actionable. A false row remains visible in the planning table but is skipped
before product validation or download. The reason is recorded in
`availability_status` and `notes`; do not bulk-change these flags.

Always select a first target explicitly. The selected names are propagated to
the separate download worker, so this command downloads and processes only
NGC 4889:

```bash
astro_env/bin/python code/run_sbf_batch.py \
  --template code/sbf-3.ipynb \
  --target-csv code/targets_additional_manifest.csv \
  --data-root data \
  --batch-root code/sbf3_runs/batch \
  --products-root code/sbf3_runs/products \
  --galaxies "NGC 4889"
```

The runner refuses an implicit selection larger than 14 targets. Consequently,
omitting `--galaxies` for this manifest raises an error instead of fetching the
114-row, 246.6-GB archive. A bare `--galaxies` also selects nothing and fails.
Names are exact and case-sensitive; quote each one separately, for example
`--galaxies "NGC 4889" "NGC 4874"`.

`--allow-bulk-targets` is an explicit override for a separately capacity-planned
volume. It is not appropriate for the current local disk. Before each transfer,
the download worker now requires enough free space for both the configured
reserve (`--min-free-gb`, 30 GiB by default) and the remaining file bytes. It
waits instead of crossing that boundary.

By default, a low-space run may remove the original signal/color FITS belonging
to already successful results. `--no-cleanup-inputs` forbids that deletion; the
downloader will then remain blocked until space is freed. The child download
worker is stopped when the parent exits normally and is also registered for
interpreter-exit cleanup.

Add `--no-download` only when every explicitly selected FITS pair already exists
under `data/<target>/`.

Required columns for an enabled row are:

- `program`, `target`, `obsid`;
- `signal_filter`, `color_filter`;
- `signal_product`, `color_product`;
- `signal_product_uri`, `color_product_uri`.

The byte-count columns let the downloader detect incomplete files. Optional
provenance and sample-selection columns include exposure times, morphology,
redshift, external distance and method, HST overlap, dust/spiral/resolved/field
flags, `calibration_family`, `science_role`, `priority` and release date.

Only `download_enabled` controls whether a row can reach the runner. Priority,
status, release date, role and distance fields are planning/post-processing
metadata; they neither sort the batch nor calibrate a distance inside
`sbf-3.ipynb`. Execution order follows the CSV. MAST filenames and
`Content-Length` values are a dated archive snapshot, not checksums, and should
be refreshed before a later large download or after archive reprocessing.

The runner still accepts the old manifest and worker flags `--f150w`/`--f090w`
for compatibility. New samples should use the generic signal/color contract.

## Pipeline boundary

`sbf-2.ipynb` is frozen and limited to F150W/F090W. `sbf-3.ipynb` accepts the
manifest filter roles, validates them against FITS headers, and requires
calibrated JWST NIRCam/NIRISS i2d `SCI` data in MJy/sr. Different pixel grids
are matched through WCS and bilinear color sampling rather than array-index
cropping.

Every `sbf-3` run requires isolated batch and product roots. Result reuse also
requires a matching notebook SHA256, input fingerprints, filters and output
directory, so different filter pairs cannot silently overwrite one another.

This is not an HST or Legacy Survey image pipeline. Those instruments still
need separate PSF and photometric adapters. A new JWST band also needs its own
SBF zero point and usually a color term: changing `signal_filter` does not make
an F150W distance calibration valid in F200W or F356W.
