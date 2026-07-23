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

### Offline input collection for GO-3055 + GO-7763

`download_go3055_go7763.py` is the standalone collector for the complete
offline processing set. Its default invocation is a local dry run: it reads the
two manifests, validates existing FITS files and reports only the missing or
incomplete queue without opening a network connection:

```bash
astro_env/bin/python code/download_go3055_go7763.py
```

The current contract contains 14 enabled GO-3055 targets and 74 enabled
GO-7763 targets (176 products in total). Five additional GO-7763 rows stay
disabled because no target-level F150W signal product exists. A real run is:

```bash
astro_env/bin/python code/download_go3055_go7763.py --download
```

That command processes all missing products from both programmes with four
concurrent downloads. Use `--workers 1` for strict sequential operation or
`--workers 6` only if four streams still do not saturate the connection. It does
not start galaxy processing. Every transfer is written as
`<name>.fits.part`, resumed with HTTP Range after interruption, checked against
the remote byte count and FITS structure, hashed with SHA-256, and only then
atomically published as `<name>.fits`. Transient errors are retried; a permanent
failure is recorded and the next product is attempted. The machine-readable
state is updated after every file in
`data/download_go3055_go7763_status.json`.

The default disk reserve is 40 GiB. The script checks the whole planned growth
before starting and checks the remaining bytes again before every response.
Use `--program 3055` or `--program 7763` only for a deliberately restricted
repair run. Repeating the same command is safe: structurally valid products are
skipped and `.part` files are resumed.

### Downloading while processing

For `run_sbf_batch.py`, always select a first target explicitly. The selected names are propagated to
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
to already successful results, but only after all five mandatory SBF-3 FITS
have been reopened successfully and recorded with sizes and SHA-256 hashes.
`status=ok` by itself is not permission to delete an input. `--no-cleanup-inputs`
forbids deletion entirely; the downloader will then remain blocked until space
is freed.

The parent downloads the current target and preloads at most one future target
(`--prefetch-targets 1`). It never starts an archive-wide background fetch.
Use `--prefetch-targets 0` to disable preloading. Child download workers are
stopped on normal exit and on an interrupt. Internally they receive an exact
manifest-row key (including URI and filters), not merely a possibly duplicated
galaxy name. Inputs shared with any still-pending filter pair are protected
from low-space cleanup.

With `--no-download`, the runner never writes an input FITS. If a selected pair
is absent, it waits and rechecks the two final paths. This is the safe mode for
consuming files from the already-running standalone collector: `.part` and
`.restart.part` files are reported as transfer progress but are never passed to
the notebook.

### Consuming the live GO-3055 + GO-7763 download

Do not start the integrated download worker beside
`download_go3055_go7763.py`: the two downloaders use different locks. Keep the
standalone collector as the only writer and start a read-only consumer in a
second terminal:

All relative path arguments are resolved against the project root, not the
current terminal directory. Therefore the same command works both from the
project root and from `code/`.

```bash
astro_env/bin/python code/run_sbf_batch.py \
  --template code/sbf-3.ipynb \
  --target-csv code/targets_go3055_manifest.csv \
  --extra-target-csv code/targets_additional_manifest.csv \
  --programs 3055 7763 \
  --data-root data \
  --batch-root runs/sbf3_go3055_go7763/batch \
  --products-root runs/sbf3_go3055_go7763/products \
  --campaign-root runs/sbf3_go3055_go7763/campaign \
  --external-download-status data/download_go3055_go7763_status.json \
  --no-download \
  --prefetch-targets 0 \
  --no-cleanup-inputs \
  --allow-bulk-targets \
  --external-download-reserve-gb 40 \
  --estimated-worker-output-gb 1 \
  --min-processing-free-gb 40 \
  --wall-time-hours 48 \
  --soft-stop-minutes 30
```

Manifest order is preserved: the 14 GO-3055 rows are consumed first, followed
by the 74 enabled GO-7763 rows. Filter roles are read per row. Thus GO-3055 uses
F150W signal + F090W color, while GO-7763 uses F150W signal + F115W color; there
is no F090W fallback for the generic GO-7763 rows.

The disk admission check is based on the remaining bytes of every unfinished
selected product, including resumable partial sizes. While inputs remain, a new
notebook starts only if free space covers those bytes, the standalone
downloader's 40-GiB reserve, and the estimated next five-FITS result. Once all
input pairs are complete, the external reserve is released automatically.

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
requires a matching stable `job_id`, notebook SHA256 and verified artifact
manifest, so different filter pairs cannot silently overwrite one another.
The `job_id` is built from program, obsid, archive product URIs, filters,
notebook SHA256 and processing-contract SHA256; local paths, `mtime` and inode
do not affect it. Local input fingerprints are still saved as attempt
provenance and detect a replaced source file.

## Autonomous campaign state

Normal parent runs now have a 48-hour hard limit and stop starting new galaxies
30 minutes before it. Change these with `--wall-time-hours` and
`--soft-stop-minutes`; set `--worker-timeout-hours` for an additional per-galaxy
limit. `SIGINT`, `SIGTERM` and `SIGHUP` stop the active process group with a
TERM-to-KILL grace period and leave the job resumable.

The durable state is stored by default under
`<batch-root>/campaign/campaign_state.sqlite` in WAL mode. It contains the
queue, attempts, events, resource samples and artifact records. A human-readable
`queue_snapshot.json`, atomic batch summaries, per-worker logs and artifact
manifests are written beside it/in the batch root. The campaign directory also
contains `campaign.log`, `campaign_events.jsonl`, `run_provenance.json`, manifest
and notebook snapshots, `invocations.jsonl`, and `campaign_report.txt`. Each
worker records the complete cell output in its target log and writes a separate
`*_cell_timings.jsonl`; source inputs and the five required output FITS receive
SHA-256 hashes. Repeating the same command resumes the newest compatible
unfinished run; `--new-run` forces a fresh run. An OS-level `parent.lock`
prevents two parent processes from resuming and mutating the same campaign
simultaneously.

RAM and free disk are sampled during workers and download waits. New workers
wait below `--min-available-ram-gb`; active workers are terminated after a
persistent low-RAM condition, immediately below
`--emergency-available-ram-gb`, above `--max-worker-rss-gb` (when nonzero), or
below `--critical-free-gb`.

A bounded example is:

```bash
astro_env/bin/python code/run_sbf_batch.py \
  --template code/sbf-3.ipynb \
  --target-csv code/targets_additional_manifest.csv \
  --data-root data \
  --batch-root code/sbf3_runs/batch \
  --products-root code/sbf3_runs/products \
  --wall-time-hours 48 \
  --soft-stop-minutes 30 \
  --galaxies "NGC 4889" "NGC 4874"
```

This is not an HST or Legacy Survey image pipeline. Those instruments still
need separate PSF and photometric adapters. A new JWST band also needs its own
SBF zero point and usually a color term: changing `signal_filter` does not make
an F150W distance calibration valid in F200W or F356W.
