# GO-7763 J-Virgo and miscellaneous JWST SBF inventory

Snapshot: 2026-07-16 (Europe/Moscow).  All archive statements below are a
point-in-time result from the official STScI/MAST services.  The companion CSV
is `/private/tmp/go7763_misc_manifest_research.csv`.

## What a CSV row means

- `target_level_i2d`: a current MAST Level-3 observation record whose expected
  target-level `*_i2d.fits` name was checked directly against the MAST download
  service.  For public products, `downloadable_public` means an unauthenticated
  HTTP HEAD returned 200; `content_length_bytes` is the returned MAST size.
- `exclusive_access_not_downloadable`: MAST exposes the observation/product
  metadata, but unauthenticated download is not currently authorized.  The
  release date is copied from MAST `t_obs_release`, not estimated from the visit
  date.
- `field_screening_candidate`: the survey field is public, but MAST target
  metadata does not identify the alleged low-redshift foreground galaxy.  No
  arbitrary tile is promoted to a galaxy product.

MAST documents Stage 3 as calibrated combined products and provides its JWST
Search/API as the authoritative discovery route:
https://archive.stsci.edu/missions-and-data/jwst

## GO-7763: J-Virgo

Official program page:
https://www.stsci.edu/jwst-program-info/program/?program=7763

Official visit-status report:
https://www.stsci.edu/jwst-program-info/visits/?program=7763

The current archive is larger than the earlier working count of 73:

- 79 distinct archived targets have public Level-3 rows.
- 385 public target-level observation/filter rows are in the CSV.
- The complete 385-product inventory is 227.09 GB (211.49 GiB).
- The normal five-product set is NIRCam F115W, F150W, F277W plus NIRISS
  F115W and F277W.
- The directly relevant NIRCam F150W signal + F115W color subset contains
  74 complete galaxy pairs and is 179.99 GB (167.63 GiB).  This, not all 385
  products, is the rational first bulk-download boundary for `sbf-3`.
- NGC 4374 (M84), NGC 4406 (M86), NGC 4486 (M87), NGC 4552 (M89), and
  IC 3459 currently have only three target-level rows each.  The four giant
  galaxies are the short observations 77--80.  IC 3459 is a separate
  three-product archive state; the reason is not inferred here, and it should
  not be silently treated as a normal five-filter target.
- Observation 27 (NGC 4425) is marked Skipped.  Its repeat, observation 81,
  is Flight Ready for a 2027-05-11 to 2027-06-08 plan window, so it is not a
  downloadable row in this snapshot.

Program status is still `Flight Ready` because observation 81 remains pending,
even though the 79 completed visits are archived.  All 385 returned Level-3
records have `dataRights=PUBLIC`.

## VV 191 and the PEARLS context

VV 191 is not an observation in GO-2738.  Current MAST associates the target
`VV-191` with GTO-1176 (`JWST Medium-Deep Fields -- Windhorst IDS GTO
Program`).  Four public NIRCam target-level products are present: F090W,
F150W, F356W, and F444W.  They total 3.00 GB (2.79 GiB) and are included as
concrete downloader rows.

Official GTO-1176 program page:
https://www.stsci.edu/jwst-program-info/program/?program=1176

Official GTO NIRCam observation table, which identifies VV-191:
https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/past-jwst-proposal-opportunities/jwst-cycle-1-guaranteed-time-observations-call-for-proposals/jwst-gto-observation-specifications/jwst-gto-nircam-observations-table

GO-2738 is the Windhorst/Hammel North Ecliptic Pole medium-deep-field program;
its current Level-3 targets are `JWST-NEP-TDS-FIELD` and `SPITZER-IDF`, not
VV-191:
https://www.stsci.edu/jwst-program-info/program/?program=2738

## COSMOS-Web and CEERS

The archive query confirms large public field datasets, but not a named
foreground galaxy at z approximately 0.07:

- COSMOS-Web (GO-1727): public NIRCam F115W/F150W/F277W/F444W tile products,
  named `CWEBTILE-*` in MAST.  Program page:
  https://www.stsci.edu/jwst-program-info/program/?program=1727
- CEERS (ERS-1345): public NIRCam F115W/F150W/F200W/F277W/F356W/F410M/F444W
  field products.  Official MAST HLSP page:
  https://stdatu.stsci.edu/hlsp/ceers
  Program page:
  https://www.stsci.edu/jwst-program-info/program/?program=1345

Neither program's MAST target-level metadata supplies a concrete low-z galaxy
identifier.  The CSV therefore contains explicit screening placeholders, not
fabricated galaxy rows.  A defensible next step is a catalog/redshift
cross-match followed by a visual footprint check, then selection of the exact
tile containing each candidate.

## Centaurus A

The concrete public NIRCam products near the Cen A nucleus are from program
12496.  The CSV includes the current central/combined association, observation
4, and tile-2 observation 7 products in F090W, F187N, F200W, F277W, F335M,
and F444W.  There is no F150W target-level product in this program.  The full
set is 156.15 GB (145.42 GiB), largely because the central short-wave mosaics
are enormous.  Observation 7 is the cheaper first diagnostic: its F090W and
F200W files together are about 2.69 GiB.

Official program page:
https://www.stsci.edu/jwst-program-info/program/?program=12496

The resolved-star crowding and dust-lane warning is a scientific suitability
judgment, not a MAST metadata flag.  These products are downloader-ready but
remain a difficult SBF test.

## M104 / Sombrero

The concrete public NIRCam products are from program 6565.  The CSV includes
observations 2, 3, and tile observation 5 in F090W, F200W, F212N, F277W,
F335M, and F444W.  The archive confirms the practical problem already noted:
F200W is available, but F150W is not.  All 18 files total 25.16 GB
(23.44 GiB); a single F090W/F200W observation pair is the sensible test unit.

Official program page:
https://www.stsci.edu/jwst-program-info/program/?program=6565

## GO-8277: M105 / NGC 3379 and NGC 4278

Official program page:
https://www.stsci.edu/jwst-program-info/program/?program=8277

Official visit-status report:
https://www.stsci.edu/jwst-program-info/visits/?program=8277

The program has a 12-month Exclusive Access Period.  Current MAST rows for both
targets have `dataRights=EXCLUSIVE_ACCESS`:

- NGC 3379 (M105), observations 13 and 14: NIRCam F115W/F200W/F277W and
  NIRISS F115W/F200W.  MAST release timestamps fall on 2027-05-09 UTC.
- NGC 4278, observation 24: the same five instrument/filter products.  MAST
  release timestamps fall on 2027-06-09 UTC.

The 15 embargoed products total 9.44 GB (8.79 GiB) in current metadata.

The CSV records the current filenames, MAST URIs, metadata sizes when returned,
and exact `t_obs_release`, but labels them non-downloadable.  Do not schedule a
pipeline run before MAST changes `dataRights` to `PUBLIC`.

## Reproducibility and caveats

Observation discovery used POST requests to:
https://mast.stsci.edu/api/v0/invoke

with service `Mast.Caom.Filtered`, `obs_collection=JWST`, the relevant
`proposal_id`, and `calib_level=3`.  Product metadata checks used
`Mast.Caom.Products`.  Public product verification used the official endpoint
`https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/<filename>`.

MAST explicitly warns that JWST filenames can change after reprocessing:
https://jwst-docs.stsci.edu/accessing-jwst-data

Therefore this CSV is downloader-ready for the snapshot date, not a permanent
identifier registry.  Re-query MAST before a large batch download.
