# GO-5989 Coma inventory notes

Inventory timestamp: **2026-07-16 15:07 UTC**.

## Bottom line

- The official Phase-II proposal contains **39 named Coma elliptical-galaxy targets**, observations 1--39.
- Current MAST discovery metadata contains **115 public association-level `i2d` products**:
  - 38 target-centered NIRCam F150W products;
  - 38 target-centered NIRCam F356W products;
  - 39 coordinated-parallel NIRISS F150W products.
- Therefore **38/39 targets are downloader-ready as an NIRCam F150W+F356W pair**.
- **NGC 4926 (observation 35) is the exception.** Its visit is officially `Archived`, and its NIRISS F150W parallel `i2d` is public, but the two expected NIRCam association products are absent from current MAST discovery. Direct HEAD requests for the expected F150W and F356W association URIs both returned HTTP 404 on 2026-07-16.
- All 115 products listed in the CSV report `PUBLIC`, and all 115 listed URIs returned HTTP 200 to a HEAD request.

The downloader-ready CSV is `/private/tmp/go5989_manifest_research.csv`. It has one row per official galaxy target and separate signal, color, and parallel product columns. Blank signal/color fields on NGC 4926 are deliberate.

## Recommended downloader interpretation

- `signal_*`: target-centered **NIRCam F150W**, the SBF measurement band proposed by GO-5989.
- `color_*`: target-centered **NIRCam F356W**, a long-wave photometry/calibration band and a candidate second image for the generalized pipeline.
- `parallel_*`: **NIRISS F150W coordinated parallel**. It is not centered on the named galaxy even though the MAST row inherits that target name. Do not silently substitute this image for the missing NIRCam target frame.
- For a clean first download/run, select `download_ready_nircam_pair=yes`; this yields 38 galaxies.
- NGC 4926 should remain a blocked row and be re-queried later. Do not manufacture its two expected filenames as downloadable inputs while MAST returns 404.

Approximate current download volume from authoritative `Content-Length` headers:

| Product class | Count | Bytes | Decimal GB |
|---|---:|---:|---:|
| NIRCam F150W target frames | 38 | 49,036,754,880 | 49.037 |
| NIRCam F356W target frames | 38 | 11,649,709,440 | 11.650 |
| NIRISS F150W parallel fields | 39 | 5,018,987,520 | 5.019 |
| Total public association `i2d` | 115 | 65,705,451,840 | 65.705 |

Thus the 38 usable NIRCam pairs alone are about **60.69 GB**. The parallel products should only be downloaded if the auxiliary fields are actually needed.

## Science-frame classification

The query found no association-level blank-sky or calibration `i2d` rows for this program. The distinction that matters is:

1. NIRCam F150W/F356W: primary science observations placed on the named galaxy; suitable inputs for the target pipeline.
2. NIRISS F150W: coordinated parallel fields intended to augment globular-cluster and dwarf-galaxy coverage; auxiliary science, not a calibration blank and not guaranteed to contain the named primary galaxy.

The proposal describes all 39 primary targets as elliptical galaxies. It additionally labels NGC 4860 as giant elliptical and NGC 4874/NGC 4889 as brightest-cluster/giant ellipticals. Those are the only morphology refinements placed in the CSV. Redshift is deliberately blank: neither the Phase-II target table nor the queried MAST observation metadata supplies a target redshift, so no secondary-catalog value was mixed into this primary-source inventory.

## Exact target-name caveats

- `NGC 4853` and `NGC 4854` are two separate official observations (28 and 27), not a range.
- `NGC 4875` and `NGC 4876` are two separate official observations (8 and 6), not a range.
- Preserve the component suffixes in `NGC 4895A` and `NGC 4841A`.
- Preserve the full identifiers `CGCG 160-223`, `CGCG 160-065`, and `2MASX J12590459+2754389`.
- The CSV retains the exact MAST spelling in `mast_target_name` alongside a human-readable `target` spelling.

## Source and query provenance

Primary STScI sources:

- Program information and current completion/embargo metadata: <https://www.stsci.edu/jwst-program-info/program/?program=5989>
- Official 88-page Phase-II proposal, target list, observing setup, and exposure plan: <https://www.stsci.edu/jwst-program-info/download/jwst/pdf/5989/>
- Official visit-status report (all 39 visits shown as `Archived`): <https://www.stsci.edu/jwst-program-info/visits/?program=5989>
- MAST API endpoint used for observation discovery: <https://mast.stsci.edu/api/v0/invoke>
- MAST file endpoint used only for HEAD/status/size checks: <https://mast.stsci.edu/api/v0.1/Download/file>

MAST discovery criteria were `obs_collection=JWST` and `proposal_id=5989`, with no cone or target-name filter. The returned association records were grouped by the observation number embedded in `obs_id`. Product filenames and `mast:JWST/product/...` URIs come directly from each record's `dataURL`; access status and byte counts were independently checked with HEAD requests. No FITS image body was downloaded.

Proposal-level observing setup is NIRCam F150W+F356W, BRIGHT2, 9 groups, 1 integration, 4 dithers, nominal total exposure 773.047/773.048 s; NIRISS parallel F150W used NISRAPID, 16 groups, 1 integration, 4 dithers. Current MAST totals are 773.048 s for every available NIRCam association and 687.152 s for every NIRISS association. The difference between the older proposal PDF's stated NIRISS total and current archive metadata is recorded rather than overwritten.
