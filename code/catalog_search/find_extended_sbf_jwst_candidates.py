#!/usr/bin/env python3
"""
Cross-match additional SBF literature samples against public JWST/NIRCam data.

This extends find_jwst_article_galaxies.py beyond the PDFs already stored in
../materials by adding several large/standard SBF samples:
- Tonry et al. SBF survey
- Mei et al. ACS Virgo SBF catalog
- Blakeslee et al. ACS Fornax SBF catalog
- Jensen et al. 2021 MASSIVE/SN Ia host SBF sample
- NGC 4993 SBF distance paper

Outputs:
- sbf2_batch_outputs/jwst_extended_sbf_literature_candidates.csv
- sbf2_batch_outputs/jwst_extended_sbf_literature_f150w_f090w.csv
- sbf2_batch_outputs/jwst_extended_sbf_literature_unprocessed_f150w_f090w.csv
- sbf2_batch_outputs/jwst_extended_sbf_literature_summary.md
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

from find_jwst_article_galaxies import mast_summary, normalize_galaxy


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sbf2_batch_outputs"

SOURCE_URLS = {
    "Tonry-SBF-survey": "https://arxiv.org/abs/astro-ph/9609113",
    "ACS-Virgo-SBF": "https://arxiv.org/abs/astro-ph/0702510",
    "ACS-Fornax-SBF": "https://arxiv.org/abs/0901.1138",
    "Jensen2021-MASSIVE-SBF": "https://arxiv.org/abs/2105.08299",
    "NGC4993-SBF": "https://arxiv.org/abs/1801.06080",
}

SBF_SAMPLES = {
    "Jensen2021-MASSIVE-SBF": [
        "IC 2597", "NGC 0057", "NGC 0315", "NGC 0383", "NGC 0410",
        "NGC 0495", "NGC 0507", "NGC 0524", "NGC 0533", "NGC 0545",
        "NGC 0547", "NGC 0665", "NGC 0708", "NGC 0741", "NGC 0777",
        "NGC 0809", "NGC 0890", "NGC 0910", "NGC 1016", "NGC 1060",
        "NGC 1129", "NGC 1167", "NGC 1200", "NGC 1201", "NGC 1259",
        "NGC 1272", "NGC 1278", "NGC 1453", "NGC 1573", "NGC 1600",
        "NGC 1684", "NGC 1700", "NGC 2258", "NGC 2274", "NGC 2340",
        "NGC 2513", "NGC 2672", "NGC 2693", "NGC 2765", "NGC 2962",
        "NGC 3158", "NGC 3392", "NGC 3504", "NGC 3842", "NGC 4036",
        "NGC 4073", "NGC 4386", "NGC 4839", "NGC 4874", "NGC 4914",
        "NGC 4993", "NGC 5322", "NGC 5353", "NGC 5490", "NGC 5557",
        "NGC 5839", "NGC 6482", "NGC 6702", "NGC 6964", "NGC 7052",
        "NGC 7242", "NGC 7619",
    ],
    "Tonry-SBF-survey": [
        "IC 4182", "NGC 1023", "NGC 1058", "NGC 1316", "NGC 1373",
        "NGC 1380", "NGC 1386", "NGC 147", "NGC 1559", "NGC 185",
        "NGC 205", "NGC 2082", "NGC 224", "NGC 3031", "NGC 3351",
        "NGC 3368", "NGC 3379", "NGC 3627", "NGC 4278", "NGC 4321",
        "NGC 4365", "NGC 4374", "NGC 4494", "NGC 4526", "NGC 4536",
        "NGC 4548", "NGC 4565", "NGC 4579", "NGC 4600", "NGC 4639",
        "NGC 4660", "NGC 4725", "NGC 5128", "NGC 5253", "NGC 7331",
        "NGC 891", "NGC 925",
    ],
    "ACS-Fornax-SBF": [
        "IC 2006", "IC 3019", "IC 3025", "IC 3032", "IC 3065",
        "IC 3101", "IC 3292", "IC 3328", "IC 3381", "IC 3383",
        "IC 3442", "IC 3461", "IC 3468", "IC 3470", "IC 3487",
        "IC 3490", "IC 3501", "IC 3509", "IC 3586", "IC 3602",
        "IC 3612", "IC 3633", "IC 3635", "IC 3652", "IC 3653",
        "IC 3735", "IC 3779", "IC 798", "IC 809", "NGC 1316",
        "NGC 1336", "NGC 1339", "NGC 1340", "NGC 1351", "NGC 1366",
        "NGC 1373", "NGC 1374", "NGC 1375", "NGC 1379", "NGC 1380",
        "NGC 1381", "NGC 1386", "NGC 1387", "NGC 1389", "NGC 1396",
        "NGC 1399", "NGC 1404", "NGC 1419", "NGC 1427", "NGC 1428",
        "NGC 1460",
    ],
    "ACS-Virgo-SBF": [
        "IC 3019", "IC 3328", "IC 3381", "IC 3442", "IC 3468",
        "IC 3653", "NGC 4365", "NGC 4374", "NGC 4379", "NGC 4382",
        "NGC 4387", "NGC 4406", "NGC 4434", "NGC 4458", "NGC 4459",
        "NGC 4472", "NGC 4473", "NGC 4476", "NGC 4478", "NGC 4482",
        "NGC 4486", "NGC 4489", "NGC 4550", "NGC 4551", "NGC 4552",
        "NGC 4564", "NGC 4578", "NGC 4621", "NGC 4638", "NGC 4649",
        "NGC 4660", "NGC 4754",
    ],
    "NGC4993-SBF": ["NGC 4993"],
}

FIELDNAMES = [
    "galaxy",
    "source_articles",
    "source_urls",
    "already_processed",
    "mast_query_status",
    "mast_error",
    "jwst_obs_count",
    "jwst_nircam_obs_count",
    "jwst_filters",
    "jwst_programs",
    "has_public_nircam_f150w",
    "has_public_nircam_f090w",
    "recommended_for_sbf_extension",
    "f150w_obsids",
    "f090w_obsids",
    "f150w_dataurls",
    "f090w_dataurls",
]


def load_previous_mast_cache() -> dict[str, dict[str, str]]:
    cache = {}
    for path in [
        OUT_DIR / "jwst_article_galaxy_candidates.csv",
        OUT_DIR / "jwst_article_galaxy_f150w_selected.csv",
    ]:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                cache[normalize_galaxy(row["galaxy"])] = row
    return cache


def processed_galaxies() -> set[str]:
    done = set()
    results = OUT_DIR / "sbf2_batch_results.csv"
    if results.exists():
        with results.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("galaxy"):
                    done.add(normalize_galaxy(row["galaxy"]))
    for path in OUT_DIR.glob("NGC_*_result.json"):
        done.add(normalize_galaxy(path.stem.replace("_result", "").replace("_", " ")))
    return done


def build_source_map() -> dict[str, set[str]]:
    sources = defaultdict(set)
    for label, galaxies in SBF_SAMPLES.items():
        for galaxy in galaxies:
            sources[normalize_galaxy(galaxy)].add(label)
    return sources


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_map = build_source_map()
    mast_cache = load_previous_mast_cache()
    processed = processed_galaxies()

    rows = []
    galaxies = sorted(source_map)
    for i, galaxy in enumerate(galaxies, 1):
        labels = sorted(source_map[galaxy])
        print(f"[{i:03d}/{len(galaxies):03d}] {galaxy}: {';'.join(labels)}", flush=True)

        if galaxy in mast_cache:
            mast = {k: mast_cache[galaxy].get(k, "") for k in FIELDNAMES if k not in {"galaxy", "source_articles", "source_urls", "already_processed"}}
        else:
            mast = mast_summary(galaxy)
            time.sleep(0.2)

        rows.append({
            "galaxy": galaxy,
            "source_articles": ";".join(labels),
            "source_urls": ";".join(SOURCE_URLS[label] for label in labels),
            "already_processed": str(galaxy in processed),
            **mast,
        })

    all_csv = OUT_DIR / "jwst_extended_sbf_literature_candidates.csv"
    with all_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    both = [r for r in rows if r["has_public_nircam_f150w"] == "True" and r["has_public_nircam_f090w"] == "True"]
    both_csv = OUT_DIR / "jwst_extended_sbf_literature_f150w_f090w.csv"
    with both_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(both)

    unprocessed = [r for r in both if r["already_processed"] != "True"]
    unprocessed_csv = OUT_DIR / "jwst_extended_sbf_literature_unprocessed_f150w_f090w.csv"
    with unprocessed_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(unprocessed)

    f150w = [r for r in rows if r["has_public_nircam_f150w"] == "True"]
    summary = OUT_DIR / "jwst_extended_sbf_literature_summary.md"
    with summary.open("w") as fh:
        fh.write("# Extended SBF literature x JWST/NIRCam availability\n\n")
        fh.write("## Literature sources scanned\n\n")
        for label, url in SOURCE_URLS.items():
            fh.write(f"- {label}: {url}\n")
        fh.write("\n## Counts\n\n")
        fh.write(f"- Literature galaxies checked: {len(rows)}\n")
        fh.write(f"- With public JWST/NIRCam F150W: {len(f150w)}\n")
        fh.write(f"- With public JWST/NIRCam F150W and F090W: {len(both)}\n")
        fh.write(f"- With both filters and not already processed: {len(unprocessed)}\n\n")
        fh.write("## Unprocessed galaxies with public F150W and F090W\n\n")
        fh.write("| galaxy | sources | programs | F150W obsids | F090W obsids |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in unprocessed:
            fh.write(
                f"| {row['galaxy']} | {row['source_articles']} | {row['jwst_programs']} | "
                f"{row['f150w_obsids']} | {row['f090w_obsids']} |\n"
            )

    print(f"Wrote {all_csv}")
    print(f"Wrote {both_csv}")
    print(f"Wrote {unprocessed_csv}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
