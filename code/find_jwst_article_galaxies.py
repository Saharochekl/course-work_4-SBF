#!/usr/bin/env python3
"""
Build a JWST availability table for galaxies mentioned in local article PDFs.

Inputs:
- ../materials/*.pdf
- article_galaxies_jwst_f150w_table.csv, if present

Outputs:
- sbf2_batch_outputs/jwst_article_galaxy_candidates.csv
- sbf2_batch_outputs/jwst_article_galaxy_f150w_selected.csv
- sbf2_batch_outputs/jwst_article_galaxy_search_summary.md
"""

from __future__ import annotations

import csv
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from astroquery.mast import Observations


ROOT = Path(__file__).resolve().parent
MATERIALS = ROOT.parent / "materials"
OUT_DIR = ROOT / "sbf2_batch_outputs"
TEXT_CACHE = OUT_DIR / "pdf_text_cache"

PDF_LABELS = {
    "1505.00400v2.pdf": "Jensen2015",
    "2101.02221v2.pdf": "Blakeslee2021",
    "2405.03743v3.pdf": "TRGB-SBF-I",
    "2502.15935v2.pdf": "Jensen2025",
    "2603.11160v1.pdf": "TRGB-SBF-IV",
}

MESSIER_TO_NGC = {
    "M49": "NGC 4472",
    "M59": "NGC 4621",
    "M60": "NGC 4649",
    "M84": "NGC 4374",
    "M86": "NGC 4406",
    "M87": "NGC 4486",
    "M89": "NGC 4552",
    "M105": "NGC 3379",
}

# Extra early-type/SBF-relevant seeds. These are only kept in the selected file
# if MAST actually returns public JWST/NIRCam F150W imaging.
FALLBACK_EARLY_TYPE_SEEDS = {
    "NGC 1023",
    "NGC 1400",
    "NGC 1407",
    "NGC 3115",
    "NGC 3377",
    "NGC 3607",
    "NGC 3608",
    "NGC 4365",
    "NGC 4382",
    "NGC 4459",
    "NGC 4473",
    "NGC 4526",
    "NGC 4564",
    "NGC 4570",
    "NGC 4660",
    "NGC 5813",
    "NGC 5845",
    "NGC 5846",
    "NGC 5866",
    "NGC 6861",
    "NGC 7619",
    "NGC 7626",
}


def normalize_galaxy(name: str) -> str:
    name = name.upper().replace("-", " ")
    m = re.match(r"\s*(NGC|IC)\s*(\d{1,5})\s*$", name)
    if not m:
        return name.strip()
    return f"{m.group(1)} {int(m.group(2))}"


def pdf_to_text(pdf: Path, out_txt: Path) -> str:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", str(pdf), str(out_txt)], check=True)
    return out_txt.read_text(errors="ignore")


def extract_article_galaxies() -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    ngc_ic_pat = re.compile(r"\b(?:NGC|IC)\s*-?\s*\d{2,5}\b", re.IGNORECASE)
    messier_pat = re.compile(r"\bM\s*([0-9]{1,3})\b", re.IGNORECASE)

    for pdf in sorted(MATERIALS.glob("*.pdf")):
        label = PDF_LABELS.get(pdf.name, pdf.stem)
        text = pdf_to_text(pdf, TEXT_CACHE / f"{pdf.stem}.txt")

        counts = Counter(normalize_galaxy(m.group(0)) for m in ngc_ic_pat.finditer(text))

        for m in messier_pat.finditer(text):
            messier = f"M{int(m.group(1))}"
            if messier in MESSIER_TO_NGC:
                counts[MESSIER_TO_NGC[messier]] += 1

        for galaxy, count in counts.items():
            entry = candidates.setdefault(
                galaxy,
                {
                    "galaxy": galaxy,
                    "source_class": "article_mentioned",
                    "source_articles": set(),
                    "mention_count": 0,
                    "aliases": set(),
                },
            )
            entry["source_articles"].add(label)
            entry["mention_count"] += count

    # Keep the previous hand-curated article list in the union, because it
    # includes table-level targets that may be hard to recover from PDF text.
    old_table = ROOT / "article_galaxies_jwst_f150w_table.csv"
    if old_table.exists():
        with old_table.open(newline="") as fh:
            for row in csv.DictReader(fh):
                galaxy = normalize_galaxy(row["galaxy"])
                entry = candidates.setdefault(
                    galaxy,
                    {
                        "galaxy": galaxy,
                        "source_class": "article_mentioned",
                        "source_articles": set(),
                        "mention_count": 0,
                        "aliases": set(),
                    },
                )
                entry["source_articles"].add("previous_curated_table")
                if row.get("source_articles"):
                    entry["source_articles"].update(s.strip() for s in row["source_articles"].split(";") if s.strip())
                if row.get("aliases"):
                    entry["aliases"].update(a.strip() for a in row["aliases"].split(";") if a.strip())

    for galaxy in FALLBACK_EARLY_TYPE_SEEDS:
        candidates.setdefault(
            galaxy,
            {
                "galaxy": galaxy,
                "source_class": "fallback_early_type_seed",
                "source_articles": set(),
                "mention_count": 0,
                "aliases": set(),
            },
        )

    return candidates


def mast_summary(galaxy: str) -> dict[str, str]:
    try:
        obs = Observations.query_object(galaxy, radius="2 arcmin")
    except Exception as exc:
        return {
            "mast_query_status": "failed",
            "mast_error": repr(exc),
            "jwst_obs_count": "0",
            "jwst_nircam_obs_count": "0",
            "jwst_filters": "",
            "jwst_programs": "",
            "has_public_nircam_f150w": "False",
            "has_public_nircam_f090w": "False",
            "f150w_obsids": "",
            "f090w_obsids": "",
            "f150w_dataurls": "",
            "f090w_dataurls": "",
            "recommended_for_sbf_extension": "False",
        }

    jwst_rows = [row for row in obs if str(row["obs_collection"]).upper() == "JWST"]
    nircam_rows = [row for row in jwst_rows if "NIRCAM" in str(row["instrument_name"]).upper()]

    filters = sorted({str(row["filters"]) for row in nircam_rows if str(row["filters"]).strip()})
    programs = sorted({str(row["proposal_id"]) for row in nircam_rows if str(row["proposal_id"]).strip()})

    def is_public_i2d(row) -> bool:
        data_url = str(row["dataURL"]) if "dataURL" in row.colnames else ""
        rights = str(row["dataRights"]).upper() if "dataRights" in row.colnames else ""
        try:
            calib_level = int(row["calib_level"])
        except Exception:
            calib_level = -1
        return rights == "PUBLIC" and calib_level >= 3 and "_i2d" in data_url.lower()

    def filt_rows(filter_name: str):
        return [
            row
            for row in nircam_rows
            if str(row["filters"]).upper() == filter_name
            and is_public_i2d(row)
        ]

    f150w = filt_rows("F150W")
    f090w = filt_rows("F090W")

    return {
        "mast_query_status": "ok",
        "mast_error": "",
        "jwst_obs_count": str(len(jwst_rows)),
        "jwst_nircam_obs_count": str(len(nircam_rows)),
        "jwst_filters": ";".join(filters),
        "jwst_programs": ";".join(programs),
        "has_public_nircam_f150w": str(bool(f150w)),
        "has_public_nircam_f090w": str(bool(f090w)),
        "f150w_obsids": ";".join(str(row["obs_id"]) for row in f150w),
        "f090w_obsids": ";".join(str(row["obs_id"]) for row in f090w),
        "f150w_dataurls": ";".join(str(row["dataURL"]) for row in f150w),
        "f090w_dataurls": ";".join(str(row["dataURL"]) for row in f090w),
        "recommended_for_sbf_extension": str(bool(f150w and f090w)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = extract_article_galaxies()

    rows = []
    for i, galaxy in enumerate(sorted(candidates), 1):
        print(f"[{i:03d}/{len(candidates):03d}] querying {galaxy}", flush=True)
        entry = candidates[galaxy]
        row = {
            "galaxy": galaxy,
            "aliases": ";".join(sorted(entry["aliases"])),
            "source_class": entry["source_class"],
            "source_articles": ";".join(sorted(entry["source_articles"])),
            "mention_count": str(entry["mention_count"]),
        }
        row.update(mast_summary(galaxy))
        rows.append(row)
        time.sleep(0.2)

    fieldnames = [
        "galaxy",
        "aliases",
        "source_class",
        "source_articles",
        "mention_count",
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

    candidates_csv = OUT_DIR / "jwst_article_galaxy_candidates.csv"
    with candidates_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = [
        row
        for row in rows
        if row["has_public_nircam_f150w"] == "True"
    ]
    selected_csv = OUT_DIR / "jwst_article_galaxy_f150w_selected.csv"
    with selected_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    article_selected = [r for r in selected if r["source_class"] == "article_mentioned"]
    fallback_selected = [r for r in selected if r["source_class"] == "fallback_early_type_seed"]
    both_filters = [r for r in selected if r["recommended_for_sbf_extension"] == "True"]

    summary = OUT_DIR / "jwst_article_galaxy_search_summary.md"
    with summary.open("w") as fh:
        fh.write("# JWST galaxy availability search\n\n")
        fh.write(f"- Local PDFs scanned: {len(list(MATERIALS.glob('*.pdf')))}\n")
        fh.write(f"- Total candidates queried: {len(rows)}\n")
        fh.write(f"- Candidates with public JWST/NIRCam F150W: {len(selected)}\n")
        fh.write(f"- Article-mentioned candidates with public JWST/NIRCam F150W: {len(article_selected)}\n")
        fh.write(f"- Fallback early-type seeds with public JWST/NIRCam F150W: {len(fallback_selected)}\n")
        fh.write(f"- Candidates with both public F150W and F090W: {len(both_filters)}\n\n")
        fh.write("## Recommended next SBF candidates\n\n")
        fh.write("These have public JWST/NIRCam F150W and F090W according to MAST, so they are the most direct extension of the current pipeline.\n\n")
        fh.write("| galaxy | source | programs | F150W obsids | F090W obsids |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in both_filters:
            fh.write(
                f"| {row['galaxy']} | {row['source_class']} / {row['source_articles']} | "
                f"{row['jwst_programs']} | {row['f150w_obsids']} | {row['f090w_obsids']} |\n"
            )

    print(f"Wrote {candidates_csv}")
    print(f"Wrote {selected_csv}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
