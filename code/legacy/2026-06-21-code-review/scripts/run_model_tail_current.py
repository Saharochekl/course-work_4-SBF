import json
import os
import shutil
import time
from pathlib import Path

for env_name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(env_name, "1")

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdgcache_codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "code" / "sbf-1.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "code" / "experiment_outputs_model_fix" / "hybrid_isophote_v1"
DATA_DIR = PROJECT_ROOT / "data" / "NGC 1380"
DATA_PSF_CACHE = DATA_DIR / "jw03055-o001_t001_nircam_clear-f150w_i2d_psf_129.fits"

SELECTED_CELLS = [
    1,
    2,
    4,
    6,
    8,
    11,
    13,
    15,
    17,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    33,
    34,
    36,
    38,
    40,
    42,
    46,
    48,
    50,
    51,
    53,
    55,
    62,
    64,
    66,
    68,
    70,
    74,
    76,
    78,
    80,
    81,
    82,
    83,
    85,
    87,
]


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def save_open_figures(prefix):
    for num in list(plt.get_fignums()):
        fig = plt.figure(num)
        fig.savefig(OUTPUT_DIR / f"{prefix}_fig{num}.png", bbox_inches="tight", dpi=140)
        plt.close(fig)


def execute_cell(nb, idx, ns):
    ns["display"] = lambda *args, **kwargs: None
    old_show = plt.show
    plt.show = lambda *args, **kwargs: save_open_figures(f"cell{idx}")
    try:
        exec("".join(nb["cells"][idx]["source"]), ns)
    finally:
        plt.show = old_show
        save_open_figures(f"cell{idx}")


def save_tables(ns):
    for name in [
        "df_resid_audit",
        "df_model_bias_profile",
        "df_model_bias_regions",
        "df_model_overlap_profile",
        "df_sbf",
        "df_sbf_compare",
        "df_annulus_summary",
        "pipeline_variants",
    ]:
        obj = ns.get(name)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def save_summary(ns):
    def region_row(df, region):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        sub = df[df["region"].eq(region)]
        if sub.empty:
            return {}
        return sub.iloc[0].to_dict()

    resid_df = ns.get("df_resid_audit")
    ann_df = ns.get("df_annulus_summary")
    overlap_df = ns.get("df_model_overlap_profile")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(OUTPUT_DIR),
        "science_model_name": ns.get("science_model_name"),
        "science_resid_name": ns.get("science_resid_name"),
        "recommended_sbf": ns.get("recommended_sbf"),
        "resid_audit": {
            "elliptical_chosen": region_row(resid_df, "elliptical_chosen"),
            "circular_inner_lit": region_row(resid_df, "circular_inner_lit"),
            "circular_outer_lit": region_row(resid_df, "circular_outer_lit"),
        },
        "annulus_summary_rows": int(len(ann_df)) if isinstance(ann_df, pd.DataFrame) else 0,
        "model_overlap_rows": int(len(overlap_df)) if isinstance(overlap_df, pd.DataFrame) else 0,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_psf_cache = OUTPUT_DIR / DATA_PSF_CACHE.name
    if DATA_PSF_CACHE.exists() and not output_psf_cache.exists():
        shutil.copy2(DATA_PSF_CACHE, output_psf_cache)
        print(f"[setup] copied PSF cache -> {output_psf_cache}")

    nb = load_notebook()
    ns = {"__builtins__": __builtins__}

    t0 = time.time()
    for idx in SELECTED_CELLS:
        cell_t0 = time.time()
        execute_cell(nb, idx, ns)
        if idx == 4:
            ns["out_dir"] = OUTPUT_DIR
            ns["DEBLEND_NPROC"] = 1
            ns["FFT_WORKERS"] = 1
        print(f"[run {time.strftime('%H:%M:%S')}] cell {idx} done in {time.time() - cell_t0:.1f}s")

    save_tables(ns)
    save_summary(ns)

    rec = ns.get("recommended_sbf") or {}
    print(f"[DONE] total {time.time() - t0:.1f}s")
    print(f"[DONE] output_dir -> {OUTPUT_DIR}")
    if rec:
        print(
            "[DONE] recommended k={:.3f}..{:.3f}, weighted corrected mbar={:.4f}, adopted sigma={:.4f}".format(
                float(rec.get("kmin", np.nan)),
                float(rec.get("kmax", np.nan)),
                float(rec.get("mbar_weighted", np.nan)),
                float(rec.get("sigma_adopted", np.nan)),
            )
        )


if __name__ == "__main__":
    main()
