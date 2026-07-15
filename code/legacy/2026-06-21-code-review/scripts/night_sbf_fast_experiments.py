import copy
import json
import os
import re
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
DATA_DIR = PROJECT_ROOT / "data" / "NGC 1380"
OUTPUT_ROOT = PROJECT_ROOT / "code" / "experiment_outputs_fast"
LOG_PATH = PROJECT_ROOT / "code" / "codex_night_run_log.md"

# These experiments are for controlled comparative screening, not final
# publication-grade uncertainties. Keep the same reduced settings for every
# variant so the relative comparison remains fair without rerunning the entire
# heavy notebook tail at full cost.
FFT_E_REALIZATIONS_MAIN_EXP = 4
FFT_E_REALIZATIONS_DIAG_EXP = 2

IMPORT_CELLS = [1, 2, 4, 6, 11]
HELPER_CELLS = [38, 40, 53, 74, 81]
PSF_CELL = 62
CLIP_CELL = 32
SCIENCE_PATH_CELL = 36
RESIDUAL_DIAG_CELLS = [76, 78]
MODEL_DIAG_CELL = 80
MEASURE_CELLS = [64, 66, 68, 82, 83, 85, 87]

K_EXPERIMENTS = [
    {
        "name": "exp01_baseline_current_k",
        "model_variant": "baseline",
        "main_k": (0.03, 0.40),
        "region_k_windows": [(0.02, 0.35), (0.03, 0.40)],
        "why": "Current notebook baseline.",
    },
    {
        "name": "exp02_conservative_lowk_cut",
        "model_variant": "baseline",
        "main_k": (0.05, 0.35),
        "region_k_windows": [(0.04, 0.30), (0.05, 0.35)],
        "why": "Stricter low-k cut against large-scale residual structure.",
    },
    {
        "name": "exp03_conservative_highk_cut",
        "model_variant": "baseline",
        "main_k": (0.03, 0.25),
        "region_k_windows": [(0.02, 0.20), (0.03, 0.25)],
        "why": "Stricter high-k cut against pixel-scale/correlated-noise power.",
    },
    {
        "name": "exp04_conservative_both_cuts",
        "model_variant": "baseline",
        "main_k": (0.05, 0.25),
        "region_k_windows": [(0.04, 0.20), (0.05, 0.25)],
        "why": "Both conservative low-k and high-k cuts.",
    },
    {
        "name": "exp05_literature_closer_lowk",
        "model_variant": "baseline",
        "main_k": (0.01, 0.25),
        "region_k_windows": [(0.005, 0.20), (0.01, 0.25)],
        "why": "Closer to Jensen-style retention of lower k while still trimming high k.",
    },
]


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def execute_cell(nb, idx, ns):
    ns["display"] = lambda *args, **kwargs: None
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        exec("".join(nb["cells"][idx]["source"]), ns)
    finally:
        plt.show = old_show


def install_deterministic_rng(ns, seed):
    np_mod = ns["np"]
    orig = np_mod.random.default_rng
    ns["_orig_default_rng"] = orig
    ns["_rng_seed"] = int(seed)
    ns["_rng_counter"] = 0

    def deterministic_default_rng(arg=None):
        if arg is not None:
            return orig(arg)
        s = int(ns.get("_rng_seed", seed))
        c = int(ns.get("_rng_counter", 0))
        ns["_rng_counter"] = c + 1
        return orig(s + c)

    np_mod.random.default_rng = deterministic_default_rng


def reset_rng(ns, seed):
    ns["_rng_seed"] = int(seed)
    ns["_rng_counter"] = 0


def parse_stream_text(nb, idx):
    texts = []
    for out in nb["cells"][idx].get("outputs", []):
        if out.get("output_type") == "stream":
            texts.append("".join(out.get("text", "")))
    return "\n".join(texts)


def rex(pattern, text, cast=float):
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"Pattern not found: {pattern}")
    return cast(m.group(1))


def prepare_base_namespace(exp_dir):
    nb = load_notebook()
    ns = {"__builtins__": __builtins__}
    for idx in IMPORT_CELLS:
        execute_cell(nb, idx, ns)
        if idx == 2:
            install_deterministic_rng(ns, seed=424242)
        if idx == 4:
            ns["out_dir"] = exp_dir
            ns["out_dir"].mkdir(parents=True, exist_ok=True)
            ns["DEBLEND_NPROC"] = 1
            ns["FFT_WORKERS"] = 1
            ns["FFT_E_REALIZATIONS_MAIN"] = FFT_E_REALIZATIONS_MAIN_EXP
            ns["FFT_E_REALIZATIONS_DIAG"] = FFT_E_REALIZATIONS_DIAG_EXP
            ns["DO_DEBLEND"] = True
            ns["SBF_PR_DO_DEBLEND"] = True

    # Seed PSF cache into writable experiment dir.
    psf_src = DATA_DIR / f"{ns['stem']}_psf_{ns['PSF_SIZE']}.fits"
    psf_dst = exp_dir / psf_src.name
    if psf_src.exists() and not psf_dst.exists():
        shutil.copy2(psf_src, psf_dst)

    # Load saved state instead of rerunning the expensive isophote/model path.
    stem = ns["stem"]
    ns["premask"] = ns["fits"].getdata(DATA_DIR / f"{stem}_dbg_mask.fits").astype(bool)
    ns["premask_src"] = ns["fits"].getdata(DATA_DIR / f"{stem}_dbg_mask_src.fits").astype(bool)
    ns["premask_segm"] = None
    ns["premask_compact_labels"] = np.array([], dtype=int)
    ns["model"] = ns["fits"].getdata(DATA_DIR / f"{stem}_sbf_model.fits").astype(np.float32)
    ns["model_full"] = ns["fits"].getdata(DATA_DIR / f"{stem}_sbf_model_full.fits").astype(np.float32)
    ns["resid_full"] = ns["fits"].getdata(DATA_DIR / f"{stem}_sbf_resid_full_science_raw.fits").astype(np.float32)
    ns["resid_full_clip"] = ns["fits"].getdata(DATA_DIR / f"{stem}_sbf_resid_full_science_clip_3p5sigma.fits").astype(np.float32)
    ns["resid_full_clip_delta"] = ns["fits"].getdata(DATA_DIR / f"{stem}_sbf_resid_full_science_clip_3p5sigma_delta.fits").astype(np.float32)
    ns["clip_tag"] = ns["CLIP_TAG_QC"]

    support = np.isfinite(ns["model"]) & (ns["model"] > 0.0)
    ys, xs = np.where(support)
    ns["y1"] = int(ys.min())
    ns["y2"] = int(ys.max()) + 1
    ns["x1"] = int(xs.min())
    ns["x2"] = int(xs.max()) + 1
    ns["model_c"] = np.asarray(ns["model"][ns["y1"]:ns["y2"], ns["x1"]:ns["x2"]], dtype=np.float32)

    cell30 = parse_stream_text(nb, 30)
    cell55 = parse_stream_text(nb, 55)
    ns["fitted_sma_max_px"] = rex(r"fitted profile sma range = [0-9.]+\.\.([0-9.]+) px", cell30)
    ns["sma_model_full_max_px"] = rex(r"built to ([0-9.]+) px", cell30)
    ns["x0_model_full"] = rex(r"fixed outer geometry: x0=([0-9.]+)", cell30)
    ns["y0_model_full"] = rex(r"fixed outer geometry: x0=[0-9.]+, y0=([0-9.]+)", cell30)
    ns["eps_model_full"] = rex(r"fixed outer geometry: .* eps=([0-9.]+)", cell30)
    ns["q_model_full"] = rex(r"fixed outer geometry: .* q=([0-9.]+)", cell30)
    ns["pa_model_full"] = rex(r"fixed outer geometry: .* pa=([0-9.]+) rad", cell30)
    ns["x0_sbf_circ"] = ns["x0_model_full"]
    ns["y0_sbf_circ"] = ns["y0_model_full"]

    ns["x0_ann"] = rex(r"final geometry: x0=([0-9.]+)", cell55)
    ns["y0_ann"] = rex(r"final geometry: x0=[0-9.]+, y0=([0-9.]+)", cell55)
    ns["sma_in"] = rex(r"sma_in=([0-9.]+)", cell55)
    ns["sma_out"] = rex(r"sma_out=([0-9.]+)", cell55)
    ns["eps_ann"] = rex(r"eps=([0-9.]+)", cell55)
    ns["q_ann"] = rex(r"q=([0-9.]+)", cell55)
    ns["pa_ann"] = rex(r"pa=([0-9.]+)", cell55)

    execute_cell(nb, SCIENCE_PATH_CELL, ns)
    for idx in HELPER_CELLS:
        execute_cell(nb, idx, ns)
    execute_cell(nb, PSF_CELL, ns)
    return nb, ns


def apply_annulus_geometry(ns):
    build_region_mask_ellipse = ns["build_region_mask_ellipse"]
    ns["annulus_ell"] = build_region_mask_ellipse(
        ns["science_resid"].shape,
        ns["x0_ann"],
        ns["y0_ann"],
        ns["q_ann"],
        ns["pa_ann"],
        ns["sma_in"],
        ns["sma_out"],
    )
    ns["mask_sbf"] = ns["premask"] | (~ns["annulus_ell"])


def run_residual_diags(nb, ns, exp_dir, include_model_diag):
    apply_annulus_geometry(ns)
    for idx in RESIDUAL_DIAG_CELLS:
        execute_cell(nb, idx, ns)
        save_figs(exp_dir, f"cell{idx}")
    if include_model_diag:
        execute_cell(nb, MODEL_DIAG_CELL, ns)
        save_figs(exp_dir, f"cell{MODEL_DIAG_CELL}")


def save_figs(exp_dir, prefix):
    for num in list(plt.get_fignums()):
        fig = plt.figure(num)
        fig.savefig(exp_dir / f"{prefix}_fig{num}.png", bbox_inches="tight", dpi=140)
        plt.close(fig)


def run_measurement(nb, ns, exp, seed):
    ns["FFT_K_RANGE_MAIN"] = tuple(exp["main_k"])
    ns["SBF_REGION_K_WINDOWS"] = [tuple(x) for x in exp["region_k_windows"]]
    reset_rng(ns, seed)
    apply_annulus_geometry(ns)
    for idx in MEASURE_CELLS:
        execute_cell(nb, idx, ns)


def extract_zero_cross(df_profile, radius_type):
    if not isinstance(df_profile, pd.DataFrame) or df_profile.empty:
        return np.nan
    sub = df_profile[df_profile["radius_type"].eq(radius_type)].sort_values("r_mid_px")
    vals = sub["resid_raw_median"].to_numpy(dtype=float)
    r = sub["r_mid_px"].to_numpy(dtype=float)
    for i in range(len(vals) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if v0 == 0:
            return float(r[i])
        if v0 * v1 < 0:
            frac = abs(v0) / (abs(v0) + abs(v1))
            return float(r[i] + frac * (r[i + 1] - r[i]))
    return np.nan


def collect_metrics(ns, exp):
    resid_df = ns.get("df_resid_audit")
    resid_rows = {}
    if isinstance(resid_df, pd.DataFrame):
        for row in resid_df.to_dict(orient="records"):
            resid_rows[row["region"]] = row

    main = copy.deepcopy(ns.get("sbf_main_diagnostics", {}))
    rec = copy.deepcopy(ns.get("recommended_sbf"))
    profile = ns.get("df_model_bias_profile")
    return {
        "experiment": exp["name"],
        "model_variant": exp["model_variant"],
        "main_k": tuple(exp["main_k"]),
        "region_k_windows": [tuple(x) for x in exp["region_k_windows"]],
        "elliptical_corrected_mbar": float(main.get("mbar_spec", np.nan)),
        "circular_inner_corrected_mbar": float(rec["mbar_inner"]) if rec else np.nan,
        "circular_outer_corrected_mbar": float(rec["mbar_outer"]) if rec else np.nan,
        "weighted_corrected_mbar": float(rec["mbar_weighted"]) if rec else np.nan,
        "formal_sigma": float(rec["sigma_weighted_formal"]) if rec else np.nan,
        "annulus_scatter": float(rec["annulus_scatter"]) if rec else np.nan,
        "adopted_sigma": float(rec["sigma_adopted"]) if rec else np.nan,
        "recommended_kmin": float(rec["kmin"]) if rec else np.nan,
        "recommended_kmax": float(rec["kmax"]) if rec else np.nan,
        "residual_regions": resid_rows,
        "zero_cross_circle_px": extract_zero_cross(profile, "circle"),
        "zero_cross_ellipse_px": extract_zero_cross(profile, "ellipse"),
    }


def residual_score(metrics):
    resid = metrics["residual_regions"]
    vals = [
        abs(resid.get("elliptical_chosen", {}).get("resid_median", np.nan)),
        abs(resid.get("circular_inner_lit", {}).get("resid_median", np.nan)),
        abs(resid.get("circular_outer_lit", {}).get("resid_median", np.nan)),
    ]
    return float(np.nansum(vals))


def write_log_header():
    if LOG_PATH.exists():
        return
    LOG_PATH.write_text("# Codex Night Run Log\n\n")


def append_log(text):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")


def log_section(metrics, decision, rationale):
    resid = metrics["residual_regions"]
    lines = [
        f"### {metrics['experiment']}",
        f"- Timestamp: {ts()}",
        f"- Model variant: `{metrics['model_variant']}`",
        f"- Main k-window: `{metrics['main_k']}`",
        f"- Region k-windows: `{metrics['region_k_windows']}`",
        f"- FFT E(k) realizations: main=`{FFT_E_REALIZATIONS_MAIN_EXP}`, region=`{FFT_E_REALIZATIONS_DIAG_EXP}`",
        f"- Decision: `{decision}`",
        f"- Why: {rationale}",
        "- Science numbers:",
        f"  elliptical corrected mbar = `{metrics['elliptical_corrected_mbar']:.4f}`",
        f"  circular inner corrected mbar = `{metrics['circular_inner_corrected_mbar']:.4f}`",
        f"  circular outer corrected mbar = `{metrics['circular_outer_corrected_mbar']:.4f}`",
        f"  weighted corrected mbar = `{metrics['weighted_corrected_mbar']:.4f}`",
        f"  formal sigma = `{metrics['formal_sigma']:.4f}`",
        f"  annulus scatter = `{metrics['annulus_scatter']:.4f}`",
        f"  adopted sigma = `{metrics['adopted_sigma']:.4f}`",
        "- Residual diagnostics:",
    ]
    for region in ["elliptical_chosen", "circular_inner_lit", "circular_outer_lit"]:
        row = resid.get(region, {})
        med = row.get("resid_median", np.nan)
        mean = row.get("resid_mean", np.nan)
        sign = "positive" if np.isfinite(med) and med > 0 else "negative" if np.isfinite(med) and med < 0 else "zero/unknown"
        lines.append(f"  {region}: median=`{med:.4e}`, mean=`{mean:.4e}`, sign=`{sign}`")
    lines.append(f"- zero-cross circle px = `{metrics['zero_cross_circle_px']:.2f}`")
    lines.append(f"- zero-cross ellipse px = `{metrics['zero_cross_ellipse_px']:.2f}`")
    append_log("\n".join(lines))


def decide_k_variant(metrics, baseline_metrics):
    if baseline_metrics is None:
        return "keep", "baseline reference"
    cur = metrics["adopted_sigma"]
    base = baseline_metrics["adopted_sigma"]
    if np.isfinite(cur) and np.isfinite(base):
        if cur < 0.95 * base:
            return "keep", "same residual image as baseline; annulus consistency improved"
        if cur > 1.10 * base:
            return "reject", "same residual image as baseline; annulus consistency worsened"
    return "keep", "same residual image as baseline; no clear degradation"


def decide_model_variant(metrics, baseline_metrics):
    cur_score = residual_score(metrics)
    base_score = residual_score(baseline_metrics)
    if np.isfinite(cur_score) and np.isfinite(base_score):
        if cur_score < 0.8 * base_score:
            if np.isfinite(metrics["adopted_sigma"]) and np.isfinite(baseline_metrics["adopted_sigma"]) and metrics["adopted_sigma"] > 1.25 * baseline_metrics["adopted_sigma"]:
                return "reject", "residual bias improved, but annulus scatter/formal error became too poor"
            return "keep", "large-scale residual bias is weaker and science numbers did not collapse"
        if cur_score > 1.2 * base_score:
            return "reject", "large-scale residual bias is stronger"
    return "keep", "residual quality similar"


def save_metrics(exp_dir, metrics):
    (exp_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def choose_best_k(metrics_list):
    finite = [m for m in metrics_list if np.isfinite(m["adopted_sigma"])]
    if not finite:
        return metrics_list[0]
    finite.sort(key=lambda m: (m["adopted_sigma"], m["annulus_scatter"]))
    return finite[0]


def apply_fitted_splice(ns):
    support = np.isfinite(ns["model"]) & (ns["model"] > 0.0)
    ns["model_full"] = np.asarray(ns["model_full"], dtype=np.float32).copy()
    ns["model_full"][support] = np.asarray(ns["model"], dtype=np.float32)[support]
    ns["resid_full"] = np.array(ns["img"] - ns["model_full"], dtype=np.float32, copy=True)
    ns["resid_full"][ns["premask"] | (~np.isfinite(ns["model_full"])) | (~np.isfinite(ns["img"]))] = np.nan
    nb = load_notebook()
    execute_cell(nb, CLIP_CELL, ns)
    execute_cell(nb, SCIENCE_PATH_CELL, ns)
    execute_cell(nb, 38, ns)
    execute_cell(nb, 40, ns)


def main():
    os.chdir(PROJECT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_log_header()

    base_dir = OUTPUT_ROOT / "shared_current_model"
    nb, ns = prepare_base_namespace(base_dir)
    run_residual_diags(nb, ns, base_dir, include_model_diag=True)

    all_metrics = []
    baseline_metrics = None
    for i, exp in enumerate(K_EXPERIMENTS):
        exp_dir = OUTPUT_ROOT / exp["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {exp['name']} {exp['main_k']}")
        run_measurement(nb, ns, exp, seed=1000 + i * 100)
        metrics = collect_metrics(ns, exp)
        decision, rationale = decide_k_variant(metrics, baseline_metrics)
        if baseline_metrics is None:
            baseline_metrics = copy.deepcopy(metrics)
        save_metrics(exp_dir, metrics)
        log_section(metrics, decision, rationale)
        all_metrics.append(metrics)

    best_k = choose_best_k(all_metrics)

    model_exp = {
        "name": "exp06_fitted_splice_best_k",
        "model_variant": "fitted_splice",
        "main_k": tuple(best_k["main_k"]),
        "region_k_windows": [tuple(x) for x in best_k["region_k_windows"]],
        "why": "Exact fitted-model values on fitted support; outer extrapolation only outside support.",
    }
    model_dir = OUTPUT_ROOT / model_exp["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    nb2, ns2 = prepare_base_namespace(model_dir)
    apply_fitted_splice(ns2)
    run_residual_diags(nb2, ns2, model_dir, include_model_diag=True)
    run_measurement(nb2, ns2, model_exp, seed=9000)
    model_metrics = collect_metrics(ns2, model_exp)
    decision, rationale = decide_model_variant(model_metrics, baseline_metrics)
    save_metrics(model_dir, model_metrics)
    log_section(model_metrics, decision, rationale)
    all_metrics.append(model_metrics)

    rows = []
    for m in all_metrics:
        rows.append({
            "experiment": m["experiment"],
            "model_variant": m["model_variant"],
            "main_kmin": m["main_k"][0],
            "main_kmax": m["main_k"][1],
            "weighted_corrected_mbar": m["weighted_corrected_mbar"],
            "elliptical_corrected_mbar": m["elliptical_corrected_mbar"],
            "circular_inner_corrected_mbar": m["circular_inner_corrected_mbar"],
            "circular_outer_corrected_mbar": m["circular_outer_corrected_mbar"],
            "formal_sigma": m["formal_sigma"],
            "annulus_scatter": m["annulus_scatter"],
            "adopted_sigma": m["adopted_sigma"],
            "residual_score": residual_score(m),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    print(f"[DONE] log -> {LOG_PATH}")
    print(f"[DONE] summary -> {OUTPUT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
