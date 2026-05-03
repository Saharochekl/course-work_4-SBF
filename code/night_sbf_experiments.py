import contextlib
import copy
import io
import json
import os
import shutil
import time
from pathlib import Path

# Safe single-thread defaults before importing heavy numeric stack.
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
OUTPUT_ROOT = PROJECT_ROOT / "code" / "experiment_outputs"
LOG_PATH = PROJECT_ROOT / "code" / "codex_night_run_log.md"

SETUP_CELLS = [
    1, 2, 4, 6, 8, 11, 13, 15, 17, 19, 22, 24, 26, 28, 30, 32, 36, 38, 40,
    42, 48, 50, 51, 53, 55, 62, 74, 81,
]
MEASURE_CELLS = [64, 66, 68, 82, 87]
RESIDUAL_DIAG_CELLS = [76, 78]
MODEL_DIAG_CELL = 80

EXPERIMENTS = [
    {
        "name": "exp01_baseline_current_k",
        "kind": "k_window",
        "model_variant": "baseline",
        "main_k": (0.03, 0.40),
        "region_k_windows": [(0.02, 0.35), (0.03, 0.40), (0.04, 0.35)],
        "why": "Current notebook baseline.",
    },
    {
        "name": "exp02_conservative_lowk_cut",
        "kind": "k_window",
        "model_variant": "baseline",
        "main_k": (0.05, 0.35),
        "region_k_windows": [(0.04, 0.30), (0.05, 0.35), (0.06, 0.30)],
        "why": "More conservative low-k cut to suppress large-scale model/sky residuals.",
    },
    {
        "name": "exp03_conservative_highk_cut",
        "kind": "k_window",
        "model_variant": "baseline",
        "main_k": (0.03, 0.25),
        "region_k_windows": [(0.02, 0.20), (0.03, 0.25), (0.04, 0.25)],
        "why": "More conservative high-k cut to reduce pixel-scale/correlated-noise sensitivity.",
    },
    {
        "name": "exp04_conservative_both_cuts",
        "kind": "k_window",
        "model_variant": "baseline",
        "main_k": (0.05, 0.25),
        "region_k_windows": [(0.04, 0.20), (0.05, 0.25), (0.06, 0.25)],
        "why": "Conservative low-k and high-k cuts together.",
    },
    {
        "name": "exp05_literature_closer_lowk",
        "kind": "k_window",
        "model_variant": "baseline",
        "main_k": (0.01, 0.25),
        "region_k_windows": [(0.005, 0.20), (0.01, 0.25), (0.015, 0.25)],
        "why": "Closer to Jensen-style retention of lower k while still trimming high k.",
    },
]


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def execute_cell(nb, idx, ns):
    src = "".join(nb["cells"][idx]["source"])
    ns["display"] = lambda *args, **kwargs: None
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        exec(src, ns)
    finally:
        plt.show = old_show


def install_deterministic_rng(ns, seed):
    np_mod = ns["np"]
    orig = np_mod.random.default_rng
    ns["_orig_default_rng"] = orig
    ns["_deterministic_rng_seed"] = int(seed)
    ns["_deterministic_rng_counter"] = 0

    def deterministic_default_rng(arg=None):
        if arg is not None:
            return orig(arg)
        current_seed = int(ns.get("_deterministic_rng_seed", seed))
        counter = int(ns.get("_deterministic_rng_counter", 0))
        ns["_deterministic_rng_counter"] = counter + 1
        return orig(current_seed + counter)

    np_mod.random.default_rng = deterministic_default_rng


def reset_rng_counter(ns, seed):
    ns["_deterministic_rng_seed"] = int(seed)
    ns["_deterministic_rng_counter"] = 0


def seed_psf_cache(ns):
    out_dir = ns["out_dir"]
    stem = ns["stem"]
    psf_name = f"{stem}_psf_{ns['PSF_SIZE']}.fits"
    src = PROJECT_ROOT / "data" / "NGC 1380" / psf_name
    dst = out_dir / psf_name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def setup_namespace(exp_dir, model_variant="baseline"):
    nb = load_notebook()
    ns = {"__builtins__": __builtins__}
    for idx in SETUP_CELLS:
        execute_cell(nb, idx, ns)
        if idx == 2:
            install_deterministic_rng(ns, seed=246810)
        if idx == 4:
            ns["out_dir"] = exp_dir
            ns["out_dir"].mkdir(parents=True, exist_ok=True)
            ns["DEBLEND_NPROC"] = 1
            ns["FFT_WORKERS"] = 1
            ns["DO_DEBLEND"] = True
            ns["SBF_PR_DO_DEBLEND"] = True
            seed_psf_cache(ns)
        if idx == 30 and model_variant == "fitted_splice":
            apply_fitted_splice_variant(ns)
            # Recompute downstream state that depends on model_full before continuing setup.
            for post_idx in [32, 36]:
                execute_cell(nb, post_idx, ns)
    return nb, ns


def apply_fitted_splice_variant(ns):
    model = np.asarray(ns["model"], dtype=np.float32)
    model_full = np.asarray(ns["model_full"], dtype=np.float32).copy()
    support = np.isfinite(model) & (model > 0.0)
    model_full[support] = model[support]
    model_full[(~np.isfinite(model_full)) | (model_full <= 0.0)] = np.nan
    ns["model_full"] = model_full
    ns["resid_full"] = np.array(ns["img"] - model_full, dtype=np.float32, copy=True)
    ns["resid_full"][ns["premask"] | (~np.isfinite(model_full)) | (~np.isfinite(ns["img"]))] = np.nan


def save_open_figures(exp_dir, tag):
    saved = []
    for fig_num in list(plt.get_fignums()):
        fig = plt.figure(fig_num)
        path = exp_dir / f"{tag}_fig{fig_num}.png"
        fig.savefig(path, bbox_inches="tight", dpi=140)
        saved.append(path)
        plt.close(fig)
    return saved


def run_residual_diagnostics(nb, ns, exp_dir, include_model_overlap=False):
    for idx in RESIDUAL_DIAG_CELLS:
        execute_cell(nb, idx, ns)
        save_open_figures(exp_dir, f"cell{idx}")
    if include_model_overlap:
        execute_cell(nb, MODEL_DIAG_CELL, ns)
        save_open_figures(exp_dir, f"cell{MODEL_DIAG_CELL}")


def run_measurement(nb, ns, exp, seed_offset=0):
    ns["FFT_K_RANGE_MAIN"] = tuple(exp["main_k"])
    ns["SBF_REGION_K_WINDOWS"] = [tuple(x) for x in exp["region_k_windows"]]
    reset_rng_counter(ns, seed=246810 + seed_offset)
    for idx in MEASURE_CELLS:
        execute_cell(nb, idx, ns)


def extract_zero_cross(df_profile, radius_type):
    if df_profile is None or len(df_profile) == 0:
        return np.nan
    sub = df_profile[df_profile["radius_type"].eq(radius_type)].sort_values("r_mid_px")
    vals = sub["resid_raw_median"].to_numpy(dtype=float)
    radii = sub["r_mid_px"].to_numpy(dtype=float)
    for i in range(len(vals) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if v0 == 0.0:
            return float(radii[i])
        if v0 * v1 < 0.0:
            frac = abs(v0) / (abs(v0) + abs(v1))
            return float(radii[i] + frac * (radii[i + 1] - radii[i]))
    return np.nan


def collect_metrics(ns, exp):
    main = copy.deepcopy(ns.get("sbf_main_diagnostics", {}))
    recommended = copy.deepcopy(ns.get("recommended_sbf"))
    resid_df = ns.get("df_resid_audit")
    bias_regions = ns.get("df_model_bias_regions")
    bias_profile = ns.get("df_model_bias_profile")

    resid_rows = {}
    if isinstance(resid_df, pd.DataFrame):
        for row in resid_df.to_dict(orient="records"):
            resid_rows[row["region"]] = row

    bias_rows = {}
    if isinstance(bias_regions, pd.DataFrame):
        for row in bias_regions.to_dict(orient="records"):
            bias_rows[row["region"]] = row

    out = {
        "experiment": exp["name"],
        "model_variant": exp["model_variant"],
        "main_k": tuple(exp["main_k"]),
        "region_k_windows": [tuple(x) for x in exp["region_k_windows"]],
        "elliptical_corrected_mbar": float(main.get("mbar_spec", np.nan)),
        "elliptical_raw_mbar": float(main.get("mbar_spec_raw", np.nan)),
        "elliptical_Pr_over_P0": float(main.get("Pr_over_P0", np.nan)),
        "weighted_corrected_mbar": float(recommended["mbar_weighted"]) if recommended else np.nan,
        "weighted_raw_mbar": float(recommended["mbar_weighted_raw"]) if recommended else np.nan,
        "formal_sigma": float(recommended["sigma_weighted_formal"]) if recommended else np.nan,
        "annulus_scatter": float(recommended["annulus_scatter"]) if recommended else np.nan,
        "adopted_sigma": float(recommended["sigma_adopted"]) if recommended else np.nan,
        "recommended_kmin": float(recommended["kmin"]) if recommended else np.nan,
        "recommended_kmax": float(recommended["kmax"]) if recommended else np.nan,
        "circular_inner_corrected_mbar": float(recommended["mbar_inner"]) if recommended else np.nan,
        "circular_outer_corrected_mbar": float(recommended["mbar_outer"]) if recommended else np.nan,
        "inner_Pr_over_P0": float(recommended["Pr_over_P0_inner"]) if recommended else np.nan,
        "outer_Pr_over_P0": float(recommended["Pr_over_P0_outer"]) if recommended else np.nan,
        "zero_cross_circle_px": extract_zero_cross(bias_profile, "circle"),
        "zero_cross_ellipse_px": extract_zero_cross(bias_profile, "ellipse"),
        "resid_abs_median_sum": float(
            np.nansum(
                [
                    abs(resid_rows.get("elliptical_chosen", {}).get("resid_median", np.nan)),
                    abs(resid_rows.get("circular_inner_lit", {}).get("resid_median", np.nan)),
                    abs(resid_rows.get("circular_outer_lit", {}).get("resid_median", np.nan)),
                ]
            )
        ),
        "residual_regions": resid_rows,
        "bias_regions": bias_rows,
    }
    return out


def compare_residual_quality(current_metrics, reference_metrics=None):
    if reference_metrics is None:
        return "reference residual state", "baseline"
    cur = current_metrics.get("resid_abs_median_sum", np.nan)
    ref = reference_metrics.get("resid_abs_median_sum", np.nan)
    if not (np.isfinite(cur) and np.isfinite(ref) and ref > 0):
        return "residual comparison unavailable", "unknown"
    ratio = cur / ref
    if ratio < 0.8:
        return "large-scale residual bias weaker; residual more noise-like", "improved"
    if ratio > 1.2:
        return "large-scale residual bias stronger; residual less noise-like", "worse"
    return "large-scale residual bias similar", "similar"


def decide_k_window_keep(metrics, baseline_metrics):
    if baseline_metrics is None:
        return "keep", "baseline reference"
    # Residual is identical for pure k-window experiments; decide on annulus consistency.
    cur = metrics.get("adopted_sigma", np.nan)
    base = baseline_metrics.get("adopted_sigma", np.nan)
    if np.isfinite(cur) and np.isfinite(base):
        if cur < base * 0.95:
            return "keep", "same residual image as baseline; annulus consistency improved"
        if cur > base * 1.10:
            return "reject", "same residual image as baseline; annulus consistency worsened"
    return "keep", "same residual image as baseline; no clear degradation"


def decide_model_keep(metrics, baseline_metrics):
    residual_note, residual_state = compare_residual_quality(metrics, baseline_metrics)
    cur_sigma = metrics.get("adopted_sigma", np.nan)
    base_sigma = baseline_metrics.get("adopted_sigma", np.nan) if baseline_metrics else np.nan
    if residual_state == "improved":
        if np.isfinite(cur_sigma) and np.isfinite(base_sigma) and cur_sigma > 1.25 * base_sigma:
            return "reject", residual_note + "; annulus scatter worsened too much"
        return "keep", residual_note
    if residual_state == "worse":
        return "reject", residual_note
    return "keep", residual_note


def metrics_to_markdown(metrics, decision, rationale):
    resid = metrics["residual_regions"]
    lines = [
        f"### {metrics['experiment']}",
        "",
        f"- Timestamp: {timestamp()}",
        f"- Model variant: `{metrics['model_variant']}`",
        f"- Main k-window: `{metrics['main_k']}`",
        f"- Region k-windows: `{metrics['region_k_windows']}`",
        f"- Decision: `{decision}`",
        f"- Why: {rationale}",
        "",
        "`Science numbers`",
        f"- Elliptical corrected mbar: `{metrics['elliptical_corrected_mbar']:.4f}`",
        f"- Circular inner corrected mbar: `{metrics['circular_inner_corrected_mbar']:.4f}`",
        f"- Circular outer corrected mbar: `{metrics['circular_outer_corrected_mbar']:.4f}`",
        f"- Weighted corrected mbar: `{metrics['weighted_corrected_mbar']:.4f}`",
        f"- Formal sigma: `{metrics['formal_sigma']:.4f}`",
        f"- Annulus scatter: `{metrics['annulus_scatter']:.4f}`",
        f"- Adopted sigma: `{metrics['adopted_sigma']:.4f}`",
        "",
        "`Residual diagnostics`",
    ]
    for region in ["elliptical_chosen", "circular_inner_lit", "circular_outer_lit"]:
        row = resid.get(region, {})
        med = row.get("resid_median", np.nan)
        mean = row.get("resid_mean", np.nan)
        sign = "positive" if np.isfinite(med) and med > 0 else "negative" if np.isfinite(med) and med < 0 else "zero/unknown"
        lines.append(
            f"- {region}: median=`{med:.4e}`, mean=`{mean:.4e}`, sign=`{sign}`"
        )
    lines += [
        f"- Circle zero-cross px: `{metrics['zero_cross_circle_px']:.2f}`",
        f"- Ellipse zero-cross px: `{metrics['zero_cross_ellipse_px']:.2f}`",
        "",
    ]
    return "\n".join(lines)


def write_log_header():
    if LOG_PATH.exists():
        return
    LOG_PATH.write_text(
        "# Codex Night Run Log\n\n"
        f"Started: {timestamp()}\n\n"
        "All experiments below were run in a single-thread / single-process mode from the project root.\n\n"
    )


def append_log(section_text):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(section_text)
        f.write("\n\n")


def save_metrics_json(exp_dir, metrics):
    out = exp_dir / "metrics_summary.json"
    clean = copy.deepcopy(metrics)
    for key in ["residual_regions", "bias_regions"]:
        if isinstance(clean.get(key), dict):
            clean[key] = clean[key]
    out.write_text(json.dumps(clean, indent=2, ensure_ascii=False))


def summarize_to_csv(all_metrics):
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
            "resid_abs_median_sum": m["resid_abs_median_sum"],
            "zero_cross_circle_px": m["zero_cross_circle_px"],
            "zero_cross_ellipse_px": m["zero_cross_ellipse_px"],
        })
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_ROOT / "night_experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def run_baseline_and_k_experiments():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_log_header()

    base_dir = OUTPUT_ROOT / "shared_current_model"
    nb, ns = setup_namespace(base_dir, model_variant="baseline")
    run_residual_diagnostics(nb, ns, base_dir, include_model_overlap=True)

    all_metrics = []
    baseline_metrics = None
    for i, exp in enumerate(EXPERIMENTS):
        print(f"[RUN] {exp['name']} :: main_k={exp['main_k']}")
        run_measurement(nb, ns, exp, seed_offset=i * 1000)
        metrics = collect_metrics(ns, exp)
        decision, rationale = decide_k_window_keep(metrics, baseline_metrics)
        if baseline_metrics is None:
            baseline_metrics = copy.deepcopy(metrics)
        exp_dir = OUTPUT_ROOT / exp["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_json(exp_dir, metrics)
        append_log(metrics_to_markdown(metrics, decision, rationale))
        all_metrics.append(metrics)
    return all_metrics, baseline_metrics


def choose_best_k_metrics(all_metrics):
    # Residuals are the same for pure k-window experiments; choose smallest adopted sigma.
    viable = [m for m in all_metrics if np.isfinite(m.get("adopted_sigma", np.nan))]
    if not viable:
        return all_metrics[0] if all_metrics else None
    viable.sort(key=lambda m: (m["adopted_sigma"], m["annulus_scatter"]))
    return viable[0]


def run_model_variant(best_k_metrics, baseline_metrics):
    exp = {
        "name": "exp06_fitted_splice_best_k",
        "kind": "model_variant",
        "model_variant": "fitted_splice",
        "main_k": tuple(best_k_metrics["main_k"]),
        "region_k_windows": [tuple(x) for x in best_k_metrics["region_k_windows"]],
        "why": "Small model tweak: use exact fitted-model values on fitted support, keep outer extrapolation only outside fitted range.",
    }
    exp_dir = OUTPUT_ROOT / exp["name"]
    nb, ns = setup_namespace(exp_dir, model_variant="baseline")
    apply_fitted_splice_variant(ns)
    for idx in [32, 36, 38, 40, 50, 51, 55]:
        execute_cell(nb, idx, ns)
    run_residual_diagnostics(nb, ns, exp_dir, include_model_overlap=True)
    run_measurement(nb, ns, exp, seed_offset=6000)
    metrics = collect_metrics(ns, exp)
    decision, rationale = decide_model_keep(metrics, baseline_metrics)
    save_metrics_json(exp_dir, metrics)
    append_log(metrics_to_markdown(metrics, decision, rationale))
    return metrics


def main():
    os.chdir(PROJECT_ROOT)
    all_metrics, baseline_metrics = run_baseline_and_k_experiments()
    best_k = choose_best_k_metrics(all_metrics)
    if best_k is not None:
        model_metrics = run_model_variant(best_k, baseline_metrics)
        all_metrics.append(model_metrics)
    csv_path = summarize_to_csv(all_metrics)
    print(f"[DONE] summary csv -> {csv_path}")
    print(f"[DONE] log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
