from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "sbf2_normalized_winsor" / "batch" / "aggregates"
MASTER_CSV = (
    PROJECT_ROOT
    / "runs"
    / "sbf2_go3055"
    / "analysis"
    / "tables"
    / "go3055_master_measurements.csv"
)
BRANCH_CSV = EXPERIMENT_DIR / "all_galaxies_combined_annuli.csv"

TRGB_COMMON_ZEROPOINT_MAG = 0.047
N_BOOTSTRAP = 300
RNG_SEED = 3055

BRANCH_LABELS = {
    "no_winsor": "Без винзорирования",
    "raw_global_3p5": "До нормировки, 3.5 sigma",
    "normalized_full_3p5": "После нормировки, вся область, 3.5 sigma",
    "normalized_union_3p5": "После нормировки, два кольца, 3.5 sigma",
}


def fit_constant(frame):
    y = frame["Mbar_F150W"].to_numpy(float)
    sy = frame["sigma_Mbar_internal"].to_numpy(float)

    def negative_log_likelihood(parameters):
        zero_point, log_scatter = parameters
        variance = sy**2 + np.exp(2 * log_scatter)
        return 0.5 * np.sum(
            np.log(2 * np.pi * variance) + (y - zero_point) ** 2 / variance
        )

    solution = minimize(
        negative_log_likelihood,
        [np.median(y), np.log(0.05)],
        method="L-BFGS-B",
        bounds=[(None, None), (np.log(1e-5), np.log(1.0))],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.x[0]), float(np.exp(solution.x[1]))


def bootstrap_zero_point_sigma(training, seed):
    rng = np.random.default_rng(seed)
    zero_points = []
    for _ in range(N_BOOTSTRAP):
        draw = training.iloc[rng.integers(0, len(training), len(training))]
        zero_points.append(fit_constant(draw)[0])
    return float(np.std(zero_points, ddof=1))


master = pd.read_csv(MASTER_CSV)
branches = pd.read_csv(BRANCH_CSV)
branches = branches[np.isclose(branches["requested_kmin"], 0.04)].copy()

required_branches = set(BRANCH_LABELS)
found_branches = set(branches["branch"])
if found_branches != required_branches:
    raise RuntimeError(
        f"Набор версий не совпал: ожидались {sorted(required_branches)}, "
        f"найдены {sorted(found_branches)}"
    )
if branches.groupby("branch")["galaxy"].nunique().ne(14).any():
    raise RuntimeError("Не во всех версиях найдено ровно 14 галактик")

reference_columns = [
    "galaxy",
    "environment",
    "A_F150W_mag",
    "sigma_A_F150W_mag",
    "mu_lit",
    "sigma_mu_lit",
    "sigma_mu_without_reddening_mag",
    "sigma_reddening_color_mag",
    "sigma_Pr_mag",
    "sigma_sky_mag",
]

rows = []
fit_rows = []
for branch_number, (branch, label) in enumerate(BRANCH_LABELS.items()):
    frame = branches.loc[branches["branch"].eq(branch)].merge(
        master[reference_columns], on="galaxy", validate="one_to_one"
    )
    frame["mbar_F150W"] = frame["mbar_weighted"] - frame["A_F150W_mag"]

    # sigma_weighted_formal already contains the fit covariance and the
    # scatter among PSF realizations.  The remaining terms follow the article.
    frame["sigma_mbar_internal"] = np.sqrt(
        frame["sigma_weighted_formal"] ** 2
        + frame["sigma_Pr_mag"] ** 2
        + frame["sigma_sky_mag"] ** 2
    )
    frame["Mbar_F150W"] = frame["mbar_F150W"] - frame["mu_lit"]
    frame["sigma_Mbar_internal"] = np.sqrt(
        frame["sigma_mbar_internal"] ** 2
        + frame["sigma_mu_without_reddening_mag"] ** 2
        + frame["sigma_reddening_color_mag"] ** 2
    )

    for target_index, target in frame.iterrows():
        training = frame.drop(index=target_index)
        zero_point, intrinsic_scatter = fit_constant(training)
        calibration_sigma = bootstrap_zero_point_sigma(
            training,
            RNG_SEED + 1000 * branch_number + sum(map(ord, target["galaxy"])),
        )

        target_sbf_sigma = np.hypot(
            target["sigma_mbar_internal"], target["sigma_A_F150W_mag"]
        )
        mu_internal_sigma = np.sqrt(
            target_sbf_sigma**2 + intrinsic_scatter**2 + calibration_sigma**2
        )
        mu_absolute_sigma = np.hypot(
            mu_internal_sigma, TRGB_COMMON_ZEROPOINT_MAG
        )
        mu_sbf = target["mbar_F150W"] - zero_point
        distance_sbf = 10 ** ((mu_sbf - 25) / 5)
        distance_trgb = 10 ** ((target["mu_lit"] - 25) / 5)
        conversion = np.log(10) / 5

        rows.append(
            {
                "galaxy": target["galaxy"],
                "environment": target["environment"],
                "branch": branch,
                "branch_label": label,
                "mu_sbf_mag": mu_sbf,
                "sigma_mu_internal_mag": mu_internal_sigma,
                "sigma_mu_absolute_mag": mu_absolute_sigma,
                "distance_sbf_mpc": distance_sbf,
                "sigma_distance_internal_mpc": (
                    conversion * distance_sbf * mu_internal_sigma
                ),
                "sigma_distance_absolute_mpc": (
                    conversion * distance_sbf * mu_absolute_sigma
                ),
                "mu_trgb_mag": target["mu_lit"],
                "sigma_mu_trgb_mag": target["sigma_mu_lit"],
                "distance_trgb_mpc": distance_trgb,
                "sigma_distance_trgb_mpc": (
                    conversion * distance_trgb * target["sigma_mu_lit"]
                ),
                "delta_distance_sbf_minus_trgb_mpc": distance_sbf - distance_trgb,
                "zero_point_loo_mag": zero_point,
                "intrinsic_scatter_loo_mag": intrinsic_scatter,
                "calibration_sigma_loo_mag": calibration_sigma,
                "target_sbf_sigma_mag": target_sbf_sigma,
            }
        )

    full_zero_point, full_intrinsic_scatter = fit_constant(frame)
    fit_rows.append(
        {
            "branch": branch,
            "branch_label": label,
            "zero_point_all_14_mag": full_zero_point,
            "intrinsic_scatter_all_14_mag": full_intrinsic_scatter,
        }
    )

long_table = pd.DataFrame(rows).sort_values(["galaxy", "branch"])
long_table.to_csv(
    EXPERIMENT_DIR / "go3055_distances_mpc_by_winsor_version_detailed.csv",
    index=False,
)

compact = (
    long_table.pivot(
        index=[
            "galaxy",
            "environment",
            "distance_trgb_mpc",
            "sigma_distance_trgb_mpc",
        ],
        columns="branch_label",
        values=[
            "distance_sbf_mpc",
            "sigma_distance_internal_mpc",
            "sigma_distance_absolute_mpc",
        ],
    )
    .reset_index()
)
compact.columns = [
    "__".join(part for part in column if part)
    if isinstance(column, tuple)
    else column
    for column in compact.columns
]
compact.to_csv(
    EXPERIMENT_DIR / "go3055_distances_mpc_by_winsor_version.csv", index=False
)

readable = compact[[
    "galaxy",
    "environment",
    "distance_trgb_mpc",
    "sigma_distance_trgb_mpc",
]].copy()
readable["TRGB Paper IV, Мпк"] = [
    f"{distance:.2f} ± {sigma:.2f}"
    for distance, sigma in zip(
        readable.pop("distance_trgb_mpc"), readable.pop("sigma_distance_trgb_mpc")
    )
]
for label in BRANCH_LABELS.values():
    distance = compact[f"distance_sbf_mpc__{label}"]
    sigma = compact[f"sigma_distance_internal_mpc__{label}"]
    readable[label] = [
        f"{value:.2f} ± {error:.2f}" for value, error in zip(distance, sigma)
    ]
readable = readable.rename(
    columns={"galaxy": "Галактика", "environment": "Скопление"}
)
readable.to_csv(
    EXPERIMENT_DIR / "go3055_distances_mpc_by_winsor_version_readable.csv",
    index=False,
)

summary = pd.DataFrame(fit_rows)
for branch, group in long_table.groupby("branch"):
    use = summary["branch"].eq(branch)
    summary.loc[use, "loo_rmse_vs_trgb_mpc"] = np.sqrt(
        np.mean(group["delta_distance_sbf_minus_trgb_mpc"] ** 2)
    )
    summary.loc[use, "loo_median_abs_vs_trgb_mpc"] = group[
        "delta_distance_sbf_minus_trgb_mpc"
    ].abs().median()
    summary.loc[use, "median_internal_sigma_distance_mpc"] = group[
        "sigma_distance_internal_mpc"
    ].median()
    summary.loc[use, "median_absolute_sigma_distance_mpc"] = group[
        "sigma_distance_absolute_mpc"
    ].median()
    combined_sigma = np.hypot(
        group["sigma_distance_internal_mpc"], group["sigma_distance_trgb_mpc"]
    )
    summary.loc[use, "within_combined_1sigma_count"] = (
        group["delta_distance_sbf_minus_trgb_mpc"].abs() <= combined_sigma
    ).sum()

summary.to_csv(
    EXPERIMENT_DIR / "go3055_distances_mpc_by_winsor_version_summary.csv",
    index=False,
)

print(readable.to_string(index=False))
print("\nСводка по версиям:")
print(summary.to_string(index=False))
