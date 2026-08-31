"""Проверка восстановления известного SBF-сигнала для четырёх ветвей.

Тест изолирует только порядок винзорирования перед FFT. Он не моделирует
ошибки изофот, фона, каталога компактных источников или поправки P_r.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.fft import fft2

from sbf2_normalized_winsor_core import (
    radial_mean_sem,
    radial_plan,
    weighted_fit,
)


BRANCHES = (
    "no_winsor",
    "raw_global_3p5",
    "normalized_full_3p5",
    "normalized_union_3p5",
)
BRANCH_LABELS = {
    "no_winsor": "No winsorization",
    "raw_global_3p5": "Raw residual, 3.5 sigma",
    "normalized_full_3p5": "Normalized full support, 3.5 sigma",
    "normalized_union_3p5": "Normalized annuli, 3.5 sigma",
}


def _limits(values: np.ndarray, sigma: float = 3.5) -> dict[str, float]:
    _, median, scale = sigma_clipped_stats(values, sigma=sigma, maxiters=5)
    median, scale = float(median), float(scale)
    return {
        "lower": median - sigma * scale,
        "upper": median + sigma * scale,
    }


def _padded_psf_fft(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if psf.shape[0] > shape[0] or psf.shape[1] > shape[1]:
        raise ValueError("PSF больше синтетического FFT-кадра")
    padded = np.zeros(shape, dtype=float)
    y0 = shape[0] // 2 - psf.shape[0] // 2
    x0 = shape[1] // 2 - psf.shape[1] // 2
    padded[y0:y0 + psf.shape[0], x0:x0 + psf.shape[1]] = psf
    return fft2(padded)


def _expectation(
    window: np.ndarray,
    psf_filter: np.ndarray,
    plan: dict,
    realizations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_use = int(window.sum())
    accumulated_power = np.zeros(window.shape, dtype=float)
    for _ in range(realizations):
        white = rng.normal(size=window.shape)
        field = np.fft.ifft2(fft2(white) * psf_filter).real
        fft_input = np.zeros_like(field)
        fft_input[window] = field[window] - np.mean(field[window])
        accumulated_power += np.abs(fft2(fft_input)) ** 2 / n_use
    profile, _, _ = radial_mean_sem(
        accumulated_power / realizations, plan, min_count=10
    )
    return profile


def _measure(
    values: np.ndarray,
    window: np.ndarray,
    expectation: np.ndarray,
    plan: dict,
) -> tuple[float, float]:
    fft_input = np.zeros_like(values)
    fft_input[window] = values[window] - np.mean(values[window])
    power = np.abs(fft2(fft_input)) ** 2 / int(window.sum())
    spectrum, spectrum_error, _ = radial_mean_sem(power, plan, min_count=10)
    use = (
        (plan["k"] >= 0.04)
        & (plan["k"] <= 0.25)
        & np.isfinite(spectrum)
        & np.isfinite(spectrum_error)
        & (spectrum_error > 0)
        & np.isfinite(expectation)
    )
    fit = weighted_fit(spectrum[use], spectrum_error[use], expectation[use])
    return fit["P0"], fit["P1"]


def _find_reference_psf(project_root: Path) -> Path:
    paths = sorted(
        (project_root / "runs" / "sbf2_go3055" / "products" / "NGC_3379")
        .glob("**/*_psf_129.fits")
    )
    if not paths:
        raise FileNotFoundError("Не найдена сохранённая PSF NGC 3379")
    return paths[-1]


def run_recovery_test(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    trials: int = 64,
    expectation_realizations: int = 96,
    seed: int = 3055,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Запускает тест и возвращает реализации, сводку и путь к рисунку."""

    project_root = Path(project_root).resolve()
    output_dir = Path(
        output_dir
        or project_root / "runs" / "sbf2_normalized_winsor" / "recovery"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    size = 512
    p0_true, p1_true = 0.90, 0.08
    yy, xx = np.indices((size, size))
    radius = np.hypot(yy - size / 2, xx - size / 2)
    model = 150 + 1800 * np.exp(-(radius / 115) ** 0.55)
    full_support = (radius >= 40) & (radius <= 230)
    inner = full_support & (radius >= 70) & (radius <= 125)
    outer = full_support & (radius >= 140) & (radius <= 205)

    mask_rng = np.random.default_rng(seed)
    for _ in range(35):
        hole_x, hole_y = mask_rng.uniform(70, 442, 2)
        hole = np.hypot(xx - hole_x, yy - hole_y) <= mask_rng.uniform(2.5, 5.0)
        full_support &= ~hole
        inner &= ~hole
        outer &= ~hole
    union = inner | outer

    psf_path = _find_reference_psf(project_root)
    with fits.open(psf_path, memmap=False) as hdul:
        psf = np.asarray(hdul[1].data, dtype=float)
    psf /= psf.sum()
    psf_filter = _padded_psf_fft(psf, model.shape)

    plans = {name: radial_plan(model.shape, 80) for name in ("inner", "outer")}
    windows = {"inner": inner, "outer": outer}
    expectations = {
        name: _expectation(
            window,
            psf_filter,
            plans[name],
            expectation_realizations,
            seed + 100 + index,
        )
        for index, (name, window) in enumerate(windows.items())
    }

    rng = np.random.default_rng(seed + 1000)
    rows = []
    for trial in range(trials):
        sbf_white = rng.normal(size=model.shape)
        sbf_field = np.fft.ifft2(fft2(sbf_white) * psf_filter).real
        white_noise = rng.normal(size=model.shape)
        normalized = (
            np.sqrt(p0_true) * sbf_field
            + np.sqrt(p1_true) * white_noise
        )
        raw = np.sqrt(model) * normalized

        raw_limits = _limits(raw[full_support])
        normalized_full_limits = _limits(normalized[full_support])
        normalized_union_limits = _limits(normalized[union])

        for branch in BRANCHES:
            if branch == "no_winsor":
                working = normalized
                changed = np.zeros_like(full_support)
            elif branch == "raw_global_3p5":
                working = np.clip(
                    raw, raw_limits["lower"], raw_limits["upper"]
                ) / np.sqrt(model)
                changed = (
                    (raw < raw_limits["lower"])
                    | (raw > raw_limits["upper"])
                )
            elif branch == "normalized_full_3p5":
                working = np.clip(
                    normalized,
                    normalized_full_limits["lower"],
                    normalized_full_limits["upper"],
                )
                changed = (
                    (normalized < normalized_full_limits["lower"])
                    | (normalized > normalized_full_limits["upper"])
                )
            else:
                working = np.clip(
                    normalized,
                    normalized_union_limits["lower"],
                    normalized_union_limits["upper"],
                )
                changed = (
                    (normalized < normalized_union_limits["lower"])
                    | (normalized > normalized_union_limits["upper"])
                )

            for ring, window in windows.items():
                p0_recovered, p1_recovered = _measure(
                    working, window, expectations[ring], plans[ring]
                )
                rows.append(
                    {
                        "trial": trial,
                        "ring": ring,
                        "branch": branch,
                        "P0_true": p0_true,
                        "P0_recovered": p0_recovered,
                        "P1_true": p1_true,
                        "P1_recovered": p1_recovered,
                        "P0_fractional_error": p0_recovered / p0_true - 1,
                        "mbar_error_mag": -2.5 * np.log10(
                            p0_recovered / p0_true
                        ),
                        "changed_fraction": changed[window].mean(),
                    }
                )

    results = pd.DataFrame(rows)
    baseline = results.loc[
        results["branch"].eq("no_winsor"),
        ["trial", "ring", "mbar_error_mag"],
    ].rename(columns={"mbar_error_mag": "mbar_error_no_winsor_mag"})
    results = results.merge(
        baseline, on=["trial", "ring"], validate="many_to_one"
    )
    results["delta_mbar_vs_no_winsor_mag"] = (
        results["mbar_error_mag"] - results["mbar_error_no_winsor_mag"]
    )
    results.to_csv(output_dir / "synthetic_branch_recovery_trials.csv", index=False)

    summary = (
        results.groupby(["branch", "ring"], sort=False)
        .agg(
            trials=("trial", "size"),
            mean_P0_fractional_error=("P0_fractional_error", "mean"),
            rms_P0_fractional_error=(
                "P0_fractional_error",
                lambda values: np.sqrt(np.mean(values**2)),
            ),
            mean_mbar_error_mag=("mbar_error_mag", "mean"),
            se_mean_mbar_error_mag=(
                "mbar_error_mag",
                lambda values: values.std(ddof=1) / np.sqrt(len(values)),
            ),
            median_mbar_error_mag=("mbar_error_mag", "median"),
            rms_mbar_error_mag=(
                "mbar_error_mag",
                lambda values: np.sqrt(np.mean(values**2)),
            ),
            mean_P1_recovered=("P1_recovered", "mean"),
            mean_changed_fraction=("changed_fraction", "mean"),
            mean_delta_mbar_vs_no_winsor_mag=(
                "delta_mbar_vs_no_winsor_mag", "mean"
            ),
            se_delta_mbar_vs_no_winsor_mag=(
                "delta_mbar_vs_no_winsor_mag",
                lambda values: values.std(ddof=1) / np.sqrt(len(values)),
            ),
        )
        .reset_index()
    )
    summary["branch_label"] = summary["branch"].map(BRANCH_LABELS)
    summary["reference_psf"] = str(psf_path)
    summary.to_csv(output_dir / "synthetic_branch_recovery_summary.csv", index=False)

    figure_path = output_dir / "synthetic_branch_recovery.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    positions = np.arange(len(BRANCHES))
    colors = {"inner": "#2563eb", "outer": "#dc2626"}
    offsets = {"inner": -0.13, "outer": 0.13}
    for ring in ("inner", "outer"):
        groups = [
            results.loc[
                results["branch"].eq(branch) & results["ring"].eq(ring),
                "P0_recovered",
            ].to_numpy()
            for branch in BRANCHES
        ]
        plot = axes[0].boxplot(
            groups,
            positions=positions + offsets[ring],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
        )
        for box in plot["boxes"]:
            box.set(facecolor=colors[ring], alpha=0.55)
        for item in plot["medians"]:
            item.set(color="black")
        axes[0].plot([], [], color=colors[ring], lw=7, alpha=0.55, label=ring)
    axes[0].axhline(p0_true, color="black", ls="--", lw=1.3, label="input $P_0$")
    axes[0].set(ylabel=r"Recovered $P_0$", title="Known-amplitude recovery")
    axes[0].legend(frameon=False)

    for ring in ("inner", "outer"):
        ring_summary = summary[summary["ring"].eq(ring)].set_index("branch")
        axes[1].errorbar(
            positions,
            [
                ring_summary.loc[branch, "mean_delta_mbar_vs_no_winsor_mag"]
                for branch in BRANCHES
            ],
            yerr=[
                ring_summary.loc[branch, "se_delta_mbar_vs_no_winsor_mag"]
                for branch in BRANCHES
            ],
            fmt="o-",
            color=colors[ring],
            capsize=3,
            label=ring,
        )
    axes[1].axhline(0, color="black", ls="--", lw=1.3)
    axes[1].set(
        ylabel=r"Additional $\Delta\overline{m}$ relative to no clipping (mag)",
        title="Bias introduced by winsorization",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(
            positions,
            ["none", "raw", "norm. full", "norm. annuli"],
            rotation=18,
        )
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    return results, summary, figure_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    _, final_summary, final_figure = run_recovery_test(root)
    print(final_summary.to_string(index=False))
    print(f"Рисунок: {final_figure}")
