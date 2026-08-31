"""Спектральный тест порядка винзорирования для готовых продуктов SBF-2.

Моделирование галактики, маска источников, кольца, PSF и поправка ``P_r``
берутся из завершённого production-прогона ``sbf-2``.  Здесь меняется только
порядок операций перед FFT:

``no_winsor``
    Остатки сразу делятся на ``sqrt(model)``.
``raw_global_3p5``
    Текущая production-ветвь: ограничение сырых остатков, затем нормировка.
``normalized_full_3p5``
    Чистая проверка порядка: нормировка раньше ограничения, но статистика порога
    всё ещё берётся по всей положительной области модели.
``normalized_union_<sigma>``
    Проверяемая ветвь: сначала нормировка, затем один общий порог по объединению
    двух рабочих колец.

Модуль намеренно ничего не делает при импорте.  Его используют отдельный
ноутбук и отдельный последовательный runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.fft import fft2, fftfreq, set_workers


# Bump whenever a numerical operation or the mandatory FITS contract changes.
# It is part of every result/cache key and prevents stale science products.
EXPERIMENT_VERSION = "sbf2-normalized-winsor-v2"
RAW_PRODUCTION_SIGMA = 3.5
RAW_PRODUCTION_MAXITERS = 5
REGION_NAMES = {
    "inner": "circular_inner_lit",
    "outer": "circular_outer_lit",
}
TARGET_STATUS_COLUMNS = [
    "galaxy", "status", "stage", "attempt", "started_at", "finished_at",
    "message", "result_path", "pipeline_version", "sigma", "kmins",
    "kmax", "e_realizations", "random_seed",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Все параметры, способные изменить численный результат эксперимента."""

    normalized_sigma: float = 3.5
    kmins: tuple[float, ...] = (0.01, 0.03, 0.04)
    kmax: float = 0.25
    k_bins: int = 80
    e_realizations: int = 64
    random_seed: int = 1489
    fft_workers: int = -1
    min_modes_per_bin: int = 10
    save_ring_fft_fits: bool = False
    save_all_branch_fits: bool = False

    @property
    def candidate_branch(self) -> str:
        return f"normalized_union_{sigma_tag(self.normalized_sigma)}"

    @property
    def branches(self) -> tuple[str, str, str, str]:
        return (
            "no_winsor",
            "raw_global_3p5",
            "normalized_full_3p5",
            self.candidate_branch,
        )


def sigma_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def experiment_config_key(config: ExperimentConfig) -> str:
    return _stable_hash({
        "version": EXPERIMENT_VERSION,
        "config": asdict(config),
    })


def find_project_root(start: str | Path | None = None) -> Path:
    """Находит корень проекта при запуске из корня или из ``code``."""

    root = Path.cwd() if start is None else Path(start)
    root = root.resolve()
    if root.name == "code":
        root = root.parent
    if not (root / "code" / "sbf-2.ipynb").is_file():
        raise FileNotFoundError(
            "Не найден корень проекта: ожидается code/sbf-2.ipynb"
        )
    return root


def galaxy_slug(galaxy: str) -> str:
    return galaxy.strip().replace(" ", "_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stable_hash(payload: Any) -> str:
    text = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path, hash_small: bool = False) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_small and stat.st_size <= 16 * 1024 * 1024:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(payload), stream, ensure_ascii=False,
                indent=2, sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(handle)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def target_status_path(output_root: str | Path) -> Path:
    return Path(output_root).resolve() / "batch" / "target_status.csv"


def _status_config(config: ExperimentConfig) -> dict[str, str]:
    return {
        "pipeline_version": EXPERIMENT_VERSION,
        "sigma": f"{config.normalized_sigma:g}",
        "kmins": ";".join(f"{value:g}" for value in config.kmins),
        "kmax": f"{config.kmax:g}",
        "e_realizations": str(config.e_realizations),
        "random_seed": str(config.random_seed),
    }


def _read_target_status(output_root: str | Path) -> pd.DataFrame:
    path = target_status_path(output_root)
    if not path.is_file():
        return pd.DataFrame(columns=TARGET_STATUS_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in TARGET_STATUS_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[TARGET_STATUS_COLUMNS].copy()


def _result_config_matches(
    result: dict[str, Any], config: ExperimentConfig
) -> bool:
    return (
        result.get("status") == "ok"
        and result.get("version") == EXPERIMENT_VERSION
        and result.get("config") == _jsonable(asdict(config))
    )


def load_completed_result(
    galaxy: str, output_root: str | Path, config: ExperimentConfig,
    source_batch_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Loads a completed target by its stable human-readable filename.

    No hash is used as a completion marker.  The latest result JSON provides
    the numerical metadata, while ``target_status.csv`` remains the readable
    campaign ledger.
    """

    path = (
        Path(output_root).resolve() / "batch"
        / f"{galaxy_slug(galaxy)}_result.json"
    )
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _result_config_matches(result, config):
        return None
    if source_batch_root is not None:
        try:
            current_source = inspect_source(galaxy, source_batch_root)
        except Exception:
            return None
        if result.get("source_key") != current_source.get("source_key"):
            return None
    if not _result_artifacts_valid(result):
        return None
    result["result_cache_hit"] = True
    # Never leak a legacy hashed result directory into the human ledger.
    result["result_path"] = str(path)
    return result


def prepare_target_status(
    galaxies: Iterable[str], output_root: str | Path,
    config: ExperimentConfig,
    source_batch_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Creates/repairs the readable status table before a campaign starts."""

    galaxies = list(galaxies)
    frame = _read_target_status(output_root)
    existing = {
        row["galaxy"]: row.to_dict()
        for _, row in frame.iterrows() if row.get("galaxy")
    }
    signature = _status_config(config)
    completed: dict[str, dict[str, Any]] = {}
    rows = []
    for galaxy in galaxies:
        old = existing.get(galaxy, {})
        same_settings = all(
            str(old.get(name, "")) == value
            for name, value in signature.items()
        )
        result = load_completed_result(
            galaxy, output_root, config, source_batch_root
        )
        if result is not None:
            completed[galaxy] = result
            status = "ok"
            stage = "complete"
            message = "completed product found"
            result_path = str(result.get("result_path", ""))
            finished_at = str(old.get("finished_at", ""))
        else:
            previous = str(old.get("status", "")) if same_settings else ""
            status = "interrupted" if previous == "running" else (
                previous if previous in {"failed", "interrupted"} else "pending"
            )
            stage = (
                "interrupted" if previous == "running"
                else str(old.get("stage", "waiting")) or "waiting"
            )
            message = (
                "previous run stopped before target completion"
                if previous == "running" else str(old.get("message", ""))
            )
            result_path = ""
            finished_at = str(old.get("finished_at", ""))
        rows.append({
            "galaxy": galaxy,
            "status": status,
            "stage": stage,
            "attempt": str(old.get("attempt", "0")) if same_settings else "0",
            "started_at": str(old.get("started_at", "")) if same_settings else "",
            "finished_at": finished_at if same_settings else "",
            "message": message,
            "result_path": result_path,
            **signature,
        })

    untouched = frame[~frame["galaxy"].isin(galaxies)]
    updated = pd.concat(
        [untouched, pd.DataFrame(rows)], ignore_index=True
    )[TARGET_STATUS_COLUMNS]
    _atomic_csv(target_status_path(output_root), updated)
    selected = updated[updated["galaxy"].isin(galaxies)].reset_index(drop=True)
    return selected, completed


def update_target_status(
    galaxy: str, status: str, output_root: str | Path,
    config: ExperimentConfig, stage: str = "", message: str = "",
    result_path: str | Path | None = None,
) -> Path:
    """Atomically updates one row in ``target_status.csv``."""

    allowed = {"pending", "running", "ok", "failed", "interrupted"}
    if status not in allowed:
        raise ValueError(f"Unknown target status: {status}")
    frame = _read_target_status(output_root)
    current = frame[frame["galaxy"].eq(galaxy)]
    old = current.iloc[-1].to_dict() if not current.empty else {}
    try:
        attempt = int(str(old.get("attempt", "0") or "0"))
    except ValueError:
        attempt = 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    started_at = str(old.get("started_at", ""))
    finished_at = str(old.get("finished_at", ""))
    if status == "running":
        if str(old.get("status", "")) != "running":
            attempt += 1
            started_at = now
        finished_at = ""
    elif status in {"ok", "failed", "interrupted"}:
        finished_at = now
    row = {
        "galaxy": galaxy,
        "status": status,
        "stage": stage,
        "attempt": str(attempt),
        "started_at": started_at,
        "finished_at": finished_at,
        "message": message,
        "result_path": "" if result_path is None else str(result_path),
        **_status_config(config),
    }
    frame = frame[~frame["galaxy"].eq(galaxy)]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    _atomic_csv(target_status_path(output_root), frame[TARGET_STATUS_COLUMNS])
    return target_status_path(output_root)


def _fits_readable(path: Path) -> bool:
    try:
        with fits.open(path, memmap=True) as hdul:
            return any(hdu.data is not None for hdu in hdul)
    except Exception:
        return False


def discover_galaxies(source_batch_root: str | Path) -> list[str]:
    """Возвращает успешные цели в порядке production-таблицы."""

    source_batch_root = Path(source_batch_root).resolve()
    table_path = source_batch_root / "sbf2_batch_results.csv"
    if table_path.is_file():
        frame = pd.read_csv(table_path)
        good = frame[frame["status"].eq("ok")]
        return good["galaxy"].drop_duplicates().tolist()

    galaxies = []
    for result_path in sorted(source_batch_root.glob("NGC_*_result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "ok":
            galaxies.append(str(result["galaxy"]))
    return galaxies


def inspect_source(galaxy: str, source_batch_root: str | Path) -> dict[str, Any]:
    """Проверяет и описывает read-only входы одного production-результата."""

    source_batch_root = Path(source_batch_root).resolve()
    result_path = source_batch_root / f"{galaxy_slug(galaxy)}_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(f"{galaxy}: production status={result.get('status')}")

    run_dir = Path(result["output_dir"]).resolve()
    stem = str(result["stem"])
    paths = {
        "signal": Path(result["signal_path"]).resolve(),
        "model": Path(result["model_full_fits"]).resolve(),
        "production_residual": Path(result["science_residual_fits"]).resolve(),
        "inner_ring": Path(result["inner_usable_residual_fits"]).resolve(),
        "outer_ring": Path(result["outer_usable_residual_fits"]).resolve(),
        "measurements": Path(result["df_sbf_csv"]).resolve(),
        "catalog_mask": run_dir / f"{stem}_sbf_catalog_mask_mcut.fits",
        "psf": run_dir / f"{stem}_psf_129.fits",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))

    fingerprints = {
        name: _file_fingerprint(
            path, hash_small=name in {"measurements", "psf"}
        )
        for name, path in paths.items()
    }
    source_payload = {
        "galaxy": galaxy,
        "job_id": result.get("job_id"),
        "background": float(result["signal_background_scalar"]),
        "paths": fingerprints,
    }
    return {
        "galaxy": galaxy,
        "result_path": str(result_path),
        "result": result,
        "run_dir": str(run_dir),
        "stem": stem,
        "paths": {name: str(path) for name, path in paths.items()},
        "fingerprints": fingerprints,
        "source_key": _stable_hash(source_payload),
    }


def inspect_sources(
    galaxies: Iterable[str], source_batch_root: str | Path
) -> pd.DataFrame:
    rows = []
    for galaxy in galaxies:
        try:
            source = inspect_source(galaxy, source_batch_root)
            rows.append({
                "galaxy": galaxy,
                "ready": True,
                "source_key": source["source_key"][:16],
                "message": "ok",
            })
        except Exception as error:
            rows.append({
                "galaxy": galaxy,
                "ready": False,
                "source_key": "",
                "message": str(error),
            })
    return pd.DataFrame(rows)


def _main_measurements(path: Path, config: ExperimentConfig) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame["region"].isin(REGION_NAMES.values())
        & frame["status"].eq("ok")
        & np.isclose(frame["kmax"], config.kmax)
    ].copy()
    if selected.empty:
        raise RuntimeError("Не найдены production-измерения круговых колец")

    for region in REGION_NAMES.values():
        region_rows = selected[selected["region"].eq(region)]
        if region_rows.empty:
            raise RuntimeError(f"Нет production-строк для {region}")
    return selected


def _production_row(frame: pd.DataFrame, ring: str, kmin: float) -> pd.Series:
    rows = frame[
        frame["region"].eq(REGION_NAMES[ring])
        & np.isclose(frame["kmin"], kmin)
    ]
    if rows.empty:
        raise RuntimeError(
            f"Нет production-строки {REGION_NAMES[ring]}, kmin={kmin}"
        )
    return rows.iloc[0]


def _build_compact_input_cache(
    source: dict[str, Any], cache_path: Path, config: ExperimentConfig
) -> None:
    """Один раз читает большие FITS и сохраняет только рабочие crops."""

    paths = {name: Path(path) for name, path in source["paths"].items()}
    result = source["result"]
    measurements = _main_measurements(paths["measurements"], config)
    geometry_rows = {
        ring: _production_row(measurements, ring, 0.04)
        for ring in REGION_NAMES
    }

    with fits.open(paths["signal"], memmap=True) as signal_hdul, \
            fits.open(paths["model"], memmap=True) as model_hdul, \
            fits.open(paths["catalog_mask"], memmap=True) as mask_hdul, \
            fits.open(paths["production_residual"], memmap=True) as production_hdul:
        science = signal_hdul["SCI"].data
        science_header = signal_hdul["SCI"].header
        model = model_hdul[0].data
        catalog_mask = mask_hdul[0].data
        production_residual = production_hdul[0].data
        background = float(result["signal_background_scalar"])

        if science.shape != model.shape or science.shape != catalog_mask.shape:
            raise RuntimeError(f"{source['galaxy']}: несовместимые размеры FITS")

        science_mask = (
            (~np.asarray(catalog_mask, dtype=bool))
            & np.isfinite(science)
            & np.isfinite(model)
            & (model > 0)
        )
        model_values = np.asarray(model[science_mask], dtype=np.float64)
        # Буквально повторяем systematics-loader: арифметика в float64,
        # затем окончательный residual в float32.
        raw_values = np.asarray(
            (np.asarray(science[science_mask], dtype=np.float64) - background)
            - model_values,
            dtype=np.float32,
        )
        _, raw_median, raw_scale = sigma_clipped_stats(
            raw_values,
            sigma=RAW_PRODUCTION_SIGMA,
            maxiters=RAW_PRODUCTION_MAXITERS,
        )
        raw_median = float(raw_median)
        raw_scale = float(raw_scale)
        raw_lower = raw_median - RAW_PRODUCTION_SIGMA * raw_scale
        raw_upper = raw_median + RAW_PRODUCTION_SIGMA * raw_scale

        normalized_full_values = raw_values / np.sqrt(model_values)
        _, normalized_full_median, normalized_full_scale = sigma_clipped_stats(
            normalized_full_values,
            sigma=RAW_PRODUCTION_SIGMA,
            maxiters=RAW_PRODUCTION_MAXITERS,
        )
        normalized_full_median = float(normalized_full_median)
        normalized_full_scale = float(normalized_full_scale)
        normalized_full_lower = (
            normalized_full_median
            - RAW_PRODUCTION_SIGMA * normalized_full_scale
        )
        normalized_full_upper = (
            normalized_full_median
            + RAW_PRODUCTION_SIGMA * normalized_full_scale
        )

        arrays: dict[str, Any] = {}
        ring_metadata: dict[str, Any] = {}
        for ring, row in geometry_rows.items():
            y0, y1 = int(row["fft_crop_y0"]), int(row["fft_crop_y1"])
            x0, x1 = int(row["fft_crop_x0"]), int(row["fft_crop_x1"])
            crop = (slice(y0, y1), slice(x0, x1))

            with fits.open(paths[f"{ring}_ring"], memmap=True) as ring_hdul:
                saved_ring = ring_hdul[0].data[crop]
                ring_window = np.isfinite(saved_ring)

            valid_crop = science_mask[crop]
            window = ring_window & valid_crop
            model_crop = np.asarray(model[crop], dtype=np.float32).copy()
            raw_crop = np.asarray(
                (np.asarray(science[crop], dtype=np.float64) - background)
                - np.asarray(model[crop], dtype=np.float64),
                dtype=np.float32,
            )
            raw_crop[~window] = np.nan
            model_crop[~window] = np.nan

            recreated_production = np.clip(
                raw_crop[window], raw_lower, raw_upper
            )
            saved_production = np.asarray(
                production_residual[crop][window], dtype=np.float32
            )
            production_difference = (
                recreated_production - saved_production
            )
            max_production_difference = float(
                np.max(np.abs(production_difference))
            )
            if max_production_difference > 1e-5:
                raise RuntimeError(
                    f"{source['galaxy']} {ring}: pixel closure старой ветви "
                    f"не пройден, max |Δ|={max_production_difference:.3e}"
                )

            expected_n = int(row["n_use"])
            if int(window.sum()) != expected_n:
                raise RuntimeError(
                    f"{source['galaxy']} {ring}: N={int(window.sum())}, "
                    f"production N={expected_n}"
                )

            arrays[f"{ring}_raw"] = raw_crop
            arrays[f"{ring}_model"] = model_crop
            arrays[f"{ring}_window"] = window.astype(np.uint8)
            ring_metadata[ring] = {
                "crop": [y0, y1, x0, x1],
                "shape": list(window.shape),
                "n_use": int(window.sum()),
                "mean_model": float(np.mean(model_crop[window])),
                "production_pixel_closure_max_abs": max_production_difference,
                "production_pixel_closure_median_abs": float(
                    np.median(np.abs(production_difference))
                ),
            }

        pixel_area = float(science_header["PIXAR_SR"]) / 2.350443e-11

    with fits.open(paths["psf"], memmap=True) as psf_hdul:
        psfs, psf_ids = [], []
        for index, hdu in enumerate(psf_hdul[1:]):
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue
            psf = np.asarray(hdu.data, dtype=np.float64).copy()
            total = float(np.sum(psf))
            if not np.isfinite(total) or total <= 0:
                continue
            psfs.append((psf / total).astype(np.float32))
            psf_ids.append(str(hdu.header.get("PSFID", hdu.name or index)))
    if not psfs:
        raise RuntimeError(f"{source['galaxy']}: в PSF FITS нет моделей")

    metadata = {
        "version": EXPERIMENT_VERSION,
        "galaxy": source["galaxy"],
        "source_key": source["source_key"],
        "source": {
            "result_path": source["result_path"],
            "job_id": result.get("job_id"),
            "paths": source["paths"],
            "fingerprints": source["fingerprints"],
        },
        "raw_global": {
            "sigma": RAW_PRODUCTION_SIGMA,
            "median": raw_median,
            "scale": raw_scale,
            "lower": float(raw_lower),
            "upper": float(raw_upper),
            "n_pixels": int(raw_values.size),
        },
        "normalized_full": {
            "sigma": RAW_PRODUCTION_SIGMA,
            "median": normalized_full_median,
            "scale": normalized_full_scale,
            "lower": float(normalized_full_lower),
            "upper": float(normalized_full_upper),
            "n_pixels": int(normalized_full_values.size),
        },
        "rings": ring_metadata,
        "pixel_area_arcsec2": pixel_area,
        "ab_zeropoint_per_pixel": float(
            -2.5 * np.log10((2.350443e-5 * pixel_area) / 3631.0)
        ),
        "production_rows": [
            _jsonable(row.to_dict()) for _, row in measurements.iterrows()
        ],
        "psf_ids": psf_ids,
    }
    arrays["psfs"] = np.stack(psfs)
    arrays["metadata_json"] = np.array(
        json.dumps(_jsonable(metadata), ensure_ascii=False, sort_keys=True)
    )
    _atomic_npz(cache_path, **arrays)


def _load_compact_cache(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        return {
            "metadata": metadata,
            "psfs": np.asarray(archive["psfs"], dtype=float),
            "rings": {
                ring: {
                    "raw": np.asarray(archive[f"{ring}_raw"], dtype=float),
                    "model": np.asarray(archive[f"{ring}_model"], dtype=float),
                    "window": np.asarray(
                        archive[f"{ring}_window"], dtype=bool
                    ),
                }
                for ring in REGION_NAMES
            },
        }


def load_or_build_compact_cache(
    source: dict[str, Any], cache_root: str | Path,
    config: ExperimentConfig, rebuild: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    cache_root = Path(cache_root).resolve()
    input_key = _stable_hash({
        "version": EXPERIMENT_VERSION,
        "source_key": source["source_key"],
        "production_kmax": config.kmax,
    })
    cache_path = (
        cache_root / "inputs"
        / f"{galaxy_slug(source['galaxy'])}_{input_key[:16]}.npz"
    )
    cache_hit = cache_path.is_file() and not rebuild
    if not cache_hit:
        print(f"[input cache] building {source['galaxy']} -> {cache_path.name}")
        _build_compact_input_cache(source, cache_path, config)
    else:
        print(f"[input cache] hit {source['galaxy']}: {cache_path.name}")
    data = _load_compact_cache(cache_path)
    if (
        data["metadata"].get("source_key") != source["source_key"]
        or data["metadata"].get("version") != EXPERIMENT_VERSION
    ):
        raise RuntimeError("Ключ компактного кэша не совпадает с входами")
    return data, cache_path, cache_hit


def radial_plan(shape: tuple[int, int], k_bins: int) -> dict[str, Any]:
    ky = fftfreq(shape[0])[:, None]
    kx = fftfreq(shape[1])[None, :]
    radius = np.hypot(kx, ky)
    edges = np.linspace(0.0, float(radius.max()), k_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ids = np.searchsorted(edges, radius.ravel(), side="right") - 1
    return {
        "k": centers,
        "ids": ids,
        "valid": (ids >= 0) & (ids < centers.size),
        "n_bins": centers.size,
    }


def radial_mean_sem(
    power: np.ndarray, plan: dict[str, Any], min_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(power, dtype=float).ravel()
    selected = plan["valid"] & np.isfinite(values)
    ids = plan["ids"][selected]
    values = values[selected]
    n_bins = int(plan["n_bins"])

    count = np.bincount(ids, minlength=n_bins).astype(int)
    total = np.bincount(ids, weights=values, minlength=n_bins)
    total2 = np.bincount(ids, weights=values**2, minlength=n_bins)
    mean = np.full(n_bins, np.nan)
    sem = np.full(n_bins, np.nan)
    enough = count >= min_count
    mean[enough] = total[enough] / count[enough]
    variance = np.zeros(n_bins)
    variance[enough] = (
        total2[enough] - total[enough] ** 2 / count[enough]
    ) / np.maximum(count[enough] - 1, 1)
    sem[enough] = np.sqrt(np.maximum(variance[enough], 0) / count[enough])
    return mean, sem, count


def _monte_carlo_expectation(
    window: np.ndarray, plan: dict[str, Any], psf_filter: np.ndarray,
    realizations: int, seed: int, min_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_use = int(window.sum())
    power_sum = np.zeros(window.shape, dtype=float)
    for _ in range(realizations):
        white = rng.normal(size=window.shape)
        field = np.real(np.fft.ifft2(fft2(white) * psf_filter))
        sampled = np.zeros_like(field)
        sampled[window] = field[window]
        sampled[window] -= np.mean(sampled[window])
        power_sum += np.abs(fft2(sampled)) ** 2 / n_use
    profile, _, _ = radial_mean_sem(
        power_sum / realizations, plan, min_count
    )
    return profile


def _project_root_from_result(result_path: str | Path) -> Path | None:
    path = Path(result_path).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "code" / "sbf-2.ipynb").is_file():
            return parent
    return None


def _existing_systematics_expectation(
    compact: dict[str, Any], config: ExperimentConfig
) -> tuple[dict[str, np.ndarray], str] | None:
    """Безопасно импортирует уже проверенный per-PSF E(k), если он идентичен."""

    if not (
        config.k_bins == 80
        and config.e_realizations == 64
        and config.random_seed == 1489
        and config.min_modes_per_bin == 10
    ):
        return None

    metadata = compact["metadata"]
    project_root = _project_root_from_result(
        metadata["source"]["result_path"]
    )
    if project_root is None:
        return None
    slug = galaxy_slug(metadata["galaxy"])
    tag = slug.lower()
    table_root = (
        project_root / "runs" / "sbf2_systematics" / slug / "tables"
    )
    expectation_path = table_root / f"{tag}_expectation_spectra.csv"
    manifest_path = table_root / f"{tag}_noise_winsor_input_manifest.csv"
    closure_path = table_root / f"{tag}_reproduction_check.csv"
    if not all(path.is_file() for path in [
        expectation_path, manifest_path, closure_path
    ]):
        return None

    manifest = pd.read_csv(manifest_path).set_index("name")
    name_map = {
        "signal": "signal",
        "model": "model",
        "catalog_mask": "catalog_mask",
        "production_residual": "production_residual",
        "inner_ring": "inner_ring",
        "outer_ring": "outer_ring",
        "measurements": "measurements",
        "psf_ensemble": "psf",
    }
    fingerprints = metadata["source"]["fingerprints"]
    for old_name, current_name in name_map.items():
        if old_name not in manifest.index:
            return None
        row = manifest.loc[old_name]
        current = fingerprints[current_name]
        if (
            Path(str(row["path"])).resolve() != Path(current["path"]).resolve()
            or int(row["size_bytes"]) != int(current["size"])
            or int(row["mtime_ns"]) != int(current["mtime_ns"])
        ):
            return None

    closure = pd.read_csv(closure_path)
    passed = closure["passed"].astype(str).str.lower().eq("true")
    if closure.empty or not passed.all():
        return None

    frame = pd.read_csv(expectation_path)
    psf_ids = metadata["psf_ids"]
    result: dict[str, np.ndarray] = {}
    for ring, ring_data in compact["rings"].items():
        plan = radial_plan(ring_data["window"].shape, config.k_bins)
        profiles = []
        for psf_id in psf_ids:
            rows = frame[
                frame["ring"].eq(ring) & frame["psf_id"].eq(psf_id)
            ].sort_values("k")
            if len(rows) != len(plan["k"]):
                return None
            if not np.allclose(rows["k"].to_numpy(float), plan["k"]):
                return None
            profiles.append(rows["E"].to_numpy(float))
        result[f"{ring}_k"] = plan["k"]
        result[f"{ring}_E"] = np.vstack(profiles)
    return result, str(expectation_path)


def load_or_build_expectation_cache(
    compact: dict[str, Any], cache_root: str | Path,
    config: ExperimentConfig, rebuild: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    metadata = compact["metadata"]
    e_payload = {
        "version": EXPERIMENT_VERSION,
        "source_key": metadata["source_key"],
        "k_bins": config.k_bins,
        "realizations": config.e_realizations,
        "seed": config.random_seed,
        "min_count": config.min_modes_per_bin,
    }
    e_key = _stable_hash(e_payload)
    cache_path = (
        Path(cache_root).resolve() / "expectation"
        / f"{galaxy_slug(metadata['galaxy'])}_{e_key[:16]}.npz"
    )
    cache_hit = cache_path.is_file() and not rebuild

    if not cache_hit:
        existing = _existing_systematics_expectation(compact, config)
        existing_arrays = None if existing is None else existing[0]
        existing_source = None if existing is None else existing[1]
        if existing_source is not None:
            print(
                f"[E(k)] importing verified systematics cache for "
                f"{metadata['galaxy']}"
            )
        arrays: dict[str, Any] = {
            "metadata_json": np.array(json.dumps({
                **e_payload, "e_key": e_key,
                "psf_ids": metadata["psf_ids"],
                "bootstrapped_from": existing_source,
            }, ensure_ascii=False, sort_keys=True))
        }
        for ring, ring_data in compact["rings"].items():
            window = ring_data["window"]
            plan = radial_plan(window.shape, config.k_bins)
            if existing_arrays is not None:
                arrays[f"{ring}_k"] = existing_arrays[f"{ring}_k"]
                arrays[f"{ring}_E"] = existing_arrays[f"{ring}_E"]
                continue

            profiles = []
            for index, psf in enumerate(compact["psfs"]):
                print(
                    f"[E(k)] {metadata['galaxy']} {ring}: "
                    f"PSF {index + 1}/{len(compact['psfs'])}"
                )
                py, px = psf.shape
                if py > window.shape[0] or px > window.shape[1]:
                    raise RuntimeError(
                        f"{metadata['galaxy']} {ring}: PSF больше FFT-crop"
                    )
                padded = np.zeros(window.shape, dtype=float)
                y0 = window.shape[0] // 2 - py // 2
                x0 = window.shape[1] // 2 - px // 2
                padded[y0:y0 + py, x0:x0 + px] = psf
                with set_workers(config.fft_workers):
                    profiles.append(_monte_carlo_expectation(
                        window, plan, fft2(padded), config.e_realizations,
                        config.random_seed + index,
                        config.min_modes_per_bin,
                    ))
            arrays[f"{ring}_k"] = plan["k"]
            arrays[f"{ring}_E"] = np.vstack(profiles)
        _atomic_npz(cache_path, **arrays)
    else:
        print(f"[E(k)] cache hit {metadata['galaxy']}: {cache_path.name}")

    with np.load(cache_path, allow_pickle=False) as archive:
        e_metadata = json.loads(str(archive["metadata_json"].item()))
        expectation = {
            "metadata": e_metadata,
            "rings": {
                ring: {
                    "k": np.asarray(archive[f"{ring}_k"], dtype=float),
                    "E": np.asarray(archive[f"{ring}_E"], dtype=float),
                }
                for ring in REGION_NAMES
            },
        }
    if expectation["metadata"].get("e_key") != e_key:
        raise RuntimeError("Ключ кэша E(k) не совпадает с параметрами")
    return expectation, cache_path, cache_hit


def weighted_fit(
    y: np.ndarray, y_error: np.ndarray, expectation: np.ndarray
) -> dict[str, Any]:
    """Взвешенный МНК для production-модели ``P0 E(k) + P1``."""

    design = np.column_stack([expectation, np.ones_like(expectation)])
    safe_error = np.maximum(y_error, 1e-12)
    weighted_design = design / safe_error[:, None]
    weighted_y = y / safe_error
    coefficients = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None
    )[0]
    prediction = design @ coefficients
    residual = y - prediction
    chi2 = float(np.sum((residual / safe_error) ** 2))
    dof = max(len(y) - 2, 1)
    chi2_reduced = chi2 / dof
    normal = weighted_design.T @ weighted_design
    covariance = np.linalg.pinv(normal) * max(chi2_reduced, 1.0)
    return {
        "P0": float(coefficients[0]),
        "P1": float(coefficients[1]),
        "P0_sigma": float(np.sqrt(max(covariance[0, 0], 0))),
        "chi2": chi2,
        "chi2_reduced": chi2_reduced,
        "aicc": float(chi2 + 4.0 + 12.0 / max(len(y) - 3, 1)),
        "condition_number": float(np.linalg.cond(weighted_design)),
        "prediction": prediction,
        "standardized_residual": residual / safe_error,
    }


def robust_mag_scatter(values: Iterable[float]) -> tuple[float, str]:
    """Буквально повторяет выбор k-window scatter из production sbf-2."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size >= 3:
        median = float(np.median(array))
        mad_sigma = float(1.4826 * np.median(np.abs(array - median)))
        std_sigma = float(np.std(array, ddof=1))
        if np.isfinite(mad_sigma) and mad_sigma > 0:
            return mad_sigma, "k-window MAD"
        if np.isfinite(std_sigma) and std_sigma > 0:
            return std_sigma, "k-window std"
    if array.size >= 2:
        std_sigma = float(np.std(array, ddof=1))
        half_range = float(0.5 * (np.max(array) - np.min(array)))
        if np.isfinite(std_sigma) and std_sigma > 0:
            return std_sigma, "k-window std"
        if np.isfinite(half_range) and half_range > 0:
            return half_range, "k-window half-range"
    return np.nan, "no k-window scatter"


def _crop_header(signal_path: Path, crop: list[int]) -> fits.Header:
    with fits.open(signal_path, memmap=True) as hdul:
        header = hdul["SCI"].header.copy()
    y0, _, x0, _ = crop
    if "CRPIX1" in header:
        header["CRPIX1"] = float(header["CRPIX1"]) - x0
    if "CRPIX2" in header:
        header["CRPIX2"] = float(header["CRPIX2"]) - y0
    return header


def _save_fft_input_fits(
    path: Path, source: dict[str, Any], metadata: dict[str, Any],
    ring: str, branch: str, normalized: np.ndarray,
    fft_input: np.ndarray, window: np.ndarray,
    clip_info: dict[str, Any], subtracted_mean: float,
    config: ExperimentConfig, experiment_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = metadata["rings"][ring]["crop"]
    header = _crop_header(Path(source["paths"]["signal"]), crop)
    header["BUNIT"] = "sqrt(MJy/sr)"
    header["GALAXY"] = metadata["galaxy"]
    header["SBFREG"] = ring
    header["SBFBRNCH"] = branch
    header["SBFVERS"] = EXPERIMENT_VERSION
    header["EXPTKEY"] = experiment_key[:16]
    header["SOURCEKY"] = source["source_key"][:16]
    header["KBINS"] = int(config.k_bins)
    header["EREAL"] = int(config.e_realizations)
    header["RNGSEED"] = int(config.random_seed)
    header["KMAX"] = float(config.kmax)
    header["NUSE"] = int(window.sum())
    header["SUBMEAN"] = float(subtracted_mean)
    header["CLIPSPC"] = str(clip_info["space"])
    if clip_info.get("sigma") is not None:
        header["CLIPSIG"] = float(clip_info["sigma"])
        header["CLIPLO"] = float(clip_info["lower"])
        header["CLIPHI"] = float(clip_info["upper"])
    header["NCLIPPED"] = int(clip_info["changed_pixels"])
    header["FRCLIP"] = float(clip_info["changed_fraction"])
    header.add_history(
        "PRIMARY: normalized, clipped if requested, mean-subtracted; "
        "NaN outside the usable ring window"
    )
    header.add_history(
        "FFTINPUT: exact array passed to FFT; zero outside the usable window"
    )
    header.add_history(
        "Requested kmin values: "
        + ", ".join(f"{value:g}" for value in config.kmins)
    )
    hdul = fits.HDUList([
        fits.PrimaryHDU(np.asarray(normalized, dtype=np.float32), header),
        fits.ImageHDU(np.asarray(fft_input, dtype=np.float64), name="FFTINPUT"),
        fits.ImageHDU(window.astype(np.uint8), name="WINDOW"),
    ])
    hdul.writeto(path, overwrite=True, checksum=True)


def _save_full_normalized_residual(
    source: dict[str, Any], run_dir: Path,
    limits: dict[str, float], config: ExperimentConfig,
    experiment_key: str,
) -> tuple[Path, dict[str, Any]]:
    """Creates the full-frame field used by the candidate SBF branch.

    The saved primary image is already the final residual divided by
    ``sqrt(model)`` and winsorized with the common two-annulus limits.  Pixels
    rejected by the final catalogue mask, invalid science pixels and pixels
    outside the positive galaxy model remain NaN.  Ring-specific mean
    subtraction is intentionally deferred until a ring is selected for FFT.
    """

    paths = {name: Path(path) for name, path in source["paths"].items()}
    background = float(source["result"]["signal_background_scalar"])
    output_dir = run_dir / "normalized_fits"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{source['stem']}_sbf_resid_catalog_mask_normalized_clip_"
        f"{sigma_tag(config.normalized_sigma)}sigma.fits"
    )

    with fits.open(paths["signal"], memmap=True) as signal_hdul, \
            fits.open(paths["model"], memmap=True) as model_hdul, \
            fits.open(paths["catalog_mask"], memmap=True) as mask_hdul:
        science = signal_hdul["SCI"].data
        model = model_hdul[0].data
        catalog_mask = mask_hdul[0].data
        if science.shape != model.shape or science.shape != catalog_mask.shape:
            raise RuntimeError(
                f"{source['galaxy']}: incompatible full-frame FITS shapes"
            )

        normalized = np.full(science.shape, np.nan, dtype=np.float32)
        n_valid = 0
        n_clipped = 0
        rows_per_chunk = 256
        for y0 in range(0, science.shape[0], rows_per_chunk):
            y1 = min(y0 + rows_per_chunk, science.shape[0])
            science_chunk = np.asarray(
                science[y0:y1], dtype=np.float64
            )
            model_chunk64 = np.asarray(model[y0:y1], dtype=np.float64)
            model_chunk32 = np.asarray(model_chunk64, dtype=np.float32)
            valid = (
                ~np.asarray(catalog_mask[y0:y1], dtype=bool)
                & np.isfinite(science_chunk)
                & np.isfinite(model_chunk64)
                & (model_chunk64 > 0)
            )
            if not np.any(valid):
                continue

            raw_chunk32 = np.asarray(
                (science_chunk - background) - model_chunk64,
                dtype=np.float32,
            )
            values = (
                np.asarray(raw_chunk32[valid], dtype=np.float64)
                / np.sqrt(np.asarray(model_chunk32[valid], dtype=np.float64))
            )
            changed = (values < limits["lower"]) | (values > limits["upper"])
            values = np.clip(values, limits["lower"], limits["upper"])
            normalized_chunk = normalized[y0:y1]
            normalized_chunk[valid] = np.asarray(values, dtype=np.float32)
            n_valid += int(valid.sum())
            n_clipped += int(changed.sum())

        header = signal_hdul["SCI"].header.copy()
        for keyword in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
            header.pop(keyword, None)

    if n_valid == 0:
        raise RuntimeError(
            f"{source['galaxy']}: final catalogue mask left no valid pixels"
        )
    clipped_fraction = n_clipped / n_valid
    header["BUNIT"] = "sqrt(MJy/sr)"
    header["GALAXY"] = source["galaxy"]
    header["SBFBRNCH"] = config.candidate_branch
    header["SBFVERS"] = EXPERIMENT_VERSION
    header["EXPTKEY"] = experiment_key[:16]
    header["SOURCEKY"] = source["source_key"][:16]
    header["BKGDSCAL"] = background
    header["NORMBY"] = "SQRTMODEL"
    header["CLIPSPC"] = "NORMUNION"
    header["CLIPSIG"] = float(config.normalized_sigma)
    header["CLIPMED"] = float(limits["median"])
    header["CLIPSCL"] = float(limits["scale"])
    header["CLIPNPIX"] = int(limits["n_pixels"])
    header["CLIPLO"] = float(limits["lower"])
    header["CLIPHI"] = float(limits["upper"])
    header["NVALID"] = int(n_valid)
    header["NCLIPPED"] = int(n_clipped)
    header["FRCLIP"] = float(clipped_fraction)
    header["MEANSUB"] = False
    header["NEXTSTEP"] = "RINGMEAN+FFT"
    header.add_history(
        "SCI-background-model divided by sqrt(model), then winsorized."
    )
    header.add_history(
        "NaN marks the final catalogue mask or invalid/non-positive model."
    )
    header.add_history(
        "This float32 PRIMARY image is the numerical source for the "
        "candidate SBF measurement."
    )
    header.add_history(
        "Per-ring mean subtraction is intentionally not applied here."
    )

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".fits", dir=output_dir
    )
    os.close(handle)
    try:
        fits.PrimaryHDU(normalized, header).writeto(
            temporary_name, overwrite=True, checksum=True
        )
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    provenance = {
        "path": str(output_path),
        "shape": list(normalized.shape),
        "dtype": str(normalized.dtype),
        "n_valid": n_valid,
        "n_clipped": n_clipped,
        "clipped_fraction": clipped_fraction,
        "normalized_by": "sqrt(model)",
        "mean_subtracted": False,
        "next_step": "select ring, subtract its mean, then FFT",
    }
    return output_path, provenance


def _normalized_threshold(
    compact: dict[str, Any], sigma: float
) -> dict[str, float]:
    inner_crop = compact["metadata"]["rings"]["inner"]["crop"]
    outer_crop = compact["metadata"]["rings"]["outer"]["crop"]
    iy0, iy1, ix0, ix1 = inner_crop
    oy0, oy1, ox0, ox1 = outer_crop
    y0, y1 = max(iy0, oy0), min(iy1, oy1)
    x0, x1 = max(ix0, ox0), min(ix1, ox1)
    if y1 > y0 and x1 > x0:
        inner_overlap = compact["rings"]["inner"]["window"][
            y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0
        ]
        outer_overlap = compact["rings"]["outer"]["window"][
            y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0
        ]
        if np.any(inner_overlap & outer_overlap):
            raise RuntimeError(
                "Рабочие кольца перекрываются: общий порог получил бы "
                "дублированные пиксели"
            )

    values = []
    for ring_data in compact["rings"].values():
        window = ring_data["window"]
        normalized = ring_data["raw"][window] / np.sqrt(
            ring_data["model"][window]
        )
        values.append(normalized)
    union = np.concatenate(values)
    _, median, scale = sigma_clipped_stats(
        union, sigma=sigma, maxiters=5
    )
    median, scale = float(median), float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Некорректный scale нормированных остатков")
    return {
        "sigma": float(sigma),
        "median": median,
        "scale": scale,
        "lower": median - sigma * scale,
        "upper": median + sigma * scale,
        "n_pixels": int(union.size),
    }


def _production_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metadata["production_rows"])


def _run_spectral_experiment(
    source: dict[str, Any], compact: dict[str, Any],
    expectation: dict[str, Any], run_dir: Path,
    config: ExperimentConfig, experiment_key: str,
) -> dict[str, Any]:
    metadata = compact["metadata"]
    production = _production_frame(metadata)
    candidate_limits = _normalized_threshold(
        compact, config.normalized_sigma
    )
    raw_limits = metadata["raw_global"]
    normalized_full_limits = metadata["normalized_full"]
    branches = config.branches

    tables_dir = run_dir / "tables"
    fits_dir = run_dir / "normalized_fits"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)

    candidate_full_path, candidate_full_provenance = (
        _save_full_normalized_residual(
            source, run_dir, candidate_limits, config, experiment_key
        )
    )

    power_rows, fit_rows, clipping_rows, fits_products = [], [], [], []
    for ring, ring_data in compact["rings"].items():
        window = ring_data["window"]
        raw = ring_data["raw"]
        model = ring_data["model"]
        n_use = int(window.sum())
        plan = radial_plan(window.shape, config.k_bins)
        e_profiles = expectation["rings"][ring]["E"]
        if not np.allclose(plan["k"], expectation["rings"][ring]["k"]):
            raise RuntimeError(f"{ring}: k-сетка не совпала с кэшем E(k)")

        y0, y1, x0, x1 = metadata["rings"][ring]["crop"]
        # Read back the actual saved FITS.  This is the contract boundary:
        # the candidate spectrum is measured from the persisted product, not
        # from an equivalent in-memory precursor.
        with fits.open(candidate_full_path, memmap=True) as candidate_hdul:
            candidate_crop = np.array(
                candidate_hdul[0].data[y0:y1, x0:x1],
                dtype=np.float64, copy=True,
            )
        if candidate_crop.shape != window.shape:
            raise RuntimeError(
                f"{metadata['galaxy']} {ring}: full-frame crop has shape "
                f"{candidate_crop.shape}, expected {window.shape}"
            )
        if not np.all(np.isfinite(candidate_crop[window])):
            raise RuntimeError(
                f"{metadata['galaxy']} {ring}: saved normalized FITS contains "
                "non-finite pixels inside the working window"
            )

        normalized_raw = np.zeros(raw.shape, dtype=float)
        normalized_raw[window] = raw[window] / np.sqrt(model[window])

        for branch in branches:
            print(f"[FFT] {metadata['galaxy']} {ring}: {branch}")
            working = normalized_raw.copy()
            if branch == "no_winsor":
                clip_space, limits = "none", None
                changed = np.zeros(raw.shape, dtype=bool)
            elif branch == "raw_global_3p5":
                clip_space, limits = "raw", raw_limits
                clipped_raw = np.clip(
                    raw[window], limits["lower"], limits["upper"]
                )
                working[window] = clipped_raw / np.sqrt(model[window])
                changed = np.zeros(raw.shape, dtype=bool)
                changed[window] = (
                    (raw[window] < limits["lower"])
                    | (raw[window] > limits["upper"])
                )
            elif branch == "normalized_full_3p5":
                clip_space, limits = "normalized_full", normalized_full_limits
                changed = np.zeros(raw.shape, dtype=bool)
                changed[window] = (
                    (working[window] < limits["lower"])
                    | (working[window] > limits["upper"])
                )
                working[window] = np.clip(
                    working[window], limits["lower"], limits["upper"]
                )
            elif branch == config.candidate_branch:
                clip_space, limits = "normalized_union", candidate_limits
                changed = np.zeros(raw.shape, dtype=bool)
                changed[window] = (
                    (working[window] < limits["lower"])
                    | (working[window] > limits["upper"])
                )
                # The candidate measurement deliberately starts from the
                # float32 full-frame FITS product saved above.  Therefore the
                # file is not merely a picture: selecting a ring and
                # subtracting its mean reproduces the actual FFT input.
                working[window] = candidate_crop[window]
            else:
                raise RuntimeError(f"Unknown experiment branch: {branch}")

            subtracted_mean = float(np.mean(working[window]))
            working[window] -= subtracted_mean
            fft_input = np.zeros(raw.shape, dtype=float)
            fft_input[window] = working[window]
            normalized_view = np.full(raw.shape, np.nan, dtype=float)
            normalized_view[window] = working[window]

            changed_pixels = int(changed[window].sum())
            changed_fraction = changed_pixels / n_use
            clip_info = {
                "space": clip_space,
                "sigma": None if limits is None else limits["sigma"],
                "threshold_median": None if limits is None else limits["median"],
                "threshold_scale": None if limits is None else limits["scale"],
                "threshold_n_pixels": None if limits is None else limits["n_pixels"],
                "lower": None if limits is None else limits["lower"],
                "upper": None if limits is None else limits["upper"],
                "changed_pixels": changed_pixels,
                "changed_fraction": changed_fraction,
            }
            clipping_rows.append({
                "galaxy": metadata["galaxy"],
                "ring": ring,
                "branch": branch,
                **clip_info,
                "subtracted_mean": subtracted_mean,
                "n_use": n_use,
            })

            save_ring_fits = config.save_all_branch_fits or (
                config.save_ring_fft_fits
                and branch == config.candidate_branch
            )
            if save_ring_fits:
                fits_path = fits_dir / (
                    f"{galaxy_slug(metadata['galaxy']).lower()}_{ring}_"
                    f"{branch}_normalized_fft_input.fits"
                )
                _save_fft_input_fits(
                    fits_path, source, metadata, ring, branch,
                    normalized_view, fft_input, window, clip_info,
                    subtracted_mean, config, experiment_key,
                )
                fits_products.append({
                    "ring": ring, "branch": branch,
                    "path": str(fits_path),
                })

            with set_workers(config.fft_workers):
                power = np.abs(fft2(fft_input)) ** 2 / n_use
            pk, pk_error, pk_count = radial_mean_sem(
                power, plan, config.min_modes_per_bin
            )
            e_median = np.nanmedian(e_profiles, axis=0)
            e_mad = 1.4826 * np.nanmedian(
                np.abs(e_profiles - e_median), axis=0
            )
            for index, k_value in enumerate(plan["k"]):
                power_rows.append({
                    "galaxy": metadata["galaxy"],
                    "ring": ring,
                    "branch": branch,
                    "k": float(k_value),
                    "Pk": float(pk[index]),
                    "Pk_error": float(pk_error[index]),
                    "Pk_count": int(pk_count[index]),
                    "E_median": float(e_median[index]),
                    "E_mad": float(e_mad[index]),
                })

            production_ring = production[
                production["region"].eq(REGION_NAMES[ring])
            ]
            pr = float(production_ring.iloc[0]["Pr"])
            zeropoint = float(metadata["ab_zeropoint_per_pixel"])
            for requested_kmin in config.kmins:
                effective_kmin = max(
                    float(requested_kmin), 10.0 / min(window.shape)
                )
                for psf_id, e_profile in zip(
                    metadata["psf_ids"], e_profiles
                ):
                    selected = (
                        (plan["k"] >= effective_kmin)
                        & (plan["k"] <= config.kmax)
                        & np.isfinite(pk)
                        & np.isfinite(pk_error)
                        & (pk_error > 0)
                        & np.isfinite(e_profile)
                        & (e_profile > 0)
                    )
                    if int(selected.sum()) < 10:
                        continue
                    fit = weighted_fit(
                        pk[selected], pk_error[selected], e_profile[selected]
                    )
                    if not np.isfinite(fit["P0"]) or fit["P0"] <= 0:
                        continue
                    fluctuation_power = fit["P0"] - pr
                    mbar = (
                        -2.5 * np.log10(fluctuation_power) + zeropoint
                        if fluctuation_power > 0 else np.nan
                    )
                    mbar_sigma = (
                        (2.5 / np.log(10.0))
                        * fit["P0_sigma"] / fluctuation_power
                        if fluctuation_power > 0 else np.nan
                    )
                    fit_rows.append({
                        "galaxy": metadata["galaxy"],
                        "ring": ring,
                        "branch": branch,
                        "requested_kmin": float(requested_kmin),
                        "effective_kmin": effective_kmin,
                        "kmax": config.kmax,
                        "psf_id": psf_id,
                        "n_fit": int(selected.sum()),
                        "P0": fit["P0"],
                        "P1": fit["P1"],
                        "P0_sigma_formal": fit["P0_sigma"],
                        "Pr": pr,
                        "P_fluctuation": fluctuation_power,
                        "mbar": mbar,
                        "mbar_sigma_formal": mbar_sigma,
                        "chi2": fit["chi2"],
                        "chi2_reduced": fit["chi2_reduced"],
                        "aicc": fit["aicc"],
                        "condition_number": fit["condition_number"],
                    })

    power_frame = pd.DataFrame(power_rows)
    fit_per_psf = pd.DataFrame(fit_rows)
    clipping_frame = pd.DataFrame(clipping_rows)
    if fit_per_psf.empty:
        raise RuntimeError(
            f"{metadata['galaxy']}: ни одна PSF не дала P0 > 0"
        )

    summary_rows = []
    group_columns = [
        "galaxy", "ring", "branch", "requested_kmin",
        "effective_kmin", "kmax",
    ]
    for keys, group in fit_per_psf.groupby(group_columns, sort=False):
        p0_values = group["P0"].to_numpy(float)
        p0 = float(np.median(p0_values))
        p0_formal = float(np.median(group["P0_sigma_formal"]))
        p0_psf = float(1.4826 * np.median(np.abs(p0_values - p0)))
        p0_total = float(np.hypot(p0_formal, p0_psf))
        pr = float(np.median(group["Pr"]))
        fluctuation_power = p0 - pr
        mbar = (
            -2.5 * np.log10(fluctuation_power)
            + float(metadata["ab_zeropoint_per_pixel"])
            if fluctuation_power > 0 else np.nan
        )
        mbar_sigma = (
            (2.5 / np.log(10.0)) * p0_total / fluctuation_power
            if fluctuation_power > 0 else np.nan
        )
        summary_rows.append({
            **dict(zip(group_columns, keys)),
            "n_psf": int(len(group)),
            "P0": p0,
            "P0_sigma_formal": p0_formal,
            "P0_psf_mad": p0_psf,
            "P0_sigma_total": p0_total,
            "P1": float(np.median(group["P1"])),
            "P1_negative_fraction": float(np.mean(group["P1"] < 0)),
            "Pr": pr,
            "P_fluctuation": fluctuation_power,
            "mbar": mbar,
            "mbar_sigma": mbar_sigma,
            "chi2_reduced": float(np.median(group["chi2_reduced"])),
            "aicc": float(np.median(group["aicc"])),
            "condition_number": float(np.median(group["condition_number"])),
        })
    fit_summary = pd.DataFrame(summary_rows)
    if fit_summary.empty:
        raise RuntimeError(f"{metadata['galaxy']}: пустая сводка спектральных фитов")
    fit_summary["k_window_scatter"] = np.nan
    fit_summary["k_window_scatter_method"] = "no k-window scatter"
    fit_summary["sigma_measurement"] = fit_summary["mbar_sigma"]
    fit_summary["sigma_measurement_method"] = "fit covariance + PSF MAD"
    for (_, ring, branch), indexes in fit_summary.groupby(
        ["galaxy", "ring", "branch"], sort=False
    ).groups.items():
        scatter, scatter_method = robust_mag_scatter(
            fit_summary.loc[indexes, "mbar"]
        )
        fit_summary.loc[indexes, "k_window_scatter"] = scatter
        fit_summary.loc[indexes, "k_window_scatter_method"] = scatter_method
        for index in indexes:
            fit_sigma = float(fit_summary.at[index, "mbar_sigma"])
            candidates = [
                value for value in [fit_sigma, scatter]
                if np.isfinite(value) and value > 0
            ]
            if candidates:
                fit_summary.at[index, "sigma_measurement"] = max(candidates)
                fit_summary.at[index, "sigma_measurement_method"] = (
                    f"max(fit covariance + PSF MAD, {scatter_method})"
                    if np.isfinite(scatter) and scatter > 0
                    else "fit covariance + PSF MAD"
                )

    closure_rows = []
    for _, row in fit_summary[
        fit_summary["branch"].eq("raw_global_3p5")
    ].iterrows():
        production_row = _production_row(
            production, str(row["ring"]), float(row["requested_kmin"])
        )
        relative_p0 = float(row["P0"] / production_row["P0"] - 1.0)
        relative_p1 = float(
            (row["P1"] - production_row["P1"])
            / max(abs(float(production_row["P1"])), 1e-12)
        )
        delta_mbar = float(row["mbar"] - production_row["mbar_spec"])
        closure_rows.append({
            "galaxy": metadata["galaxy"],
            "ring": row["ring"],
            "requested_kmin": row["requested_kmin"],
            "production_P0": float(production_row["P0"]),
            "recreated_P0": float(row["P0"]),
            "relative_P0_difference": relative_p0,
            "production_P1": float(production_row["P1"]),
            "recreated_P1": float(row["P1"]),
            "relative_P1_difference": relative_p1,
            "production_mbar": float(production_row["mbar_spec"]),
            "recreated_mbar": float(row["mbar"]),
            "delta_mbar": delta_mbar,
            "passed": bool(
                abs(delta_mbar) <= 0.005
                and abs(relative_p0) <= 0.005
                and abs(relative_p1) <= 0.005
            ),
        })
    closure = pd.DataFrame(closure_rows, columns=[
        "galaxy", "ring", "requested_kmin",
        "production_P0", "recreated_P0", "relative_P0_difference",
        "production_P1", "recreated_P1", "relative_P1_difference",
        "production_mbar", "recreated_mbar", "delta_mbar", "passed",
    ])

    combined_rows = []
    for (branch, kmin), group in fit_summary.groupby(
        ["branch", "requested_kmin"], sort=False
    ):
        ring_rows = group.set_index("ring")
        if not set(REGION_NAMES).issubset(ring_rows.index):
            continue
        inner, outer = ring_rows.loc["inner"], ring_rows.loc["outer"]
        sigmas = np.array([
            inner["sigma_measurement"], outer["sigma_measurement"]
        ], float)
        values = np.array([inner["mbar"], outer["mbar"]], float)
        valid = np.isfinite(sigmas) & (sigmas > 0) & np.isfinite(values)
        weights = np.zeros(2, dtype=float)
        weights[valid] = 1.0 / sigmas[valid] ** 2
        weighted = (
            float(np.sum(weights * values) / np.sum(weights))
            if np.sum(weights) > 0 else np.nan
        )
        formal = (
            float(np.sqrt(1.0 / np.sum(weights)))
            if np.sum(weights) > 0 else np.nan
        )
        combined_rows.append({
            "galaxy": metadata["galaxy"],
            "branch": branch,
            "requested_kmin": float(kmin),
            "mbar_inner": float(inner["mbar"]),
            "sigma_inner_fit_psf": float(inner["mbar_sigma"]),
            "sigma_inner": float(inner["sigma_measurement"]),
            "sigma_inner_method": inner["sigma_measurement_method"],
            "mbar_outer": float(outer["mbar"]),
            "sigma_outer_fit_psf": float(outer["mbar_sigma"]),
            "sigma_outer": float(outer["sigma_measurement"]),
            "sigma_outer_method": outer["sigma_measurement_method"],
            "mbar_weighted": weighted,
            "sigma_weighted_formal": formal,
            "annulus_difference": float(outer["mbar"] - inner["mbar"]),
            "annulus_half_difference": float(
                0.5 * abs(outer["mbar"] - inner["mbar"])
            ),
            "sigma_adopted_internal": float(max(
                formal,
                0.5 * abs(outer["mbar"] - inner["mbar"]),
            )),
        })
    combined = pd.DataFrame(combined_rows)
    if combined.empty:
        raise RuntimeError(f"{metadata['galaxy']}: не удалось объединить кольца")

    table_paths = {
        "power_spectra": tables_dir / "power_spectra.csv",
        "fit_per_psf": tables_dir / "fit_per_psf.csv",
        "fit_summary": tables_dir / "fit_summary.csv",
        "clipping": tables_dir / "clipping_summary.csv",
        "production_closure": tables_dir / "production_closure.csv",
        "combined_annuli": tables_dir / "combined_annuli.csv",
    }
    for frame, path in [
        (power_frame, table_paths["power_spectra"]),
        (fit_per_psf, table_paths["fit_per_psf"]),
        (fit_summary, table_paths["fit_summary"]),
        (clipping_frame, table_paths["clipping"]),
        (closure, table_paths["production_closure"]),
        (combined, table_paths["combined_annuli"]),
    ]:
        frame.to_csv(path, index=False)

    return {
        "table_paths": {name: str(path) for name, path in table_paths.items()},
        "full_normalized_residual_fits": str(candidate_full_path),
        "full_normalized_residual": candidate_full_provenance,
        "normalized_fits": fits_products,
        "closure_passed": bool(not closure.empty and closure["passed"].all()),
        "candidate_limits": candidate_limits,
    }


def _result_artifacts_valid(result: dict[str, Any]) -> bool:
    table_paths = [Path(path) for path in result.get("table_paths", {}).values()]
    full_path_text = result.get("full_normalized_residual_fits")
    full_path = Path(full_path_text) if full_path_text else None
    fits_paths = [
        Path(item["path"]) for item in result.get("normalized_fits", [])
    ]
    return (
        bool(table_paths) and full_path is not None
        and all(path.is_file() and path.stat().st_size > 0 for path in table_paths)
        and _fits_readable(full_path)
        and all(_fits_readable(path) for path in fits_paths)
    )


def process_target(
    galaxy: str,
    source_batch_root: str | Path,
    output_root: str | Path,
    config: ExperimentConfig | None = None,
    force: bool = False,
    rebuild_input_cache: bool = False,
    rebuild_expectation_cache: bool = False,
) -> dict[str, Any]:
    """Последовательно пересчитывает только спектральную часть одной цели."""

    config = ExperimentConfig() if config is None else config
    output_root = Path(output_root).resolve()
    source = inspect_source(galaxy, source_batch_root)
    experiment_payload = {
        "version": EXPERIMENT_VERSION,
        "source_key": source["source_key"],
        "config": asdict(config),
    }
    config_key = experiment_config_key(config)
    experiment_key = _stable_hash(experiment_payload)
    run_dir = (
        output_root / "products" / galaxy_slug(galaxy)
        / experiment_key[:16]
    )
    result_path = (
        output_root / "batch" / "results"
        / f"{galaxy_slug(galaxy)}_result.json"
    )
    latest_result_path = (
        output_root / "batch" / f"{galaxy_slug(galaxy)}_result.json"
    )
    legacy_result_path = (
        output_root / "batch" / "results" / config_key[:16]
        / f"{galaxy_slug(galaxy)}_result.json"
    )

    cached_result_paths = [
        result_path, latest_result_path, legacy_result_path,
    ]
    for cached_path in (cached_result_paths if not force else []):
        if not cached_path.is_file():
            continue
        old_result = json.loads(cached_path.read_text(encoding="utf-8"))
        if (
            old_result.get("status") == "ok"
            and old_result.get("version") == EXPERIMENT_VERSION
            and old_result.get("config") == _jsonable(asdict(config))
            and old_result.get("source_key") == source["source_key"]
            and _result_artifacts_valid(old_result)
        ):
            old_result["result_cache_hit"] = True
            old_result["result_path"] = str(result_path)
            _atomic_json(result_path, old_result)
            _atomic_json(latest_result_path, old_result)
            update_target_status(
                galaxy, "ok", output_root, config, stage="complete",
                message="existing result reused", result_path=result_path,
            )
            return old_result

    cache_root = output_root / "cache"
    update_target_status(
        galaxy, "running", output_root, config, stage="input cache"
    )
    compact, compact_path, compact_hit = load_or_build_compact_cache(
        source, cache_root, config, rebuild=rebuild_input_cache
    )
    update_target_status(
        galaxy, "running", output_root, config, stage="E(k) cache"
    )
    expectation, expectation_path, expectation_hit = (
        load_or_build_expectation_cache(
            compact, cache_root, config, rebuild=rebuild_expectation_cache
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    update_target_status(
        galaxy, "running", output_root, config, stage="spectral measurement"
    )
    products = _run_spectral_experiment(
        source, compact, expectation, run_dir, config, experiment_key
    )
    result = {
        "status": "ok",
        "galaxy": galaxy,
        "version": EXPERIMENT_VERSION,
        "config_key": config_key,
        "experiment_key": experiment_key,
        "config": asdict(config),
        "candidate_branch": config.candidate_branch,
        "source_result": source["result_path"],
        "source_job_id": source["result"].get("job_id"),
        "source_key": source["source_key"],
        "n_psf": int(len(compact["metadata"]["psf_ids"])),
        "run_dir": str(run_dir),
        "compact_cache": str(compact_path),
        "expectation_cache": str(expectation_path),
        "compact_cache_hit": compact_hit,
        "expectation_cache_hit": expectation_hit,
        "expectation_bootstrapped_from": expectation["metadata"].get(
            "bootstrapped_from"
        ),
        "result_cache_hit": False,
        **products,
    }
    result["result_path"] = str(result_path)
    _atomic_json(run_dir / "provenance.json", {
        **experiment_payload,
        "source": source,
        "compact_cache": str(compact_path),
        "expectation_cache": str(expectation_path),
        "products": products,
    })
    _atomic_json(result_path, result)
    _atomic_json(latest_result_path, result)
    update_target_status(
        galaxy, "ok", output_root, config, stage="complete",
        message="ok", result_path=result_path,
    )
    return result


def load_result_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(path)
        for name, path in result["table_paths"].items()
    }


def evaluation_criteria() -> pd.DataFrame:
    """Заранее объявленные правила; красоту гистограммы сюда не подмешиваем."""

    return pd.DataFrame([
        {
            "priority": 1,
            "check": "Production closure",
            "rule": "|Δm̄| ≤ 0.005 mag и |ΔP0/P0| ≤ 0.5% для старой ветви",
            "role": "hard gate",
        },
        {
            "priority": 2,
            "check": "Physical fit",
            "rule": "P0 − Pr > 0 во всех кольцах; все PSF дали конечный фит",
            "role": "hard gate",
        },
        {
            "priority": 3,
            "check": "k-window stability",
            "rule": "размах m̄ по kmin=0.01/0.03/0.04 желательно ≤ 0.03 mag",
            "role": "stability",
        },
        {
            "priority": 4,
            "check": "Annulus sensitivity",
            "rule": "расхождение колец не должно ухудшиться > 0.02 mag относительно no-winsor",
            "role": "diagnostic; градиент может быть физическим",
        },
        {
            "priority": 5,
            "check": "Intervention balance",
            "rule": "доли изменённых пикселей inner/outer публикуются; перекос > 1 п.п. — флаг",
            "role": "diagnostic, не критерий истинности",
        },
        {
            "priority": 6,
            "check": "Injection recovery",
            "rule": "целевой bias < 0.01 mag; |bias| > 0.02 mag — отказ от ветви",
            "role": "required before adoption",
        },
        {
            "priority": 7,
            "check": "All-14 calibration",
            "rule": "сравнить нуль-пункт, intrinsic scatter и LOO RMS для четырёх ветвей",
            "role": "final scientific decision",
        },
    ])


def build_evaluation_table(
    results: Iterable[dict[str, Any]], output_root: str | Path,
    main_kmin: float = 0.04,
) -> pd.DataFrame:
    """Сводит измеримые критерии; не объявляет кандидата «правильным»."""

    results = list(results)
    rows = []
    for result in results:
        if result.get("status") != "ok":
            rows.append({
                "galaxy": result.get("galaxy", "unknown"),
                "status": "failed",
                "message": result.get("error", "unknown error"),
            })
            continue
        tables = load_result_tables(result)
        fit = tables["fit_summary"]
        fit_per_psf = tables["fit_per_psf"]
        combined = tables["combined_annuli"]
        clipping = tables["clipping"]
        candidate = result["candidate_branch"]

        main = combined[np.isclose(combined["requested_kmin"], main_kmin)]
        by_branch = main.set_index("branch")
        required_branches = {
            "no_winsor", "raw_global_3p5",
            "normalized_full_3p5", candidate,
        }
        missing_branches = sorted(required_branches - set(by_branch.index))
        if missing_branches:
            rows.append({
                "galaxy": result["galaxy"],
                "status": "missing_branch_results",
                "production_closure": bool(result.get("closure_passed", False)),
                "candidate_physical_fit": False,
                "message": "missing: " + ", ".join(missing_branches),
            })
            continue
        candidate_rows = fit[fit["branch"].eq(candidate)]
        candidate_psf_rows = fit_per_psf[
            fit_per_psf["branch"].eq(candidate)
        ]
        expected_psf_rows = (
            2 * int(result["n_psf"])
            * len(result["config"]["kmins"])
        )
        physical = bool(
            not candidate_rows.empty
            and (candidate_rows["P_fluctuation"] > 0).all()
            and (candidate_rows["n_psf"] == int(result["n_psf"])).all()
            and len(candidate_psf_rows) == expected_psf_rows
            and np.isfinite(candidate_psf_rows["P_fluctuation"]).all()
            and (candidate_psf_rows["P_fluctuation"] > 0).all()
        )
        candidate_k = combined[combined["branch"].eq(candidate)]
        k_span = float(
            candidate_k["mbar_weighted"].max()
            - candidate_k["mbar_weighted"].min()
        )
        candidate_clip = clipping[clipping["branch"].eq(candidate)].set_index("ring")
        inner_fraction = float(candidate_clip.loc["inner", "changed_fraction"])
        outer_fraction = float(candidate_clip.loc["outer", "changed_fraction"])
        closure_ok = bool(result.get("closure_passed", False))
        status = (
            "invalid_production_closure" if not closure_ok
            else "invalid_candidate_fit" if not physical
            else "ready_for_scientific_comparison"
        )
        rows.append({
            "galaxy": result["galaxy"],
            "status": status,
            "production_closure": closure_ok,
            "candidate_physical_fit": physical,
            "mbar_no_winsor": float(by_branch.loc["no_winsor", "mbar_weighted"]),
            "mbar_raw_global_3p5": float(by_branch.loc["raw_global_3p5", "mbar_weighted"]),
            "mbar_normalized_full_3p5": float(
                by_branch.loc["normalized_full_3p5", "mbar_weighted"]
            ),
            "mbar_candidate": float(by_branch.loc[candidate, "mbar_weighted"]),
            "candidate_minus_no_winsor": float(
                by_branch.loc[candidate, "mbar_weighted"]
                - by_branch.loc["no_winsor", "mbar_weighted"]
            ),
            "candidate_minus_raw_global_3p5": float(
                by_branch.loc[candidate, "mbar_weighted"]
                - by_branch.loc["raw_global_3p5", "mbar_weighted"]
            ),
            "normalized_full_minus_raw_global_3p5": float(
                by_branch.loc["normalized_full_3p5", "mbar_weighted"]
                - by_branch.loc["raw_global_3p5", "mbar_weighted"]
            ),
            "annulus_abs_no_winsor": abs(float(
                by_branch.loc["no_winsor", "annulus_difference"]
            )),
            "annulus_abs_raw_global_3p5": abs(float(
                by_branch.loc["raw_global_3p5", "annulus_difference"]
            )),
            "annulus_abs_candidate": abs(float(
                by_branch.loc[candidate, "annulus_difference"]
            )),
            "candidate_k_window_span": k_span,
            "candidate_changed_inner": inner_fraction,
            "candidate_changed_outer": outer_fraction,
            "candidate_changed_abs_difference": abs(
                inner_fraction - outer_fraction
            ),
        })
    frame = pd.DataFrame(rows)
    config_keys = {
        result.get("config_key") for result in results
        if result.get("status") == "ok"
    }
    config_keys.discard(None)
    if len(config_keys) > 1:
        raise ValueError("Нельзя смешивать разные config_key в одной сводке")
    output_path = (
        Path(output_root).resolve() / "batch" / "aggregates"
        / "evaluation.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def write_aggregate_tables(
    results: Iterable[dict[str, Any]], output_root: str | Path
) -> dict[str, str]:
    results = list(results)
    output_root = Path(output_root).resolve()
    config_keys = {
        result.get("config_key") for result in results
        if result.get("status") == "ok"
    }
    config_keys.discard(None)
    if len(config_keys) > 1:
        raise ValueError("Нельзя смешивать разные config_key в одной сводке")
    batch_root = output_root / "batch" / "aggregates"
    batch_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table_name in [
        "fit_summary", "clipping", "production_closure", "combined_annuli"
    ]:
        frames = [
            load_result_tables(result)[table_name]
            for result in results if result.get("status") == "ok"
        ]
        if not frames:
            continue
        path = batch_root / f"all_galaxies_{table_name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        paths[table_name] = str(path)
    return paths


def load_matching_results(
    output_root: str | Path, config: ExperimentConfig,
    source_batch_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Загружает все готовые цели данной конфигурации для устойчивой сводки."""

    key = experiment_config_key(config)
    result_root = Path(output_root).resolve() / "batch" / "results"
    results = []
    for path in sorted(result_root.glob("NGC_*_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        source_is_current = True
        if source_batch_root is not None and result.get("galaxy"):
            try:
                current_source = inspect_source(
                    result["galaxy"], source_batch_root
                )
                source_is_current = (
                    current_source["source_key"] == result.get("source_key")
                )
            except Exception:
                source_is_current = False
        if (
            result.get("status") == "ok"
            and result.get("config_key") == key
            and source_is_current
            and _result_artifacts_valid(result)
        ):
            results.append(result)
    return results


def plot_normalized_inputs(
    result: dict[str, Any], branch: str | None = None
):
    """Shows the full normalized residual used by the candidate branch."""

    import matplotlib.pyplot as plt

    branch = result["candidate_branch"] if branch is None else branch
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")

    if branch == result["candidate_branch"]:
        path = Path(result["full_normalized_residual_fits"])
        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data
            sample = np.array(
                data[::4, ::4], dtype=np.float32, copy=True
            )
            values = sample[np.isfinite(sample)]
            if values.size == 0:
                raise RuntimeError(f"{path}: no finite pixels")
            limit = float(np.nanpercentile(np.abs(values), 99.5))
            fig, axis = plt.subplots(
                1, 1, figsize=(12, 6), constrained_layout=True
            )
            axis.imshow(
                sample, origin="lower", cmap=cmap,
                vmin=-limit, vmax=limit, interpolation="nearest",
            )
        axis.set_title(
            f"{result['galaxy']}: normalized and winsorized residual"
        )
        axis.set_axis_off()
        return fig

    selected = {
        item["ring"]: Path(item["path"])
        for item in result["normalized_fits"]
        if item["branch"] == branch
    }
    if not set(REGION_NAMES).issubset(selected):
        raise FileNotFoundError(
            f"Ring FITS for {branch} were not saved; enable "
            "save_ring_fft_fits=True for the candidate or "
            "save_all_branch_fits=True for every branch"
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, ring in zip(axes, ["inner", "outer"]):
        data = np.asarray(fits.getdata(selected[ring]), dtype=float)
        values = data[np.isfinite(data)]
        limit = float(np.nanpercentile(np.abs(values), 99.5))
        axis.imshow(data, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        axis.set_title(f"{ring} ring")
        axis.set_axis_off()
    fig.suptitle(f"{result['galaxy']}: {branch}")
    return fig


def plot_power_comparison(result: dict[str, Any], kmin: float = 0.04):
    """Сравнивает три ветви при неизменных PSF, маске и k-окне."""

    import matplotlib.pyplot as plt

    tables = load_result_tables(result)
    power = tables["power_spectra"]
    summary = tables["fit_summary"]
    colors = {
        "no_winsor": "black",
        "raw_global_3p5": "#d95f02",
        "normalized_full_3p5": "#7570b3",
        result["candidate_branch"]: "#1b9e77",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, ring in zip(axes, ["inner", "outer"]):
        for branch, color in colors.items():
            spectrum = power[
                power["ring"].eq(ring) & power["branch"].eq(branch)
            ]
            fit = summary[
                summary["ring"].eq(ring)
                & summary["branch"].eq(branch)
                & np.isclose(summary["requested_kmin"], kmin)
            ].iloc[0]
            axis.plot(
                spectrum["k"], spectrum["Pk"], ".", ms=3,
                color=color, alpha=0.65,
            )
            model = fit["P0"] * spectrum["E_median"] + fit["P1"]
            axis.plot(spectrum["k"], model, color=color, label=branch)
        axis.set_xlim(kmin, float(result["config"]["kmax"]))
        axis.set_xlabel(r"$k$ (pixel$^{-1}$)")
        axis.set_ylabel(r"$P(k)$")
        axis.set_title(f"{ring} ring")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(result["galaxy"])
    return fig
