#!/usr/bin/env python3
import atexit
import argparse
import builtins
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from astropy.io import fits


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "sbf-2.ipynb"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_BATCH_ROOT = SCRIPT_DIR / "sbf2_batch_outputs"
DEFAULT_TARGET_CSV = SCRIPT_DIR / "targets_go3055_manifest.csv"
MAST_DOWNLOAD_PREFIX = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
DEFAULT_SIGNAL_FILTER = "F150W"
DEFAULT_COLOR_FILTER = "F090W"
CURRENT_NOTEBOOK_FILTER_PAIR = (DEFAULT_SIGNAL_FILTER, DEFAULT_COLOR_FILTER)
SBF2_NOTEBOOK_FAMILY = "sbf2"
SBF3_NOTEBOOK_FAMILY = "sbf3"
MAX_IMPLICIT_TARGETS = 14


TARGETS = [
    {
        "name": "NGC 1380",
        "f150w": "jw03055-o001_t001_nircam_clear-f150w_i2d.fits",
        "f090w": "jw03055-o001_t001_nircam_clear-f090w_i2d.fits",
        "f150w_size": 1210685760,
        "f090w_size": 1210423680,
    },
    {
        "name": "NGC 1404",
        "f150w": "jw03055-o003_t003_nircam_clear-f150w_i2d.fits",
        "f090w": "jw03055-o003_t003_nircam_clear-f090w_i2d.fits",
        "f150w_size": 1216696320,
        "f090w_size": 1216696320,
    },
]


def urlquote(value):
    from urllib.parse import quote

    return quote(value, safe="")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def as_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    return value


def bytes_gb(value):
    try:
        return float(value) / 1024**3
    except Exception:
        return float("nan")


def disk_stats(path):
    usage = shutil.disk_usage(path)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "total_gb": bytes_gb(usage.total),
        "used_gb": bytes_gb(usage.used),
        "free_gb": bytes_gb(usage.free),
    }


def memory_stats():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total": int(vm.total),
            "available": int(vm.available),
            "used": int(vm.used),
            "percent": float(vm.percent),
            "total_gb": bytes_gb(vm.total),
            "available_gb": bytes_gb(vm.available),
            "used_gb": bytes_gb(vm.used),
        }
    except Exception:
        if sys.platform == "darwin":
            try:
                pages = {}
                vm_stat = subprocess.check_output(["vm_stat"], text=True)
                page_size = 4096
                for line in vm_stat.splitlines():
                    if "page size of" in line:
                        parts = line.split("page size of", 1)[1].split("bytes", 1)[0]
                        page_size = int(parts.strip())
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    pages[key.strip()] = int(value.strip().strip(".").replace(".", ""))
                free_pages = pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
                available = free_pages * page_size
                return {
                    "available": available,
                    "available_gb": bytes_gb(available),
                }
            except Exception:
                pass
        return {}


def log_resources(label, data_root):
    disk = disk_stats(data_root)
    mem = memory_stats()
    mem_text = "unknown"
    if mem:
        if "available_gb" in mem:
            mem_text = f"available={mem['available_gb']:.1f} GB"
        if "used_gb" in mem and "total_gb" in mem:
            mem_text = (
                f"used={mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB, "
                f"available={mem.get('available_gb', float('nan')):.1f} GB"
            )
    print(
        f"[{timestamp()}] [RESOURCE] {label}: "
        f"disk_free={disk['free_gb']:.1f}/{disk['total_gb']:.1f} GB, RAM {mem_text}"
    )
    return disk, mem


def fits_is_readable(path):
    try:
        with fits.open(path, memmap=True) as hdul:
            _ = hdul[0].header
            if "SCI" in hdul:
                _ = hdul["SCI"].header
        return True, ""
    except Exception as exc:
        return False, str(exc)


def wait_for_input(path, expected_size=None, poll_seconds=60, timeout_seconds=0):
    path = Path(path)
    start = time.time()
    last_size = None
    stable_count = 0

    while True:
        elapsed = time.time() - start
        if timeout_seconds and elapsed > timeout_seconds:
            raise TimeoutError(f"timeout waiting for {path}")

        if not path.exists():
            print(f"[{timestamp()}] waiting for {path} (missing)")
            time.sleep(poll_seconds)
            continue

        size = path.stat().st_size
        if expected_size and size != expected_size:
            readable, read_error = fits_is_readable(path)
            nearly_complete = size >= int(0.995 * expected_size)
            if readable and nearly_complete:
                print(
                    f"[{timestamp()}] input ready with size warning: {path} "
                    f"({size}/{expected_size} bytes)"
                )
                return path
            pct = 100.0 * size / expected_size if expected_size else 0.0
            print(
                f"[{timestamp()}] waiting for {path.name}: "
                f"{size}/{expected_size} bytes ({pct:.1f}%)"
            )
            if read_error and pct > 95.0:
                print(f"[{timestamp()}] FITS read check: {read_error}")
            time.sleep(poll_seconds)
            continue

        if not expected_size:
            if size == last_size:
                stable_count += 1
            else:
                stable_count = 0
            last_size = size
            if stable_count < 2:
                print(f"[{timestamp()}] waiting for stable size {path.name}: {size} bytes")
                time.sleep(poll_seconds)
                continue

        readable, read_error = fits_is_readable(path)
        if not readable:
            print(f"[{timestamp()}] waiting for readable FITS {path.name}: {read_error}")
            time.sleep(poll_seconds)
            continue

        print(f"[{timestamp()}] input ready: {path} ({size} bytes)")
        return path


def is_input_ready(path, expected_size=None):
    path = Path(path)
    if not path.exists():
        return False
    size = path.stat().st_size
    if expected_size and size != expected_size:
        readable, _ = fits_is_readable(path)
        return readable and size >= int(0.995 * expected_size)
    readable, _ = fits_is_readable(path)
    return readable


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def notebook_family(template_path):
    """Identify a known SBF notebook from its metadata or stable content."""
    try:
        notebook = json.loads(Path(template_path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    metadata = notebook.get("metadata") or {}
    declared = metadata.get("sbf_pipeline_family")
    pipeline_metadata = metadata.get("sbf_pipeline")
    if declared is None and isinstance(pipeline_metadata, dict):
        declared = pipeline_metadata.get("family")
    if declared is not None:
        normalized = str(declared).strip().lower().replace("_", "-")
        if normalized in {"sbf2", "sbf-2", "2"}:
            return SBF2_NOTEBOOK_FAMILY
        if normalized in {"sbf3", "sbf-3", "3"}:
            return SBF3_NOTEBOOK_FAMILY

    code_sources = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                heading = line.strip().lower()
                if heading.startswith("# sbf-2"):
                    return SBF2_NOTEBOOK_FAMILY
                if heading.startswith("# sbf-3"):
                    return SBF3_NOTEBOOK_FAMILY
        elif cell.get("cell_type") == "code":
            code_sources.append(source)

    code = "\n".join(code_sources)
    if 'output_header["SBFVER"] = ("sbf-3"' in code:
        return SBF3_NOTEBOOK_FAMILY
    if "f150w_path = Path" in code and "f090w_path = Path" in code:
        return SBF2_NOTEBOOK_FAMILY
    return None


def input_fingerprint(path):
    """Return a cheap local identity without reading a potentially huge FITS."""
    if path is None:
        return {
            "resolved_path": None,
            "size": None,
            "mtime_ns": None,
            "device": None,
            "inode": None,
        }
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {
            "resolved_path": str(resolved),
            "size": None,
            "mtime_ns": None,
            "device": None,
            "inode": None,
        }
    return {
        "resolved_path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def input_pair_key(
    signal_path,
    color_path,
    signal_fingerprint=None,
    color_fingerprint=None,
):
    signal_fingerprint = signal_fingerprint or input_fingerprint(signal_path)
    color_fingerprint = color_fingerprint or input_fingerprint(color_path)
    payload = json.dumps(
        {
            "signal": signal_fingerprint,
            "color": color_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def target_output_dir(
    galaxy,
    signal_path,
    color_path=None,
    products_root=None,
    signal_filter=None,
    color_filter=None,
    signal_fingerprint=None,
    color_fingerprint=None,
):
    if products_root is None:
        return Path(signal_path).resolve().parent
    pair_label = f"{signal_filter or 'signal'}__{color_filter or 'color'}"
    pair_key = input_pair_key(
        signal_path,
        color_path,
        signal_fingerprint=signal_fingerprint,
        color_fingerprint=color_fingerprint,
    )
    return (
        Path(products_root).resolve()
        / slug(galaxy)
        / slug(pair_label)
        / pair_key
    )


def result_json_path(batch_root, galaxy, identity=None):
    if identity and identity.get("template_family") == SBF3_NOTEBOOK_FAMILY:
        run_label = "__".join(
            [
                slug(galaxy),
                slug(identity["signal_filter"]),
                slug(identity["color_filter"]),
                identity["input_pair_key"],
            ]
        )
        return Path(batch_root) / f"{run_label}_result.json"
    return Path(batch_root) / f"{slug(galaxy)}_result.json"


def expected_run_identity(
    template_path,
    target,
    signal_path,
    color_path,
    products_root=None,
):
    template_path = Path(template_path).resolve()
    signal_path = Path(signal_path).resolve()
    color_path = Path(color_path).resolve()
    signal_fingerprint = input_fingerprint(signal_path)
    color_fingerprint = input_fingerprint(color_path)
    pair_key = input_pair_key(
        signal_path,
        color_path,
        signal_fingerprint=signal_fingerprint,
        color_fingerprint=color_fingerprint,
    )
    return {
        "template_name": template_path.name,
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "template_family": notebook_family(template_path),
        "signal_filter": target["signal_filter"],
        "color_filter": target["color_filter"],
        "signal_path": str(signal_path),
        "color_path": str(color_path),
        "signal_fingerprint": signal_fingerprint,
        "color_fingerprint": color_fingerprint,
        "input_pair_key": pair_key,
        "out_dir": str(
            target_output_dir(
                target["name"],
                signal_path,
                color_path,
                products_root=products_root,
                signal_filter=target["signal_filter"],
                color_filter=target["color_filter"],
                signal_fingerprint=signal_fingerprint,
                color_fingerprint=color_fingerprint,
            )
        ),
    }


def result_matches_identity(result, identity):
    return all(result.get(key) == value for key, value in identity.items())


def final_result_for(target, batch_root, identity=None, allow_legacy=False):
    path = result_json_path(batch_root, target["name"], identity=identity)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except Exception:
        return None
    if result.get("status") != "ok":
        return None
    if identity is not None and not result_matches_identity(result, identity):
        if not allow_legacy:
            return None
        if result.get("template_sha256") is not None:
            return None
    return result


def f150_to_f090_filename(filename):
    lower = filename.lower()
    if "f150w" not in lower:
        raise ValueError(f"cannot derive F090W filename from {filename}")
    idx = lower.index("f150w")
    return filename[:idx] + "f090w" + filename[idx + len("f150w") :]


def product_download_url(filename):
    return MAST_DOWNLOAD_PREFIX + urlquote(f"mast:JWST/product/{filename}")


def product_uri_download_url(uri, filename):
    value = str(uri or "").strip()
    if not value:
        return product_download_url(filename)
    if value.startswith(("https://", "http://")):
        return value
    if value.startswith("mast:"):
        return MAST_DOWNLOAD_PREFIX + urlquote(value)
    raise ValueError(f"unsupported product URI for {filename}: {value!r}")


def optional_int(value):
    value = str(value or "").strip()
    return int(value) if value else None


def manifest_row_enabled(row):
    """Return whether a manifest row is actionable by the downloader.

    Existing manifests predate this flag and therefore remain enabled by
    default.  Extended inventories can keep proprietary, scheduled, or
    demonstration-only targets in the same CSV with ``download_enabled=false``
    without making the batch runner attempt nonexistent or embargoed products.
    """
    value = str(row.get("download_enabled") or "").strip().lower()
    if not value:
        return True
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    target = str(row.get("target") or row.get("galaxy") or "<unknown>").strip()
    raise ValueError(
        f"{target}: invalid download_enabled={row.get('download_enabled')!r}"
    )


def normalize_target(target):
    """Return the generic two-filter target contract.

    Legacy F150W/F090W keys remain accepted so old manifests and the small
    built-in fallback target list stay reproducible.
    """
    item = dict(target)
    item["name"] = str(item.get("name") or item.get("target") or "").strip()
    item["signal_filter"] = str(
        item.get("signal_filter") or DEFAULT_SIGNAL_FILTER
    ).strip().upper()
    item["color_filter"] = str(
        item.get("color_filter") or DEFAULT_COLOR_FILTER
    ).strip().upper()
    item["signal_product"] = str(
        item.get("signal_product") or item.get("f150w") or ""
    ).strip()
    item["color_product"] = str(
        item.get("color_product") or item.get("f090w") or ""
    ).strip()
    item["signal_url"] = item.get("signal_url") or item.get("f150w_url")
    item["color_url"] = item.get("color_url") or item.get("f090w_url")
    item["signal_size"] = item.get("signal_size", item.get("f150w_size"))
    item["color_size"] = item.get("color_size", item.get("f090w_size"))
    if not item["name"]:
        raise ValueError("target has no name")
    if not item["signal_product"] or not item["color_product"]:
        raise ValueError(f"{item['name']}: signal_product and color_product are required")
    return item


def validate_notebook_filter_pair(template_path, signal_filter, color_filter):
    pair = (signal_filter.strip().upper(), color_filter.strip().upper())
    family = notebook_family(template_path)
    if family == SBF2_NOTEBOOK_FAMILY and pair != CURRENT_NOTEBOOK_FILTER_PAIR:
        expected = "/".join(CURRENT_NOTEBOOK_FILTER_PAIR)
        actual = "/".join(pair)
        raise ValueError(
            f"{Path(template_path).name} is an sbf-2 notebook and is still "
            f"validated only for {expected}; got {actual}. "
            "Use the filter-aware sbf-3.ipynb for other pairs."
        )


def validate_run_layout(template_path, batch_root, products_root):
    if notebook_family(template_path) != SBF3_NOTEBOOK_FAMILY:
        return
    if products_root is None:
        raise ValueError(
            "sbf-3.ipynb requires --products-root so it cannot overwrite "
            "the frozen sbf-2 products"
        )
    if Path(batch_root).resolve() == DEFAULT_BATCH_ROOT.resolve():
        raise ValueError(
            "sbf-3.ipynb requires a separate --batch-root; refusing to mix "
            "sbf-2 and sbf-3 result JSON files"
        )


def read_targets_from_csv(csv_path, data_root):
    rows = []
    with Path(csv_path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not manifest_row_enabled(row):
                continue
            is_generic = "signal_product" in row
            galaxy = str(row.get("target") or row.get("galaxy") or "").strip()
            if is_generic:
                signal_product = str(row.get("signal_product") or "").strip()
                color_product = str(row.get("color_product") or "").strip()
                if not galaxy or not signal_product or not color_product:
                    continue
                signal_url = product_uri_download_url(
                    row.get("signal_product_uri"), signal_product
                )
                color_url = product_uri_download_url(
                    row.get("color_product_uri"), color_product
                )
                signal_size = optional_int(row.get("signal_content_length_bytes"))
                color_size = optional_int(row.get("color_content_length_bytes"))
                signal_filter = row.get("signal_filter") or DEFAULT_SIGNAL_FILTER
                color_filter = row.get("color_filter") or DEFAULT_COLOR_FILTER
            else:
                signal_product = str(row.get("expected_f150w_i2d") or "").strip()
                if not galaxy or not signal_product:
                    continue
                color_product = f150_to_f090_filename(signal_product)
                signal_url = row.get("mast_download_url") or product_download_url(
                    signal_product
                )
                color_url = product_download_url(color_product)
                signal_size = optional_int(row.get("f150w_content_length_bytes"))
                color_size = None
                signal_filter = DEFAULT_SIGNAL_FILTER
                color_filter = DEFAULT_COLOR_FILTER
            target_dir = Path(data_root) / galaxy
            rows.append(normalize_target({
                    "name": galaxy,
                    "program": row.get("program") or row.get("jwst_program"),
                    "obsid": row.get("obsid"),
                    "signal_filter": signal_filter,
                    "color_filter": color_filter,
                    "signal_product": signal_product,
                    "color_product": color_product,
                    "signal_url": signal_url,
                    "color_url": color_url,
                    "signal_size": signal_size,
                    "color_size": color_size,
                    "signal_exposure_time_s": row.get("signal_exposure_time_s"),
                    "color_exposure_time_s": row.get("color_exposure_time_s"),
                    "morphology": row.get("morphology"),
                    "redshift": row.get("redshift"),
                    "external_distance_modulus": row.get(
                        "external_distance_modulus"
                    ),
                    "distance_method": row.get("distance_method"),
                    "hst_overlap": row.get("hst_overlap"),
                    "dust_flag": row.get("dust_flag"),
                    "spiral_flag": row.get("spiral_flag"),
                    "resolved_flag": row.get("resolved_flag"),
                    "field_contamination_flag": row.get(
                        "field_contamination_flag"
                    ),
                    "calibration_family": row.get("calibration_family"),
                    "availability_status": row.get("availability_status"),
                    "science_role": row.get("science_role"),
                    "priority": row.get("priority"),
                    "public_release_date": row.get("public_release_date"),
                    "notes": row.get("notes"),
                    "target_dir": str(target_dir),
                    "source_csv": str(csv_path),
                }))
    return rows


def merge_known_targets(targets):
    known = {target["name"]: normalize_target(target) for target in TARGETS}
    merged = []
    for target in targets:
        item = normalize_target(target)
        if item["name"] in known:
            known_item = known[item["name"]]
            for key, value in known_item.items():
                if item.get(key) in (None, ""):
                    item[key] = value
            for role in ("signal", "color"):
                size_key = f"{role}_size"
                product_key = f"{role}_product"
                if (
                    item.get(size_key) in (None, "")
                    or item.get(product_key) == known_item.get(product_key)
                ):
                    item[size_key] = known_item.get(size_key, item.get(size_key))
        merged.append(item)
    return merged


def local_target_files(target, data_root):
    target = normalize_target(target)
    root = Path(data_root) / target["name"]
    return {
        "signal": root / target["signal_product"],
        "color": root / target["color_product"],
    }


def target_paths(target, data_root):
    files = local_target_files(target, data_root)
    return files["signal"], files["color"]


def notebook_code_cells(template_path):
    data = json.loads(Path(template_path).read_text())
    cells = []
    for cell_no, cell in enumerate(data["cells"], start=1):
        if cell.get("cell_type") == "code":
            cells.append((cell_no, "".join(cell.get("source", []))))
    return cells


def make_display(namespace):
    def display(obj):
        printer = namespace.get("print", builtins.print)
        if hasattr(obj, "to_string"):
            printer("\n" + obj.to_string())
        else:
            printer(repr(obj))

    return display


def override_target_namespace(
    namespace,
    galaxy,
    signal_path,
    color_path,
    signal_filter=DEFAULT_SIGNAL_FILTER,
    color_filter=DEFAULT_COLOR_FILTER,
    out_dir=None,
):
    signal_path = Path(signal_path).resolve()
    color_path = Path(color_path).resolve()
    namespace["TARGET_GALAXY"] = galaxy
    namespace["SIGNAL_FILTER"] = signal_filter
    namespace["COLOR_FILTER"] = color_filter
    namespace["signal_path"] = signal_path
    namespace["color_path"] = color_path
    # Compatibility bridge for the current notebook. Its numerical operations
    # are filter-agnostic even though these two variable names are historical.
    namespace["f150w_path"] = signal_path
    namespace["f090w_path"] = color_path
    namespace["out_dir"] = (
        Path(out_dir).resolve() if out_dir is not None else signal_path.parent
    )
    namespace["stem"] = signal_path.stem
    namespace["out_dir"].mkdir(parents=True, exist_ok=True)


def result_paths(out_dir, stem, pipeline_label="sbf2"):
    out_dir = Path(out_dir)
    return {
        "model_full_fits": out_dir / f"{stem}_sbf_model_full.fits",
        "science_residual_fits": out_dir / f"{stem}_sbf_resid_full_science.fits",
        "science_residual_raw_fits": out_dir / f"{stem}_sbf_resid_full_science_raw.fits",
        "inner_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_inner_lit_usable.fits",
        "outer_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_outer_lit_usable.fits",
        "df_sbf_csv": out_dir / f"{stem}_{pipeline_label}_df_sbf.csv",
        "annulus_summary_csv": out_dir
        / f"{stem}_{pipeline_label}_annulus_summary.csv",
    }


def execute_template_for_target(
    template_path,
    galaxy,
    signal_path,
    color_path,
    batch_root,
    signal_filter=DEFAULT_SIGNAL_FILTER,
    color_filter=DEFAULT_COLOR_FILTER,
    output_dir=None,
):
    template_path = Path(template_path).resolve()
    pipeline_label = template_path.stem.replace("-", "").replace("_", "")
    signal_fingerprint = input_fingerprint(signal_path)
    color_fingerprint = input_fingerprint(color_path)
    pair_key = input_pair_key(
        signal_path,
        color_path,
        signal_fingerprint=signal_fingerprint,
        color_fingerprint=color_fingerprint,
    )
    namespace = {
        "__name__": "__sbf_notebook_exec__",
        "__file__": str(template_path),
    }
    override_target_namespace(
        namespace,
        galaxy,
        signal_path,
        color_path,
        signal_filter=signal_filter,
        color_filter=color_filter,
        out_dir=output_dir,
    )
    namespace["display"] = make_display(namespace)
    code_cells = notebook_code_cells(template_path)

    for cell_no, source in code_cells:
        print(f"[{timestamp()}] executing {template_path.name} cell {cell_no}")
        try:
            exec(compile(source, f"{template_path}:cell-{cell_no}", "exec"), namespace)
        except Exception:
            print(f"[{timestamp()}] failed in {template_path.name} cell {cell_no}")
            raise

        # The frozen sbf-2 parameter cell overwrites injected values. Reapply
        # them only for that legacy cell; sbf-3 preserves preseeded globals.
        if "f150w_path = Path" in source and "f090w_path = Path" in source:
            override_target_namespace(
                namespace,
                galaxy,
                signal_path,
                color_path,
                signal_filter=signal_filter,
                color_filter=color_filter,
                out_dir=output_dir,
            )
            namespace["display"] = make_display(namespace)
            print(f"[{timestamp()}] target override: {galaxy}")
            print(f"[{timestamp()}] {signal_filter} signal -> {namespace['signal_path']}")
            print(f"[{timestamp()}] {color_filter} color -> {namespace['color_path']}")

    recommended = namespace.get("recommended_sbf")
    if not recommended:
        raise RuntimeError(f"{template_path.name} finished without recommended_sbf")

    out_dir = Path(namespace["out_dir"])
    stem = namespace["stem"]
    paths = result_paths(out_dir, stem, pipeline_label=pipeline_label)

    df_sbf = namespace.get("df_sbf")
    if df_sbf is not None:
        df_sbf.to_csv(paths["df_sbf_csv"], index=False)

    df_annulus_summary = namespace.get("df_annulus_summary")
    if df_annulus_summary is not None:
        df_annulus_summary.to_csv(paths["annulus_summary_csv"], index=False)

    result = {
        "galaxy": galaxy,
        "status": "ok",
        "template_name": template_path.name,
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "template_family": notebook_family(template_path),
        "signal_filter": signal_filter,
        "color_filter": color_filter,
        "color_name": f"{color_filter}-{signal_filter}",
        "signal_path": str(Path(signal_path).resolve()),
        "color_path": str(Path(color_path).resolve()),
        "signal_fingerprint": signal_fingerprint,
        "color_fingerprint": color_fingerprint,
        "input_pair_key": pair_key,
        "out_dir": str(out_dir.resolve()),
        "stem": stem,
    }
    namespace_metadata = {
        "input_family": "INPUT_FAMILY",
        "signal_bunit": "signal_bunit",
        "color_bunit": "color_bunit",
        "color_sampling_mode": "color_sampling_mode",
        "color_grid_aligned": "color_grid_aligned",
        "color_grid_max_offset_px": "color_grid_max_offset_px",
        "signal_background_scalar": "bg_scalar",
        "color_background_scalar": "color_bg_scalar",
        "science_pixel_scale_arcsec": "science_pixel_scale_arcsec",
        "psf_pixel_scale_arcsec": "psf_pixel_scale_arcsec",
        "psf_scale_rel_error": "psf_scale_rel_error",
        "opd_delta_days": "opd_delta_days",
    }
    for result_key, namespace_key in namespace_metadata.items():
        if namespace_key in namespace:
            result[result_key] = as_builtin(namespace[namespace_key])
    if signal_filter == "F150W" and color_filter == "F090W":
        result["f150w_path"] = result["signal_path"]
        result["f090w_path"] = result["color_path"]
    for key, value in as_builtin(recommended).items():
        result[f"recommended_{key}"] = value
    for key, value in paths.items():
        result[key] = str(value.resolve())
        result[f"{key}_exists"] = Path(value).exists()

    color_summary = namespace.get("df_color_summary")
    if color_summary is not None and len(color_summary) > 0:
        try:
            row0 = color_summary.iloc[0].to_dict()
            color_value = as_builtin(
                row0.get("color_index", row0.get("color_F090W_F150W"))
            )
            result["color_index"] = color_value
            result["color_name"] = row0.get("color_name", result["color_name"])
            result[f"color_{color_filter}_{signal_filter}"] = color_value
            result["color_sigma_proxy"] = as_builtin(row0.get("sigma_proxy"))
        except Exception:
            pass

    batch_root.mkdir(parents=True, exist_ok=True)
    result_json = result_json_path(batch_root, galaxy, identity=result)
    result_json.write_text(json.dumps(as_builtin(result), ensure_ascii=False, indent=2))
    print(f"[{timestamp()}] wrote result {result_json}")
    return result


def run_worker(args):
    template_path = Path(args.template).resolve()
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    runtime_cache = batch_root / ".runtime_cache"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache / "xdg"))
    signal_filter = args.signal_filter.strip().upper()
    color_filter = args.color_filter.strip().upper()
    worker_target = {
        "name": args.galaxy,
        "signal_filter": signal_filter,
        "color_filter": color_filter,
    }
    worker_identity = expected_run_identity(
        template_path,
        worker_target,
        args.signal,
        args.color,
        products_root=args.products_root,
    )
    log_path = result_json_path(
        batch_root, args.galaxy, identity=worker_identity
    ).with_suffix(".log")
    output_dir = target_output_dir(
        args.galaxy,
        args.signal,
        args.color,
        products_root=args.products_root,
        signal_filter=signal_filter,
        color_filter=color_filter,
        signal_fingerprint=worker_identity["signal_fingerprint"],
        color_fingerprint=worker_identity["color_fingerprint"],
    )

    with log_path.open("a") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[{timestamp()}] worker start: {args.galaxy}")
            try:
                validate_run_layout(
                    template_path, batch_root, args.products_root
                )
                validate_notebook_filter_pair(
                    template_path, signal_filter, color_filter
                )
                result = execute_template_for_target(
                    template_path,
                    args.galaxy,
                    Path(args.signal).resolve(),
                    Path(args.color).resolve(),
                    batch_root,
                    signal_filter=signal_filter,
                    color_filter=color_filter,
                    output_dir=output_dir,
                )
                print(
                    f"[{timestamp()}] worker done: {args.galaxy} "
                    f"mbar={result.get('recommended_mbar_weighted')} "
                    f"sigma={result.get('recommended_sigma_adopted')}"
                )
                return 0
            except Exception as exc:
                err = {
                    "galaxy": args.galaxy,
                    "status": "failed",
                    **worker_identity,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                err_path = result_json_path(
                    batch_root, args.galaxy, identity=err
                )
                err_path.write_text(json.dumps(err, ensure_ascii=False, indent=2))
                print(err["traceback"])
                print(f"[{timestamp()}] worker failed: {args.galaxy}")
                return 1


def write_summary(results, batch_root):
    batch_root = Path(batch_root)
    csv_path = batch_root / "sbf2_batch_results.csv"
    json_path = batch_root / "sbf2_batch_results.json"
    json_path.write_text(json.dumps(as_builtin(results), ensure_ascii=False, indent=2))

    keys = []
    for result in results:
        for key in result:
            if key not in keys:
                keys.append(key)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in keys})

    print(f"[{timestamp()}] summary CSV  -> {csv_path}")
    print(f"[{timestamp()}] summary JSON -> {json_path}")
    return csv_path, json_path


def link_residuals(results, batch_root):
    residual_dir = Path(batch_root) / "residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.get("status") != "ok":
            continue
        galaxy_slug = slug(result["galaxy"])
        for key in [
            "science_residual_fits",
            "inner_usable_residual_fits",
            "outer_usable_residual_fits",
        ]:
            src = Path(result.get(key, ""))
            if not src.exists():
                continue
            dst = residual_dir / f"{galaxy_slug}_{key}.fits"
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
            except Exception:
                pass


def download_one(url, dest, expected_size=None, chunk_size=1024 * 1024, timeout=120):
    import urllib.error
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if is_input_ready(dest, expected_size):
        return True, "already-ready"

    headers = {"Accept-Encoding": "identity"}
    start = dest.stat().st_size if dest.exists() else 0
    if start:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if start and response.status == 200:
                start = 0
                dest.unlink(missing_ok=True)
            mode = "ab" if start else "wb"
            with dest.open(mode) as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and is_input_ready(dest, expected_size):
            return True, "range-complete"
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, repr(exc)

    if is_input_ready(dest, expected_size):
        return True, "downloaded"
    size = dest.stat().st_size if dest.exists() else 0
    return False, f"incomplete after transfer: {size}/{expected_size}"


def ensure_disk_space_for_downloads(
    data_root,
    completed_results,
    min_free_gb,
    cleanup_enabled=True,
    required_bytes=0,
):
    disk, _ = log_resources("disk-check", data_root)
    required_free_gb = min_free_gb + bytes_gb(required_bytes or 0)
    if disk["free_gb"] >= required_free_gb:
        return True
    if not cleanup_enabled:
        print(
            f"[{timestamp()}] [DISK] download blocked: free space "
            f"{disk['free_gb']:.1f} GB is below the required "
            f"{required_free_gb:.1f} GB (reserve + next transfer); "
            "cleanup disabled"
        )
        return False

    print(
        f"[{timestamp()}] [DISK] free space below requirement "
        f"({disk['free_gb']:.1f} < {required_free_gb:.1f} GB), removing source inputs "
        "for completed galaxies"
    )
    for result in completed_results:
        if result.get("status") != "ok":
            continue
        input_paths = [
            result.get("signal_path") or result.get("f150w_path"),
            result.get("color_path") or result.get("f090w_path"),
        ]
        for value in input_paths:
            path = Path(value or "")
            if not path.exists():
                continue
            try:
                size_gb = bytes_gb(path.stat().st_size)
                path.unlink()
                print(f"[{timestamp()}] [DISK] removed {path} ({size_gb:.2f} GB)")
            except Exception as exc:
                print(f"[{timestamp()}] [DISK] failed to remove {path}: {exc}")
        disk, _ = log_resources("disk-check-after-cleanup", data_root)
        if disk["free_gb"] >= required_free_gb:
            return True

    print(
        f"[{timestamp()}] [DISK] download blocked: free space "
        f"{disk['free_gb']:.1f} GB is below the required "
        f"{required_free_gb:.1f} GB after cleanup"
    )
    return False


def load_completed_results(batch_root):
    completed = []
    for result_file in sorted(Path(batch_root).glob("*_result.json")):
        try:
            result = json.loads(result_file.read_text())
        except Exception:
            continue
        if result.get("status") == "ok":
            completed.append(result)
    return completed


def download_targets_until_stopped(
    targets,
    data_root,
    batch_root,
    completed_results,
    min_free_gb,
    cleanup_enabled,
    retry_sleep,
    stop_when_all_ready=False,
):
    status_path = Path(batch_root) / "download_status.json"
    while True:
        all_ready = True
        for target in targets:
            files = local_target_files(target, data_root)
            jobs = [
                (
                    target["signal_filter"],
                    target.get("signal_url"),
                    files["signal"],
                    target.get("signal_size"),
                ),
                (
                    target["color_filter"],
                    target.get("color_url"),
                    files["color"],
                    target.get("color_size"),
                ),
            ]
            for band, url, dest, expected_size in jobs:
                if is_input_ready(dest, expected_size):
                    continue
                all_ready = False
                current_size = dest.stat().st_size if dest.exists() else 0
                remaining_bytes = (
                    max(int(expected_size) - current_size, 0)
                    if expected_size
                    else 0
                )
                while True:
                    completed_results = load_completed_results(batch_root)
                    if ensure_disk_space_for_downloads(
                        data_root,
                        completed_results,
                        min_free_gb=min_free_gb,
                        cleanup_enabled=cleanup_enabled,
                        required_bytes=remaining_bytes,
                    ):
                        break
                    status = {
                        "time": timestamp(),
                        "target": target["name"],
                        "band": band,
                        "path": str(dest),
                        "ok": False,
                        "message": "blocked: insufficient disk space",
                        "size": current_size,
                        "expected_size": expected_size,
                    }
                    status_path.write_text(
                        json.dumps(status, ensure_ascii=False, indent=2)
                    )
                    time.sleep(retry_sleep)
                print(f"[{timestamp()}] [DOWNLOAD] {target['name']} {band} -> {dest}")
                ok, msg = download_one(url, dest, expected_size=expected_size)
                size = dest.stat().st_size if dest.exists() else 0
                status = {
                    "time": timestamp(),
                    "target": target["name"],
                    "band": band,
                    "path": str(dest),
                    "ok": ok,
                    "message": msg,
                    "size": size,
                    "expected_size": expected_size,
                }
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
                if ok:
                    print(f"[{timestamp()}] [DOWNLOAD] ready: {target['name']} {band} ({msg})")
                else:
                    print(
                        f"[{timestamp()}] [DOWNLOAD] not ready: {target['name']} {band}: "
                        f"{msg}; retry later"
                    )
                    time.sleep(retry_sleep)

        if all_ready or stop_when_all_ready:
            if all_ready:
                print(f"[{timestamp()}] [DOWNLOAD] all selected target inputs are ready")
            return all_ready


def start_download_manager(args, targets, completed_results):
    if args.no_download:
        return None
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--download-worker",
        "--target-csv",
        str(Path(args.target_csv).resolve()),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--batch-root",
        str(Path(args.batch_root).resolve()),
        "--download-retry-seconds",
        str(args.download_retry_seconds),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    if args.no_cleanup_inputs:
        cmd.append("--no-cleanup-inputs")
    if targets:
        # The download worker re-reads the manifest in a separate process.
        # Pass the already selected parent target set explicitly; otherwise a
        # narrow ``--galaxies`` run could download every enabled manifest row.
        cmd.append("--galaxies")
        cmd.extend(target["name"] for target in targets)
    print(f"[{timestamp()}] starting download manager: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    atexit.register(stop_download_manager, process)
    return process


def stop_download_manager(process, grace_seconds=10):
    if process is None or process.poll() is not None:
        return
    print(f"[{timestamp()}] stopping download manager")
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def select_targets(targets, galaxies, allow_bulk_targets=False):
    if galaxies is None:
        if len(targets) > MAX_IMPLICIT_TARGETS and not allow_bulk_targets:
            raise RuntimeError(
                f"manifest contains {len(targets)} actionable targets; refusing "
                "implicit bulk selection. Pass exact names with --galaxies. "
                "Use --allow-bulk-targets only after a storage-capacity check."
            )
        return list(targets)
    wanted = set(galaxies)
    return [target for target in targets if target["name"] in wanted]


def run_parent(args):
    template = Path(args.template).resolve()
    data_root = Path(args.data_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    products_root = (
        Path(args.products_root).resolve() if args.products_root else None
    )
    validate_run_layout(template, batch_root, products_root)
    template_family = notebook_family(template)
    batch_root.mkdir(parents=True, exist_ok=True)

    if args.target_csv:
        targets = merge_known_targets(read_targets_from_csv(args.target_csv, data_root))
    else:
        targets = [normalize_target(t) for t in TARGETS]

    targets = select_targets(
        targets,
        args.galaxies,
        allow_bulk_targets=args.allow_bulk_targets,
    )
    if not targets:
        raise RuntimeError(f"no targets selected: {args.galaxies}")

    completed_results = []
    results = []
    identities = {}
    for target in targets:
        signal_path, color_path = target_paths(target, data_root)
        identities[target["name"]] = expected_run_identity(
            template,
            target,
            signal_path,
            color_path,
            products_root=products_root,
        )
        allow_legacy = (
            template_family == SBF2_NOTEBOOK_FAMILY and products_root is None
        )
        existing = final_result_for(
            target,
            batch_root,
            identity=identities[target["name"]],
            allow_legacy=allow_legacy,
        )
        if existing is not None:
            completed_results.append(existing)
            results.append(existing)
            print(f"[{timestamp()}] reusing completed result for {target['name']}")
    download_proc = start_download_manager(args, targets, completed_results)
    for target in targets:
        signal_path, color_path = target_paths(target, data_root)
        identities[target["name"]] = expected_run_identity(
            template,
            target,
            signal_path,
            color_path,
            products_root=products_root,
        )
        allow_legacy = (
            template_family == SBF2_NOTEBOOK_FAMILY and products_root is None
        )
        existing = final_result_for(
            target,
            batch_root,
            identity=identities[target["name"]],
            allow_legacy=allow_legacy,
        )
        if existing is not None:
            if not any(r.get("galaxy") == existing.get("galaxy") for r in results):
                results.append(existing)
            write_summary(results, batch_root)
            link_residuals(results, batch_root)
            continue

        log_resources(f"before-wait {target['name']}", data_root)
        wait_for_input(
            signal_path,
            expected_size=target.get("signal_size"),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        wait_for_input(
            color_path,
            expected_size=target.get("color_size"),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        # Files may have been downloaded after the initial cache check. Capture
        # their real local fingerprints before choosing output/result paths.
        identities[target["name"]] = expected_run_identity(
            template,
            target,
            signal_path,
            color_path,
            products_root=products_root,
        )

        while True:
            _, mem = log_resources(f"before-worker {target['name']}", data_root)
            available_gb = mem.get("available_gb") if mem else None
            if available_gb is None or available_gb >= args.min_available_ram_gb:
                break
            print(
                f"[{timestamp()}] [RESOURCE] waiting for RAM before {target['name']}: "
                f"available={available_gb:.1f} GB < {args.min_available_ram_gb:.1f} GB"
            )
            gc.collect()
            time.sleep(args.poll_seconds)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--galaxy",
            target["name"],
            "--signal",
            str(signal_path),
            "--color",
            str(color_path),
            "--signal-filter",
            target["signal_filter"],
            "--color-filter",
            target["color_filter"],
            "--template",
            str(template),
            "--batch-root",
            str(batch_root),
        ]
        if products_root is not None:
            cmd.extend(["--products-root", str(products_root)])
        print(f"[{timestamp()}] starting worker: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        log_resources(f"after-worker {target['name']}", data_root)

        result_path = result_json_path(
            batch_root,
            target["name"],
            identity=identities[target["name"]],
        )
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "galaxy": target["name"],
                "status": "failed",
                "error": f"worker exited {proc.returncode} without result json",
            }
        if proc.returncode != 0 and result.get("status") == "ok":
            result["status"] = "failed"
            result["error"] = f"worker exited {proc.returncode}"
        results.append(result)
        if result.get("status") == "ok":
            completed_results.append(result)
        write_summary(results, batch_root)
        link_residuals(results, batch_root)
        ensure_disk_space_for_downloads(
            data_root,
            completed_results,
            min_free_gb=args.min_free_gb,
            cleanup_enabled=not args.no_cleanup_inputs,
        )
        gc.collect()

    write_summary(results, batch_root)
    link_residuals(results, batch_root)
    stop_download_manager(download_proc)
    return 0 if all(r.get("status") == "ok" for r in results) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument(
        "--products-root",
        default=None,
        help="isolated per-target directory for notebook-generated products",
    )
    parser.add_argument("--target-csv", default=str(DEFAULT_TARGET_CSV))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--download-retry-seconds", type=int, default=120)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--min-available-ram-gb", type=float, default=8.0)
    parser.add_argument("--galaxies", nargs="*", default=None)
    parser.add_argument(
        "--allow-bulk-targets",
        action="store_true",
        help=(
            "allow an unfiltered manifest with more than "
            f"{MAX_IMPLICIT_TARGETS} actionable targets"
        ),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-cleanup-inputs", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--download-worker", action="store_true")
    parser.add_argument("--galaxy", default=None)
    parser.add_argument(
        "--signal",
        "--f150w",
        dest="signal",
        default=None,
        help="SBF signal image; --f150w is a backwards-compatible alias",
    )
    parser.add_argument(
        "--color",
        "--f090w",
        dest="color",
        default=None,
        help="second image used for color; --f090w is a backwards-compatible alias",
    )
    parser.add_argument("--signal-filter", default=DEFAULT_SIGNAL_FILTER)
    parser.add_argument("--color-filter", default=DEFAULT_COLOR_FILTER)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.download_worker:
        targets = merge_known_targets(read_targets_from_csv(args.target_csv, args.data_root))
        targets = select_targets(
            targets,
            args.galaxies,
            allow_bulk_targets=args.allow_bulk_targets,
        )
        if not targets:
            raise RuntimeError("download worker has no explicitly selected targets")
        completed_results = load_completed_results(args.batch_root)
        download_targets_until_stopped(
            targets,
            Path(args.data_root).resolve(),
            Path(args.batch_root).resolve(),
            completed_results,
            min_free_gb=args.min_free_gb,
            cleanup_enabled=not args.no_cleanup_inputs,
            retry_sleep=args.download_retry_seconds,
        )
        return 0
    if args.worker:
        if not args.galaxy or not args.signal or not args.color:
            raise SystemExit("--worker requires --galaxy, --signal and --color")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
