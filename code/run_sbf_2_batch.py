#!/usr/bin/env python3
"""Offline GO-3055 batch runner for the Jensen-like ``sbf-2.ipynb`` pipeline.

This file is deliberately separate from ``run_sbf_batch.py``.  It accepts only
the 14 GO-3055 F150W/F090W targets, writes products into an isolated run tree,
never downloads or removes science inputs by default, and keeps scientific QC
separate from the mechanical ``done`` state.
"""

import atexit
import argparse
import builtins
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from sbf_campaign_state import CampaignState, canonical_sha256, stable_job_id
from sbf_campaign_runtime import (
    Deadline,
    SignalController,
    atomic_write_json,
    atomic_write_text,
    build_artifact_manifest,
    launch_process_group,
    supervise_process,
    terminate_process_group,
)
from sbf_target_status import (
    PRIMARY_QUANTITY,
    ensure_target_rows,
    measurement_method as target_status_measurement_method,
    read_target_status,
    science_status_fields,
    reusable_result_from_status,
    target_status_key,
    update_target_status,
    validate_reusable_result,
    write_target_status,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "sbf-2.ipynb"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "sbf2_go3055"
DEFAULT_BATCH_ROOT = DEFAULT_RUN_ROOT / "batch"
DEFAULT_PRODUCTS_ROOT = DEFAULT_RUN_ROOT / "products"
DEFAULT_CAMPAIGN_ROOT = DEFAULT_RUN_ROOT / "campaign"
DEFAULT_TARGET_CSV = SCRIPT_DIR / "targets_go3055_manifest.csv"
DEFAULT_PAPER_IV_METADATA = DEFAULT_DATA_ROOT / "go3055_paper_iv_metadata.csv"
DEFAULT_WSS_OPD_DIR = DEFAULT_DATA_ROOT / "wss_opd"
DEFAULT_STPSF_DATA_DIR = Path.home() / "data" / "stpsf-data"
TARGET_STATUS_FILENAME = "target_status.csv"
GO3055_QC_FILENAME = "go3055_qc.csv"
MAST_DOWNLOAD_PREFIX = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
DEFAULT_SIGNAL_FILTER = "F150W"
DEFAULT_COLOR_FILTER = "F090W"
CURRENT_NOTEBOOK_FILTER_PAIR = (DEFAULT_SIGNAL_FILTER, DEFAULT_COLOR_FILTER)
SBF2_NOTEBOOK_FAMILY = "sbf2"
SBF3_NOTEBOOK_FAMILY = "sbf3"
MAX_IMPLICIT_TARGETS = 14
_ACTIVE_CAMPAIGN_LOCK = None
SBF3_REQUIRED_FITS_KEYS = (
    "clean_model_fits",
    "clean_isophotes_fits",
    "full_residual_fits",
    "working_residual_fits",
    "working_annuli_residual_fits",
)
SBF2_REQUIRED_FITS_KEYS = (
    "model_full_fits",
    "science_residual_fits",
    "science_residual_raw_fits",
    "inner_usable_residual_fits",
    "outer_usable_residual_fits",
)
SBF2_REQUIRED_TABLE_KEYS = (
    "df_sbf_csv",
    "annulus_summary_csv",
)


CLI_PATH_ARGUMENTS = (
    "template",
    "data_root",
    "batch_root",
    "products_root",
    "campaign_root",
    "target_csv",
    "external_download_status",
    "wss_opd_dir",
    "stpsf_data_dir",
    "signal",
    "color",
)


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


def resolve_cli_path(value):
    """Resolve relative CLI paths against the repository, never the shell CWD."""
    if value in (None, ""):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def normalize_cli_paths(args):
    """Make one command behave identically from PROJECT_ROOT and code/."""
    for name in CLI_PATH_ARGUMENTS:
        value = getattr(args, name, None)
        if value not in (None, ""):
            setattr(args, name, resolve_cli_path(value))
    extras = getattr(args, "extra_target_csv", None)
    if extras:
        args.extra_target_csv = [resolve_cli_path(value) for value in extras]
    return args


def append_jsonl(path, payload):
    """Append one durable, human-readable JSON event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(as_builtin(payload), ensure_ascii=False, sort_keys=False)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit_campaign_event(
    event_log_path,
    event_type,
    *,
    state=None,
    run_id=None,
    job_id=None,
    attempt_id=None,
    payload=None,
):
    """Mirror a campaign event to JSONL and, when available, SQLite."""
    event_payload = as_builtin(payload or {})
    record = {
        "timestamp": timestamp(),
        "timestamp_unix": time.time(),
        "event_type": event_type,
        "run_id": run_id,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "payload": event_payload,
    }
    if event_log_path is not None:
        append_jsonl(event_log_path, record)
    if state is not None and run_id is not None:
        state.append_event(
            run_id,
            job_id=job_id,
            attempt_id=attempt_id,
            event_type=event_type,
            payload=event_payload,
        )
    return record


def acquire_campaign_lock(campaign_root):
    """Hold a non-blocking OS lock so two parents cannot own one campaign."""
    import fcntl

    lock_path = Path(campaign_root) / "parent.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"campaign already has an active parent ({owner}): {lock_path}"
        )
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {"pid": os.getpid(), "started": timestamp()},
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_campaign_lock(handle):
    if handle is None or handle.closed:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def as_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return as_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
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
            if not hdul:
                return False, "FITS contains no HDUs"
            hdul.verify("exception")
            for hdu in hdul:
                _ = hdu.header
                if hdu.data is not None and hdu.data.size:
                    _ = hdu.data.reshape(-1)[-1]
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
    job_id=None,
):
    if products_root is None:
        return Path(signal_path).resolve().parent
    pair_label = f"{signal_filter or 'signal'}__{color_filter or 'color'}"
    pair_key = (
        str(job_id).removeprefix("job-")[:12]
        if job_id
        else input_pair_key(
            signal_path,
            color_path,
            signal_fingerprint=signal_fingerprint,
            color_fingerprint=color_fingerprint,
        )
    )
    return (
        Path(products_root).resolve()
        / slug(galaxy)
        / slug(pair_label)
        / pair_key
    )


def result_json_path(batch_root, galaxy, identity=None):
    if identity and identity.get("template_family") == SBF3_NOTEBOOK_FAMILY:
        run_key = (
            str(identity["job_id"]).removeprefix("job-")[:12]
            if identity.get("job_id")
            else identity["input_pair_key"]
        )
        run_label = "__".join(
            [
                slug(galaxy),
                slug(identity["signal_filter"]),
                slug(identity["color_filter"]),
                run_key,
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
    job_id=None,
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
    identity = {
        "galaxy": target["name"],
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
                job_id=job_id,
            )
        ),
    }
    if job_id:
        identity["job_id"] = str(job_id)
    return identity


def result_matches_identity(result, identity):
    """Match reusable science inputs while treating notebook details as provenance."""
    if "galaxy" in identity and (
        " ".join(str(result.get("galaxy") or "").upper().split())
        != " ".join(str(identity["galaxy"]).upper().split())
    ):
        return False
    for key in ("signal_filter", "color_filter"):
        if str(result.get(key) or "").strip().upper() != str(
            identity.get(key) or ""
        ).strip().upper():
            return False
    if identity.get("job_id") and result.get("job_id") != identity.get("job_id"):
        return False
    for role in ("signal", "color"):
        current = identity.get(f"{role}_fingerprint") or {}
        recorded = result.get(f"{role}_fingerprint") or {}
        if current.get("size") is None:
            # A completed target remains reusable after deliberate input cleanup.
            current_name = Path(identity.get(f"{role}_path") or "").name
            recorded_name = Path(result.get(f"{role}_path") or "").name
            if current_name != recorded_name:
                return False
            continue
        if recorded == current:
            continue
        recorded_sha = result.get(f"{role}_sha256")
        current_path = Path(identity.get(f"{role}_path") or "")
        if not recorded_sha or not current_path.is_file():
            return False
        if sha256_file(current_path) != recorded_sha:
            return False
    return True


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
                    "signal_product_uri": row.get("signal_product_uri")
                    or f"mast:JWST/product/{signal_product}",
                    "color_product_uri": row.get("color_product_uri")
                    or f"mast:JWST/product/{color_product}",
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


def target_csv_paths(target_csv=None, extra_target_csvs=None):
    """Resolve one legacy manifest plus any additional manifests."""
    values = []
    if isinstance(target_csv, (list, tuple)):
        values.extend(target_csv)
    elif target_csv:
        values.append(target_csv)
    values.extend(extra_target_csvs or [])
    if not values:
        values.append(DEFAULT_TARGET_CSV)

    paths = []
    seen = set()
    for value in values:
        path = Path(resolve_cli_path(value))
        key = str(path)
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def canonical_program(value):
    text = str(value or "").strip().upper()
    if text.startswith("GO-"):
        text = text[3:]
    elif text.startswith("GO"):
        text = text[2:]
    text = text.strip()
    if text.isdigit():
        return str(int(text))
    return text


def archive_target_identity(target):
    target = normalize_target(target)
    return canonical_sha256(
        {
            "target": " ".join(target["name"].casefold().split()),
            "program": canonical_program(target.get("program")),
            "obsid": str(target.get("obsid") or "").strip().casefold(),
            "product_uris": {
                role: str(uri or "").strip()
                for role, uri in campaign_product_uris(target).items()
            },
            "filters": {
                "signal": target["signal_filter"],
                "color": target["color_filter"],
            },
        }
    )


def deduplicate_manifest_targets(targets, data_root):
    """Deduplicate identical rows and reject ambiguous local destinations."""
    unique = []
    by_identity = {}
    destination_claims = {}

    for raw_target in targets:
        target = normalize_target(raw_target)
        target["program"] = canonical_program(target.get("program"))
        identity = archive_target_identity(target)
        previous = by_identity.get(identity)
        if previous is not None:
            conflicting = []
            for key in (
                "signal_product",
                "color_product",
                "signal_size",
                "color_size",
            ):
                left = previous.get(key)
                right = target.get(key)
                if left not in (None, "") and right not in (None, "") and left != right:
                    conflicting.append(f"{key}={left!r}/{right!r}")
            if conflicting:
                raise ValueError(
                    f"conflicting duplicate archive target {target['name']}: "
                    + ", ".join(conflicting)
                )
            sources = list(previous.get("source_csvs") or [])
            for source in (previous.get("source_csv"), target.get("source_csv")):
                if source and source not in sources:
                    sources.append(source)
            previous["source_csvs"] = sources
            continue

        files = local_target_files(target, data_root)
        uris = campaign_product_uris(target)
        for role, path in files.items():
            destination = str(path.resolve())
            claim = {
                "uri": str(uris.get(role) or "").strip(),
                "filter": target[f"{role}_filter"],
                "target": target["name"],
                "role": role,
            }
            old_claim = destination_claims.get(destination)
            if old_claim is not None and (
                old_claim["uri"] != claim["uri"]
                or old_claim["filter"] != claim["filter"]
            ):
                raise ValueError(
                    f"local FITS destination has conflicting claims: {destination}; "
                    f"{old_claim} versus {claim}"
                )
            destination_claims[destination] = claim

        sources = [target["source_csv"]] if target.get("source_csv") else []
        target["source_csvs"] = sources
        by_identity[identity] = target
        unique.append(target)
    return unique


def load_manifest_targets(target_csv, data_root, extra_target_csvs=None):
    paths = target_csv_paths(target_csv, extra_target_csvs)
    targets = []
    for path in paths:
        targets.extend(read_targets_from_csv(path, data_root))
    return deduplicate_manifest_targets(merge_known_targets(targets), data_root)


def merge_known_targets(targets):
    known = {target["name"]: normalize_target(target) for target in TARGETS}
    merged = []
    for target in targets:
        item = normalize_target(target)
        if item["name"] in known:
            known_item = known[item["name"]]
            for key, value in known_item.items():
                if key in {"signal_size", "color_size"}:
                    continue
                if item.get(key) in (None, ""):
                    item[key] = value
            for role in ("signal", "color"):
                size_key = f"{role}_size"
                product_key = f"{role}_product"
                if (
                    item.get(size_key) in (None, "")
                    and item.get(product_key) == known_item.get(product_key)
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
            source = "".join(cell.get("source", []))
            cells.append(
                (
                    cell_no,
                    str(cell.get("id") or ""),
                    hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    source,
                )
            )
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
    apply_paper_iv_metadata(namespace, galaxy)


def apply_paper_iv_metadata(namespace, galaxy, metadata_path=DEFAULT_PAPER_IV_METADATA):
    """Inject Paper IV extinction and quality metadata for one GO-3055 target."""
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"GO-3055 Paper IV metadata is missing: {metadata_path}"
        )

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["galaxy"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("galaxy")
        }
    if galaxy not in rows:
        raise KeyError(f"{galaxy}: no Paper IV metadata row in {metadata_path}")

    row = rows[galaxy]
    numeric = (
        "A_F090W",
        "E_BV",
        "A_F150W",
        "sigma_E_BV",
        "sigma_A_F090W",
        "sigma_A_F150W",
        "sigma_color_extinction",
    )
    for key in numeric:
        namespace[key] = float(row[key])
    namespace["PAPER_IV_HIGH_QUALITY"] = (
        str(row.get("paper_iv_high_quality", "")).strip().lower()
        in {"1", "true", "yes", "y"}
    )
    namespace["PAPER_IV_NAME"] = row.get("paper_iv_name", galaxy)
    namespace["EXTINCTION_SOURCE"] = row.get("source", "Paper IV Table 2")
    namespace["EXTINCTION_METADATA_PATH"] = str(metadata_path.resolve())


def result_paths(
    out_dir,
    stem,
    pipeline_label="sbf2",
    notebook_family_label=None,
):
    out_dir = Path(out_dir)
    family = str(notebook_family_label or pipeline_label).lower()
    family = family.replace("-", "").replace("_", "")
    if family == SBF3_NOTEBOOK_FAMILY:
        return {
            "clean_model_fits": out_dir / f"{stem}_01_модель_чистая.fits",
            "clean_isophotes_fits": out_dir
            / f"{stem}_02_изофоты_чистые.fits",
            "full_residual_fits": out_dir / f"{stem}_03_остатки_общие.fits",
            "working_residual_fits": out_dir
            / f"{stem}_04_остатки_общие_рабочие.fits",
            "working_annuli_residual_fits": out_dir
            / f"{stem}_05_остатки_общие_рабочие_два_кольца.fits",
            "df_sbf_csv": out_dir / f"{stem}_{pipeline_label}_df_sbf.csv",
            "annulus_summary_csv": out_dir
            / f"{stem}_{pipeline_label}_annulus_summary.csv",
        }
    return {
        "model_full_fits": out_dir / f"{stem}_sbf_model_full.fits",
        # This is the actual FFT input after sigma capping and the final
        # compact-source catalogue mask.  The earlier ``full_science`` image
        # remains available below as a diagnostic, but must not be advertised
        # as the residual used for the SBF measurement.
        "science_residual_fits": out_dir
        / f"{stem}_sbf_resid_catalog_mask_clip_3p5sigma.fits",
        "science_residual_pre_catalog_fits": out_dir
        / f"{stem}_sbf_resid_full_science.fits",
        "science_residual_raw_fits": out_dir / f"{stem}_sbf_resid_full_science_raw.fits",
        "inner_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_inner_lit_usable.fits",
        "outer_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_outer_lit_usable.fits",
        "df_sbf_csv": out_dir / f"{stem}_{pipeline_label}_df_sbf.csv",
        "annulus_summary_csv": out_dir
        / f"{stem}_{pipeline_label}_annulus_summary.csv",
        "power_spectrum_csv": out_dir / f"{stem}_sbf_power_spectra.csv",
    }


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _csv_rows(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _row_is_true(value):
    return str(value).strip().casefold() in {"1", "true", "yes", "ok"}


def validate_sbf2_tables(paths):
    """Validate the two numerical tables, including their science rows."""
    required_columns = {
        "df_sbf_csv": {
            "region",
            "kmin",
            "kmax",
            "measurement_ok",
            "mbar_spec",
            "P_fluc",
        },
        "annulus_summary_csv": {
            "kmin",
            "kmax",
            "mbar_inner",
            "mbar_outer",
            "mbar_weighted",
            "sigma_adopted",
        },
    }
    tables = {}
    errors = []
    for key, columns_required in required_columns.items():
        path = Path(paths[key])
        table = {
            "path": str(path.resolve()),
            "ok": False,
            "row_count": 0,
            "columns": [],
        }
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = list(reader.fieldnames or [])
            table["row_count"] = len(rows)
            table["columns"] = columns
            missing = sorted(columns_required.difference(columns))
            if missing:
                raise ValueError(f"missing columns: {', '.join(missing)}")
            if not rows:
                raise ValueError("table has no data rows")
            table["rows"] = rows
            table["ok"] = True
        except Exception as exc:
            table["error"] = str(exc)
            errors.append(f"{key}: {exc}")
        tables[key] = table

    measurements = tables.get("df_sbf_csv", {})
    if measurements.get("ok"):
        rows = measurements["rows"]
        for region in ("circular_inner_lit", "circular_outer_lit"):
            main = next(
                (
                    row
                    for row in rows
                    if row.get("region") == region
                    and _finite_float(row.get("kmin")) is not None
                    and _finite_float(row.get("kmax")) is not None
                    and np.isclose(float(row["kmin"]), 0.04, atol=1e-9)
                    and np.isclose(float(row["kmax"]), 0.25, atol=1e-9)
                ),
                None,
            )
            if main is None:
                errors.append(f"df_sbf_csv: missing {region} main k-window")
                continue
            if not _row_is_true(main.get("measurement_ok")):
                errors.append(f"df_sbf_csv: {region} measurement is not ok")
            if _finite_float(main.get("mbar_spec")) is None:
                errors.append(f"df_sbf_csv: {region} mbar_spec is not finite")
            p_fluc = _finite_float(main.get("P_fluc"))
            if p_fluc is None or p_fluc <= 0.0:
                errors.append(f"df_sbf_csv: {region} P_fluc is not positive")

    summary = tables.get("annulus_summary_csv", {})
    if summary.get("ok"):
        main = next(
            (
                row
                for row in summary["rows"]
                if _finite_float(row.get("kmin")) is not None
                and _finite_float(row.get("kmax")) is not None
                and np.isclose(float(row["kmin"]), 0.04, atol=1e-9)
                and np.isclose(float(row["kmax"]), 0.25, atol=1e-9)
            ),
            None,
        )
        if main is None:
            errors.append("annulus_summary_csv: missing main k-window")
        else:
            for column in (
                "mbar_inner",
                "mbar_outer",
                "mbar_weighted",
                "sigma_adopted",
            ):
                if _finite_float(main.get(column)) is None:
                    errors.append(
                        f"annulus_summary_csv: {column} is not finite"
                    )

    for table in tables.values():
        table.pop("rows", None)
    if errors:
        for table in tables.values():
            table["ok"] = False
    return {"ok": not errors, "tables": tables, "errors": errors}


def annotate_sbf2_fits(
    paths,
    namespace,
    galaxy,
    signal_filter,
    color_filter,
):
    """Add enough ASCII FITS provenance to interpret every required product."""
    recommended = namespace.get("recommended_sbf") or {}
    kmin = _finite_float(recommended.get("kmin"))
    kmax = _finite_float(recommended.get("kmax"))
    measurements = namespace.get("df_sbf")
    region_rows = {}
    if measurements is not None and hasattr(measurements, "to_dict"):
        for row in measurements.to_dict(orient="records"):
            region = row.get("region")
            if region not in {"circular_inner_lit", "circular_outer_lit"}:
                continue
            row_kmin = _finite_float(row.get("kmin"))
            row_kmax = _finite_float(row.get("kmax"))
            if (
                row_kmin is not None
                and row_kmax is not None
                and kmin is not None
                and kmax is not None
                and np.isclose(row_kmin, kmin, atol=1e-9)
                and np.isclose(row_kmax, kmax, atol=1e-9)
            ):
                region_rows[region] = row

    stage_metadata = {
        "model_full_fits": {
            "stage": "MODEL",
            "source": "measured_isophotes",
            "masksrc": False,
        },
        "science_residual_raw_fits": {
            "stage": "RESIDRAW",
            "source": "data_minus_model",
            "masksrc": False,
        },
        "science_residual_fits": {
            "stage": "RESIDFIN",
            "source": "catalog_mask_clip_3p5sigma",
            "masksrc": True,
            "sigclip": 3.5,
        },
        "inner_usable_residual_fits": {
            "stage": "RINGIN",
            "source": "catalog_mask_clip_3p5sigma",
            "masksrc": True,
            "sigclip": 3.5,
            "region": "circular_inner_lit",
            "rinasec": 8.2,
            "routasec": 16.4,
        },
        "outer_usable_residual_fits": {
            "stage": "RINGOUT",
            "source": "catalog_mask_clip_3p5sigma",
            "masksrc": True,
            "sigclip": 3.5,
            "region": "circular_outer_lit",
            "rinasec": 16.4,
            "routasec": 32.8,
        },
    }
    for key in SBF2_REQUIRED_FITS_KEYS:
        path = Path(paths[key])
        metadata = stage_metadata[key]
        with fits.open(path, mode="update", memmap=True) as hdul:
            header = hdul[0].header
            header["OBJECT"] = (str(galaxy), "SBF target")
            header["PROGRAM"] = ("GO-3055", "JWST observing program")
            header["SIGFILT"] = (str(signal_filter), "SBF signal filter")
            header["COLFILT"] = (str(color_filter), "SBF color filter")
            header["SBFMETH"] = ("JENSEN2", "Jensen-like sbf-2 pipeline")
            header["SBFSTAGE"] = (metadata["stage"], "SBF product stage")
            header["RESIDSRC"] = (
                metadata["source"],
                "Residual/model source used for product",
            )
            header["MASKSRC"] = (
                bool(metadata["masksrc"]),
                "Compact-source catalogue mask applied",
            )
            if namespace.get("PIPELINE_VERSION") is not None:
                header["PIPEVER"] = (
                    str(namespace["PIPELINE_VERSION"]),
                    "SBF pipeline version",
                )
            if namespace.get("opd_corr_id") is not None:
                header["OPDCORR"] = (
                    str(namespace["opd_corr_id"]),
                    "Time-matched WSS OPD identifier",
                )
            if namespace.get("psf_method_id") is not None:
                header["PSFMETH"] = (
                    str(namespace["psf_method_id"]),
                    "PSF construction method",
                )
            if namespace.get("psf_input_count") is not None:
                header["PSFN"] = (
                    int(namespace["psf_input_count"]),
                    "PSFs used in ensemble",
                )
            elif namespace.get("psf_library") is not None:
                header["PSFN"] = (
                    int(len(namespace["psf_library"])),
                    "PSFs used in ensemble",
                )
            for keyword, namespace_key in (
                ("SBFXCEN", "x0_sbf_circ"),
                ("SBFYCEN", "y0_sbf_circ"),
            ):
                value = _finite_float(namespace.get(namespace_key))
                if value is not None:
                    header[keyword] = (value, "SBF centre in full-frame pixels")
            if metadata.get("sigclip") is not None:
                header["SIGCLIP"] = (
                    float(metadata["sigclip"]),
                    "Symmetric residual cap in sigma",
                )
            region = metadata.get("region")
            if region:
                header["SBFREG"] = (region, "SBF measurement region")
                header["RINASEC"] = (
                    float(metadata["rinasec"]),
                    "Inner circular radius in arcsec",
                )
                header["ROUTASEC"] = (
                    float(metadata["routasec"]),
                    "Outer circular radius in arcsec",
                )
                row = region_rows.get(region, {})
                n_use = _finite_float(
                    row.get("n_use", row.get("usable_pixels"))
                )
                if n_use is not None:
                    header["NUSE"] = (
                        int(n_use),
                        "Finite unmasked pixels used by FFT",
                    )
                if kmin is not None:
                    header["KMIN"] = (kmin, "Recommended normalized k minimum")
                if kmax is not None:
                    header["KMAX"] = (kmax, "Recommended normalized k maximum")
            header.add_history(
                "Created by run_sbf_2_batch.py from sbf-2.ipynb; "
                "see result JSON and CSV tables for full provenance."
            )
            hdul.flush(output_verify="exception")


def evaluate_go3055_qc(result):
    """Return numerical GO-3055 QC without changing the execution status."""
    measurements = _csv_rows(result.get("df_sbf_csv"))
    summaries = _csv_rows(result.get("annulus_summary_csv"))
    kmin = _finite_float(result.get("recommended_kmin"))
    kmax = _finite_float(result.get("recommended_kmax"))
    if kmin is None:
        kmin = 0.04
    if kmax is None:
        kmax = 0.25

    region_names = ("circular_inner_lit", "circular_outer_lit")
    flags = []
    region_metrics = {}

    def same_window(row):
        row_kmin = _finite_float(row.get("kmin"))
        row_kmax = _finite_float(row.get("kmax"))
        return (
            row_kmin is not None
            and row_kmax is not None
            and np.isclose(row_kmin, kmin, atol=1e-9)
            and np.isclose(row_kmax, kmax, atol=1e-9)
        )

    for region in region_names:
        short = "inner" if "inner" in region else "outer"
        rows = [row for row in measurements if row.get("region") == region]
        main = next((row for row in rows if same_window(row)), None)
        metrics = {
            "mbar": None,
            "mbar_sigma": None,
            "usable_fraction": None,
            "model_coverage": _finite_float(
                result.get(f"{short}_model_coverage")
            ),
            "n_use": None,
            "P_fluc": None,
            "P1": None,
            "P0_fit_sigma": None,
            "fit_significance": None,
            "jensen_sbf_to_white_noise": None,
            "Pr_over_P0": None,
            "corr": None,
            "k_span_mag": None,
            "psf_n_used": None,
            "psf_success_fraction": None,
            "psf_scatter_mag": None,
            "hard_invalid": False,
        }
        hard_reasons = []
        if main is None:
            hard_reasons.append("main_k_window_missing")
        else:
            metrics["mbar"] = _finite_float(main.get("mbar_spec"))
            metrics["mbar_sigma"] = _finite_float(main.get("mbar_fit_sigma"))
            metrics["usable_fraction"] = _finite_float(
                main.get("usable_fraction")
            )
            metrics["n_use"] = _finite_float(
                main.get("n_use", main.get("usable_pixels"))
            )
            metrics["P_fluc"] = _finite_float(main.get("P_fluc"))
            metrics["P1"] = _finite_float(main.get("P1"))
            metrics["P0_fit_sigma"] = _finite_float(main.get("P0_fit_sigma"))
            metrics["Pr_over_P0"] = _finite_float(main.get("Pr_over_P0"))
            metrics["corr"] = _finite_float(main.get("corr"))
            metrics["psf_n_used"] = _finite_float(main.get("psf_n_used"))
            metrics["psf_scatter_mag"] = _finite_float(
                main.get("psf_scatter_mag")
            )
            if "measurement_ok" in main and not _row_is_true(
                main.get("measurement_ok")
            ):
                hard_reasons.append("measurement_not_ok")
            if metrics["mbar"] is None:
                hard_reasons.append("mbar_nonfinite")
            if metrics["P_fluc"] is None or metrics["P_fluc"] <= 0.0:
                hard_reasons.append("P_fluc_nonpositive")
            if metrics["n_use"] is None or metrics["n_use"] < 5000:
                hard_reasons.append("too_few_pixels")
            if metrics["psf_n_used"] is not None and metrics["psf_n_used"] <= 0:
                hard_reasons.append("no_usable_psf")
            if (
                metrics["P_fluc"] is not None
                and metrics["P0_fit_sigma"] is not None
                and metrics["P0_fit_sigma"] > 0.0
            ):
                metrics["fit_significance"] = (
                    metrics["P_fluc"] / metrics["P0_fit_sigma"]
                )
            if metrics["P1"] is not None and metrics["P1"] > 0.0:
                metrics["jensen_sbf_to_white_noise"] = (
                    metrics["P_fluc"] / metrics["P1"]
                )

        valid_mbars = [
            value
            for row in rows
            if ("measurement_ok" not in row or _row_is_true(row["measurement_ok"]))
            for value in [_finite_float(row.get("mbar_spec"))]
            if value is not None
        ]
        if len(valid_mbars) >= 2:
            metrics["k_span_mag"] = max(valid_mbars) - min(valid_mbars)

        psf_input_count = _finite_float(result.get("psf_input_count"))
        if (
            metrics["psf_n_used"] is not None
            and psf_input_count is not None
            and psf_input_count > 0
        ):
            metrics["psf_success_fraction"] = (
                metrics["psf_n_used"] / psf_input_count
            )

        metrics["hard_invalid"] = bool(hard_reasons)
        for reason in hard_reasons:
            flags.append(f"{short}:invalid:{reason}")

        usable = metrics["usable_fraction"]
        if usable is not None and usable < 0.70:
            level = "review" if usable < 0.50 else "warn"
            flags.append(f"{short}:{level}:usable_fraction={usable:.3f}")
        coverage = metrics["model_coverage"]
        if coverage is None:
            flags.append(f"{short}:review:model_coverage_missing")
        elif coverage < 0.95:
            level = "review" if coverage < 0.75 else "warn"
            flags.append(f"{short}:{level}:model_coverage={coverage:.3f}")
        k_span = metrics["k_span_mag"]
        if k_span is not None and k_span > 0.10:
            level = "review" if k_span > 0.20 else "warn"
            flags.append(f"{short}:{level}:k_span={k_span:.3f}mag")
        pr_ratio = metrics["Pr_over_P0"]
        if pr_ratio is not None and pr_ratio > 0.10:
            level = "review" if pr_ratio > 0.20 else "warn"
            flags.append(f"{short}:{level}:Pr_over_P0={pr_ratio:.3f}")
        corr = metrics["corr"]
        if corr is not None and corr < 0.50:
            level = "review" if corr < 0.30 else "warn"
            flags.append(f"{short}:{level}:corr={corr:.3f}")
        fit_significance = metrics["fit_significance"]
        if fit_significance is not None and fit_significance < 5.0:
            level = "review" if fit_significance < 3.0 else "warn"
            flags.append(
                f"{short}:{level}:fit_significance={fit_significance:.2f}"
            )
        jensen_ratio = metrics["jensen_sbf_to_white_noise"]
        if metrics["P1"] is None or metrics["P1"] <= 0.0:
            flags.append(f"{short}:review:white_noise_P1_nonpositive")
        elif jensen_ratio is not None and jensen_ratio < 5.0:
            level = "review" if jensen_ratio < 3.0 else "warn"
            flags.append(
                f"{short}:{level}:jensen_Pfluc_over_P1={jensen_ratio:.2f}"
            )
        psf_n = metrics["psf_n_used"]
        psf_fraction = metrics["psf_success_fraction"]
        if (
            (psf_n is not None and psf_n < 3)
            or (psf_fraction is not None and psf_fraction < 0.60)
        ):
            level = (
                "review"
                if (psf_n is not None and psf_n <= 1)
                or (psf_fraction is not None and psf_fraction < 0.30)
                else "warn"
            )
            flags.append(
                f"{short}:{level}:psf={psf_n}/{psf_input_count}"
            )
        psf_scatter = metrics["psf_scatter_mag"]
        if psf_scatter is not None and psf_scatter > 0.05:
            level = "review" if psf_scatter > 0.10 else "warn"
            flags.append(
                f"{short}:{level}:psf_scatter={psf_scatter:.3f}mag"
            )
        region_metrics[short] = metrics

    summary_main = next((row for row in summaries if same_window(row)), None)
    delta_mag = None
    delta_z = None
    if summary_main is not None:
        inner_mbar = _finite_float(summary_main.get("mbar_inner"))
        outer_mbar = _finite_float(summary_main.get("mbar_outer"))
        inner_sigma = _finite_float(summary_main.get("sigma_inner"))
        outer_sigma = _finite_float(summary_main.get("sigma_outer"))
        if inner_mbar is not None and outer_mbar is not None:
            delta_mag = abs(inner_mbar - outer_mbar)
        if (
            delta_mag is not None
            and inner_sigma is not None
            and outer_sigma is not None
            and np.hypot(inner_sigma, outer_sigma) > 0.0
        ):
            delta_z = delta_mag / np.hypot(inner_sigma, outer_sigma)
    if delta_mag is not None:
        significance_text = (
            f",{delta_z:.1f}sigma" if delta_z is not None else ",sigma_unknown"
        )
        if delta_mag > 0.30 and (delta_z is None or delta_z >= 3.0):
            flags.append(
                f"annuli:review:delta={delta_mag:.3f}mag{significance_text}"
            )
        elif delta_mag > 0.15 and (delta_z is None or delta_z >= 2.0):
            flags.append(
                f"annuli:warn:delta={delta_mag:.3f}mag{significance_text}"
            )

    hard_invalid_count = sum(
        bool(region_metrics[name]["hard_invalid"]) for name in ("inner", "outer")
    )
    if hard_invalid_count >= 1:
        qc_status = "invalid"
    elif any(":review:" in flag for flag in flags):
        qc_status = "review"
    elif flags:
        qc_status = "warn"
    else:
        qc_status = "pass"
    compact = qc_status
    if flags:
        compact += ":" + "|".join(flags)
    return {
        "status": qc_status,
        "summary": compact,
        "flags": flags,
        "metrics": {
            "kmin": kmin,
            "kmax": kmax,
            "annulus_delta_mag": delta_mag,
            "annulus_delta_z": delta_z,
            "inner": region_metrics["inner"],
            "outer": region_metrics["outer"],
        },
    }


def write_go3055_qc(results, batch_root):
    """Write one compact, human-facing QC row per completed galaxy."""
    fieldnames = [
        "galaxy",
        "program",
        "signal_filter",
        "color_filter",
        "execution_status",
        "qc_status",
        "qc_flags",
        "recommended_mbar_weighted",
        "recommended_sigma_adopted",
        "annulus_delta_mag",
        "annulus_delta_z",
        "inner_mbar",
        "inner_mbar_sigma",
        "inner_P_fluc",
        "inner_P1",
        "inner_usable_fraction",
        "inner_model_coverage",
        "inner_k_span_mag",
        "inner_Pr_over_P0",
        "inner_corr",
        "inner_fit_significance",
        "inner_jensen_sbf_to_white_noise",
        "inner_psf_n_used",
        "inner_psf_success_fraction",
        "inner_psf_scatter_mag",
        "outer_mbar",
        "outer_mbar_sigma",
        "outer_P_fluc",
        "outer_P1",
        "outer_usable_fraction",
        "outer_model_coverage",
        "outer_k_span_mag",
        "outer_Pr_over_P0",
        "outer_corr",
        "outer_fit_significance",
        "outer_jensen_sbf_to_white_noise",
        "outer_psf_n_used",
        "outer_psf_success_fraction",
        "outer_psf_scatter_mag",
        "opd_corr_id",
        "opd_delta_days",
        "isophote_fit_mode",
        "isophote_count",
        "fitted_sma_max_px",
        "result_json",
        "worker_log_path",
        "error",
    ]
    summary_json = Path(batch_root) / "sbf2_batch_results.json"
    if summary_json.is_file():
        try:
            stored_results = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(stored_results, list):
                results = stored_results
        except Exception:
            pass
    rows = []
    for result in sorted(results, key=lambda item: str(item.get("galaxy", ""))):
        qc = result.get("go3055_qc")
        if result.get("status") == "ok" and not isinstance(qc, dict):
            qc = evaluate_go3055_qc(result)
        if not isinstance(qc, dict):
            qc = {
                "status": "execution_failed",
                "flags": [str(result.get("error") or "worker_failed")],
                "metrics": {},
            }
        metrics = qc.get("metrics") or {}
        inner = metrics.get("inner") or {}
        outer = metrics.get("outer") or {}
        row = {
            "galaxy": result.get("galaxy"),
            "program": result.get("program", "3055"),
            "signal_filter": result.get("signal_filter"),
            "color_filter": result.get("color_filter"),
            "execution_status": result.get("status"),
            "qc_status": qc.get("status"),
            "qc_flags": "|".join(qc.get("flags") or []),
            "recommended_mbar_weighted": result.get(
                "recommended_mbar_weighted"
            ),
            "recommended_sigma_adopted": result.get(
                "recommended_sigma_adopted"
            ),
            "annulus_delta_mag": metrics.get("annulus_delta_mag"),
            "annulus_delta_z": metrics.get("annulus_delta_z"),
            "opd_corr_id": result.get("opd_corr_id"),
            "opd_delta_days": result.get("opd_delta_days"),
            "isophote_fit_mode": result.get("isophote_fit_mode"),
            "isophote_count": result.get("isophote_count"),
            "fitted_sma_max_px": result.get("fitted_sma_max_px"),
            "result_json": result.get("result_json"),
            "worker_log_path": result.get("worker_log_path"),
            "error": result.get("error"),
        }
        for prefix, values in (("inner", inner), ("outer", outer)):
            for output_name, metric_name in (
                ("mbar", "mbar"),
                ("mbar_sigma", "mbar_sigma"),
                ("P_fluc", "P_fluc"),
                ("P1", "P1"),
                ("usable_fraction", "usable_fraction"),
                ("model_coverage", "model_coverage"),
                ("k_span_mag", "k_span_mag"),
                ("Pr_over_P0", "Pr_over_P0"),
                ("corr", "corr"),
                ("fit_significance", "fit_significance"),
                (
                    "jensen_sbf_to_white_noise",
                    "jensen_sbf_to_white_noise",
                ),
                ("psf_n_used", "psf_n_used"),
                ("psf_success_fraction", "psf_success_fraction"),
                ("psf_scatter_mag", "psf_scatter_mag"),
            ):
                row[f"{prefix}_{output_name}"] = values.get(metric_name)
        rows.append({key: as_builtin(row.get(key)) for key in fieldnames})

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    output_path = Path(batch_root) / GO3055_QC_FILENAME
    atomic_write_text(output_path, buffer.getvalue())
    return output_path


def execute_template_for_target(
    template_path,
    galaxy,
    signal_path,
    color_path,
    batch_root,
    signal_filter=DEFAULT_SIGNAL_FILTER,
    color_filter=DEFAULT_COLOR_FILTER,
    output_dir=None,
    job_id=None,
    input_sha256=None,
    cell_timings_path=None,
    worker_log_path=None,
    attempt_output_key=None,
):
    template_path = Path(template_path).resolve()
    pipeline_label = template_path.stem.replace("-", "").replace("_", "")
    pipeline_family = notebook_family(template_path)
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

    for cell_no, cell_id, cell_sha256, source in code_cells:
        cell_started = time.monotonic()
        cell_base = {
            "timestamp": timestamp(),
            "worker_pid": os.getpid(),
            "job_id": job_id,
            "galaxy": galaxy,
            "notebook": template_path.name,
            "cell_no": cell_no,
            "cell_id": cell_id or None,
            "cell_sha256": cell_sha256,
        }
        print(
            f"[{timestamp()}] [CELL_START] {template_path.name} cell {cell_no} "
            f"id={cell_id or '-'} sha256={cell_sha256[:12]}"
        )
        if cell_timings_path is not None:
            append_jsonl(
                cell_timings_path,
                {**cell_base, "event_type": "CELL_START", "status": "running"},
            )
        try:
            exec(compile(source, f"{template_path}:cell-{cell_no}", "exec"), namespace)
        except Exception as exc:
            duration = time.monotonic() - cell_started
            print(
                f"[{timestamp()}] [CELL_FAILED] {template_path.name} cell {cell_no} "
                f"after {duration:.3f}s: {exc!r}"
            )
            if cell_timings_path is not None:
                append_jsonl(
                    cell_timings_path,
                    {
                        **cell_base,
                        "timestamp": timestamp(),
                        "event_type": "CELL_FAILED",
                        "status": "failed",
                        "duration_seconds": duration,
                        "error": repr(exc),
                    },
                )
            raise
        duration = time.monotonic() - cell_started
        print(
            f"[{timestamp()}] [CELL_END] {template_path.name} cell {cell_no} "
            f"status=ok duration={duration:.3f}s"
        )
        if cell_timings_path is not None:
            append_jsonl(
                cell_timings_path,
                {
                    **cell_base,
                    "timestamp": timestamp(),
                    "event_type": "CELL_END",
                    "status": "ok",
                    "duration_seconds": duration,
                },
            )

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
    paths = result_paths(
        out_dir,
        stem,
        pipeline_label=pipeline_label,
        notebook_family_label=pipeline_family,
    )

    if pipeline_family == SBF3_NOTEBOOK_FAMILY:
        product_errors = []
        for key in SBF3_REQUIRED_FITS_KEYS:
            product_path = paths[key]
            if not product_path.exists():
                product_errors.append(f"{key}: missing {product_path}")
                continue
            readable, read_error = fits_is_readable(product_path)
            if not readable:
                product_errors.append(f"{key}: unreadable {product_path}: {read_error}")
        if product_errors:
            raise RuntimeError(
                "sbf-3 did not produce the required five FITS products: "
                + " | ".join(product_errors)
            )

    df_sbf = namespace.get("df_sbf")
    if df_sbf is not None:
        df_sbf.to_csv(paths["df_sbf_csv"], index=False)

    df_annulus_summary = namespace.get("df_annulus_summary")
    if df_annulus_summary is not None:
        df_annulus_summary.to_csv(paths["annulus_summary_csv"], index=False)

    sbf2_table_validation = None
    if pipeline_family == SBF2_NOTEBOOK_FAMILY:
        product_errors = []
        for key in SBF2_REQUIRED_FITS_KEYS:
            product_path = paths[key]
            if not product_path.is_file():
                product_errors.append(f"{key}: missing {product_path}")
                continue
            readable, read_error = fits_is_readable(product_path)
            if not readable:
                product_errors.append(
                    f"{key}: unreadable {product_path}: {read_error}"
                )
        sbf2_table_validation = validate_sbf2_tables(paths)
        product_errors.extend(sbf2_table_validation["errors"])
        if not product_errors:
            try:
                annotate_sbf2_fits(
                    paths,
                    namespace,
                    galaxy,
                    signal_filter,
                    color_filter,
                )
            except Exception as exc:
                product_errors.append(f"FITS provenance update failed: {exc}")
        if product_errors:
            raise RuntimeError(
                "sbf-2 did not produce the required five FITS and two tables: "
                + " | ".join(product_errors)
            )

    result = {
        "galaxy": galaxy,
        "status": "ok",
        "template_name": template_path.name,
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "template_family": pipeline_family,
        "program": "3055" if pipeline_family == SBF2_NOTEBOOK_FAMILY else None,
        "signal_filter": signal_filter,
        "color_filter": color_filter,
        "color_name": f"{color_filter}-{signal_filter}",
        "signal_path": str(Path(signal_path).resolve()),
        "color_path": str(Path(color_path).resolve()),
        "signal_fingerprint": signal_fingerprint,
        "color_fingerprint": color_fingerprint,
        "input_pair_key": pair_key,
        "out_dir": str(out_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "stem": stem,
    }
    if attempt_output_key:
        result["attempt_output_key"] = str(attempt_output_key)
    if input_sha256:
        result["signal_sha256"] = input_sha256.get("signal")
        result["color_sha256"] = input_sha256.get("color")
    if cell_timings_path is not None:
        result["cell_timings_path"] = str(Path(cell_timings_path).resolve())
    if worker_log_path is not None:
        result["worker_log_path"] = str(Path(worker_log_path).resolve())
    if job_id:
        result["job_id"] = str(job_id)
    namespace_metadata = {
        "pipeline_version": "PIPELINE_VERSION",
        "siaf_prd_version": "LOCAL_SIAF_PRD_VERSION",
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
        "psf_native_scale_rel_error": "psf_native_scale_rel_error",
        "psf_method_id": "psf_method_id",
        "psf_method_limitations": "psf_method_limitations",
        "psf_detector_set": "psf_detector_set",
        "psf_input_count": "psf_input_count",
        "psf_selected_extension": "psf_selected_ext",
        "opd_corr_id": "opd_corr_id",
        "opd_signed_delta_days": "opd_signed_delta_days",
        "opd_delta_days": "opd_delta_days",
        "color_opd_corr_id": "f090_opd_corr_id",
        "color_opd_path": "f090_opd_path",
        "color_opd_signed_delta_days": "f090_signed_delta_days",
        "color_opd_delta_days": "f090_delta_days",
        "color_psf_selected_extension": "color_psf_f090_ext",
        "selected_sbf_region": "selected_sbf_region",
        "selected_sbf_selection_method": "selected_sbf_selection_method",
        "selected_color_index": "selected_color",
        "A_F090W": "A_F090W",
        "A_F150W": "A_F150W",
        "E_BV": "E_BV",
        "sigma_E_BV": "sigma_E_BV",
        "sigma_A_F090W": "sigma_A_F090W",
        "sigma_A_F150W": "sigma_A_F150W",
        "sigma_color_extinction": "sigma_color_extinction",
        "paper_iv_high_quality": "PAPER_IV_HIGH_QUALITY",
        "paper_iv_name": "PAPER_IV_NAME",
        "extinction_source": "EXTINCTION_SOURCE",
        "extinction_metadata_path": "EXTINCTION_METADATA_PATH",
    }
    for result_key, namespace_key in namespace_metadata.items():
        if namespace_key in namespace:
            result[result_key] = as_builtin(namespace[namespace_key])
    for result_key, namespace_key in (
        ("isophote_fit_mode", "iso_fit_mode_used"),
        ("isophote_fit_signature", "iso_fit_signature_used"),
        ("fitted_sma_min_px", "fitted_sma_min_px"),
        ("fitted_sma_max_px", "fitted_sma_max_px"),
    ):
        if namespace_key in namespace:
            result[result_key] = as_builtin(namespace[namespace_key])
    if namespace.get("isolist") is not None:
        result["isophote_count"] = len(namespace["isolist"])
    for short_name, namespace_key in (
        ("inner", "inner_cov"),
        ("outer", "outer_cov"),
    ):
        coverage = namespace.get(namespace_key)
        if coverage is not None and len(coverage) >= 5:
            result[f"{short_name}_model_coverage"] = as_builtin(coverage[3])
            result[f"{short_name}_premask_usable_coverage"] = as_builtin(
                coverage[4]
            )
    if namespace.get("psf_library") is not None:
        result["psf_input_count"] = len(namespace["psf_library"])
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
            selected_color_rows = color_summary
            if "selected_for_final" in color_summary.columns:
                selected_color_rows = color_summary[
                    color_summary["selected_for_final"].fillna(False).astype(bool)
                ]
            if selected_color_rows.empty:
                raise ValueError("no color row selected for final result")
            row0 = selected_color_rows.iloc[0].to_dict()
            color_value = as_builtin(
                row0.get("color_index", row0.get("color_F090W_F150W"))
            )
            result["color_index"] = color_value
            result["color_name"] = row0.get("color_name", result["color_name"])
            result[f"color_{color_filter}_{signal_filter}"] = color_value
            result["color_sigma_proxy"] = as_builtin(row0.get("sigma_proxy"))
            for column in (
                "color_F090W_F150W_observed",
                "color_F090W_F150W_extinction_corrected",
                "sigma_color_measurement",
                "sigma_color_extinction",
                "extinction_applied",
            ):
                if column in row0:
                    result[column] = as_builtin(row0.get(column))
        except Exception:
            pass

    result_json = result_json_path(batch_root, galaxy, identity=result)
    result["result_json"] = str(result_json.resolve())
    if pipeline_family in {SBF2_NOTEBOOK_FAMILY, SBF3_NOTEBOOK_FAMILY}:
        required_keys = (
            SBF3_REQUIRED_FITS_KEYS
            if pipeline_family == SBF3_NOTEBOOK_FAMILY
            else SBF2_REQUIRED_FITS_KEYS
        )
        required_artifacts = {key: Path(result[key]) for key in required_keys}
        fits_manifest = build_artifact_manifest(
            required_artifacts,
            base_dir=out_dir,
            include_sha256=(pipeline_family == SBF3_NOTEBOOK_FAMILY),
            validate_fits=True,
            require_astropy=True,
        )
        artifact_manifest = fits_manifest
        expected_artifact_count = len(required_keys)
        if pipeline_family == SBF2_NOTEBOOK_FAMILY:
            if sbf2_table_validation is None:
                sbf2_table_validation = validate_sbf2_tables(paths)
            table_artifacts = {
                key: Path(result[key]) for key in SBF2_REQUIRED_TABLE_KEYS
            }
            table_manifest = build_artifact_manifest(
                table_artifacts,
                base_dir=out_dir,
                include_sha256=False,
                validate_fits=False,
            )
            table_validation = sbf2_table_validation["tables"]
            for item in table_manifest["artifacts"]:
                validation = table_validation.get(item["name"], {})
                item["csv_valid"] = bool(validation.get("ok"))
                item["row_count"] = validation.get("row_count", 0)
                item["columns"] = validation.get("columns", [])
                if validation.get("error"):
                    item["csv_error"] = validation["error"]
                item["ok"] = bool(item.get("ok") and item["csv_valid"])
            artifact_manifest = dict(fits_manifest)
            artifact_manifest["artifacts"] = (
                fits_manifest["artifacts"] + table_manifest["artifacts"]
            )
            artifact_manifest["count"] = len(artifact_manifest["artifacts"])
            artifact_manifest["fits_count"] = len(required_keys)
            artifact_manifest["table_count"] = len(SBF2_REQUIRED_TABLE_KEYS)
            artifact_manifest["ok"] = bool(
                fits_manifest["ok"]
                and table_manifest["ok"]
                and sbf2_table_validation["ok"]
                and all(
                    item.get("ok") for item in table_manifest["artifacts"]
                )
            )
            expected_artifact_count += len(SBF2_REQUIRED_TABLE_KEYS)
        if (
            not artifact_manifest["ok"]
            or artifact_manifest["count"] != expected_artifact_count
        ):
            raise RuntimeError(
                "required products failed the final artifact gate: "
                + " | ".join(
                    f"{item['name']}: "
                    f"{item.get('fits_error') or item.get('csv_error') or item.get('error')}"
                    for item in artifact_manifest["artifacts"]
                    if not item.get("ok")
                )
            )
        manifest_path = (
            Path(batch_root)
            / "artifact_manifests"
            / f"{result_json.stem}_artifacts.json"
        )
        atomic_write_json(manifest_path, artifact_manifest)
        result["artifacts_verified"] = True
        result["artifacts_verified_at"] = artifact_manifest["created_at_utc"]
        result["artifact_manifest_path"] = str(manifest_path.resolve())
        result["artifact_count"] = artifact_manifest["count"]
        result["fits_artifact_count"] = len(required_keys)
        result["table_artifact_count"] = (
            len(SBF2_REQUIRED_TABLE_KEYS)
            if pipeline_family == SBF2_NOTEBOOK_FAMILY
            else 0
        )
        result["artifact_manifest"] = artifact_manifest["artifacts"]

    if pipeline_family == SBF2_NOTEBOOK_FAMILY:
        result["go3055_qc"] = evaluate_go3055_qc(result)
        result["qc_status"] = result["go3055_qc"]["status"]
        result["qc_flags"] = result["go3055_qc"]["flags"]

    batch_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_json, as_builtin(result), sort_keys=False)
    print(f"[{timestamp()}] wrote result {result_json}")
    return result


def run_worker(args):
    normalize_cli_paths(args)
    template_path = Path(args.template).resolve()
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    runtime_cache = batch_root / ".runtime_cache"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = runtime_cache / "matplotlib"
    xdg_cache = runtime_cache / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ["STPSF_PATH"] = str(Path(args.stpsf_data_dir).resolve())
    os.environ["SBF_WSS_OPD_DIR"] = str(Path(args.wss_opd_dir).resolve())
    os.environ["SBF_PSF_MAX_OPD_DELTA_DAYS"] = str(args.max_opd_delta_days)
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
        job_id=args.job_id,
    )
    output_dir = target_output_dir(
        args.galaxy,
        args.signal,
        args.color,
        products_root=args.products_root,
        signal_filter=signal_filter,
        color_filter=color_filter,
        signal_fingerprint=worker_identity["signal_fingerprint"],
        color_fingerprint=worker_identity["color_fingerprint"],
        job_id=args.job_id,
    )
    if args.products_root and args.attempt_output_key:
        output_dir = (
            Path(args.products_root).resolve()
            / slug(args.galaxy)
            / slug(f"{signal_filter}__{color_filter}")
            / slug(args.attempt_output_key)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.attempt_output_key:
        log_path = output_dir / "worker.log"
        cell_timings_path = output_dir / "cell_timings.jsonl"
        log_mode = "w"
    else:
        log_path = result_json_path(
            batch_root, args.galaxy, identity=worker_identity
        ).with_suffix(".log")
        cell_timings_path = log_path.with_name(
            f"{log_path.stem}_cell_timings.jsonl"
        )
        log_mode = "a"

    with log_path.open(log_mode) as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 88)
            print(
                f"[{timestamp()}] worker start: {args.galaxy}; pid={os.getpid()}; "
                f"job_id={args.job_id or '-'}"
            )
            print(f"[{timestamp()}] command: {shlex.join(sys.argv)}")
            print(f"[{timestamp()}] template: {template_path} sha256={sha256_file(template_path)}")
            print(f"[{timestamp()}] signal: {signal_filter} {Path(args.signal).resolve()}")
            print(f"[{timestamp()}] color:  {color_filter} {Path(args.color).resolve()}")
            print(f"[{timestamp()}] output directory: {output_dir}")
            print(f"[{timestamp()}] cell timings: {cell_timings_path}")
            try:
                validate_run_layout(
                    template_path, batch_root, args.products_root
                )
                validate_notebook_filter_pair(
                    template_path, signal_filter, color_filter
                )
                print(
                    f"[{timestamp()}] input identity (no full-file SHA): "
                    f"signal={worker_identity['signal_fingerprint']} "
                    f"color={worker_identity['color_fingerprint']}"
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
                    job_id=args.job_id,
                    input_sha256=None,
                    cell_timings_path=cell_timings_path,
                    worker_log_path=log_path,
                    attempt_output_key=args.attempt_output_key,
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
                    "worker_log_path": str(log_path.resolve()),
                    "cell_timings_path": str(cell_timings_path.resolve()),
                    "attempt_output_key": args.attempt_output_key,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                err_path = result_json_path(
                    batch_root, args.galaxy, identity=err
                )
                atomic_write_json(err_path, err, sort_keys=False)
                print(err["traceback"])
                print(f"[{timestamp()}] worker failed: {args.galaxy}")
                return 1


def write_summary(results, batch_root):
    batch_root = Path(batch_root)
    csv_path = batch_root / "sbf2_batch_results.csv"
    json_path = batch_root / "sbf2_batch_results.json"

    def stable_result_key(result):
        # This runner is deliberately restricted to the single GO-3055
        # F150W+F090W measurement per galaxy.  Failed/signal-interrupted
        # records may be written before the worker has copied filter metadata
        # into the result.  Including the filters in the key would therefore
        # leave a stale failure next to the later successful result.
        return (
            canonical_program(result.get("program") or "3055"),
            " ".join(str(result.get("galaxy") or "").casefold().split()),
        )

    merged = {}
    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(previous, list):
                for result in previous:
                    if isinstance(result, dict) and result.get("galaxy"):
                        merged[stable_result_key(result)] = result
        except Exception:
            pass
    for result in results:
        if isinstance(result, dict) and result.get("galaxy"):
            merged[stable_result_key(result)] = result
    results = sorted(
        merged.values(),
        key=lambda result: (
            canonical_program(result.get("program") or "3055"),
            str(result.get("galaxy") or ""),
        ),
    )
    atomic_write_json(json_path, as_builtin(results), sort_keys=False)

    keys = []
    for result in results:
        for key in result:
            if key not in keys:
                keys.append(key)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result.get(key, "") for key in keys})
    atomic_write_text(csv_path, buffer.getvalue())

    print(f"[{timestamp()}] summary CSV  -> {csv_path}")
    print(f"[{timestamp()}] summary JSON -> {json_path}")
    return csv_path, json_path


def link_residuals(results, batch_root):
    residual_dir = Path(batch_root) / "residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    name_counts = {}
    for result in results:
        if result.get("status") == "ok":
            key = slug(result["galaxy"])
            name_counts[key] = name_counts.get(key, 0) + 1
    for result in results:
        if result.get("status") != "ok":
            continue
        base_galaxy_slug = slug(result["galaxy"])
        galaxy_slug = base_galaxy_slug
        if name_counts.get(base_galaxy_slug, 0) > 1:
            run_key = str(result.get("job_id") or result.get("input_pair_key") or "run")
            run_key = run_key.removeprefix("job-")[:12]
            galaxy_slug = "__".join(
                [
                    base_galaxy_slug,
                    slug(result.get("signal_filter") or "signal"),
                    slug(result.get("color_filter") or "color"),
                    run_key,
                ]
            )
        if result.get("template_family") == SBF3_NOTEBOOK_FAMILY:
            residual_keys = [
                "working_residual_fits",
                "working_annuli_residual_fits",
            ]
        else:
            residual_keys = [
                "science_residual_fits",
                "inner_usable_residual_fits",
                "outer_usable_residual_fits",
            ]
        for key in residual_keys:
            if name_counts.get(base_galaxy_slug, 0) > 1:
                legacy = residual_dir / f"{base_galaxy_slug}_{key}.fits"
                if legacy.is_symlink():
                    legacy.unlink()
            source_path = result.get(key)
            if not source_path:
                continue
            src = Path(source_path)
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

    # A partially transferred file must never masquerade as a finished input.
    # Keep it under a .part name and publish it atomically only after the FITS
    # readability/size gate has passed.  Adopt partial files left by older
    # versions of this downloader so interrupted campaigns can still resume.
    partial = dest.with_name(f"{dest.name}.part")
    if dest.exists() and not partial.exists():
        os.replace(dest, partial)

    headers = {"Accept-Encoding": "identity"}
    start = partial.stat().st_size if partial.exists() else 0
    if start:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if start and response.status == 200:
                start = 0
                partial.unlink(missing_ok=True)
            mode = "ab" if start else "wb"
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and is_input_ready(partial, expected_size):
            os.replace(partial, dest)
            return True, "range-complete"
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, repr(exc)

    if is_input_ready(partial, expected_size):
        os.replace(partial, dest)
        return True, "downloaded"
    size = partial.stat().st_size if partial.exists() else 0
    return False, f"incomplete after transfer: {size}/{expected_size}"


def ensure_disk_space_for_downloads(
    data_root,
    completed_results,
    min_free_gb,
    cleanup_enabled=True,
    required_bytes=0,
    protected_input_paths=None,
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
    confined_root = Path(data_root).resolve()
    protected = {
        Path(path).resolve() for path in (protected_input_paths or [])
    }
    for result in completed_results:
        artifacts_valid = result_artifacts_still_valid(result)
        if result.get("status") != "ok" or not artifacts_valid:
            if result.get("status") == "ok":
                print(
                    f"[{timestamp()}] [DISK] preserving inputs for "
                    f"{result.get('galaxy', '<unknown>')}: artifacts are not "
                    "currently verified"
                )
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
                resolved_path = path.resolve()
                resolved_path.relative_to(confined_root)
            except ValueError:
                print(
                    f"[{timestamp()}] [DISK] refusing to remove input outside "
                    f"data root: {path}"
                )
                continue
            if resolved_path in protected:
                print(
                    f"[{timestamp()}] [DISK] preserving shared input still "
                    f"needed by a pending job: {path}"
                )
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


def result_artifacts_still_valid(result):
    """Recheck the deletion authority; a stale boolean is never sufficient."""
    if not result.get("artifacts_verified"):
        return False
    if result.get("template_family") != SBF3_NOTEBOOK_FAMILY:
        return False
    artifacts = {
        key: result.get(key)
        for key in SBF3_REQUIRED_FITS_KEYS
        if result.get(key)
    }
    if len(artifacts) != len(SBF3_REQUIRED_FITS_KEYS):
        return False
    try:
        out_dir = Path(result["out_dir"]).resolve()
        for value in artifacts.values():
            Path(value).resolve().relative_to(out_dir)
    except (KeyError, TypeError, ValueError):
        return False
    try:
        current = build_artifact_manifest(
            artifacts,
            include_sha256=True,
            validate_fits=True,
            require_astropy=True,
        )
    except Exception:
        return False
    if not current.get("ok"):
        return False
    recorded = {
        item.get("name"): item.get("sha256")
        for item in result.get("artifact_manifest", [])
    }
    return all(
        recorded.get(item["name"]) == item.get("sha256")
        for item in current["artifacts"]
    )


def load_completed_results(batch_root, allowed_job_ids=None):
    completed = []
    for result_file in sorted(Path(batch_root).glob("*_result.json")):
        try:
            result = json.loads(result_file.read_text())
        except Exception:
            continue
        if allowed_job_ids is not None and result.get("job_id") not in allowed_job_ids:
            continue
        if result.get("status") == "ok" and result.get("artifacts_verified"):
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
    eligible_cleanup_job_ids=None,
    protected_input_paths=None,
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
                partial = dest.with_name(f"{dest.name}.part")
                current_size = (
                    dest.stat().st_size
                    if dest.exists()
                    else partial.stat().st_size
                    if partial.exists()
                    else 0
                )
                remaining_bytes = (
                    max(int(expected_size) - current_size, 0)
                    if expected_size
                    else 0
                )
                while True:
                    completed_results = load_completed_results(
                        batch_root,
                        allowed_job_ids=eligible_cleanup_job_ids,
                    )
                    if ensure_disk_space_for_downloads(
                        data_root,
                        completed_results,
                        min_free_gb=min_free_gb,
                        cleanup_enabled=cleanup_enabled,
                        required_bytes=remaining_bytes,
                        protected_input_paths=protected_input_paths,
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
                    atomic_write_json(status_path, status, sort_keys=False)
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
                atomic_write_json(status_path, status, sort_keys=False)
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


def start_download_manager(
    args, targets, completed_results, protected_targets=None
):
    if args.no_download:
        return None
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--download-worker",
        "--data-root",
        resolve_cli_path(args.data_root),
        "--batch-root",
        resolve_cli_path(args.batch_root),
        "--download-retry-seconds",
        str(args.download_retry_seconds),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    manifests = target_csv_paths(
        getattr(args, "target_csv", None),
        getattr(args, "extra_target_csv", None),
    )
    cmd.extend(["--target-csv", str(manifests[0])])
    for manifest in manifests[1:]:
        cmd.extend(["--extra-target-csv", str(manifest)])
    programs = getattr(args, "programs", None)
    if programs:
        cmd.append("--programs")
        cmd.extend(programs)
    if args.no_cleanup_inputs:
        cmd.append("--no-cleanup-inputs")
    cleanup_job_ids = sorted(
        {
            result["job_id"]
            for result in completed_results
            if result.get("job_id") and result.get("artifacts_verified")
        }
    )
    if cleanup_job_ids:
        cmd.append("--cleanup-job-ids")
        cmd.extend(cleanup_job_ids)
    protected_paths = []
    for protected_target in (protected_targets or targets):
        try:
            files = local_target_files(
                protected_target, Path(resolve_cli_path(args.data_root))
            )
        except ValueError:
            # Keeps the helper usable by dry CLI/unit callers that only carry
            # names; real manifest targets always have both product names.
            continue
        protected_paths.extend(str(path.resolve()) for path in files.values())
    if protected_paths:
        cmd.append("--protected-inputs")
        cmd.extend(sorted(set(protected_paths)))
    if targets:
        # The download worker re-reads the manifest in a separate process.
        # Pass the already selected parent target set explicitly; otherwise a
        # narrow ``--galaxies`` run could download every enabled manifest row.
        cmd.append("--target-keys")
        cmd.extend(target_manifest_key(target) for target in targets)
        cmd.append("--galaxies")
        cmd.extend(target["name"] for target in targets)
    print(f"[{timestamp()}] starting download manager: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, start_new_session=(os.name != "nt"))
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


def target_manifest_key(target):
    return canonical_sha256(
        {
            "name": target.get("name"),
            "program": target.get("program"),
            "obsid": target.get("obsid"),
            "signal_product": target.get("signal_product"),
            "color_product": target.get("color_product"),
            "signal_filter": target.get("signal_filter"),
            "color_filter": target.get("color_filter"),
            "product_uris": campaign_product_uris(target),
        }
    )


def select_targets(
    targets, galaxies, allow_bulk_targets=False, target_keys=None, programs=None
):
    if programs is not None:
        wanted_programs = {canonical_program(value) for value in programs}
        available_programs = {
            canonical_program(target.get("program")) for target in targets
        }
        unknown = sorted(wanted_programs - available_programs)
        if unknown:
            raise ValueError(
                "requested programs are absent from the selected manifests: "
                + ", ".join(unknown)
            )
        targets = [
            target
            for target in targets
            if canonical_program(target.get("program")) in wanted_programs
        ]
    if target_keys is not None:
        wanted_keys = set(target_keys)
        targets = [
            target for target in targets if target_manifest_key(target) in wanted_keys
        ]
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


def validate_go3055_scope(args, targets, template):
    """Refuse accidental use of this article runner outside its frozen scope."""
    errors = []
    if notebook_family(template) != SBF2_NOTEBOOK_FAMILY:
        errors.append(f"template is not sbf-2: {template}")
    if getattr(args, "extra_target_csv", None):
        errors.append("--extra-target-csv is disabled in the GO-3055 runner")
    requested_programs = {
        canonical_program(value) for value in (getattr(args, "programs", None) or [])
    }
    if requested_programs != {"3055"}:
        errors.append(
            "the GO-3055 runner requires exactly --programs 3055"
        )
    if len(targets) > 14:
        errors.append(f"selected {len(targets)} targets; GO-3055 contains 14")
    for target in targets:
        program = canonical_program(target.get("program"))
        pair = (target.get("signal_filter"), target.get("color_filter"))
        if program != "3055":
            errors.append(f"{target['name']}: program is GO-{program}, not GO-3055")
        if pair != CURRENT_NOTEBOOK_FILTER_PAIR:
            errors.append(
                f"{target['name']}: filter pair is {pair}, expected "
                f"{CURRENT_NOTEBOOK_FILTER_PAIR}"
            )
    if errors:
        raise ValueError("GO-3055 scope validation failed: " + " | ".join(errors))


def _observation_time_from_header(header):
    for key in ("MJD-AVG", "MJD-BEG", "MJD-OBS"):
        value = header.get(key)
        if value is not None:
            try:
                return Time(float(value), format="mjd", scale="utc")
            except Exception:
                pass
    for key in ("DATE-AVG", "DATE-BEG", "DATE-OBS", "DATE"):
        value = header.get(key)
        if not value:
            continue
        text_value = str(value)
        if key == "DATE-OBS" and "T" not in text_value and header.get("TIME-OBS"):
            text_value = f"{text_value}T{header['TIME-OBS']}"
        try:
            return Time(text_value, scale="utc")
        except Exception:
            pass
    return None


def _observation_time_from_fits(path):
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if "SCI" in hdul:
            science_time = _observation_time_from_header(hdul["SCI"].header)
            if science_time is not None:
                return science_time
        return _observation_time_from_header(hdul[0].header)


def validated_local_opds(wss_opd_dir):
    """Return complete, timestamped WSS OPDs; partial files are ignored."""
    valid = []
    rejected = []
    for path in sorted(Path(wss_opd_dir).glob("*.fits")):
        try:
            if path.stat().st_size <= 0 or path.stat().st_size % 2880:
                raise ValueError("size is not a complete FITS block")
            with fits.open(path, mode="readonly", memmap=True) as hdul:
                hdul.verify("exception")
                if "RESULT_PHASE" not in hdul:
                    raise ValueError("RESULT_PHASE extension is absent")
                if not hdul["RESULT_PHASE"].shape:
                    raise ValueError("RESULT_PHASE extension is empty")
                opd_time = _observation_time_from_header(hdul[0].header)
            if opd_time is None:
                raise ValueError("physical observation time is absent")
            valid.append((path.resolve(), opd_time))
        except Exception as exc:
            rejected.append({"path": str(path.resolve()), "error": repr(exc)})
    return valid, rejected


def validate_offline_dependencies(args, targets, data_root, batch_root):
    """Fail before a long run when local inputs or time-matched OPDs are absent."""
    stpsf_data_dir = Path(args.stpsf_data_dir).resolve()
    wss_opd_dir = Path(args.wss_opd_dir).resolve()
    if not stpsf_data_dir.is_dir():
        raise RuntimeError(f"local STPSF reference data are absent: {stpsf_data_dir}")
    if not wss_opd_dir.is_dir():
        raise RuntimeError(f"project WSS OPD directory is absent: {wss_opd_dir}")

    missing_inputs = []
    input_header_errors = []
    for target in targets:
        if not target_inputs_ready(target, data_root):
            missing_inputs.append(target["name"])
            continue
        files = local_target_files(target, data_root)
        headers = {}
        for role, path in files.items():
            header = fits.getheader(path, 0)
            headers[role] = header
            actual_filter = str(header.get("FILTER", "")).strip().upper()
            expected_filter = target[f"{role}_filter"]
            if actual_filter != expected_filter:
                input_header_errors.append(
                    f"{target['name']} {role}: FITS FILTER={actual_filter}, "
                    f"expected {expected_filter}"
                )
        signal_instrument = str(
            headers["signal"].get("INSTRUME", "")
        ).strip().upper()
        color_instrument = str(
            headers["color"].get("INSTRUME", "")
        ).strip().upper()
        if signal_instrument != "NIRCAM" or color_instrument != signal_instrument:
            input_header_errors.append(
                f"{target['name']}: incompatible instruments "
                f"{signal_instrument}/{color_instrument}"
            )
    if args.no_download and missing_inputs:
        raise RuntimeError(
            "offline GO-3055 run has missing or unreadable inputs: "
            + ", ".join(missing_inputs)
        )
    if input_header_errors:
        raise RuntimeError(
            "GO-3055 FITS header validation failed: "
            + " | ".join(input_header_errors)
        )

    opds, rejected = validated_local_opds(wss_opd_dir)
    if not opds:
        raise RuntimeError(f"no complete WSS OPD FITS in {wss_opd_dir}")
    coverage_rows = []
    uncovered = []
    for target in targets:
        for role, path in local_target_files(target, data_root).items():
            if not path.is_file():
                continue
            science_time = _observation_time_from_fits(path)
            if science_time is None:
                uncovered.append(f"{target['name']} {role}: science time absent")
                continue
            candidates = sorted(
                [
                    (
                    abs(float((opd_time - science_time).to_value("day"))),
                    float((opd_time - science_time).to_value("day")),
                    opd_path,
                    )
                    for opd_path, opd_time in opds
                ],
                key=lambda item: item[0],
            )
            delta_abs, delta_signed, opd_path = candidates[0]
            ready = delta_abs <= args.max_opd_delta_days
            coverage_rows.append(
                {
                    "galaxy": target["name"],
                    "role": role,
                    "filter": target[f"{role}_filter"],
                    "science_path": str(path.resolve()),
                    "science_time": science_time.isot,
                    "opd_path": str(opd_path),
                    "opd_signed_delta_days": delta_signed,
                    "opd_delta_days": delta_abs,
                    "ready": ready,
                }
            )
            if not ready:
                uncovered.append(
                    f"{target['name']} {target[f'{role}_filter']}: "
                    f"nearest OPD is {delta_abs:.2f} d away"
                )
    report = {
        "created_at": timestamp(),
        "stpsf_data_dir": str(stpsf_data_dir),
        "wss_opd_dir": str(wss_opd_dir),
        "max_opd_delta_days": args.max_opd_delta_days,
        "valid_opd_count": len(opds),
        "valid_opds": [str(path) for path, _ in opds],
        "rejected_opds": rejected,
        "coverage": coverage_rows,
        "ready": not uncovered,
        "errors": uncovered,
    }
    report_path = Path(batch_root) / "offline_dependency_check.json"
    atomic_write_json(report_path, report, sort_keys=False)
    print(f"[{timestamp()}] offline dependency report -> {report_path}")
    if uncovered:
        raise RuntimeError(
            "offline WSS OPD coverage failed: " + " | ".join(uncovered)
        )
    return report


def campaign_product_uris(target):
    signal_product = target.get("signal_product")
    color_product = target.get("color_product")
    return {
        "signal": target.get("signal_product_uri")
        or (f"mast:JWST/product/{signal_product}" if signal_product else None),
        "color": target.get("color_product_uri")
        or (f"mast:JWST/product/{color_product}" if color_product else None),
    }


def target_inputs_ready(target, data_root):
    signal_path, color_path = target_paths(target, data_root)
    return is_input_ready(signal_path, target.get("signal_size")) and is_input_ready(
        color_path, target.get("color_size")
    )


def remaining_input_growth(targets, data_root):
    """Estimate bytes still needed to publish all selected final FITS files."""
    remaining_bytes = 0
    unknown_products = []
    for target in targets:
        files = local_target_files(target, data_root)
        for role, final_path in files.items():
            expected = optional_int(target.get(f"{role}_size"))
            if is_input_ready(final_path, expected):
                continue
            candidates = [
                final_path,
                final_path.with_name(f"{final_path.name}.part"),
                final_path.with_name(f"{final_path.name}.restart.part"),
            ]
            current = max(
                (path.stat().st_size for path in candidates if path.exists()),
                default=0,
            )
            if expected is None:
                unknown_products.append(
                    {
                        "target": target["name"],
                        "role": role,
                        "path": str(final_path),
                    }
                )
            else:
                remaining_bytes += max(expected - current, 0)
    return {
        "remaining_bytes": remaining_bytes,
        "remaining_gb": bytes_gb(remaining_bytes),
        "unknown_product_count": len(unknown_products),
        "unknown_products": unknown_products,
    }


def read_json_snapshot(path):
    path = Path(path)
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, repr(exc)


def input_wait_snapshot(target, data_root):
    files = local_target_files(target, data_root)
    snapshot = {}
    for role, final_path in files.items():
        expected = optional_int(target.get(f"{role}_size"))
        final_exists = final_path.exists()
        final_size = final_path.stat().st_size if final_exists else 0
        final_ready = is_input_ready(final_path, expected)
        fits_error = None
        if final_exists and not final_ready:
            _, fits_error = fits_is_readable(final_path)
        part_path = final_path.with_name(f"{final_path.name}.part")
        restart_path = final_path.with_name(f"{final_path.name}.restart.part")
        sidecar_path = final_path.with_name(f"{final_path.name}.part.json")
        part_size = part_path.stat().st_size if part_path.exists() else 0
        restart_size = restart_path.stat().st_size if restart_path.exists() else 0
        transfer_size = max(part_size, restart_size)
        percent = (
            100.0 * transfer_size / expected
            if expected and transfer_size
            else None
        )
        sidecar, sidecar_error = read_json_snapshot(sidecar_path)
        snapshot[role] = {
            "filter": target[f"{role}_filter"],
            "final_path": str(final_path.resolve()),
            "expected_size_bytes": expected,
            "final_exists": final_exists,
            "final_size_bytes": final_size,
            "final_ready": final_ready,
            "final_fits_error": fits_error,
            "part_path": str(part_path.resolve()),
            "part_exists": part_path.exists(),
            "part_size_bytes": part_size,
            "restart_part_path": str(restart_path.resolve()),
            "restart_part_exists": restart_path.exists(),
            "restart_part_size_bytes": restart_size,
            "transfer_percent": percent,
            "part_metadata": sidecar,
            "part_metadata_error": (
                None if sidecar_error == "missing" else sidecar_error
            ),
        }
    return snapshot


def external_download_snapshot(status_path, target):
    if status_path is None:
        return None
    path = Path(status_path)
    status, error = read_json_snapshot(path)
    base = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "mtime": path.stat().st_mtime if path.exists() else None,
        "read_error": None if error == "missing" else error,
    }
    if not isinstance(status, dict):
        return base
    wanted_program = canonical_program(target.get("program"))
    wanted_target = " ".join(target["name"].casefold().split())
    wanted_obsid = str(target.get("obsid") or "").strip().casefold()
    matching_results = []
    for result in status.get("results") or []:
        if canonical_program(result.get("program")) != wanted_program:
            continue
        if " ".join(str(result.get("target") or "").casefold().split()) != wanted_target:
            continue
        result_obsid = str(result.get("obsid") or "").strip().casefold()
        if wanted_obsid and result_obsid and result_obsid != wanted_obsid:
            continue
        matching_results.append(result)
    return {
        **base,
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "programs": status.get("programs"),
        "interrupted": status.get("interrupted"),
        "counts": status.get("counts"),
        "matching_results": matching_results,
    }


def format_input_progress(target, inputs):
    parts = []
    for role in ("signal", "color"):
        item = inputs[role]
        if item["final_ready"]:
            state_text = f"ready {item['final_size_bytes']} B"
        elif item["part_exists"] or item["restart_part_exists"]:
            transferred = max(
                item["part_size_bytes"], item["restart_part_size_bytes"]
            )
            percent = item.get("transfer_percent")
            state_text = f"part {transferred} B"
            if percent is not None:
                state_text += f" ({percent:.1f}%)"
        elif item["final_exists"]:
            state_text = f"invalid final {item['final_size_bytes']} B"
        else:
            state_text = "missing"
        parts.append(f"{role}:{item['filter']}={state_text}")
    return f"{target['name']} GO-{canonical_program(target.get('program'))}: " + "; ".join(parts)


def verified_campaign_result(target, batch_root, identity):
    """Load verified work by target/input identity, independent of notebook SHA.

    Version-1 job ids included the notebook and campaign digests.  The fallback
    scan adopts only successful results with the same galaxy, filters and input
    files, then republishes a tiny result receipt under the version-2 job id.
    Failed or interrupted receipts are never candidates.
    """
    batch_root = Path(batch_root)
    direct_path = result_json_path(batch_root, target["name"], identity=identity)
    prefix = "__".join(
        [
            slug(target["name"]),
            slug(identity["signal_filter"]),
            slug(identity["color_filter"]),
        ]
    )
    candidate_paths = [direct_path]
    candidate_paths.extend(
        path
        for path in sorted(batch_root.glob(f"{prefix}__*_result.json"))
        if path != direct_path
    )

    legacy_identity = dict(identity)
    legacy_identity.pop("job_id", None)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            result = json.loads(path.read_text())
        except Exception:
            continue
        if result.get("status") != "ok":
            continue
        if identity.get("template_family") == SBF2_NOTEBOOK_FAMILY:
            validated = validate_reusable_result(path, target)
            if validated is None:
                continue
            result = validated
        match_identity = identity if path == direct_path else legacy_identity
        if not result_matches_identity(result, match_identity):
            continue
        if result.get("template_family") == SBF3_NOTEBOOK_FAMILY:
            if not result_artifacts_still_valid(result):
                continue

        if path != direct_path and identity.get("job_id"):
            adopted = dict(result)
            adopted["producer_job_id"] = result.get("job_id")
            adopted["producer_template_sha256"] = result.get("template_sha256")
            adopted["job_id"] = identity["job_id"]
            adopted["reused_from_result_json"] = str(path.resolve())
            adopted["reused_for_template_sha256"] = identity.get("template_sha256")
            adopted["reused_at"] = timestamp()
            adopted["reused_at_unix"] = time.time()
            atomic_write_json(direct_path, as_builtin(adopted), sort_keys=False)
            return adopted
        return result
    return None


def wait_for_campaign_inputs(
    target,
    data_root,
    process,
    deadline,
    signal_controller,
    poll_seconds,
    timeout_seconds=0,
    state=None,
    run_id=None,
    job_id=None,
    emergency_ram_gb=0,
    critical_free_gb=0,
    external_status_path=None,
    event_log_path=None,
):
    started = time.monotonic()
    report_interval = max(0.1, float(poll_seconds))
    next_report = started
    initial_inputs = input_wait_snapshot(target, data_root)
    emit_campaign_event(
        event_log_path,
        "INPUT_WAIT_STARTED",
        state=state,
        run_id=run_id,
        job_id=job_id,
        payload={
            "target": target["name"],
            "program": canonical_program(target.get("program")),
            "obsid": target.get("obsid"),
            "filters": {
                "signal": target["signal_filter"],
                "color": target["color_filter"],
            },
            "inputs": initial_inputs,
            "external_downloader": external_download_snapshot(
                external_status_path, target
            ),
        },
    )

    def abort(reason):
        inputs = input_wait_snapshot(target, data_root)
        emit_campaign_event(
            event_log_path,
            "INPUT_WAIT_ABORTED",
            state=state,
            run_id=run_id,
            job_id=job_id,
            payload={
                "target": target["name"],
                "reason": reason,
                "elapsed_seconds": time.monotonic() - started,
                "inputs": inputs,
                "external_downloader": external_download_snapshot(
                    external_status_path, target
                ),
            },
        )
        print(f"[{timestamp()}] [INPUT_WAIT_ABORTED] {target['name']}: {reason}")
        return False, reason

    while True:
        if target_inputs_ready(target, data_root):
            inputs = input_wait_snapshot(target, data_root)
            emit_campaign_event(
                event_log_path,
                "INPUT_READY",
                state=state,
                run_id=run_id,
                job_id=job_id,
                payload={
                    "target": target["name"],
                    "elapsed_seconds": time.monotonic() - started,
                    "inputs": inputs,
                    "external_downloader": external_download_snapshot(
                        external_status_path, target
                    ),
                },
            )
            print(f"[{timestamp()}] [INPUT_READY] {format_input_progress(target, inputs)}")
            return True, "ready"
        if signal_controller.stop_requested:
            return abort("signal")
        if deadline.hard_expired:
            return abort("deadline")
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            return abort("input-timeout")
        if process is not None and process.poll() is not None:
            return abort(f"downloader-exited-{process.returncode}")
        now = time.monotonic()
        if now >= next_report:
            disk, mem = log_resources(f"waiting-input {target['name']}", data_root)
            inputs = input_wait_snapshot(target, data_root)
            downloader = external_download_snapshot(external_status_path, target)
            elapsed = now - started
            print(
                f"[{timestamp()}] [INPUT_WAIT] elapsed={elapsed:.0f}s; "
                f"{format_input_progress(target, inputs)}"
            )
            payload = {
                "target": target["name"],
                "elapsed_seconds": elapsed,
                "inputs": inputs,
                "disk": disk,
                "memory": mem,
                "external_downloader": downloader,
            }
            emit_campaign_event(
                event_log_path,
                "INPUT_WAIT_HEARTBEAT",
                state=state,
                run_id=run_id,
                job_id=job_id,
                payload=payload,
            )
            if state is not None and run_id is not None:
                state.record_resource_sample(
                    run_id,
                    job_id=job_id,
                    ram_total_bytes=mem.get("total") if mem else None,
                    ram_available_bytes=mem.get("available") if mem else None,
                    disk_total_bytes=disk.get("total"),
                    disk_free_bytes=disk.get("free"),
                    metrics={"phase": "download-wait", **payload},
                )
            if (
                emergency_ram_gb > 0
                and mem.get("available_gb") is not None
                and mem["available_gb"] < emergency_ram_gb
            ):
                return abort("resource-emergency-ram")
            if critical_free_gb > 0 and disk["free_gb"] < critical_free_gb:
                return abort("resource-critical-disk")
            next_report = now + report_interval
        remaining = deadline.remaining()
        sleep_seconds = min(5.0, max(0.1, float(poll_seconds)))
        if remaining is not None:
            sleep_seconds = min(sleep_seconds, max(0.1, remaining))
        signal_controller.wait(sleep_seconds)


def wait_for_worker_capacity(
    args,
    data_root,
    deadline,
    signal_controller,
    state=None,
    run_id=None,
    job_id=None,
    pending_targets=None,
):
    while True:
        if signal_controller.stop_requested or deadline.hard_expired:
            return False
        disk, mem = log_resources("before-worker", data_root)
        available_gb = mem.get("available_gb") if mem else None
        enough_ram = (
            available_gb is None or available_gb >= args.min_available_ram_gb
        )
        input_growth = remaining_input_growth(pending_targets or [], data_root)
        download_reserve_gb = float(
            getattr(args, "external_download_reserve_gb", 0.0) or 0.0
        )
        if (
            input_growth["remaining_bytes"] <= 0
            and input_growth["unknown_product_count"] == 0
        ):
            download_reserve_gb = 0.0
        estimated_output_gb = float(
            getattr(args, "estimated_worker_output_gb", 0.0) or 0.0
        )
        archive_required_free_gb = (
            download_reserve_gb
            + input_growth["remaining_gb"]
            + estimated_output_gb
        )
        processing_floor_gb = float(
            getattr(args, "min_processing_free_gb", 0.0) or 0.0
        )
        required_free_gb = max(
            args.critical_free_gb,
            processing_floor_gb + estimated_output_gb,
            archive_required_free_gb,
        )
        enough_disk = disk["free_gb"] >= required_free_gb
        if state is not None and run_id is not None:
            state.record_resource_sample(
                run_id,
                job_id=job_id,
                ram_total_bytes=mem.get("total") if mem else None,
                ram_available_bytes=mem.get("available") if mem else None,
                disk_total_bytes=disk.get("total"),
                disk_free_bytes=disk.get("free"),
                metrics={
                    "phase": "worker-capacity",
                    "disk": disk,
                    "memory": mem,
                    "input_growth": input_growth,
                    "download_reserve_gb": download_reserve_gb,
                    "estimated_worker_output_gb": estimated_output_gb,
                    "processing_floor_gb": processing_floor_gb,
                    "required_free_gb": required_free_gb,
                    "admitted": enough_ram and enough_disk,
                },
            )
        if enough_ram and enough_disk:
            print(
                f"[{timestamp()}] [RESOURCE] worker admitted: "
                f"disk={disk['free_gb']:.1f} GB, required={required_free_gb:.1f} GB; "
                f"remaining inputs={input_growth['remaining_gb']:.1f} GB"
            )
            return True
        print(
            f"[{timestamp()}] [RESOURCE] worker blocked: "
            f"RAM={available_gb if available_gb is not None else 'unknown'} GB, "
            f"disk={disk['free_gb']:.1f} GB (required={required_free_gb:.1f} GB; "
            f"remaining inputs={input_growth['remaining_gb']:.1f} GB; "
            f"download reserve={download_reserve_gb:.1f} GB; "
            f"processing floor={processing_floor_gb:.1f} GB; "
            f"next products={estimated_output_gb:.1f} GB)"
        )
        gc.collect()
        remaining = deadline.remaining()
        sleep_seconds = min(5.0, max(0.1, float(args.poll_seconds)))
        if remaining is not None:
            sleep_seconds = min(sleep_seconds, max(0.1, remaining))
        signal_controller.wait(sleep_seconds)


def record_resource_sample(state, run_id, job_id, attempt_id, sample):
    ram = sample.get("system_ram") or {}
    worker = sample.get("worker") or {}
    swap = sample.get("swap") or {}
    disk = sample.get("disk") or {}
    state.record_resource_sample(
        run_id,
        job_id=job_id,
        attempt_id=attempt_id,
        sampled_at=sample.get("timestamp_unix"),
        ram_total_bytes=ram.get("total_bytes"),
        ram_available_bytes=sample.get("available_ram_bytes"),
        process_rss_bytes=worker.get("rss_bytes"),
        children_rss_bytes=worker.get("children_rss_bytes"),
        swap_used_bytes=sample.get("swap_used_bytes"),
        disk_total_bytes=disk.get("total_bytes"),
        disk_free_bytes=sample.get("disk_free_bytes"),
        metrics=sample,
    )


def validate_parent_args(args):
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.resource_sample_seconds <= 0:
        raise ValueError("--resource-sample-seconds must be positive")
    if args.soft_stop_minutes < 0:
        raise ValueError("--soft-stop-minutes must not be negative")
    if args.emergency_available_ram_gb > args.min_available_ram_gb:
        raise ValueError(
            "--emergency-available-ram-gb must not exceed "
            "--min-available-ram-gb"
        )
    nonnegative = {
        "--wall-time-hours": args.wall_time_hours,
        "--worker-timeout-hours": args.worker_timeout_hours,
        "--min-free-gb": args.min_free_gb,
        "--min-available-ram-gb": args.min_available_ram_gb,
        "--emergency-available-ram-gb": args.emergency_available_ram_gb,
        "--max-worker-rss-gb": args.max_worker_rss_gb,
        "--critical-free-gb": args.critical_free_gb,
        "--min-processing-free-gb": getattr(
            args, "min_processing_free_gb", 0.0
        ),
        "--external-download-reserve-gb": getattr(
            args, "external_download_reserve_gb", 0.0
        ),
        "--estimated-worker-output-gb": getattr(
            args, "estimated_worker_output_gb", 0.0
        ),
        "--max-opd-delta-days": getattr(args, "max_opd_delta_days", 0.0),
        "--worker-term-grace-seconds": args.worker_term_grace_seconds,
        "--worker-kill-grace-seconds": args.worker_kill_grace_seconds,
        "--timeout-seconds": args.timeout_seconds,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise ValueError(f"must not be negative: {', '.join(invalid)}")
    if args.poll_seconds <= 0 or args.download_retry_seconds <= 0:
        raise ValueError("--poll-seconds and --download-retry-seconds must be positive")


def campaign_root_from_args(args):
    batch_root = Path(args.batch_root).resolve()
    return (
        Path(args.campaign_root).resolve()
        if getattr(args, "campaign_root", None)
        else batch_root / "campaign"
    )


def command_provenance(args):
    argv = getattr(args, "_argv", None)
    if argv is None:
        command = list(sys.argv)
    else:
        command = [str(Path(__file__).resolve()), *map(str, argv)]
    return {
        "argv": command,
        "shell_command": shlex.join(command),
        "resolved_args": as_builtin(
            {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            }
        ),
    }


def git_provenance(path):
    result = {"root": str(Path(path).resolve()), "head": None, "status": None}
    try:
        result["head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["status"] = subprocess.run(
            ["git", "status", "--short"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def package_provenance():
    packages = (
        "numpy",
        "astropy",
        "photutils",
        "stpsf",
        "scipy",
        "matplotlib",
        "pandas",
        "psutil",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_run_provenance(
    campaign_root,
    *,
    args,
    run_id,
    template,
    manifests,
    targets,
    external_status_path,
):
    campaign_root = Path(campaign_root)
    snapshots_root = campaign_root / "input_contract_snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for manifest in manifests:
        digest = sha256_file(manifest)
        snapshot = snapshots_root / f"{manifest.stem}__{digest[:12]}{manifest.suffix}"
        if not snapshot.exists():
            atomic_write_text(snapshot, manifest.read_text(encoding="utf-8"))
        manifest_records.append(
            {
                "path": str(manifest),
                "sha256": digest,
                "size_bytes": manifest.stat().st_size,
                "snapshot": str(snapshot.resolve()),
            }
        )
    template_digest = sha256_file(template)
    template_snapshot = (
        snapshots_root
        / f"{template.stem}__{template_digest[:12]}{template.suffix}"
    )
    if not template_snapshot.exists():
        atomic_write_text(template_snapshot, template.read_text(encoding="utf-8"))

    provenance = {
        "created_at": timestamp(),
        "created_at_unix": time.time(),
        "run_id": run_id,
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "command": command_provenance(args),
        "environment": {
            "packages": package_provenance(),
            "STPSF_PATH": os.environ.get("STPSF_PATH"),
            "CRDS_PATH": os.environ.get("CRDS_PATH"),
            "CRDS_CONTEXT": os.environ.get("CRDS_CONTEXT"),
        },
        "git": git_provenance(PROJECT_ROOT),
        "template": {
            "path": str(template),
            "sha256": template_digest,
            "family": notebook_family(template),
            "snapshot": str(template_snapshot.resolve()),
        },
        "manifests": manifest_records,
        "programs": [canonical_program(value) for value in (getattr(args, "programs", None) or [])],
        "external_download_status": (
            str(Path(external_status_path).resolve())
            if external_status_path is not None
            else None
        ),
        "selected_target_count": len(targets),
        "selected_targets": targets,
        "required_sbf3_fits": list(SBF3_REQUIRED_FITS_KEYS),
    }
    path = campaign_root / "run_provenance.json"
    atomic_write_json(path, as_builtin(provenance), sort_keys=False)
    append_jsonl(campaign_root / "invocations.jsonl", provenance)
    return path


def write_campaign_report(campaign_root, snapshot, target_jobs, results_by_job):
    lines = [
        f"SBF campaign report: {snapshot['run']['run_id']}",
        f"Generated: {timestamp()}",
        f"State: {snapshot['run']['state']}",
        f"Counts: {json.dumps(snapshot['counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "Targets:",
    ]
    for target, job in target_jobs:
        stored = next(
            (
                item
                for item in snapshot.get("jobs", [])
                if item.get("job_id") == job["job_id"]
            ),
            job,
        )
        result = results_by_job.get(job["job_id"], {})
        lines.append(
            f"- {target['name']} | GO-{canonical_program(target.get('program'))} "
            f"| {target['signal_filter']}+{target['color_filter']} "
            f"| {stored.get('state')} | job_id={job['job_id']}"
        )
        if result.get("error"):
            lines.append(f"  error: {result['error']}")
        if result.get("artifact_manifest_path"):
            lines.append(f"  artifacts: {result['artifact_manifest_path']}")
        if result.get("cell_timings_path"):
            lines.append(f"  cell timings: {result['cell_timings_path']}")
    report_path = Path(campaign_root) / "campaign_report.txt"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def _run_parent_impl(args):
    global _ACTIVE_CAMPAIGN_LOCK
    validate_parent_args(args)
    template = Path(args.template).resolve()
    data_root = Path(args.data_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    products_root = (
        Path(args.products_root).resolve() if args.products_root else None
    )
    validate_run_layout(template, batch_root, products_root)
    template_family = notebook_family(template)
    batch_root.mkdir(parents=True, exist_ok=True)

    raw_target_csv = getattr(args, "target_csv", None)
    raw_extra_target_csv = getattr(args, "extra_target_csv", None)
    manifests = (
        target_csv_paths(raw_target_csv, raw_extra_target_csv)
        if raw_target_csv or raw_extra_target_csv
        else []
    )
    if manifests:
        targets = load_manifest_targets(
            raw_target_csv,
            data_root,
            raw_extra_target_csv,
        )
    else:
        targets = [normalize_target(t) for t in TARGETS]

    targets = select_targets(
        targets,
        args.galaxies,
        allow_bulk_targets=args.allow_bulk_targets,
        target_keys=args.target_keys,
        programs=getattr(args, "programs", None),
    )
    if not targets:
        raise RuntimeError(f"no targets selected: {args.galaxies}")
    validate_go3055_scope(args, targets, template)
    validate_offline_dependencies(
        args,
        targets,
        data_root,
        batch_root,
    )

    campaign_root = campaign_root_from_args(args)
    event_log_path = campaign_root / "campaign_events.jsonl"
    external_status_path = None
    if getattr(args, "external_download_status", None):
        external_status_path = Path(args.external_download_status).resolve()
    elif args.no_download:
        external_status_path = (
            data_root / "download_go3055_go7763_status.json"
        ).resolve()
    campaign_lock = acquire_campaign_lock(campaign_root)
    _ACTIVE_CAMPAIGN_LOCK = campaign_lock
    atexit.register(release_campaign_lock, campaign_lock)
    campaign_config = {
        # Version 2 separates durable target/input identity from notebook SHA.
        # The config bump starts one clean run while verified old products are
        # adopted below instead of being recomputed.
        "schema_version": 2,
        "job_identity_version": 2,
        "notebook_family": template_family,
        "required_fits": list(SBF3_REQUIRED_FITS_KEYS)
        if template_family == SBF3_NOTEBOOK_FAMILY
        else list(SBF2_REQUIRED_FITS_KEYS),
    }
    state = CampaignState(campaign_root)
    run = state.create_or_resume_run(
        template_sha256=sha256_file(template),
        config=campaign_config,
        run_id=args.run_id,
        wall_time_seconds=(
            None if args.wall_time_hours <= 0 else args.wall_time_hours * 3600
        ),
        soft_stop_seconds=args.soft_stop_minutes * 60,
        metadata={
            "template": str(template),
            "batch_root": str(batch_root),
            "products_root": str(products_root) if products_root else None,
            "target_csv": str(manifests[0]) if len(manifests) == 1 else None,
            "target_csvs": [str(path) for path in manifests],
            "programs": [
                canonical_program(value)
                for value in (getattr(args, "programs", None) or [])
            ],
            "external_download_status": (
                str(external_status_path) if external_status_path else None
            ),
            "download_mode": "external-consumer" if args.no_download else "integrated",
        },
        resume=not args.new_run,
    )
    run_id = run["run_id"]
    args._active_campaign_state = state
    args._active_run_id = run_id
    provenance_path = write_run_provenance(
        campaign_root,
        args=args,
        run_id=run_id,
        template=template,
        manifests=manifests,
        targets=targets,
        external_status_path=external_status_path,
    )
    emit_campaign_event(
        event_log_path,
        "CAMPAIGN_INVOCATION",
        state=state,
        run_id=run_id,
        payload={
            "resumed": bool(run.get("resume_count")),
            "provenance_path": str(provenance_path),
            "target_count": len(targets),
            "programs": sorted(
                {canonical_program(target.get("program")) for target in targets}
            ),
        },
    )
    print(
        f"[{timestamp()}] campaign {run_id}: {len(targets)} targets; "
        f"manifests={', '.join(str(path) for path in manifests)}"
    )
    print(f"[{timestamp()}] provenance -> {provenance_path}")
    print(f"[{timestamp()}] event log  -> {event_log_path}")
    if external_status_path is not None:
        print(f"[{timestamp()}] external downloader status -> {external_status_path}")
    for position, target in enumerate(targets, start=1):
        print(
            f"[{timestamp()}] queue {position:03d}/{len(targets):03d}: "
            f"GO-{canonical_program(target.get('program'))} {target['name']} "
            f"signal={target['signal_filter']} color={target['color_filter']}"
        )
    recovered = state.recover_incomplete_work(run_id)
    if recovered:
        print(f"[{timestamp()}] recovered {len(recovered)} interrupted jobs")

    job_specs = []
    for position, target in enumerate(targets):
        job_specs.append(
            {
                "target": target["name"],
                "program": target.get("program"),
                "obsid": target.get("obsid"),
                "product_uris": campaign_product_uris(target),
                "filters": {
                    "signal": target["signal_filter"],
                    "color": target["color_filter"],
                },
                "payload": target,
                "priority": int(target.get("priority") or 0),
                "queue_position": position,
            }
        )
    jobs = state.upsert_jobs(run_id, job_specs)
    target_jobs = list(zip(targets, jobs))

    # This compact CSV is the human-facing and SHA-independent completion
    # ledger.  SQLite remains the detailed attempt/event store, while only a
    # physically validated ``done`` row suppresses another notebook run.
    target_status_path = campaign_root / TARGET_STATUS_FILENAME
    target_status_rows = read_target_status(target_status_path)
    ensure_target_rows(target_status_rows, targets)
    sqlite_to_public_status = {
        "RUNNING": "running",
        "VERIFYING": "running",
        "FAILED": "failed",
        "RETRY_WAIT": "failed",
        "SKIPPED": "skipped",
        "CANCELLED": "skipped",
    }
    for target, job in target_jobs:
        status_row = target_status_rows[target_status_key(target)]
        if args.force_reprocess:
            update_target_status(
                target_status_rows,
                target,
                "pending",
                result_value="",
                result_unit="",
                selected_region="",
                selection_method="",
                qc="",
                result_json="",
                error="forced reprocessing requested",
            )
            continue
        if status_row.get("status") == "done":
            continue
        update_target_status(
            target_status_rows,
            target,
            sqlite_to_public_status.get(job["state"], "pending"),
            result_value="",
            result_unit="",
            selected_region="",
            selection_method="",
            qc="",
            result_json="",
            error=job.get("last_error") or "",
        )
    write_target_status(target_status_path, target_status_rows)
    print(f"[{timestamp()}] target status -> {target_status_path}")

    def persist_target_status(target, public_status, **details):
        if public_status != "done":
            for field in (
                "result_value",
                "result_unit",
                "selected_region",
                "selection_method",
                "qc",
                "result_json",
            ):
                details.setdefault(field, "")
        if details.get("error") is not None:
            details["error"] = str(details["error"])[:2000]
        row = update_target_status(
            target_status_rows,
            target,
            public_status,
            **details,
        )
        write_target_status(target_status_path, target_status_rows)
        return row

    selected_job_ids = {job["job_id"] for job in jobs}
    for stored_job in state.list_jobs(run_id):
        if (
            stored_job["job_id"] not in selected_job_ids
            and stored_job["state"] != "SUCCEEDED"
        ):
            state.transition_job(
                run_id,
                stored_job["job_id"],
                "SKIPPED",
                force=True,
                details={"reason": "not selected by resumed invocation"},
            )

    now_wall = time.time()
    remaining = (
        None
        if run.get("deadline_at") is None
        else max(0.0, run["deadline_at"] - now_wall)
    )
    reserve = (
        0.0
        if run.get("deadline_at") is None or run.get("soft_stop_at") is None
        else max(0.0, run["deadline_at"] - run["soft_stop_at"])
    )
    deadline = Deadline(
        wall_time_seconds=remaining,
        soft_stop_seconds=reserve,
    )
    completed_results = []
    results_by_job = {}
    identities = {}

    for target, job in target_jobs:
        signal_path, color_path = target_paths(target, data_root)
        identity = expected_run_identity(
            template,
            target,
            signal_path,
            color_path,
            products_root=products_root,
            job_id=job["job_id"],
        )
        identities[job["job_id"]] = identity
        status_row = target_status_rows[target_status_key(target)]
        existing = None
        if not args.force_reprocess:
            existing = reusable_result_from_status(target_status_rows, target)
        reuse_source = (
            status_row.get("result_json") if existing is not None else None
        )
        reuse_kind = "target-status" if existing is not None else None
        if existing is None and not args.force_reprocess:
            existing = verified_campaign_result(target, batch_root, identity)
            if existing is not None:
                reuse_source = str(
                    result_json_path(
                        batch_root, target["name"], identity=identity
                    ).resolve()
                )
                reuse_kind = "campaign-result"
        if existing is not None:
            existing = dict(existing)
            existing["go3055_qc"] = evaluate_go3055_qc(existing)
            existing["qc_status"] = existing["go3055_qc"]["status"]
            existing["qc_flags"] = existing["go3055_qc"]["flags"]
            producer_job_id = existing.get("producer_job_id") or existing.get(
                "job_id"
            )
            existing["producer_job_id"] = producer_job_id
            existing["job_id"] = job["job_id"]
            existing["reused_from_result_json"] = reuse_source
            existing["reuse_kind"] = reuse_kind
            completed_results.append(existing)
            results_by_job[job["job_id"]] = existing
            for artifact in existing.get("artifact_manifest", []):
                state.record_artifact(
                    run_id,
                    job["job_id"],
                    attempt_id=None,
                    kind=artifact["name"],
                    path=artifact["path"],
                    size_bytes=artifact.get("size_bytes"),
                    sha256=artifact.get("sha256"),
                    verified=artifact.get("ok"),
                    metadata={
                        "fits_valid": artifact.get("fits_valid"),
                        "csv_valid": artifact.get("csv_valid"),
                        "row_count": artifact.get("row_count"),
                        "reused": True,
                        "producer_job_id": producer_job_id,
                    },
                )
            emit_campaign_event(
                event_log_path,
                "RESULT_REUSED",
                state=state,
                run_id=run_id,
                job_id=job["job_id"],
                payload={
                    "target": target["name"],
                    "producer_job_id": producer_job_id,
                    "producer_template_sha256": existing.get(
                        "producer_template_sha256",
                        existing.get("template_sha256"),
                    ),
                    "current_template_sha256": identity.get("template_sha256"),
                    "reused_from_result_json": existing.get(
                        "reused_from_result_json"
                    ),
                    "artifact_count": len(existing.get("artifact_manifest", [])),
                    "reuse_kind": reuse_kind,
                    "target_status_path": str(target_status_path),
                },
            )
            science_fields = science_status_fields(existing)
            persist_target_status(
                target,
                "done",
                method=target_status_measurement_method(existing),
                **science_fields,
                result_json=reuse_source,
                qc=existing["go3055_qc"]["summary"],
                error="",
            )
            if job["state"] != "SUCCEEDED":
                state.transition_job(
                    run_id, job["job_id"], "SUCCEEDED", force=True,
                    details={"reason": "verified result reused"},
                )
        elif status_row.get("status") == "done":
            # A textual ``done`` flag is authoritative only while its result
            # and required products remain physically readable.  Demote a
            # stale row instead of silently suppressing necessary work.
            persist_target_status(
                target,
                "pending",
                qc="",
                error="stored done result failed product validation",
            )
            if job["state"] == "SUCCEEDED":
                state.transition_job(
                    run_id,
                    job["job_id"],
                    "PENDING",
                    force=True,
                    error="stored done result failed product validation",
                )
        elif job["state"] == "SUCCEEDED":
            persist_target_status(
                target,
                "pending",
                qc="",
                error="stored success failed artifact verification",
            )
            state.transition_job(
                run_id, job["job_id"], "PENDING", force=True,
                error="stored success failed artifact verification",
            )
        elif job["state"] in {"SKIPPED", "CANCELLED"}:
            persist_target_status(target, "pending", error="")
            state.transition_job(
                run_id,
                job["job_id"],
                "PENDING",
                force=True,
                details={"reason": "selected again"},
            )

    prefetch_proc = None
    prefetch_job_id = None
    stop_reason = None
    with SignalController() as signal_controller:
        try:
            for index, (target, job) in enumerate(target_jobs):
                job_id = job["job_id"]
                if job_id in results_by_job:
                    continue
                if prefetch_proc is not None and prefetch_job_id != job_id:
                    stop_download_manager(prefetch_proc)
                    prefetch_proc = None
                    prefetch_job_id = None
                if signal_controller.stop_requested:
                    stop_reason = signal_controller.signal_name or "signal"
                    break
                if not deadline.may_start():
                    stop_reason = "soft-deadline"
                    state.set_run_state(run_id, "SOFT_STOPPED")
                    break

                signal_path, color_path = target_paths(target, data_root)
                identity = expected_run_identity(
                    template, target, signal_path, color_path,
                    products_root=products_root, job_id=job_id,
                )
                identities[job_id] = identity

                current_job = state.get_job(run_id, job_id)
                if current_job["attempt_count"] >= args.max_attempts:
                    state.transition_job(
                        run_id, job_id, "FAILED", force=True,
                        error="maximum attempt count reached",
                    )
                    results_by_job[job_id] = {
                        "job_id": job_id,
                        "galaxy": target["name"],
                        "status": "failed",
                        "error": "maximum attempt count reached",
                    }
                    persist_target_status(
                        target,
                        "failed",
                        error="maximum attempt count reached",
                    )
                    continue

                if target_inputs_ready(target, data_root):
                    state.transition_job(run_id, job_id, "READY", force=True)
                    ready_inputs = input_wait_snapshot(target, data_root)
                    emit_campaign_event(
                        event_log_path,
                        "INPUT_READY_INITIAL",
                        state=state,
                        run_id=run_id,
                        job_id=job_id,
                        payload={
                            "target": target["name"],
                            "program": canonical_program(target.get("program")),
                            "inputs": ready_inputs,
                        },
                    )
                    print(
                        f"[{timestamp()}] [INPUT_READY_INITIAL] "
                        f"{format_input_progress(target, ready_inputs)}"
                    )
                    if prefetch_job_id == job_id:
                        stop_download_manager(prefetch_proc)
                        prefetch_proc = None
                        prefetch_job_id = None
                else:
                    state.transition_job(run_id, job_id, "DOWNLOADING", force=True)
                    if prefetch_job_id != job_id:
                        stop_download_manager(prefetch_proc)
                        protected_targets = [
                            candidate_target
                            for candidate_target, candidate_job in target_jobs
                            if candidate_job["job_id"] not in results_by_job
                        ]
                        prefetch_proc = start_download_manager(
                            args,
                            [target],
                            completed_results,
                            protected_targets=protected_targets,
                        )
                        prefetch_job_id = job_id
                    ready, reason = wait_for_campaign_inputs(
                        target, data_root, prefetch_proc, deadline,
                        signal_controller, args.poll_seconds, args.timeout_seconds,
                        state=state, run_id=run_id, job_id=job_id,
                        emergency_ram_gb=args.emergency_available_ram_gb,
                        critical_free_gb=args.critical_free_gb,
                        external_status_path=external_status_path,
                        event_log_path=event_log_path,
                    )
                    if not ready:
                        state.transition_job(
                            run_id, job_id, "INTERRUPTED", force=True, error=reason
                        )
                        persist_target_status(target, "pending", error=reason)
                        stop_reason = reason
                        break
                    stop_download_manager(prefetch_proc)
                    prefetch_proc = None
                    prefetch_job_id = None
                    state.transition_job(run_id, job_id, "READY", force=True)

                if not wait_for_worker_capacity(
                    args, data_root, deadline, signal_controller,
                    state=state, run_id=run_id, job_id=job_id,
                    pending_targets=[
                        candidate_target
                        for candidate_target, candidate_job in target_jobs
                        if candidate_job["job_id"] not in results_by_job
                    ],
                ):
                    state.transition_job(
                        run_id, job_id, "INTERRUPTED", force=True,
                        error="stopped while waiting for resources",
                    )
                    persist_target_status(
                        target,
                        "pending",
                        error="stopped while waiting for resources",
                    )
                    stop_reason = "resource-wait-stopped"
                    break
                if not deadline.may_start():
                    state.transition_job(run_id, job_id, "READY", force=True)
                    persist_target_status(
                        target, "pending", error="soft deadline reached"
                    )
                    state.set_run_state(run_id, "SOFT_STOPPED")
                    stop_reason = "soft-deadline"
                    break

                # At most one future target is downloaded while this worker runs.
                if args.prefetch_targets == 1:
                    for next_target, next_job in target_jobs[index + 1 :]:
                        if next_job["job_id"] in results_by_job:
                            continue
                        if not target_inputs_ready(next_target, data_root):
                            state.transition_job(
                                run_id,
                                next_job["job_id"],
                                "DOWNLOADING",
                                force=True,
                                details={"reason": "single-target prefetch"},
                            )
                            protected_targets = [
                                candidate_target
                                for candidate_target, candidate_job in target_jobs
                                if candidate_job["job_id"] not in results_by_job
                            ]
                            prefetch_proc = start_download_manager(
                                args,
                                [next_target],
                                completed_results,
                                protected_targets=protected_targets,
                            )
                            prefetch_job_id = next_job["job_id"]
                        break

                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--job-id", job_id,
                    "--galaxy", target["name"],
                    "--signal", str(signal_path),
                    "--color", str(color_path),
                    "--signal-filter", target["signal_filter"],
                    "--color-filter", target["color_filter"],
                    "--template", str(template),
                    "--batch-root", str(batch_root),
                    "--wss-opd-dir", str(Path(args.wss_opd_dir).resolve()),
                    "--stpsf-data-dir", str(Path(args.stpsf_data_dir).resolve()),
                    "--max-opd-delta-days", str(args.max_opd_delta_days),
                ]
                attempt_output_key = (
                    f"{run_id}__attempt-{current_job['attempt_count'] + 1}"
                    f"__{int(time.time())}"
                )
                cmd.extend(["--attempt-output-key", attempt_output_key])
                if products_root is not None:
                    cmd.extend(["--products-root", str(products_root)])
                print(f"[{timestamp()}] starting worker: {' '.join(cmd)}")
                emit_campaign_event(
                    event_log_path,
                    "WORKER_START_REQUESTED",
                    state=state,
                    run_id=run_id,
                    job_id=job_id,
                    payload={
                        "target": target["name"],
                        "program": canonical_program(target.get("program")),
                        "filters": {
                            "signal": target["signal_filter"],
                            "color": target["color_filter"],
                        },
                        "command": cmd,
                        "signal_path": str(signal_path),
                        "color_path": str(color_path),
                    },
                )
                state.transition_job(run_id, job_id, "RUNNING", force=True)
                persist_target_status(
                    target,
                    "running",
                    method=template_family or "sbf2",
                    quantity=PRIMARY_QUANTITY,
                    error="",
                )
                worker_log = result_json_path(
                    batch_root, target["name"], identity=identity
                ).with_suffix(".log")
                try:
                    proc = launch_process_group(cmd)
                except Exception as exc:
                    attempt = state.record_attempt_start(
                        run_id, job_id, command=cmd, pid=None, log_path=worker_log
                    )
                    state.record_attempt_end(
                        attempt["attempt_id"], state="FAILED", error=repr(exc)
                    )
                    state.transition_job(
                        run_id, job_id, "FAILED", force=True, error=repr(exc)
                    )
                    results_by_job[job_id] = {
                        "job_id": job_id,
                        "galaxy": target["name"],
                        "status": "failed",
                        "error": f"worker could not start: {exc!r}",
                    }
                    persist_target_status(
                        target,
                        "failed",
                        error=f"worker could not start: {exc!r}",
                    )
                    continue
                try:
                    attempt = state.record_attempt_start(
                        run_id, job_id, command=cmd, pid=proc.pid,
                        log_path=worker_log,
                    )
                except BaseException:
                    terminate_process_group(
                        proc,
                        term_grace_seconds=args.worker_term_grace_seconds,
                        kill_grace_seconds=args.worker_kill_grace_seconds,
                    )
                    state.transition_job(
                        run_id, job_id, "INTERRUPTED", force=True,
                        error="failed to persist worker attempt",
                    )
                    persist_target_status(
                        target,
                        "pending",
                        error="failed to persist worker attempt",
                    )
                    raise
                timeout_seconds = (
                    None
                    if args.worker_timeout_hours <= 0
                    else args.worker_timeout_hours * 3600
                )
                try:
                    supervision = supervise_process(
                        proc,
                        deadline=deadline,
                        signal_controller=signal_controller,
                        timeout_seconds=timeout_seconds,
                        sample_interval_seconds=args.resource_sample_seconds,
                        disk_path=data_root,
                        min_available_ram_gib=args.min_available_ram_gb,
                        emergency_available_ram_gib=args.emergency_available_ram_gb,
                        max_worker_rss_gib=(
                            None
                            if args.max_worker_rss_gb <= 0
                            else args.max_worker_rss_gb
                        ),
                        min_free_disk_gib=args.critical_free_gb,
                        callback=lambda sample, jid=job_id, aid=attempt["attempt_id"]: (
                            record_resource_sample(state, run_id, jid, aid, sample)
                        ),
                        term_grace_seconds=args.worker_term_grace_seconds,
                        kill_grace_seconds=args.worker_kill_grace_seconds,
                    )
                except BaseException as exc:
                    terminate_process_group(
                        proc,
                        term_grace_seconds=args.worker_term_grace_seconds,
                        kill_grace_seconds=args.worker_kill_grace_seconds,
                    )
                    state.record_attempt_end(
                        attempt["attempt_id"],
                        state="INTERRUPTED",
                        exit_code=proc.poll(),
                        error=f"supervisor failed: {exc!r}",
                    )
                    state.transition_job(
                        run_id,
                        job_id,
                        "INTERRUPTED",
                        force=True,
                        error=f"supervisor failed: {exc!r}",
                    )
                    persist_target_status(
                        target,
                        "pending",
                        error=f"supervisor failed: {exc!r}",
                    )
                    raise
                emit_campaign_event(
                    event_log_path,
                    "WORKER_SUPERVISION_ENDED",
                    state=state,
                    run_id=run_id,
                    job_id=job_id,
                    attempt_id=attempt["attempt_id"],
                    payload=supervision.as_dict(),
                )

                result = verified_campaign_result(target, batch_root, identity)
                if supervision.ok and result is not None:
                    result["go3055_qc"] = evaluate_go3055_qc(result)
                    result["qc_status"] = result["go3055_qc"]["status"]
                    result["qc_flags"] = result["go3055_qc"]["flags"]
                    state.transition_job(run_id, job_id, "VERIFYING", force=True)
                    for artifact in result.get("artifact_manifest", []):
                        state.record_artifact(
                            run_id, job_id, attempt_id=attempt["attempt_id"],
                            kind=artifact["name"], path=artifact["path"],
                            size_bytes=artifact.get("size_bytes"),
                            sha256=artifact.get("sha256"), verified=artifact.get("ok"),
                            metadata={
                                "fits_valid": artifact.get("fits_valid"),
                                "csv_valid": artifact.get("csv_valid"),
                                "row_count": artifact.get("row_count"),
                            },
                        )
                    state.record_attempt_end(
                        attempt["attempt_id"], state="SUCCEEDED",
                        exit_code=supervision.returncode,
                        metadata={"supervision": supervision.as_dict()},
                    )
                    state.transition_job(run_id, job_id, "SUCCEEDED")
                    completed_results.append(result)
                    results_by_job[job_id] = result
                    science_fields = science_status_fields(result)
                    persist_target_status(
                        target,
                        "done",
                        method=target_status_measurement_method(result),
                        **science_fields,
                        result_json=result_json_path(
                            batch_root, target["name"], identity=identity
                        ),
                        qc=result["go3055_qc"]["summary"],
                        error="",
                    )
                    emit_campaign_event(
                        event_log_path,
                        "ARTIFACTS_VERIFIED",
                        state=state,
                        run_id=run_id,
                        job_id=job_id,
                        attempt_id=attempt["attempt_id"],
                        payload={
                            "target": target["name"],
                            "artifact_count": result.get("artifact_count"),
                            "artifact_manifest_path": result.get(
                                "artifact_manifest_path"
                            ),
                            "artifacts": result.get("artifact_manifest"),
                        },
                    )
                else:
                    error = supervision.detail or (
                        f"worker exited {supervision.returncode}; "
                        "required artifacts are absent or invalid"
                    )
                    attempt_state = (
                        "INTERRUPTED"
                        if supervision.reason in {"signal", "deadline"}
                        else "FAILED"
                    )
                    state.record_attempt_end(
                        attempt["attempt_id"], state=attempt_state,
                        exit_code=supervision.returncode, error=error,
                        metadata={"supervision": supervision.as_dict()},
                    )
                    refreshed = state.get_job(run_id, job_id)
                    if (
                        attempt_state == "FAILED"
                        and refreshed["attempt_count"] < args.max_attempts
                        and deadline.may_start()
                        and not signal_controller.stop_requested
                    ):
                        state.transition_job(
                            run_id, job_id, "RETRY_WAIT", force=True, error=error
                        )
                        persist_target_status(target, "failed", error=error)
                        # Requeue at the tail.  The stable job_id keeps the
                        # second attempt attached to the same database row.
                        target_jobs.append((target, state.get_job(run_id, job_id)))
                    else:
                        state.transition_job(
                            run_id, job_id, attempt_state, force=True, error=error
                        )
                        persist_target_status(
                            target,
                            "pending" if attempt_state == "INTERRUPTED" else "failed",
                            error=error,
                        )
                        results_by_job[job_id] = {
                            "job_id": job_id,
                            "galaxy": target["name"],
                            "status": "failed",
                            "error": error,
                            "supervision": supervision.as_dict(),
                        }
                    if attempt_state == "INTERRUPTED":
                        stop_reason = supervision.reason
                        break

                current_results = list(results_by_job.values())
                write_summary(current_results, batch_root)
                write_go3055_qc(current_results, batch_root)
                link_residuals(current_results, batch_root)
                state.snapshot_queue(run_id)
                ensure_disk_space_for_downloads(
                    data_root, completed_results,
                    min_free_gb=args.min_free_gb,
                    cleanup_enabled=not args.no_cleanup_inputs,
                    protected_input_paths={
                        str(path.resolve())
                        for candidate_target, candidate_job in target_jobs
                        if candidate_job["job_id"] not in results_by_job
                        for path in local_target_files(
                            candidate_target, data_root
                        ).values()
                    },
                )
                gc.collect()
        finally:
            stop_download_manager(prefetch_proc)

    final_results = list(results_by_job.values())
    write_summary(final_results, batch_root)
    qc_path = write_go3055_qc(final_results, batch_root)
    link_residuals(final_results, batch_root)
    jobs_final = state.list_jobs(run_id)
    states = {job["state"] for job in jobs_final}
    if signal_controller.stop_requested or stop_reason in {"signal", "deadline"}:
        state.set_run_state(run_id, "INTERRUPTED", error=stop_reason)
    elif stop_reason == "soft-deadline":
        state.set_run_state(run_id, "SOFT_STOPPED")
    elif states <= {"SUCCEEDED", "SKIPPED"}:
        state.set_run_state(run_id, "COMPLETED")
    elif "FAILED" in states:
        state.set_run_state(run_id, "FAILED", error="one or more jobs failed")
    else:
        state.set_run_state(run_id, "INTERRUPTED", error=stop_reason)
    snapshot = state.snapshot_queue(run_id)
    report_path = write_campaign_report(
        campaign_root, snapshot, target_jobs, results_by_job
    )
    emit_campaign_event(
        event_log_path,
        "CAMPAIGN_FINISHED",
        state=state,
        run_id=run_id,
        payload={
            "state": snapshot["run"]["state"],
            "counts": snapshot["counts"],
            "stop_reason": stop_reason,
            "report_path": str(report_path),
        },
    )
    release_campaign_lock(campaign_lock)
    _ACTIVE_CAMPAIGN_LOCK = None
    print(
        f"[{timestamp()}] campaign {run_id}: {snapshot['run']['state']} "
        f"{snapshot['counts']}"
    )
    print(f"[{timestamp()}] GO-3055 QC -> {qc_path}")
    print(f"[{timestamp()}] campaign report -> {report_path}")
    return 0 if snapshot["run"]["state"] == "COMPLETED" else 1


def run_parent(args):
    global _ACTIVE_CAMPAIGN_LOCK
    normalize_cli_paths(args)
    validate_parent_args(args)
    campaign_root = campaign_root_from_args(args)
    campaign_root.mkdir(parents=True, exist_ok=True)
    log_path = campaign_root / "campaign.log"
    event_log_path = campaign_root / "campaign_events.jsonl"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 96)
            print(
                f"[{timestamp()}] parent invocation start; pid={os.getpid()}; "
                f"cwd={Path.cwd()}"
            )
            print(f"[{timestamp()}] command: {command_provenance(args)['shell_command']}")
            print(f"[{timestamp()}] persistent parent log -> {log_path}")
            emit_campaign_event(
                event_log_path,
                "PARENT_INVOCATION_STARTED",
                payload={
                    "pid": os.getpid(),
                    "cwd": str(Path.cwd()),
                    "command": command_provenance(args),
                    "campaign_log": str(log_path),
                },
            )
            try:
                returncode = _run_parent_impl(args)
            except BaseException as exc:
                trace = traceback.format_exc()
                print(trace)
                active_state = getattr(args, "_active_campaign_state", None)
                active_run_id = getattr(args, "_active_run_id", None)
                if active_state is not None and active_run_id is not None:
                    try:
                        active_state.set_run_state(
                            active_run_id,
                            "INTERRUPTED",
                            error=f"parent crash: {exc!r}",
                        )
                    except Exception as state_exc:
                        print(
                            f"[{timestamp()}] failed to persist parent crash in SQLite: "
                            f"{state_exc!r}"
                        )
                emit_campaign_event(
                    event_log_path,
                    "PARENT_CRASH",
                    state=active_state,
                    run_id=active_run_id,
                    payload={"error": repr(exc), "traceback": trace},
                )
                atomic_write_text(
                    campaign_root / "parent_crash.txt",
                    f"[{timestamp()}] {exc!r}\n\n{trace}",
                )
                raise
            finally:
                release_campaign_lock(_ACTIVE_CAMPAIGN_LOCK)
                _ACTIVE_CAMPAIGN_LOCK = None
            emit_campaign_event(
                event_log_path,
                "PARENT_INVOCATION_ENDED",
                payload={"returncode": returncode},
            )
            print(f"[{timestamp()}] parent invocation end; returncode={returncode}")
            return returncode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument(
        "--products-root",
        default=str(DEFAULT_PRODUCTS_ROOT),
        help="isolated per-target directory for notebook-generated products",
    )
    parser.add_argument("--target-csv", default=str(DEFAULT_TARGET_CSV))
    parser.add_argument(
        "--extra-target-csv",
        action="append",
        default=[],
        help="additional target manifest; may be repeated",
    )
    parser.add_argument(
        "--programs",
        nargs="+",
        default=["3055"],
        help="fixed to GO-3055 in this dedicated runner",
    )
    parser.add_argument(
        "--external-download-status",
        default=None,
        help=(
            "telemetry JSON written by an independent downloader; final FITS "
            "validation remains the readiness authority"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--download-retry-seconds", type=int, default=120)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--min-available-ram-gb", type=float, default=0.0)
    parser.add_argument(
        "--campaign-root",
        default=str(DEFAULT_CAMPAIGN_ROOT),
        help="SQLite state and queue snapshots",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="do not resume the newest compatible unfinished campaign",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help=(
            "start a new run and bypass completion reuse for the selected "
            "targets; ordinary restarts should omit this"
        ),
    )
    parser.add_argument("--wall-time-hours", type=float, default=48.0)
    parser.add_argument("--soft-stop-minutes", type=float, default=60.0)
    parser.add_argument("--worker-timeout-hours", type=float, default=12.0)
    parser.add_argument("--resource-sample-seconds", type=float, default=30.0)
    parser.add_argument("--emergency-available-ram-gb", type=float, default=0.0)
    parser.add_argument(
        "--max-worker-rss-gb",
        type=float,
        default=0.0,
        help="0 disables the per-worker RSS ceiling",
    )
    parser.add_argument("--critical-free-gb", type=float, default=40.0)
    parser.add_argument(
        "--min-processing-free-gb",
        type=float,
        default=40.0,
        help=(
            "free-space floor to preserve after the estimated next notebook "
            "products are written"
        ),
    )
    parser.add_argument(
        "--external-download-reserve-gb",
        type=float,
        default=0.0,
        help=(
            "free-space reserve owned by an independent downloader; combined "
            "with the calculated remaining manifest bytes"
        ),
    )
    parser.add_argument(
        "--estimated-worker-output-gb",
        type=float,
        default=6.0,
        help="space reserved before starting the next sbf-2 result",
    )
    parser.add_argument("--worker-term-grace-seconds", type=float, default=300.0)
    parser.add_argument("--worker-kill-grace-seconds", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--prefetch-targets",
        type=int,
        choices=(0, 1),
        default=0,
        help="hard cap on future targets downloaded while a worker runs",
    )
    parser.add_argument("--galaxies", nargs="*", default=None)
    parser.add_argument(
        "--target-keys",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cleanup-job-ids",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protected-inputs",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-bulk-targets",
        action="store_true",
        help=(
            "allow an unfiltered manifest with more than "
            f"{MAX_IMPLICIT_TARGETS} actionable targets"
        ),
    )
    download_mode = parser.add_mutually_exclusive_group()
    download_mode.add_argument(
        "--no-download",
        dest="no_download",
        action="store_true",
        default=True,
        help="offline mode (default)",
    )
    download_mode.add_argument(
        "--allow-download",
        dest="no_download",
        action="store_false",
        help="explicitly allow the inherited downloader",
    )
    cleanup_mode = parser.add_mutually_exclusive_group()
    cleanup_mode.add_argument(
        "--no-cleanup-inputs",
        dest="no_cleanup_inputs",
        action="store_true",
        default=True,
        help="never remove GO-3055 science inputs (default)",
    )
    cleanup_mode.add_argument(
        "--allow-input-cleanup",
        dest="no_cleanup_inputs",
        action="store_false",
        help="explicitly allow removal of verified completed inputs",
    )
    parser.add_argument("--wss-opd-dir", default=str(DEFAULT_WSS_OPD_DIR))
    parser.add_argument(
        "--stpsf-data-dir",
        default=str(DEFAULT_STPSF_DATA_DIR),
    )
    parser.add_argument(
        "--max-opd-delta-days",
        type=float,
        default=7.0,
        help="maximum allowed science-to-WSS-OPD date separation",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--download-worker", action="store_true")
    parser.add_argument("--galaxy", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--attempt-output-key",
        default=None,
        help=argparse.SUPPRESS,
    )
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
    args = parser.parse_args(argv)
    args._argv = list(sys.argv[1:] if argv is None else argv)
    if args.force_reprocess:
        args.new_run = True
    return args


def main():
    global _ACTIVE_CAMPAIGN_LOCK
    args = parse_args()
    normalize_cli_paths(args)
    if args.download_worker:
        targets = load_manifest_targets(
            args.target_csv,
            args.data_root,
            args.extra_target_csv,
        )
        targets = select_targets(
            targets,
            args.galaxies,
            allow_bulk_targets=args.allow_bulk_targets,
            target_keys=args.target_keys,
            programs=args.programs,
        )
        if not targets:
            raise RuntimeError("download worker has no explicitly selected targets")
        completed_results = load_completed_results(
            args.batch_root,
            allowed_job_ids=set(args.cleanup_job_ids or []),
        )
        download_targets_until_stopped(
            targets,
            Path(args.data_root).resolve(),
            Path(args.batch_root).resolve(),
            completed_results,
            min_free_gb=args.min_free_gb,
            cleanup_enabled=not args.no_cleanup_inputs,
            retry_sleep=args.download_retry_seconds,
            eligible_cleanup_job_ids=set(args.cleanup_job_ids or []),
            protected_input_paths=set(args.protected_inputs or []),
        )
        return 0
    if args.worker:
        if not args.galaxy or not args.signal or not args.color:
            raise SystemExit("--worker requires --galaxy, --signal and --color")
        return run_worker(args)
    try:
        return run_parent(args)
    finally:
        release_campaign_lock(_ACTIVE_CAMPAIGN_LOCK)
        _ACTIVE_CAMPAIGN_LOCK = None


if __name__ == "__main__":
    raise SystemExit(main())
