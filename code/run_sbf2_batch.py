#!/usr/bin/env python3
import argparse
import builtins
import csv
import gc
import json
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
DEFAULT_TARGET_CSV = SCRIPT_DIR / "article_galaxies_jwst_f150w_selected.csv"
MAST_DOWNLOAD_PREFIX = "https://mast.stsci.edu/api/v0.1/Download/file?uri="


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


def final_result_for(target, batch_root):
    path = Path(batch_root) / f"{slug(target['name'])}_result.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except Exception:
        return None
    if result.get("status") != "ok":
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


def read_targets_from_csv(csv_path, data_root):
    rows = []
    with Path(csv_path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            galaxy = row["galaxy"].strip()
            f150w = row["expected_f150w_i2d"].strip()
            if not galaxy or not f150w:
                continue
            f090w = f150_to_f090_filename(f150w)
            target_dir = Path(data_root) / galaxy
            rows.append(
                {
                    "name": galaxy,
                    "f150w": f150w,
                    "f090w": f090w,
                    "f150w_url": row.get("mast_download_url") or product_download_url(f150w),
                    "f090w_url": product_download_url(f090w),
                    "f150w_size": int(row["f150w_content_length_bytes"])
                    if row.get("f150w_content_length_bytes")
                    else None,
                    "f090w_size": None,
                    "target_dir": str(target_dir),
                    "source_csv": str(csv_path),
                }
            )
    return rows


def merge_known_targets(targets):
    known = {target["name"]: target for target in TARGETS}
    merged = []
    for target in targets:
        item = dict(target)
        if item["name"] in known:
            for key, value in known[item["name"]].items():
                if item.get(key) in (None, ""):
                    item[key] = value
            item["f150w_size"] = known[item["name"]].get("f150w_size", item.get("f150w_size"))
            item["f090w_size"] = known[item["name"]].get("f090w_size", item.get("f090w_size"))
        merged.append(item)
    return merged


def local_target_files(target, data_root):
    root = Path(data_root) / target["name"]
    return {
        "f150w": root / target["f150w"],
        "f090w": root / target["f090w"],
    }


def target_paths(target, data_root):
    files = local_target_files(target, data_root)
    return files["f150w"], files["f090w"]


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


def override_target_namespace(namespace, galaxy, f150w_path, f090w_path):
    f150w_path = Path(f150w_path).resolve()
    f090w_path = Path(f090w_path).resolve()
    namespace["TARGET_GALAXY"] = galaxy
    namespace["f150w_path"] = f150w_path
    namespace["f090w_path"] = f090w_path
    namespace["out_dir"] = f150w_path.parent
    namespace["stem"] = f150w_path.stem
    namespace["out_dir"].mkdir(parents=True, exist_ok=True)


def result_paths(out_dir, stem):
    out_dir = Path(out_dir)
    return {
        "model_full_fits": out_dir / f"{stem}_sbf_model_full.fits",
        "science_residual_fits": out_dir / f"{stem}_sbf_resid_full_science.fits",
        "science_residual_raw_fits": out_dir / f"{stem}_sbf_resid_full_science_raw.fits",
        "inner_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_inner_lit_usable.fits",
        "outer_usable_residual_fits": out_dir
        / f"{stem}_sbf_resid_science_circular_outer_lit_usable.fits",
        "df_sbf_csv": out_dir / f"{stem}_sbf2_df_sbf.csv",
        "annulus_summary_csv": out_dir / f"{stem}_sbf2_annulus_summary.csv",
    }


def execute_template_for_target(template_path, galaxy, f150w_path, f090w_path, batch_root):
    namespace = {
        "__name__": "__sbf2_notebook_exec__",
        "__file__": str(template_path),
    }
    namespace["display"] = make_display(namespace)
    code_cells = notebook_code_cells(template_path)

    for cell_no, source in code_cells:
        print(f"[{timestamp()}] executing sbf-2 cell {cell_no}")
        try:
            exec(compile(source, f"{template_path}:cell-{cell_no}", "exec"), namespace)
        except Exception:
            print(f"[{timestamp()}] failed in sbf-2 cell {cell_no}")
            raise

        if "f150w_path = Path" in source and "f090w_path = Path" in source:
            override_target_namespace(namespace, galaxy, f150w_path, f090w_path)
            namespace["display"] = make_display(namespace)
            print(f"[{timestamp()}] target override: {galaxy}")
            print(f"[{timestamp()}] F150W -> {namespace['f150w_path']}")
            print(f"[{timestamp()}] F090W -> {namespace['f090w_path']}")

    recommended = namespace.get("recommended_sbf")
    if not recommended:
        raise RuntimeError("sbf-2 finished without recommended_sbf")

    out_dir = Path(namespace["out_dir"])
    stem = namespace["stem"]
    paths = result_paths(out_dir, stem)

    df_sbf = namespace.get("df_sbf")
    if df_sbf is not None:
        df_sbf.to_csv(paths["df_sbf_csv"], index=False)

    df_annulus_summary = namespace.get("df_annulus_summary")
    if df_annulus_summary is not None:
        df_annulus_summary.to_csv(paths["annulus_summary_csv"], index=False)

    result = {
        "galaxy": galaxy,
        "status": "ok",
        "f150w_path": str(Path(f150w_path).resolve()),
        "f090w_path": str(Path(f090w_path).resolve()),
        "out_dir": str(out_dir.resolve()),
        "stem": stem,
    }
    for key, value in as_builtin(recommended).items():
        result[f"recommended_{key}"] = value
    for key, value in paths.items():
        result[key] = str(value.resolve())
        result[f"{key}_exists"] = Path(value).exists()

    color_summary = namespace.get("df_color_summary")
    if color_summary is not None and len(color_summary) > 0:
        try:
            row0 = color_summary.iloc[0].to_dict()
            result["color_F090W_F150W"] = as_builtin(row0.get("color_F090W_F150W"))
            result["color_sigma_proxy"] = as_builtin(row0.get("sigma_proxy"))
        except Exception:
            pass

    batch_root.mkdir(parents=True, exist_ok=True)
    result_json = batch_root / f"{slug(galaxy)}_result.json"
    result_json.write_text(json.dumps(as_builtin(result), ensure_ascii=False, indent=2))
    print(f"[{timestamp()}] wrote result {result_json}")
    return result


def run_worker(args):
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    log_path = batch_root / f"{slug(args.galaxy)}.log"

    with log_path.open("a") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[{timestamp()}] worker start: {args.galaxy}")
            try:
                result = execute_template_for_target(
                    Path(args.template).resolve(),
                    args.galaxy,
                    Path(args.f150w).resolve(),
                    Path(args.f090w).resolve(),
                    batch_root,
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
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                err_path = batch_root / f"{slug(args.galaxy)}_result.json"
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


def ensure_disk_space_for_downloads(data_root, completed_results, min_free_gb, cleanup_enabled=True):
    disk, _ = log_resources("disk-check", data_root)
    if disk["free_gb"] >= min_free_gb:
        return
    if not cleanup_enabled:
        print(
            f"[{timestamp()}] [DISK] free space below threshold "
            f"({disk['free_gb']:.1f} < {min_free_gb:.1f} GB), cleanup disabled"
        )
        return

    print(
        f"[{timestamp()}] [DISK] free space below threshold "
        f"({disk['free_gb']:.1f} < {min_free_gb:.1f} GB), removing source inputs "
        "for completed galaxies"
    )
    for result in completed_results:
        if result.get("status") != "ok":
            continue
        for key in ("f150w_path", "f090w_path"):
            path = Path(result.get(key, ""))
            if not path.exists():
                continue
            try:
                size_gb = bytes_gb(path.stat().st_size)
                path.unlink()
                print(f"[{timestamp()}] [DISK] removed {path} ({size_gb:.2f} GB)")
            except Exception as exc:
                print(f"[{timestamp()}] [DISK] failed to remove {path}: {exc}")
        disk, _ = log_resources("disk-check-after-cleanup", data_root)
        if disk["free_gb"] >= min_free_gb:
            return


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
                ("f150w", target.get("f150w_url"), files["f150w"], target.get("f150w_size")),
                ("f090w", target.get("f090w_url"), files["f090w"], target.get("f090w_size")),
            ]
            for band, url, dest, expected_size in jobs:
                if is_input_ready(dest, expected_size):
                    continue
                all_ready = False
                completed_results = load_completed_results(batch_root)
                ensure_disk_space_for_downloads(
                    data_root,
                    completed_results,
                    min_free_gb=min_free_gb,
                    cleanup_enabled=cleanup_enabled,
                )
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
    print(f"[{timestamp()}] starting download manager: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def run_parent(args):
    template = Path(args.template).resolve()
    data_root = Path(args.data_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    if args.target_csv:
        targets = merge_known_targets(read_targets_from_csv(args.target_csv, data_root))
    else:
        targets = [dict(t) for t in TARGETS]

    wanted = set(args.galaxies) if args.galaxies else None
    targets = [t for t in targets if wanted is None or t["name"] in wanted]
    if not targets:
        raise RuntimeError(f"no targets selected: {args.galaxies}")

    completed_results = []
    results = []
    for target in targets:
        existing = final_result_for(target, batch_root)
        if existing is not None:
            completed_results.append(existing)
            results.append(existing)
            print(f"[{timestamp()}] reusing completed result for {target['name']}")
    download_proc = start_download_manager(args, targets, completed_results)
    for target in targets:
        existing = final_result_for(target, batch_root)
        if existing is not None:
            if not any(r.get("galaxy") == existing.get("galaxy") for r in results):
                results.append(existing)
            write_summary(results, batch_root)
            link_residuals(results, batch_root)
            continue

        f150w_path, f090w_path = target_paths(target, data_root)
        log_resources(f"before-wait {target['name']}", data_root)
        wait_for_input(
            f150w_path,
            expected_size=target.get("f150w_size"),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        wait_for_input(
            f090w_path,
            expected_size=target.get("f090w_size"),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
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
            "--f150w",
            str(f150w_path),
            "--f090w",
            str(f090w_path),
            "--template",
            str(template),
            "--batch-root",
            str(batch_root),
        ]
        print(f"[{timestamp()}] starting worker: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        log_resources(f"after-worker {target['name']}", data_root)

        result_path = batch_root / f"{slug(target['name'])}_result.json"
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
    if download_proc is not None:
        if download_proc.poll() is None:
            print(f"[{timestamp()}] waiting for download manager to finish")
            try:
                download_proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[{timestamp()}] download manager still running; leaving it alive")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--target-csv", default=str(DEFAULT_TARGET_CSV))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--download-retry-seconds", type=int, default=120)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--min-available-ram-gb", type=float, default=8.0)
    parser.add_argument("--galaxies", nargs="*", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-cleanup-inputs", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--download-worker", action="store_true")
    parser.add_argument("--galaxy", default=None)
    parser.add_argument("--f150w", default=None)
    parser.add_argument("--f090w", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download_worker:
        targets = merge_known_targets(read_targets_from_csv(args.target_csv, args.data_root))
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
        if not args.galaxy or not args.f150w or not args.f090w:
            raise SystemExit("--worker requires --galaxy, --f150w and --f090w")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
