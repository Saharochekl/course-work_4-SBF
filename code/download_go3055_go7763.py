#!/usr/bin/env python3
"""Download missing JWST/i2d files for GO-3055 and GO-7763.

The default mode is a local dry run: manifests and existing FITS files are
checked, but no network request is made.  Add ``--download`` to process every
missing or incomplete enabled product.  Four files are downloaded concurrently
by default; use ``--workers 1`` for strictly sequential operation.

Incomplete transfers are kept as ``.part`` files and resumed with HTTP Range.
A completed file is published atomically only after its size and FITS structure
have been checked.  Console messages and help are in Russian; Python identifiers
deliberately follow the usual English naming convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import builtins


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
MANIFESTS = {
    "3055": PROJECT_ROOT / "code" / "targets_go3055_manifest.csv",
    "7763": PROJECT_ROOT / "code" / "targets_additional_manifest.csv",
}
MAST_DOWNLOAD_ENDPOINT = "https://mast.stsci.edu/api/v0.1/Download/file"
USER_AGENT = "course-work-SBF/2.0"
FITS_BLOCK_SIZE = 2880
LOCAL_MANIFEST_SIZE_TOLERANCE = 0.01
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.I)
UNSATISFIED_RANGE_RE = re.compile(r"^bytes\s+\*/(\d+)$", re.I)

_orig_print = builtins.print
def print(*args, **kwargs):
    _orig_print(f"[{time.strftime('%H:%M:%S')}]", *args, **kwargs)


@dataclass(frozen=True)
class Product:
    program: str
    target: str
    obsid: str
    role: str
    filter_name: str
    product_uri: str
    file_name: str
    destination: Path
    manifest_size: int | None

    @property
    def key(self) -> str:
        return f"{self.program}:{self.obsid}:{self.role}:{self.file_name}"

    @property
    def url(self) -> str:
        return mast_download_url(self.product_uri)


@dataclass(frozen=True)
class FitsValidation:
    ready: bool
    reason: str
    size: int
    structural_size: int | None = None
    manifest_size_drift: bool = False


@dataclass(frozen=True)
class ProductPlan:
    product: Product
    status: str
    reason: str
    current_size: int
    remaining_size: int


@dataclass(frozen=True)
class RemoteInfo:
    total_size: int | None
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class DownloadResult:
    key: str
    program: str
    target: str
    obsid: str
    role: str
    filter_name: str
    destination: str
    status: str
    message: str
    size: int
    remote_size: int | None = None
    manifest_size: int | None = None
    sha256: str | None = None
    attempts: int = 0
    downloaded_bytes: int = 0


@dataclass
class QueueProgress:
    total_bytes: int
    completed_bytes: int = 0
    worker_count: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self.total_bytes, self.completed_bytes, self.worker_count

    def finish_product(self, product: Product, result: DownloadResult) -> None:
        with self._lock:
            planned = product.manifest_size or result.remote_size or result.size
            if result.status == "failed":
                self.completed_bytes += planned
                return
            actual = result.remote_size or result.size or planned
            self.total_bytes += actual - planned
            self.completed_bytes += actual


class DownloadError(RuntimeError):
    """Base class for one-product download failures."""


class PermanentDownloadError(DownloadError):
    """A retry cannot fix this response or local configuration."""


class RetryableDownloadError(DownloadError):
    """The current partial file can normally be resumed."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class DiskSpaceError(PermanentDownloadError):
    """There is not enough free disk space for the requested transfer."""


class DownloadCancelled(DownloadError):
    """The parent process asked active workers to preserve partial files and stop."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: str | None, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unknown boolean value: {value!r}")


def parse_optional_int(value: str | None) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def mast_download_url(product_uri: str) -> str:
    return f"{MAST_DOWNLOAD_ENDPOINT}?{urlencode({'uri': product_uri})}"


def read_products(
    programs: set[str],
    data_dir: Path,
    manifests: dict[str, Path] | None = None,
) -> tuple[list[Product], list[dict[str, str]]]:
    """Read enabled product pairs and return disabled rows separately."""

    manifest_paths = manifests or MANIFESTS
    products: list[Product] = []
    disabled_rows: list[dict[str, str]] = []

    for program in ("3055", "7763"):
        if program not in programs:
            continue
        path = manifest_paths[program]
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("program", "").strip() != program:
                    continue
                if not parse_bool(row.get("download_enabled"), default=True):
                    disabled_rows.append(row)
                    continue

                target = row.get("target", "").strip()
                obsid = row.get("obsid", "").strip()
                if not target or not obsid:
                    raise ValueError(f"Incomplete target row in {path}: {row!r}")

                for role in ("signal", "color"):
                    file_name = row.get(f"{role}_product", "").strip()
                    product_uri = row.get(f"{role}_product_uri", "").strip()
                    filter_name = row.get(f"{role}_filter", "").strip().upper()
                    if not file_name or not product_uri or not filter_name:
                        raise ValueError(
                            f"Enabled row {program}/{target}/{obsid} has no complete "
                            f"{role} product"
                        )
                    products.append(
                        Product(
                            program=program,
                            target=target,
                            obsid=obsid,
                            role=role,
                            filter_name=filter_name,
                            product_uri=product_uri,
                            file_name=file_name,
                            destination=data_dir / target / file_name,
                            manifest_size=parse_optional_int(
                                row.get(f"{role}_content_length_bytes")
                            ),
                        )
                    )

    deduplicated: list[Product] = []
    by_destination: dict[Path, Product] = {}
    for product in products:
        previous = by_destination.get(product.destination)
        if previous is None:
            by_destination[product.destination] = product
            deduplicated.append(product)
            continue
        if previous.product_uri != product.product_uri:
            raise ValueError(
                f"Conflicting archive products use {product.destination}: "
                f"{previous.product_uri!r} and {product.product_uri!r}"
            )

    return deduplicated, disabled_rows


def _normalise_program(value: object) -> str:
    text = str(value or "").strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def validate_fits(
    path: Path,
    product: Product | None = None,
    authoritative_size: int | None = None,
    use_manifest_tolerance: bool = True,
) -> FitsValidation:
    """Check a JWST FITS without reading all image pixels into memory."""

    path = Path(path)
    if not path.is_file():
        return FitsValidation(False, "file is missing", 0)

    size = path.stat().st_size
    if size <= 0:
        return FitsValidation(False, "empty file", size)
    if size % FITS_BLOCK_SIZE:
        return FitsValidation(
            False,
            f"size {size} is not a multiple of the FITS block ({FITS_BLOCK_SIZE})",
            size,
        )
    if authoritative_size is not None and size != authoritative_size:
        return FitsValidation(
            False,
            f"size {size} differs from remote size {authoritative_size}",
            size,
        )

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - depends on the invoking Python
        raise RuntimeError(
            "Astropy is required for FITS validation. Run this script with "
            "astro_env/bin/python."
        ) from exc

    structural_size: int | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with fits.open(
                path,
                mode="readonly",
                memmap=True,
                lazy_load_hdus=False,
                checksum=True,
            ) as hdul:
                hdul.verify("exception")
                if "SCI" not in hdul:
                    return FitsValidation(False, "SCI extension is missing", size)
                science = hdul["SCI"]
                if science.data is None or not science.shape or any(
                    int(axis) <= 0 for axis in science.shape
                ):
                    return FitsValidation(False, "SCI extension is empty", size)

                first_index = tuple(0 for _ in science.shape)
                last_index = tuple(-1 for _ in science.shape)
                _ = science.data[first_index]
                _ = science.data[last_index]

                ends: list[int] = []
                for index in range(len(hdul)):
                    info = hdul.fileinfo(index)
                    if info:
                        ends.append(int(info["datLoc"]) + int(info["datSpan"]))
                structural_size = max(ends, default=0)

                if product is not None:
                    header = hdul[0].header
                    if _normalise_program(header.get("PROGRAM")) != product.program:
                        return FitsValidation(
                            False,
                            f"PROGRAM={header.get('PROGRAM')!r}, expected {product.program}",
                            size,
                            structural_size,
                        )
                    if str(header.get("FILTER", "")).strip().upper() != product.filter_name:
                        return FitsValidation(
                            False,
                            f"FILTER={header.get('FILTER')!r}, expected {product.filter_name}",
                            size,
                            structural_size,
                        )
                    header_name = str(header.get("FILENAME", "")).strip()
                    if header_name and header_name != product.file_name:
                        return FitsValidation(
                            False,
                            f"FILENAME={header_name!r}, expected {product.file_name!r}",
                            size,
                            structural_size,
                        )

            warning_text = " | ".join(str(item.message) for item in caught)
            bad_warning_markers = (
                "truncated",
                "missing end",
                "checksum verification failed",
                "datasum verification failed",
            )
            if any(marker in warning_text.lower() for marker in bad_warning_markers):
                return FitsValidation(False, warning_text, size, structural_size)
    except Exception as exc:
        return FitsValidation(False, f"FITS validation failed: {exc}", size, structural_size)

    if structural_size != size:
        return FitsValidation(
            False,
            f"FITS structure ends at {structural_size}, file size is {size}",
            size,
            structural_size,
        )

    manifest_drift = False
    if (
        use_manifest_tolerance
        and authoritative_size is None
        and product is not None
        and product.manifest_size
    ):
        manifest_drift = size != product.manifest_size
        relative_drift = abs(size - product.manifest_size) / product.manifest_size
        if relative_drift > LOCAL_MANIFEST_SIZE_TOLERANCE:
            return FitsValidation(
                False,
                f"complete FITS size {size} is too far from manifest size "
                f"{product.manifest_size}",
                size,
                structural_size,
                manifest_size_drift=True,
            )

    reason = "valid FITS"
    if manifest_drift and product is not None:
        reason += f"; manifest size drift {size - int(product.manifest_size or 0):+d} bytes"
    return FitsValidation(True, reason, size, structural_size, manifest_drift)


def partial_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def partial_metadata_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part.json")


def restart_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.restart.part")


def quarantine_partial(destination: Path) -> Path | None:
    """Preserve an untrusted partial instead of deleting user data."""

    part = partial_path(destination)
    if not part.exists():
        return None
    candidate = destination.with_name(f"{destination.name}.stale.part")
    if candidate.exists():
        candidate = destination.with_name(
            f"{destination.name}.stale.{time.time_ns()}.part"
        )
    os.replace(part, candidate)
    remove_partial_metadata(destination)
    return candidate


def remove_quarantined_partials(destination: Path) -> None:
    for path in destination.parent.glob(f"{destination.name}.stale*.part"):
        path.unlink(missing_ok=True)


def classify_product(product: Product) -> ProductPlan:
    final_check = validate_fits(product.destination, product=product)
    if final_check.ready:
        return ProductPlan(product, "ready", final_check.reason, final_check.size, 0)

    part = partial_path(product.destination)
    part_check = validate_fits(part, product=product)
    if part_check.ready:
        return ProductPlan(
            product,
            "complete-part",
            "complete FITS is waiting under .part",
            part_check.size,
            0,
        )

    candidates = [
        path.stat().st_size
        for path in (product.destination, part)
        if path.is_file()
    ]
    current_size = max(candidates, default=0)
    remaining = (
        max(product.manifest_size - current_size, 0)
        if product.manifest_size is not None
        else 0
    )
    if product.destination.exists() or part.exists():
        reason = final_check.reason if product.destination.exists() else part_check.reason
        return ProductPlan(product, "incomplete", reason, current_size, remaining)
    return ProductPlan(
        product,
        "missing",
        "file is missing",
        0,
        product.manifest_size or 0,
    )


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_partial_metadata(destination: Path) -> dict[str, object] | None:
    path = partial_metadata_path(destination)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_partial_metadata(product: Product, remote: RemoteInfo) -> None:
    atomic_write_json(
        partial_metadata_path(product.destination),
        {
            "product_uri": product.product_uri,
            "remote_size": remote.total_size,
            "etag": remote.etag,
            "last_modified": remote.last_modified,
            "updated_at": utc_now(),
        },
    )


def remove_partial_metadata(destination: Path) -> None:
    partial_metadata_path(destination).unlink(missing_ok=True)


def parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    match = CONTENT_RANGE_RE.match(str(value or "").strip())
    if not match:
        return None
    start, end, total = match.groups()
    return int(start), int(end), None if total == "*" else int(total)


def parse_unsatisfied_total(value: str | None) -> int | None:
    match = UNSATISFIED_RANGE_RE.match(str(value or "").strip())
    return int(match.group(1)) if match else None


def response_remote_info(response, total_size: int | None) -> RemoteInfo:
    return RemoteInfo(
        total_size=total_size,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
    )


def ensure_identity_encoding(headers) -> None:
    encoding = str(headers.get("Content-Encoding", "identity")).strip().lower()
    if encoding not in {"", "identity"}:
        raise PermanentDownloadError(f"unexpected Content-Encoding: {encoding}")


def retry_after_seconds(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def ensure_disk_space(
    data_dir: Path,
    required_bytes: int,
    reserve_gib: float,
) -> None:
    free = shutil.disk_usage(data_dir).free
    reserve = int(reserve_gib * 1024**3)
    if free - max(int(required_bytes), 0) < reserve:
        raise DiskSpaceError(
            f"not enough disk space: free={free / 1024**3:.1f} GiB, "
            f"required transfer={required_bytes / 1024**3:.1f} GiB, "
            f"reserve={reserve_gib:.1f} GiB"
        )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(math.ceil(seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}д {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def progress_description(
    current_bytes: int,
    file_total: int | None,
    bytes_received: int,
    elapsed: float,
    queue_progress: QueueProgress | None,
    planned_file_size: int | None,
) -> str:
    speed = bytes_received / elapsed if bytes_received > 0 and elapsed > 0 else 0.0
    if file_total:
        percent = min(100.0, 100.0 * current_bytes / file_total)
        main = (
            f"{current_bytes / 1e9:.2f}/{file_total / 1e9:.2f} GB "
            f"({percent:.1f}%)"
        )
        file_remaining = max(file_total - current_bytes, 0)
        file_eta = file_remaining / speed if speed > 0 else None
    else:
        main = f"{current_bytes / 1e9:.2f} GB"
        file_eta = None

    speed_text = f"{speed / 1e6:.2f} MB/s" if speed > 0 else "скорость —"
    queue_eta: float | None = None
    if queue_progress is not None and speed > 0:
        queue_total, queue_completed, worker_count = queue_progress.snapshot()
        planned = planned_file_size or file_total or 0
        effective_total = file_total or planned
        adjusted_queue_total = queue_total + (effective_total - planned)
        queue_done = queue_completed + min(
            current_bytes, effective_total or current_bytes
        )
        # This is deliberately approximate: active streams can have different
        # rates, but multiplying by their count is much closer than reporting
        # the ETA of one stream as the ETA of the entire parallel queue.
        queue_eta = max(adjusted_queue_total - queue_done, 0) / (
            speed * max(worker_count, 1)
        )

    return (
        f"{main}; {speed_text}; ETA файла {format_eta(file_eta)}; "
        f"ETA очереди ~{format_eta(queue_eta)}"
    )


def _response_total(response, requested_start: int) -> tuple[int | None, bool]:
    """Return remote total and whether the response is safe to append."""

    status = int(getattr(response, "status", response.getcode()))
    ensure_identity_encoding(response.headers)
    if status == 206:
        parsed = parse_content_range(response.headers.get("Content-Range"))
        if parsed is None:
            raise RetryableDownloadError("206 response has no valid Content-Range")
        start, end, total = parsed
        content_length = parse_optional_int(response.headers.get("Content-Length"))
        if (
            start != requested_start
            or total is None
            or end != total - 1
            or (content_length is not None and content_length != end - start + 1)
        ):
            raise RetryableDownloadError(
                f"inconsistent Content-Range: {start}-{end}/{total}; "
                f"requested start={requested_start}, Content-Length={content_length}"
            )
        return total, requested_start > 0
    if status == 200:
        length = parse_optional_int(response.headers.get("Content-Length"))
        return length, False
    raise PermanentDownloadError(f"unexpected HTTP status {status}")


def prepare_partial(product: Product, timeout: float) -> tuple[Path, int]:
    """Return only a partial whose remote identity was recorded by this script."""

    destination = product.destination
    part = partial_path(destination)
    restart = restart_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = timeout

    final_check = validate_fits(destination, product=product)
    if final_check.ready:
        return part, 0

    part_check = validate_fits(part, product=product)
    if part_check.ready:
        os.replace(part, destination)
        remove_partial_metadata(destination)
        restart.unlink(missing_ok=True)
        remove_quarantined_partials(destination)
        return part, 0

    restart_check = validate_fits(restart, product=product)
    if restart_check.ready:
        os.replace(restart, destination)
        part.unlink(missing_ok=True)
        remove_partial_metadata(destination)
        remove_quarantined_partials(destination)
        return part, 0

    if restart.is_file():
        if not part.is_file() or restart.stat().st_size > part.stat().st_size:
            os.replace(restart, part)
            remove_partial_metadata(destination)
        else:
            restart.unlink()

    # Never move or overwrite an invalid final file before a verified
    # replacement exists. Old files remain available for manual inspection.
    if not part.is_file() or part.stat().st_size == 0:
        return part, 0

    metadata = load_partial_metadata(destination)
    metadata_matches = bool(
        metadata
        and metadata.get("product_uri") == product.product_uri
        and metadata.get("remote_size") is not None
        and (metadata.get("etag") or metadata.get("last_modified"))
    )
    if metadata_matches:
        return part, part.stat().st_size

    # A legacy partial has no trustworthy ETag/Last-Modified. Preserve it, but
    # start a clean transfer instead of mixing two archive revisions.
    quarantine_partial(destination)
    return part, 0


def _metadata_conflicts(metadata: dict[str, object] | None, remote: RemoteInfo) -> bool:
    if not metadata:
        return False
    old_size = metadata.get("remote_size")
    if old_size is None or remote.total_size is None or int(old_size) != remote.total_size:
        return True
    old_etag = metadata.get("etag")
    if old_etag:
        return remote.etag != old_etag
    old_modified = metadata.get("last_modified")
    if old_modified:
        return remote.last_modified != old_modified
    return True


def adopt_restart_progress(
    product: Product,
    restart: Path,
    remote: RemoteInfo | None,
) -> None:
    """Keep the more advanced safe prefix after a failed full restart."""

    if not restart.is_file() or remote is None:
        return
    part = partial_path(product.destination)
    if not part.is_file() or restart.stat().st_size > part.stat().st_size:
        os.replace(restart, part)
        save_partial_metadata(product, remote)
    else:
        restart.unlink()


def transfer_once(
    product: Product,
    data_dir: Path,
    reserve_gib: float,
    timeout: float,
    chunk_size: int,
    progress_interval: float = 10.0,
    queue_progress: QueueProgress | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, int | None, int]:
    """Perform one HTTP attempt and return status, remote size, bytes received."""

    part, start = prepare_partial(product, timeout)
    ready_after_preparation = validate_fits(product.destination, product=product)
    if ready_after_preparation.ready:
        return "already-ready", ready_after_preparation.size, 0
    metadata = load_partial_metadata(product.destination)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    }
    if start:
        headers["Range"] = f"bytes={start}-"
        if metadata:
            if metadata.get("etag"):
                headers["If-Range"] = str(metadata["etag"])
            elif metadata.get("last_modified"):
                headers["If-Range"] = str(metadata["last_modified"])

    estimated_total = product.manifest_size
    estimated_remaining = (
        max(estimated_total - start, 0) if estimated_total is not None else 0
    )
    ensure_disk_space(data_dir, estimated_remaining, reserve_gib)

    request = Request(product.url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416:
            total = parse_unsatisfied_total(exc.headers.get("Content-Range"))
            if total is not None and part.is_file() and part.stat().st_size == total:
                check = validate_fits(
                    part,
                    product=product,
                    authoritative_size=total,
                    use_manifest_tolerance=False,
                )
                if check.ready:
                    os.replace(part, product.destination)
                    remove_partial_metadata(product.destination)
                    remove_quarantined_partials(product.destination)
                    return "resumed", total, 0
            quarantine_partial(product.destination)
            raise RetryableDownloadError("remote rejected stale partial file with HTTP 416")
        retry_after = retry_after_seconds(exc.headers.get("Retry-After"))
        if exc.code in RETRYABLE_HTTP_CODES:
            raise RetryableDownloadError(
                f"HTTP {exc.code}: {exc.reason}", retry_after=retry_after
            ) from exc
        raise PermanentDownloadError(f"HTTP {exc.code}: {exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RetryableDownloadError(str(exc)) from exc

    bytes_received = 0
    requested_start = start
    resumed_response = False
    remote: RemoteInfo | None = None
    active_path = part
    restart_in_use = False
    try:
        with response:
            remote_total, resumed_response = _response_total(response, requested_start)
            remote = response_remote_info(response, remote_total)

            if resumed_response and _metadata_conflicts(metadata, remote):
                quarantine_partial(product.destination)
                raise RetryableDownloadError("remote file changed; restarting from byte zero")

            write_start = requested_start if resumed_response else 0
            required = (
                max(remote_total - write_start, 0)
                if remote_total is not None
                else max((product.manifest_size or 0) - write_start, 0)
            )
            ensure_disk_space(data_dir, required, reserve_gib)

            restart_in_use = bool(requested_start and not resumed_response)
            active_path = restart_path(product.destination) if restart_in_use else part
            mode = "ab" if resumed_response else "wb"
            next_progress = time.monotonic() + progress_interval
            transfer_started = time.monotonic()
            last_reported_bytes = -1
            with active_path.open(mode) as output:
                if not restart_in_use:
                    save_partial_metadata(product, remote)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise DownloadCancelled("остановлено пользователем")
                    block = response.read(chunk_size)
                    if not block:
                        break
                    output.write(block)
                    bytes_received += len(block)
                    if time.monotonic() >= next_progress:
                        current = write_start + bytes_received
                        progress = progress_description(
                            current_bytes=current,
                            file_total=remote_total,
                            bytes_received=bytes_received,
                            elapsed=time.monotonic() - transfer_started,
                            queue_progress=queue_progress,
                            planned_file_size=product.manifest_size,
                        )
                        print(
                            f"    [{product.target} {product.filter_name}] {progress}",
                            flush=True,
                        )
                        last_reported_bytes = bytes_received
                        next_progress = time.monotonic() + progress_interval
                output.flush()
                os.fsync(output.fileno())
                if bytes_received and bytes_received != last_reported_bytes:
                    current = write_start + bytes_received
                    progress = progress_description(
                        current_bytes=current,
                        file_total=remote_total,
                        bytes_received=bytes_received,
                        elapsed=time.monotonic() - transfer_started,
                        queue_progress=queue_progress,
                        planned_file_size=product.manifest_size,
                    )
                    print(
                        f"    [{product.target} {product.filter_name}] {progress}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise
    except DownloadCancelled:
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise
    except (RetryableDownloadError, PermanentDownloadError):
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise
    except (URLError, TimeoutError, OSError) as exc:
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise RetryableDownloadError(str(exc)) from exc
    except Exception as exc:
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise RetryableDownloadError(f"connection interrupted: {exc}") from exc

    final_size = active_path.stat().st_size if active_path.exists() else 0
    if remote_total is not None and final_size != remote_total:
        if restart_in_use:
            adopt_restart_progress(product, active_path, remote)
        raise RetryableDownloadError(
            f"incomplete response: local={final_size}, remote={remote_total} bytes"
        )

    check = validate_fits(
        active_path,
        product=product,
        authoritative_size=remote_total,
        use_manifest_tolerance=remote_total is None,
    )
    if not check.ready:
        if restart_in_use:
            active_path.unlink(missing_ok=True)
        else:
            quarantine_partial(product.destination)
        raise RetryableDownloadError(f"downloaded FITS is invalid: {check.reason}")

    os.replace(active_path, product.destination)
    part.unlink(missing_ok=True)
    restart_path(product.destination).unlink(missing_ok=True)
    remove_partial_metadata(product.destination)
    remove_quarantined_partials(product.destination)
    try:
        directory_fd = os.open(product.destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass

    status = "resumed" if resumed_response and requested_start else "downloaded"
    return status, remote_total, bytes_received


def download_product(
    product: Product,
    data_dir: Path,
    reserve_gib: float,
    max_attempts: int,
    timeout: float,
    chunk_size: int,
    queue_progress: QueueProgress | None = None,
    cancel_event: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_source: Callable[[], float] = random.random,
) -> DownloadResult:
    """Download one product with bounded retries; never starts another transfer."""

    existing = validate_fits(product.destination, product=product)
    if existing.ready:
        return DownloadResult(
            key=product.key,
            program=product.program,
            target=product.target,
            obsid=product.obsid,
            role=product.role,
            filter_name=product.filter_name,
            destination=str(product.destination),
            status="already-ready",
            message=existing.reason,
            size=existing.size,
            manifest_size=product.manifest_size,
        )

    total_received = 0
    for attempt in range(1, max_attempts + 1):
        try:
            status, remote_size, received = transfer_once(
                product=product,
                data_dir=data_dir,
                reserve_gib=reserve_gib,
                timeout=timeout,
                chunk_size=chunk_size,
                queue_progress=queue_progress,
                cancel_event=cancel_event,
            )
            total_received += received
            final_check = validate_fits(
                product.destination,
                product=product,
                authoritative_size=remote_size,
                use_manifest_tolerance=remote_size is None,
            )
            if not final_check.ready:
                raise RetryableDownloadError(
                    f"published file failed validation: {final_check.reason}"
                )
            digest = sha256_file(product.destination)
            drift = ""
            if (
                remote_size is not None
                and product.manifest_size is not None
                and remote_size != product.manifest_size
            ):
                drift = (
                    f"; remote/manifest size drift "
                    f"{remote_size - product.manifest_size:+d} bytes"
                )
            return DownloadResult(
                key=product.key,
                program=product.program,
                target=product.target,
                obsid=product.obsid,
                role=product.role,
                filter_name=product.filter_name,
                destination=str(product.destination),
                status=status,
                message=f"FITS verified; sha256={digest}{drift}",
                size=product.destination.stat().st_size,
                remote_size=remote_size,
                manifest_size=product.manifest_size,
                sha256=digest,
                attempts=attempt,
                downloaded_bytes=total_received,
            )
        except DownloadCancelled:
            raise
        except DiskSpaceError:
            raise
        except PermanentDownloadError as exc:
            return DownloadResult(
                key=product.key,
                program=product.program,
                target=product.target,
                obsid=product.obsid,
                role=product.role,
                filter_name=product.filter_name,
                destination=str(product.destination),
                status="failed",
                message=str(exc),
                size=product.destination.stat().st_size
                if product.destination.exists()
                else 0,
                manifest_size=product.manifest_size,
                attempts=attempt,
                downloaded_bytes=total_received,
            )
        except RetryableDownloadError as exc:
            if attempt >= max_attempts:
                return DownloadResult(
                    key=product.key,
                    program=product.program,
                    target=product.target,
                    obsid=product.obsid,
                    role=product.role,
                    filter_name=product.filter_name,
                    destination=str(product.destination),
                    status="failed",
                    message=f"after {attempt} attempts: {exc}",
                    size=product.destination.stat().st_size
                    if product.destination.exists()
                    else 0,
                    manifest_size=product.manifest_size,
                    attempts=attempt,
                    downloaded_bytes=total_received,
                )
            delay = exc.retry_after
            if delay is None:
                delay = min(120.0, 5.0 * (2 ** (attempt - 1)))
                delay *= 0.9 + 0.2 * random_source()
            print(
                f"    попытка {attempt}/{max_attempts} не удалась: {exc}; "
                f"повтор через {delay:.1f} с",
                flush=True,
            )
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    raise DownloadCancelled("остановлено пользователем")
            else:
                sleep(delay)

    raise AssertionError("unreachable retry loop")


def format_gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def print_plan(
    plans: list[ProductPlan],
    disabled_rows: list[dict[str, str]],
    data_dir: Path,
    reserve_gib: float,
) -> None:
    ready = [plan for plan in plans if plan.status == "ready"]
    pending = [plan for plan in plans if plan.status != "ready"]
    remaining = sum(plan.remaining_size for plan in pending)
    transfer_volume = sum(
        plan.product.manifest_size or plan.remaining_size for plan in pending
    )
    free = shutil.disk_usage(data_dir).free

    print(
        f"Включено: {len(plans) // 2} целей, {len(plans)} файлов. "
        f"Отключено неполных пар: {len(disabled_rows)}."
    )
    print(
        f"Готово: {len(ready)} файла. В очереди: {len(pending)}. "
        f"Оценка оставшегося объёма: {format_gib(remaining)}."
    )
    print(
        f"Верхняя оценка сетевого трафика: {format_gib(transfer_volume)}."
    )
    print(
        f"Свободно сейчас: {format_gib(free)}; оценка после загрузки: "
        f"{format_gib(max(free - remaining, 0))}; неприкосновенный резерв: {reserve_gib:.1f} GiB."
    )
    for index, plan in enumerate(pending, start=1):
        product = plan.product
        print(
            f"  {index:03d}. [{plan.status.upper():10}] GO-{product.program} "
            f"{product.target:12} {product.filter_name:6} "
            f"remaining~{format_gib(plan.remaining_size):>10} -> "
            f"{product.destination}"
        )
        if plan.status == "incomplete":
            print(f"       причина: {plan.reason}")


def serialise_result(result: DownloadResult) -> dict[str, object]:
    return asdict(result)


def write_status(
    path: Path,
    started_at: str,
    programs: set[str],
    data_dir: Path,
    results: Iterable[DownloadResult],
    download_workers: int = 1,
    interrupted: bool = False,
) -> None:
    records = list(results)
    counts: dict[str, int] = {}
    for result in records:
        counts[result.status] = counts.get(result.status, 0) + 1
    atomic_write_json(
        path,
        {
            "started_at": started_at,
            "updated_at": utc_now(),
            "programs": sorted(programs),
            "data_dir": str(data_dir.resolve()),
            "download_workers": download_workers,
            "sequential_downloads": download_workers <= 1,
            "parallel_downloads": download_workers > 1,
            "interrupted": interrupted,
            "counts": counts,
            "results": [serialise_result(result) for result in records],
        },
    )


def acquire_run_lock(data_dir: Path):
    """Prevent two standalone collectors from writing the same products."""

    import fcntl

    path = data_dir / ".download_go3055_go7763.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise PermanentDownloadError(
            f"уже запущен другой download_go3055_go7763.py ({path})"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={utc_now()}\n")
    handle.flush()
    return handle


def release_run_lock(handle) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program",
        choices=("3055", "7763", "both"),
        default="both",
        help="выбрать программу; по умолчанию обе",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="запустить реальную загрузку",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="число одновременно скачиваемых файлов; по умолчанию 4",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="корневой каталог с папками целей",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=40.0,
        help="минимальный свободный остаток; по умолчанию 40 GiB",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=8,
        help="максимум HTTP-попыток на файл; по умолчанию 8",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="тайм-аут сокета в секундах; по умолчанию 120",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=1,
        help="размер блока чтения в MiB; по умолчанию 8",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="путь к JSON-отчёту; по умолчанию он лежит в DATA_DIR",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers должен быть от 1 до 16")
    if args.reserve_gib < 0:
        raise SystemExit("--reserve-gib не может быть отрицательным")
    if args.attempts < 1:
        raise SystemExit("--attempts должен быть не меньше 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout должен быть положительным")
    if args.chunk_mib < 1:
        raise SystemExit("--chunk-mib должен быть не меньше 1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    programs = {"3055", "7763"} if args.program == "both" else {args.program}
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    status_file = args.status_file or data_dir / "download_go3055_go7763_status.json"

    products, disabled_rows = read_products(programs, data_dir)
    plans = [classify_product(product) for product in products]
    print_plan(plans, disabled_rows, data_dir, args.reserve_gib)

    if not args.download:
        print("Сухой прогон завершён. Сеть не использовалась. Для старта добавьте --download.")
        return 0

    pending = [plan.product for plan in plans if plan.status != "ready"]
    required_growth = sum(plan.remaining_size for plan in plans if plan.status != "ready")
    worker_count = min(args.workers, len(pending)) if pending else 0
    queue_progress = QueueProgress(
        total_bytes=sum(product.manifest_size or 0 for product in pending),
        worker_count=max(worker_count, 1),
    )
    try:
        ensure_disk_space(data_dir, required_growth, args.reserve_gib)
        lock_handle = acquire_run_lock(data_dir)
    except (DiskSpaceError, PermanentDownloadError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2

    started_at = utc_now()
    results: list[DownloadResult] = [
        DownloadResult(
            key=plan.product.key,
            program=plan.product.program,
            target=plan.product.target,
            obsid=plan.product.obsid,
            role=plan.product.role,
            filter_name=plan.product.filter_name,
            destination=str(plan.product.destination),
            status="already-ready",
            message=plan.reason,
            size=plan.current_size,
            manifest_size=plan.product.manifest_size,
        )
        for plan in plans
        if plan.status == "ready"
    ]
    cancel_event = threading.Event()
    executor: ThreadPoolExecutor | None = None
    future_to_product: dict[Future[DownloadResult], Product] = {}

    def run_product(index: int, product: Product) -> DownloadResult:
        if cancel_event.is_set():
            raise DownloadCancelled("остановлено пользователем")
        print(
            f"[старт {index}/{len(pending)}] GO-{product.program} "
            f"{product.target} {product.filter_name}: {product.file_name}",
            flush=True,
        )
        return download_product(
            product=product,
            data_dir=data_dir,
            reserve_gib=args.reserve_gib,
            max_attempts=args.attempts,
            timeout=args.timeout,
            chunk_size=args.chunk_mib * 1024**2,
            queue_progress=queue_progress,
            cancel_event=cancel_event,
        )

    try:
        write_status(
            status_file,
            started_at,
            programs,
            data_dir,
            results,
            download_workers=worker_count,
        )
        if pending:
            print(
                f"Запуск: {worker_count} одновременных загрузок; "
                f"файлов в очереди: {len(pending)}.",
                flush=True,
            )
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="jwst-download",
            )
            future_to_product = {
                executor.submit(run_product, index, product): product
                for index, product in enumerate(pending, start=1)
            }

        completed = 0
        for future in as_completed(future_to_product):
            product = future_to_product[future]
            result = future.result()
            results.append(result)
            queue_progress.finish_product(product, result)
            completed += 1
            print(
                f"[готово {completed}/{len(pending)}] {product.target} "
                f"{product.filter_name}: {result.status}; {result.message}",
                flush=True,
            )
            write_status(
                status_file,
                started_at,
                programs,
                data_dir,
                results,
                download_workers=worker_count,
            )

        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
    except KeyboardInterrupt:
        cancel_event.set()
        for future in future_to_product:
            future.cancel()
        if executor is not None:
            print("Останавливаю активные загрузки...", file=sys.stderr, flush=True)
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        write_status(
            status_file,
            started_at,
            programs,
            data_dir,
            results,
            download_workers=worker_count,
            interrupted=True,
        )
        print(
            "Прервано. Активные .part сохранены и будут докачаны при следующем запуске.",
            file=sys.stderr,
        )
        release_run_lock(lock_handle)
        return 130
    except DiskSpaceError as exc:
        cancel_event.set()
        for future in future_to_product:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        write_status(
            status_file,
            started_at,
            programs,
            data_dir,
            results,
            download_workers=worker_count,
        )
        print(f"Остановлено до исчерпания дискового резерва: {exc}", file=sys.stderr)
        release_run_lock(lock_handle)
        return 2
    except BaseException:
        cancel_event.set()
        for future in future_to_product:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        release_run_lock(lock_handle)
        raise

    failures = [result for result in results if result.status == "failed"]
    print(
        f"Завершено. Готово/успешно: {len(results) - len(failures)}; "
        f"ошибок: {len(failures)}. Отчёт: {status_file}"
    )
    release_run_lock(lock_handle)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
