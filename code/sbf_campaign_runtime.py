#!/usr/bin/env python3
"""Runtime primitives for long-running SBF processing campaigns.

The module intentionally has no mandatory third-party dependencies.  ``psutil``
and ``astropy`` are used when present, but campaign state and process cleanup
remain usable without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union


PathLike = Union[str, os.PathLike[str]]
BytesThreshold = Optional[Union[int, float]]


__all__ = [
    "Deadline",
    "SignalController",
    "TerminationResult",
    "SupervisionResult",
    "launch_process_group",
    "terminate_process_group",
    "collect_resource_sample",
    "supervise_process",
    "atomic_write_text",
    "atomic_write_json",
    "sha256_file",
    "validate_fits_artifacts",
    "build_artifact_manifest",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_nonnegative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value < 0 or value == float("inf") or value != value:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass
class Deadline:
    """Monotonic wall-time deadline with a soft stop before the hard limit.

    ``soft_stop_seconds`` is a reserve measured backwards from the hard
    deadline.  Once that point is reached, :meth:`may_start` returns ``False``
    while already running work may continue until ``hard_expired``.
    """

    wall_time_seconds: Optional[float]
    soft_stop_seconds: float = 0.0
    started_monotonic: Optional[float] = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.wall_time_seconds = _finite_nonnegative(
            self.wall_time_seconds, "wall_time_seconds"
        )
        self.soft_stop_seconds = _finite_nonnegative(
            self.soft_stop_seconds, "soft_stop_seconds"
        ) or 0.0
        if self.started_monotonic is None:
            self.started_monotonic = float(self.clock())
        else:
            self.started_monotonic = float(self.started_monotonic)

    @classmethod
    def from_hours(
        cls,
        wall_time_hours: Optional[float],
        soft_stop_minutes: float = 0.0,
        **kwargs: Any,
    ) -> "Deadline":
        wall_seconds = None if wall_time_hours is None else float(wall_time_hours) * 3600.0
        return cls(
            wall_time_seconds=wall_seconds,
            soft_stop_seconds=float(soft_stop_minutes) * 60.0,
            **kwargs,
        )

    @property
    def hard_at(self) -> Optional[float]:
        if self.wall_time_seconds is None:
            return None
        return float(self.started_monotonic) + self.wall_time_seconds

    @property
    def soft_at(self) -> Optional[float]:
        if self.hard_at is None:
            return None
        return max(
            float(self.started_monotonic),
            self.hard_at - self.soft_stop_seconds,
        )

    def remaining(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds left until the hard deadline, or ``None`` if unlimited."""
        if self.hard_at is None:
            return None
        current = float(self.clock() if now is None else now)
        return max(0.0, self.hard_at - current)

    def soft_remaining(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds left before new work should stop being started."""
        if self.soft_at is None:
            return None
        current = float(self.clock() if now is None else now)
        return max(0.0, self.soft_at - current)

    @property
    def hard_expired(self) -> bool:
        return self.hard_at is not None and float(self.clock()) >= self.hard_at

    @property
    def soft_stop_reached(self) -> bool:
        return self.soft_at is not None and float(self.clock()) >= self.soft_at

    def is_hard_expired(self, now: Optional[float] = None) -> bool:
        if self.hard_at is None:
            return False
        current = float(self.clock() if now is None else now)
        return current >= self.hard_at

    def may_start(
        self,
        estimated_seconds: float = 0.0,
        reserve_seconds: float = 0.0,
        now: Optional[float] = None,
    ) -> bool:
        """Return whether a new unit of work fits before the soft stop."""
        estimated = _finite_nonnegative(estimated_seconds, "estimated_seconds") or 0.0
        reserve = _finite_nonnegative(reserve_seconds, "reserve_seconds") or 0.0
        if self.soft_at is None:
            return True
        current = float(self.clock() if now is None else now)
        return current + estimated + reserve < self.soft_at

    def as_dict(self) -> dict[str, Any]:
        now = float(self.clock())
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "soft_stop_seconds": self.soft_stop_seconds,
            "started_monotonic": self.started_monotonic,
            "hard_at_monotonic": self.hard_at,
            "soft_at_monotonic": self.soft_at,
            "remaining_seconds": self.remaining(now),
            "soft_remaining_seconds": self.soft_remaining(now),
            "hard_expired": self.is_hard_expired(now),
            "soft_stop_reached": self.soft_at is not None and now >= self.soft_at,
        }


class SignalController:
    """Context-managed stop flag for SIGINT, SIGTERM and SIGHUP.

    Previous handlers are restored on exit.  Installing twice is harmless, and
    a partial installation is rolled back if registration fails.
    """

    def __init__(self, signals: Optional[Iterable[int]] = None) -> None:
        default_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            default_signals.append(signal.SIGHUP)
        self.signals = tuple(dict.fromkeys(default_signals if signals is None else signals))
        self._previous: dict[int, Any] = {}
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._installed = False
        self._signum: Optional[int] = None
        self._received_at: Optional[str] = None
        self._count = 0

    def _handler(self, signum: int, _frame: Any) -> None:
        with self._lock:
            if self._signum is None:
                self._signum = int(signum)
                self._received_at = _utc_now()
            self._count += 1
            self._event.set()

    def install(self) -> "SignalController":
        if self._installed:
            return self
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be installed in the main thread")
        try:
            for signum in self.signals:
                self._previous[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
        except BaseException:
            self.restore()
            raise
        self._installed = True
        return self

    def restore(self) -> None:
        for signum, previous in reversed(tuple(self._previous.items())):
            try:
                signal.signal(signum, previous)
            except (OSError, RuntimeError, ValueError):
                pass
        self._previous.clear()
        self._installed = False

    def request_stop(self, signum: Optional[int] = None) -> None:
        """Set the stop flag manually; useful for orchestration and tests."""
        with self._lock:
            if self._signum is None and signum is not None:
                self._signum = int(signum)
                self._received_at = _utc_now()
            self._count += 1
            self._event.set()

    def clear(self) -> None:
        with self._lock:
            self._signum = None
            self._received_at = None
            self._count = 0
            self._event.clear()

    @property
    def stop_requested(self) -> bool:
        return self._event.is_set()

    @property
    def signum(self) -> Optional[int]:
        return self._signum

    @property
    def signal_name(self) -> Optional[str]:
        if self._signum is None:
            return None
        try:
            return signal.Signals(self._signum).name
        except (ValueError, AttributeError):
            return str(self._signum)

    @property
    def received_at(self) -> Optional[str]:
        return self._received_at

    @property
    def count(self) -> int:
        return self._count

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def __enter__(self) -> "SignalController":
        return self.install()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.restore()


@dataclass
class TerminationResult:
    pid: int
    already_exited: bool = False
    term_sent: bool = False
    kill_sent: bool = False
    returncode: Optional[int] = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def launch_process_group(
    command: Sequence[Union[str, os.PathLike[str]]],
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    **popen_kwargs: Any,
) -> subprocess.Popen[Any]:
    """Launch a command in a new process group/session.

    On POSIX this uses ``start_new_session``.  On Windows it uses
    ``CREATE_NEW_PROCESS_GROUP`` so the entire worker tree can be stopped.
    """
    if not command:
        raise ValueError("command must not be empty")
    args = [os.fspath(part) for part in command]
    if os.name == "nt":
        flags = int(popen_kwargs.pop("creationflags", 0))
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        popen_kwargs["creationflags"] = flags
    else:
        popen_kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(args, cwd=cwd, env=env, **popen_kwargs)


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    end = time.monotonic() + max(0.0, timeout)
    while True:
        if predicate():
            return True
        remaining = end - time.monotonic()
        if remaining <= 0:
            return predicate()
        time.sleep(min(0.1, remaining))


def _posix_group_alive(pgid: Optional[int]) -> bool:
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_windows_tree(
    process: subprocess.Popen[Any],
    grace_seconds: float,
    errors: list[str],
) -> tuple[bool, list[Any]]:
    survivors: list[Any] = []
    try:
        import psutil  # type: ignore

        root = psutil.Process(process.pid)
        tree = root.children(recursive=True) + [root]
        for member in reversed(tree):
            try:
                member.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, survivors = psutil.wait_procs(tree, timeout=max(0.0, grace_seconds))
        return True, survivors
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"psutil terminate: {exc}")
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        return True, survivors
    except Exception as exc:
        errors.append(f"CTRL_BREAK_EVENT: {exc}")
    try:
        process.terminate()
        return True, survivors
    except Exception as exc:
        errors.append(f"terminate: {exc}")
        return False, survivors


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    term_grace_seconds: float = 30.0,
    kill_grace_seconds: float = 5.0,
) -> TerminationResult:
    """Stop a worker tree with TERM, wait, then force KILL if necessary."""
    term_grace_seconds = _finite_nonnegative(
        term_grace_seconds, "term_grace_seconds"
    ) or 0.0
    kill_grace_seconds = _finite_nonnegative(
        kill_grace_seconds, "kill_grace_seconds"
    ) or 0.0
    started = time.monotonic()
    result = TerminationResult(pid=int(process.pid))
    if process.poll() is not None:
        result.already_exited = True
        result.returncode = process.returncode
        return result

    if os.name == "nt":
        sent, survivors = _terminate_windows_tree(
            process, term_grace_seconds, result.errors
        )
        result.term_sent = sent
        exited = _wait_until(lambda: process.poll() is not None, term_grace_seconds)
        if not exited or survivors:
            result.kill_sent = True
            if survivors:
                for member in survivors:
                    try:
                        member.kill()
                    except Exception as exc:
                        result.errors.append(f"psutil kill: {exc}")
            try:
                if process.poll() is None:
                    process.kill()
            except Exception as exc:
                result.errors.append(f"kill: {exc}")
            _wait_until(lambda: process.poll() is not None, kill_grace_seconds)
    else:
        pgid: Optional[int]
        try:
            pgid = os.getpgid(process.pid)
            if pgid == os.getpgrp():
                result.errors.append("worker shares supervisor process group; using PID only")
                pgid = None
        except ProcessLookupError:
            pgid = None
        except Exception as exc:
            result.errors.append(f"getpgid: {exc}")
            pgid = None

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                process.terminate()
            result.term_sent = True
        except ProcessLookupError:
            pass
        except Exception as exc:
            result.errors.append(f"SIGTERM: {exc}")

        def group_finished() -> bool:
            root_done = process.poll() is not None
            return root_done and not _posix_group_alive(pgid)

        if not _wait_until(group_finished, term_grace_seconds):
            result.kill_sent = True
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except ProcessLookupError:
                pass
            except Exception as exc:
                result.errors.append(f"SIGKILL: {exc}")
            _wait_until(group_finished, kill_grace_seconds)

    try:
        result.returncode = process.poll()
        if result.returncode is None:
            result.returncode = process.wait(timeout=0)
    except Exception:
        result.returncode = process.poll()
    result.duration_seconds = time.monotonic() - started
    return result


def _bytes_gib(value: Optional[int]) -> Optional[float]:
    return None if value is None else float(value) / 1024.0**3


def _read_linux_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts:
                multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
                values[key] = int(parts[0]) * multiplier
    except (OSError, ValueError):
        pass
    return values


def _fallback_system_memory() -> tuple[dict[str, Any], dict[str, Any]]:
    ram: dict[str, Any] = {}
    swap: dict[str, Any] = {}
    if sys.platform.startswith("linux"):
        mem = _read_linux_meminfo()
        total = mem.get("MemTotal")
        available = mem.get("MemAvailable", mem.get("MemFree"))
        if total is not None:
            ram["total_bytes"] = total
        if available is not None:
            ram["available_bytes"] = available
        if total is not None and available is not None:
            ram["used_bytes"] = max(0, total - available)
            ram["percent"] = 100.0 * ram["used_bytes"] / total if total else 0.0
        swap_total = mem.get("SwapTotal")
        swap_free = mem.get("SwapFree")
        if swap_total is not None:
            swap["total_bytes"] = swap_total
        if swap_free is not None:
            swap["free_bytes"] = swap_free
        if swap_total is not None and swap_free is not None:
            swap["used_bytes"] = max(0, swap_total - swap_free)
            swap["percent"] = 100.0 * swap["used_bytes"] / swap_total if swap_total else 0.0
    elif sys.platform == "darwin":
        try:
            total = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            )
            ram["total_bytes"] = total
            output = subprocess.check_output(["vm_stat"], text=True)
            page_size = 4096
            pages: dict[str, int] = {}
            for line in output.splitlines():
                if "page size of" in line:
                    page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0])
                elif ":" in line:
                    key, raw = line.split(":", 1)
                    pages[key.strip()] = int(raw.strip().rstrip("."))
            available = page_size * (
                pages.get("Pages free", 0)
                + pages.get("Pages inactive", 0)
                + pages.get("Pages speculative", 0)
                + pages.get("Pages purgeable", 0)
            )
            ram.update(
                available_bytes=available,
                used_bytes=max(0, total - available),
                percent=100.0 * max(0, total - available) / total if total else 0.0,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    for section in (ram, swap):
        for key in ("total", "available", "used", "free"):
            byte_key = f"{key}_bytes"
            if byte_key in section:
                section[f"{key}_gib"] = _bytes_gib(section[byte_key])
    return ram, swap


def _linux_process_tree_rss(pid: int) -> tuple[Optional[int], Optional[int], int]:
    seen: set[int] = set()

    def walk(current: int) -> tuple[int, int]:
        if current in seen:
            return 0, 0
        seen.add(current)
        rss = 0
        try:
            for line in Path(f"/proc/{current}/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
        child_total = 0
        descendants = 0
        try:
            raw_children = Path(
                f"/proc/{current}/task/{current}/children"
            ).read_text(encoding="ascii")
            children = [int(value) for value in raw_children.split()]
        except (OSError, ValueError):
            children = []
        for child in children:
            child_rss, child_count = walk(child)
            child_total += child_rss
            descendants += 1 + child_count
        return rss + child_total, descendants

    total, count = walk(pid)
    if not seen or total == 0:
        return None, None, count
    try:
        root_total, _ = _linux_process_tree_rss_root(pid)
    except Exception:
        root_total = 0
    return root_total or None, max(0, total - root_total), count


def _linux_process_tree_rss_root(pid: int) -> tuple[int, bool]:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024, True
    except (OSError, ValueError, IndexError):
        pass
    return 0, False


def collect_resource_sample(
    process: Union[subprocess.Popen[Any], int],
    disk_path: PathLike = ".",
) -> dict[str, Any]:
    """Collect one system + worker-tree resource sample.

    The returned dictionary contains both nested sections and flat byte fields
    used by :func:`supervise_process`.
    """
    pid = int(process if isinstance(process, int) else process.pid)
    sample: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "monotonic": time.monotonic(),
        "pid": pid,
        "collector": "fallback",
    }
    try:
        usage = shutil.disk_usage(disk_path)
        disk = {
            "path": str(Path(disk_path).resolve()),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "total_gib": _bytes_gib(usage.total),
            "used_gib": _bytes_gib(usage.used),
            "free_gib": _bytes_gib(usage.free),
        }
    except OSError as exc:
        disk = {"path": str(disk_path), "error": str(exc)}

    ram: dict[str, Any]
    swap: dict[str, Any]
    worker: dict[str, Any] = {"pid": pid}
    try:
        import psutil  # type: ignore

        sample["collector"] = "psutil"
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        ram = {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "percent": float(vm.percent),
        }
        swap = {
            "total_bytes": int(sm.total),
            "used_bytes": int(sm.used),
            "free_bytes": int(sm.free),
            "percent": float(sm.percent),
        }
        try:
            root = psutil.Process(pid)
            root_rss = int(root.memory_info().rss)
            children = root.children(recursive=True)
            children_rss = 0
            live_children = 0
            for child in children:
                try:
                    children_rss += int(child.memory_info().rss)
                    live_children += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            worker.update(
                rss_bytes=root_rss,
                children_rss_bytes=children_rss,
                total_rss_bytes=root_rss + children_rss,
                child_count=live_children,
                status=root.status(),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            worker["error"] = str(exc)
    except ImportError:
        ram, swap = _fallback_system_memory()
        if sys.platform.startswith("linux"):
            root_rss, children_rss, child_count = _linux_process_tree_rss(pid)
            if root_rss is not None:
                worker.update(
                    rss_bytes=root_rss,
                    children_rss_bytes=children_rss or 0,
                    total_rss_bytes=root_rss + (children_rss or 0),
                    child_count=child_count,
                )
    except Exception as exc:
        ram, swap = _fallback_system_memory()
        sample["collector_error"] = str(exc)

    for section in (ram, swap, worker):
        for key in ("total", "available", "used", "free", "rss", "children_rss", "total_rss"):
            byte_key = f"{key}_bytes"
            if byte_key in section:
                section[f"{key}_gib"] = _bytes_gib(section[byte_key])

    sample.update(system_ram=ram, swap=swap, disk=disk, worker=worker)
    sample["available_ram_bytes"] = ram.get("available_bytes")
    sample["swap_used_bytes"] = swap.get("used_bytes")
    sample["disk_free_bytes"] = disk.get("free_bytes")
    sample["worker_rss_bytes"] = worker.get("rss_bytes")
    sample["worker_children_rss_bytes"] = worker.get("children_rss_bytes")
    sample["worker_total_rss_bytes"] = worker.get("total_rss_bytes")
    return sample


@dataclass
class SupervisionResult:
    reason: str
    returncode: Optional[int]
    started_monotonic: float
    ended_monotonic: float
    sample_count: int
    last_sample: Optional[dict[str, Any]] = None
    detail: Optional[str] = None
    signal_name: Optional[str] = None
    termination: Optional[dict[str, Any]] = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_monotonic - self.started_monotonic)

    @property
    def completed(self) -> bool:
        return self.reason == "completed"

    @property
    def ok(self) -> bool:
        return self.completed and self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duration_seconds"] = self.duration_seconds
        result["completed"] = self.completed
        result["ok"] = self.ok
        return result


def _threshold_bytes(bytes_value: BytesThreshold, gib_value: BytesThreshold) -> Optional[int]:
    if bytes_value is not None and gib_value is not None:
        raise ValueError("specify a threshold either in bytes or GiB, not both")
    if gib_value is not None:
        gib = _finite_nonnegative(float(gib_value), "GiB threshold")
        return None if gib is None else int(gib * 1024.0**3)
    if bytes_value is None:
        return None
    value = _finite_nonnegative(float(bytes_value), "byte threshold")
    return None if value is None else int(value)


def supervise_process(
    process: subprocess.Popen[Any],
    *,
    deadline: Optional[Deadline] = None,
    signal_controller: Optional[SignalController] = None,
    timeout_seconds: Optional[float] = None,
    sample_interval_seconds: float = 30.0,
    disk_path: PathLike = ".",
    min_available_ram_bytes: BytesThreshold = None,
    emergency_available_ram_bytes: BytesThreshold = None,
    max_worker_rss_bytes: BytesThreshold = None,
    min_free_disk_bytes: BytesThreshold = None,
    min_available_ram_gib: BytesThreshold = None,
    emergency_available_ram_gib: BytesThreshold = None,
    max_worker_rss_gib: BytesThreshold = None,
    min_free_disk_gib: BytesThreshold = None,
    low_ram_samples_before_stop: int = 3,
    callback: Optional[Callable[[dict[str, Any]], None]] = None,
    resource_sampler: Optional[
        Callable[[Union[subprocess.Popen[Any], int], PathLike], dict[str, Any]]
    ] = None,
    term_grace_seconds: float = 30.0,
    kill_grace_seconds: float = 5.0,
) -> SupervisionResult:
    """Monitor a worker until completion or an enforced stop condition.

    Result reasons are ``completed``, ``deadline``, ``signal``, ``timeout`` or
    ``resource``.  Low RAM must persist for ``low_ram_samples_before_stop``
    samples; emergency RAM, disk and worker-RSS limits stop immediately.
    """
    timeout_seconds = _finite_nonnegative(timeout_seconds, "timeout_seconds")
    sample_interval_seconds = _finite_nonnegative(
        sample_interval_seconds, "sample_interval_seconds"
    ) or 0.0
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than zero")
    if low_ram_samples_before_stop < 1:
        raise ValueError("low_ram_samples_before_stop must be at least 1")

    min_ram = _threshold_bytes(min_available_ram_bytes, min_available_ram_gib)
    emergency_ram = _threshold_bytes(
        emergency_available_ram_bytes, emergency_available_ram_gib
    )
    max_rss = _threshold_bytes(max_worker_rss_bytes, max_worker_rss_gib)
    min_disk = _threshold_bytes(min_free_disk_bytes, min_free_disk_gib)
    if min_ram is not None and emergency_ram is not None and emergency_ram > min_ram:
        raise ValueError("emergency RAM threshold must not exceed minimum RAM threshold")

    sampler = resource_sampler or collect_resource_sample
    started = time.monotonic()
    next_sample = started
    sample_count = 0
    low_ram_count = 0
    last_sample: Optional[dict[str, Any]] = None
    reason: Optional[str] = None
    detail: Optional[str] = None
    signal_name: Optional[str] = None

    while True:
        returncode = process.poll()
        if returncode is not None:
            reason = "completed"
            break

        now = time.monotonic()
        if signal_controller is not None and signal_controller.stop_requested:
            reason = "signal"
            signal_name = signal_controller.signal_name
            detail = signal_name or "stop requested"
            break
        if deadline is not None and deadline.is_hard_expired(now):
            reason = "deadline"
            detail = "hard wall-time deadline expired"
            break
        if timeout_seconds is not None and now - started >= timeout_seconds:
            reason = "timeout"
            detail = f"worker timeout after {timeout_seconds:.3f} seconds"
            break

        if now >= next_sample:
            try:
                last_sample = sampler(process, disk_path)
            except Exception as exc:
                last_sample = {
                    "timestamp_utc": _utc_now(),
                    "monotonic": now,
                    "pid": process.pid,
                    "collector_error": str(exc),
                }
            sample_count += 1
            next_sample = now + sample_interval_seconds

            available = last_sample.get("available_ram_bytes")
            free_disk = last_sample.get("disk_free_bytes")
            worker_rss = last_sample.get("worker_total_rss_bytes")
            warnings: list[str] = []

            if min_ram is not None and available is not None and available < min_ram:
                low_ram_count += 1
                warnings.append(
                    f"available RAM {available} below minimum {min_ram} "
                    f"({low_ram_count}/{low_ram_samples_before_stop})"
                )
            else:
                low_ram_count = 0

            violation: Optional[str] = None
            if emergency_ram is not None and available is not None and available < emergency_ram:
                violation = f"available RAM {available} below emergency threshold {emergency_ram}"
            elif max_rss is not None and worker_rss is not None and worker_rss > max_rss:
                violation = f"worker RSS {worker_rss} above maximum {max_rss}"
            elif min_disk is not None and free_disk is not None and free_disk < min_disk:
                violation = f"free disk {free_disk} below minimum {min_disk}"
            elif min_ram is not None and low_ram_count >= low_ram_samples_before_stop:
                violation = f"available RAM remained below minimum {min_ram}"

            if warnings:
                last_sample["warnings"] = warnings
            if violation:
                last_sample["violation"] = violation

            if callback is not None:
                try:
                    callback(last_sample)
                except Exception as exc:
                    last_sample["callback_error"] = str(exc)

            if violation:
                reason = "resource"
                detail = violation
                break

        sleep_for = max(0.01, next_sample - time.monotonic())
        if deadline is not None:
            remaining = deadline.remaining()
            if remaining is not None:
                sleep_for = min(sleep_for, max(0.01, remaining))
        if timeout_seconds is not None:
            timeout_remaining = timeout_seconds - (time.monotonic() - started)
            sleep_for = min(sleep_for, max(0.01, timeout_remaining))
        if signal_controller is not None:
            signal_controller.wait(sleep_for)
        else:
            time.sleep(sleep_for)

    termination: Optional[dict[str, Any]] = None
    if reason != "completed" and process.poll() is None:
        termination = terminate_process_group(
            process,
            term_grace_seconds=term_grace_seconds,
            kill_grace_seconds=kill_grace_seconds,
        ).as_dict()
    ended = time.monotonic()
    return SupervisionResult(
        reason=reason or "completed",
        returncode=process.poll(),
        started_monotonic=started,
        ended_monotonic=ended,
        sample_count=sample_count,
        last_sample=last_sample,
        detail=detail,
        signal_name=signal_name,
        termination=termination,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: Optional[int] = None,
) -> Path:
    """Atomically replace a text file and fsync file + parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(text, str):
        raise TypeError("text must be str")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        elif destination.exists():
            os.chmod(temporary, destination.stat().st_mode & 0o777)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_write_json(
    path: PathLike,
    value: Any,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> Path:
    """Serialize JSON and atomically replace the destination file."""
    payload = json.dumps(
        value,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )
    if indent is not None:
        payload += "\n"
    return atomic_write_text(path, payload)


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _named_artifacts(
    artifacts: Union[
        PathLike,
        Mapping[str, PathLike],
        Iterable[Union[PathLike, tuple[str, PathLike]]],
    ]
) -> list[tuple[str, Path]]:
    if isinstance(artifacts, Mapping):
        return [(str(name), Path(path)) for name, path in artifacts.items()]
    if isinstance(artifacts, (str, os.PathLike)):
        path = Path(artifacts)
        return [(path.name, path)]
    result: list[tuple[str, Path]] = []
    for item in artifacts:
        if isinstance(item, tuple) and len(item) == 2:
            name, path = item
            result.append((str(name), Path(path)))
        else:
            path = Path(item)  # type: ignore[arg-type]
            result.append((path.name, path))
    return result


def validate_fits_artifacts(
    artifacts: Union[
        PathLike,
        Mapping[str, PathLike],
        Iterable[Union[PathLike, tuple[str, PathLike]]],
    ],
    *,
    require_astropy: bool = False,
    minimum_size_bytes: int = 1,
) -> dict[str, Any]:
    """Validate FITS existence, non-empty size and readability.

    If Astropy is unavailable, metadata checks are accepted unless
    ``require_astropy=True``.  Importing Astropy is deliberately deferred.
    """
    if minimum_size_bytes < 1:
        raise ValueError("minimum_size_bytes must be at least 1")
    named = _named_artifacts(artifacts)
    try:
        from astropy.io import fits  # type: ignore

        astropy_available = True
    except ImportError:
        fits = None
        astropy_available = False

    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, path in named:
        entry: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": path.is_file(),
            "readable": False,
            "fits_checked": False,
            "ok": False,
        }
        if not entry["exists"]:
            entry["error"] = "file does not exist"
        else:
            try:
                size = path.stat().st_size
                entry["size_bytes"] = size
                if size < minimum_size_bytes:
                    entry["error"] = (
                        f"file is too small ({size} < {minimum_size_bytes} bytes)"
                    )
                elif astropy_available:
                    entry["fits_checked"] = True
                    try:
                        assert fits is not None
                        with fits.open(path, mode="readonly", memmap=True) as hdul:
                            if len(hdul) == 0:
                                raise ValueError("FITS contains no HDUs")
                            hdul.verify("exception")
                            for hdu in hdul:
                                _ = hdu.header
                                _ = getattr(hdu, "shape", None)
                        entry["readable"] = True
                        entry["ok"] = True
                    except Exception as exc:
                        entry["error"] = f"FITS is not readable: {exc}"
                elif require_astropy:
                    entry["error"] = "Astropy is required but not installed"
                else:
                    entry["readable"] = os.access(path, os.R_OK)
                    entry["ok"] = bool(entry["readable"])
                    entry["validation"] = "metadata_only"
                    if not entry["readable"]:
                        entry["error"] = "file is not readable"
            except OSError as exc:
                entry["error"] = str(exc)
        if not entry["ok"]:
            errors.append(f"{name}: {entry.get('error', 'invalid FITS')}")
        entries.append(entry)
    return {
        "ok": not errors,
        "count": len(entries),
        "valid_count": sum(bool(entry["ok"]) for entry in entries),
        "astropy_available": astropy_available,
        "artifacts": entries,
        "errors": errors,
    }


def build_artifact_manifest(
    artifacts: Union[
        PathLike,
        Mapping[str, PathLike],
        Iterable[Union[PathLike, tuple[str, PathLike]]],
    ],
    *,
    base_dir: Optional[PathLike] = None,
    include_sha256: bool = True,
    validate_fits: bool = False,
    require_astropy: bool = False,
) -> dict[str, Any]:
    """Build a durable manifest containing size, timestamps and SHA-256."""
    named = _named_artifacts(artifacts)
    base = Path(base_dir).resolve() if base_dir is not None else None
    validation_by_name: dict[str, dict[str, Any]] = {}
    if validate_fits:
        validation = validate_fits_artifacts(
            named, require_astropy=require_astropy
        )
        validation_by_name = {
            str(item["name"]): item for item in validation["artifacts"]
        }

    entries: list[dict[str, Any]] = []
    all_ok = True
    for name, path in named:
        absolute = path.resolve()
        entry: dict[str, Any] = {
            "name": name,
            "path": str(absolute),
            "exists": path.is_file(),
        }
        if base is not None:
            try:
                entry["relative_path"] = str(absolute.relative_to(base))
            except ValueError:
                entry["relative_path"] = None
        if path.is_file():
            stat = path.stat()
            entry.update(
                size_bytes=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
                mtime_utc=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="milliseconds"),
            )
            if include_sha256:
                try:
                    entry["sha256"] = sha256_file(path)
                except OSError as exc:
                    entry["sha256"] = None
                    entry["error"] = str(exc)
        else:
            entry["error"] = "file does not exist"

        if validate_fits:
            checked = validation_by_name.get(name, {})
            entry["fits_valid"] = bool(checked.get("ok", False))
            if checked.get("error"):
                entry["fits_error"] = checked["error"]
        entry_ok = bool(entry["exists"] and entry.get("size_bytes", 0) > 0)
        if include_sha256:
            entry_ok = entry_ok and bool(entry.get("sha256"))
        if validate_fits:
            entry_ok = entry_ok and bool(entry.get("fits_valid"))
        entry["ok"] = entry_ok
        all_ok = all_ok and entry_ok
        entries.append(entry)

    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "base_dir": str(base) if base is not None else None,
        "count": len(entries),
        "ok": all_ok,
        "artifacts": entries,
    }
