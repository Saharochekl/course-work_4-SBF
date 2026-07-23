#!/usr/bin/env python3
"""Durable SQLite state for long-running SBF processing campaigns.

The module deliberately depends only on the Python standard library.  A
``CampaignState`` instance stores its database below the supplied run root and
opens a short-lived SQLite connection for every operation.  This makes the API
safe to use from the parent runner, a resource-monitor thread, and separately
spawned worker processes.

All mutating operations use ``BEGIN IMMEDIATE`` transactions.  SQLite runs in
WAL mode with foreign-key checks and a busy timeout enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_DB_NAME = "campaign_state.sqlite"
DEFAULT_SNAPSHOT_NAME = "queue_snapshot.json"

JOB_STATES = frozenset(
    {
        "PENDING",
        "DOWNLOADING",
        "READY",
        "RUNNING",
        "VERIFYING",
        "SUCCEEDED",
        "RETRY_WAIT",
        "FAILED",
        "INTERRUPTED",
        "SKIPPED",
        "CANCELLED",
    }
)

TERMINAL_JOB_STATES = frozenset({"SUCCEEDED", "SKIPPED", "CANCELLED"})

JOB_TRANSITIONS = {
    "PENDING": {
        "DOWNLOADING",
        "READY",
        "RUNNING",
        "FAILED",
        "SKIPPED",
        "CANCELLED",
    },
    "DOWNLOADING": {
        "READY",
        "RETRY_WAIT",
        "FAILED",
        "INTERRUPTED",
        "CANCELLED",
    },
    "READY": {"RUNNING", "FAILED", "SKIPPED", "CANCELLED"},
    "RUNNING": {
        "VERIFYING",
        "SUCCEEDED",
        "RETRY_WAIT",
        "FAILED",
        "INTERRUPTED",
        "CANCELLED",
    },
    "VERIFYING": {
        "SUCCEEDED",
        "RETRY_WAIT",
        "FAILED",
        "INTERRUPTED",
        "CANCELLED",
    },
    "RETRY_WAIT": {
        "PENDING",
        "DOWNLOADING",
        "READY",
        "RUNNING",
        "FAILED",
        "CANCELLED",
    },
    "FAILED": {"RETRY_WAIT", "PENDING", "READY", "RUNNING", "CANCELLED"},
    "INTERRUPTED": {"PENDING", "READY", "RUNNING", "FAILED", "CANCELLED"},
    "SUCCEEDED": set(),
    "SKIPPED": set(),
    "CANCELLED": set(),
}

RUN_STATES = frozenset(
    {
        "RUNNING",
        "SOFT_STOPPED",
        "INTERRUPTED",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }
)
RESUMABLE_RUN_STATES = frozenset(
    {"RUNNING", "SOFT_STOPPED", "INTERRUPTED", "FAILED"}
)
TERMINAL_RUN_STATES = frozenset({"COMPLETED", "CANCELLED"})


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    campaign_key TEXT NOT NULL,
    state TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    deadline_at REAL,
    soft_stop_at REAL,
    resume_count INTEGER NOT NULL DEFAULT 0,
    owner_pid INTEGER,
    owner_host TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign
    ON runs(campaign_key, created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    program TEXT,
    obsid TEXT,
    target TEXT NOT NULL,
    product_uris_json TEXT NOT NULL,
    filters_json TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    queue_position INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    last_error TEXT,
    PRIMARY KEY (run_id, job_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_queue
    ON jobs(run_id, state, priority DESC, queue_position, created_at);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    state TEXT NOT NULL,
    command_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    pid INTEGER,
    log_path TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    exit_code INTEGER,
    error TEXT,
    UNIQUE (run_id, job_id, attempt_no),
    FOREIGN KEY (run_id, job_id)
        REFERENCES jobs(run_id, job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_job
    ON attempts(run_id, job_id, attempt_no DESC);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    attempt_id INTEGER,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    sha256 TEXT,
    verified INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (run_id, job_id, path),
    FOREIGN KEY (run_id, job_id)
        REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_job
    ON artifacts(run_id, job_id, kind);

CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    job_id TEXT,
    attempt_id INTEGER,
    sampled_at REAL NOT NULL,
    ram_total_bytes INTEGER,
    ram_available_bytes INTEGER,
    process_rss_bytes INTEGER,
    children_rss_bytes INTEGER,
    swap_used_bytes INTEGER,
    disk_total_bytes INTEGER,
    disk_free_bytes INTEGER,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, job_id)
        REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_resources_time
    ON resource_samples(run_id, sampled_at);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    job_id TEXT,
    attempt_id INTEGER,
    created_at REAL NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, job_id)
        REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_events_time
    ON events(run_id, created_at, event_id);
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON suitable for hashing and storage."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def canonical_sha256(value: Any) -> str:
    """Hash an arbitrary JSON-compatible value deterministically."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_scalar(value: Any, *, upper: bool = False) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text.upper() if upper else text


def _normalise_named_values(value: Any, *, upper_values: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).strip().lower(): _normalise_scalar(item, upper=upper_values)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, str):
        return [_normalise_scalar(value, upper=upper_values)]
    if isinstance(value, Sequence):
        return [_normalise_scalar(item, upper=upper_values) for item in value]
    raise TypeError("expected a mapping, sequence, or string")


def stable_job_id(
    *,
    program: Any,
    obsid: Any,
    target: Any,
    product_uris: Mapping[str, Any] | Sequence[Any] | str,
    filters: Mapping[str, Any] | Sequence[Any] | str,
    template_sha256: str,
    config_sha256: str,
) -> str:
    """Build a stable job id from the target and archive input identities.

    Local paths, file size, inode, and modification time are intentionally not
    accepted.  Mappings are preferred for ``product_uris`` and ``filters`` so
    roles such as ``signal`` and ``color`` remain explicit.  Template and run
    configuration digests are accepted for backwards API compatibility, but
    deliberately remain provenance: editing the notebook must not turn an
    already verified target into a different job.
    """
    target_value = _normalise_scalar(target, upper=True)
    if not target_value:
        raise ValueError("target must not be empty")
    payload = {
        "identity_version": 2,
        "program": _normalise_scalar(program),
        "obsid": _normalise_scalar(obsid),
        "target": target_value,
        "product_uris": _normalise_named_values(product_uris),
        "filters": _normalise_named_values(filters, upper_values=True),
    }
    return "job-" + canonical_sha256(payload)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Atomically replace ``path`` with durable, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some platforms/filesystems.
            pass
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


class CampaignState:
    """Transactional campaign state stored below a processing run root."""

    def __init__(
        self,
        run_root: str | os.PathLike[str],
        *,
        db_name: str = DEFAULT_DB_NAME,
        busy_timeout_seconds: float = 30.0,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_root / db_name
        self.busy_timeout_ms = max(1, int(busy_timeout_seconds * 1000))
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return connection

    def _initialise(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _reader(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in tuple(result):
            if key.endswith("_json") and result[key] is not None:
                decoded_key = key[:-5]
                result[decoded_key] = json.loads(result.pop(key))
        for key in ("verified",):
            if key in result and result[key] is not None:
                result[key] = bool(result[key])
        return result

    @staticmethod
    def _event_tx(
        connection: sqlite3.Connection,
        run_id: str,
        *,
        event_type: str,
        level: str = "INFO",
        message: str | None = None,
        job_id: str | None = None,
        attempt_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
        created_at: float | None = None,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO events (
                run_id, job_id, attempt_id, created_at, level,
                event_type, message, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                job_id,
                attempt_id,
                time.time() if created_at is None else float(created_at),
                str(level).upper(),
                str(event_type).upper(),
                message,
                canonical_json(dict(payload or {})),
            ),
        )
        return int(cursor.lastrowid)

    def create_or_resume_run(
        self,
        *,
        template_sha256: str,
        config: Mapping[str, Any] | None = None,
        config_sha256: str | None = None,
        run_id: str | None = None,
        wall_time_seconds: float | None = None,
        soft_stop_seconds: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Create a campaign run or resume a compatible unfinished one.

        With no explicit ``run_id``, the newest compatible unfinished run in
        this run root is resumed.  A stored deadline is never silently reset
        when resuming.  Pass ``resume=False`` to force creation of a new run.
        """
        now = time.time()
        config_value = dict(config or {})
        actual_config_sha = config_sha256 or canonical_sha256(config_value)
        campaign_key = canonical_sha256(
            {
                "template_sha256": str(template_sha256).lower(),
                "config_sha256": str(actual_config_sha).lower(),
            }
        )
        if wall_time_seconds is not None and wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive")
        if soft_stop_seconds < 0:
            raise ValueError("soft_stop_seconds must not be negative")

        with self._transaction() as connection:
            existing = None
            if run_id is not None:
                existing = connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
            elif resume:
                placeholders = ",".join("?" for _ in RESUMABLE_RUN_STATES)
                existing = connection.execute(
                    f"""
                    SELECT * FROM runs
                    WHERE campaign_key = ? AND state IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (campaign_key, *sorted(RESUMABLE_RUN_STATES)),
                ).fetchone()

            if existing is not None:
                if existing["campaign_key"] != campaign_key:
                    raise ValueError(
                        f"run {existing['run_id']} has a different template/config identity"
                    )
                if existing["state"] in TERMINAL_RUN_STATES:
                    raise ValueError(
                        f"run {existing['run_id']} is terminal: {existing['state']}"
                    )
                stored_metadata = json.loads(existing["metadata_json"])
                stored_metadata.update(metadata or {})
                connection.execute(
                    """
                    UPDATE runs
                    SET state = 'RUNNING', updated_at = ?, ended_at = NULL,
                        resume_count = resume_count + 1,
                        owner_pid = ?, owner_host = ?, metadata_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        now,
                        os.getpid(),
                        socket.gethostname(),
                        canonical_json(stored_metadata),
                        existing["run_id"],
                    ),
                )
                self._event_tx(
                    connection,
                    existing["run_id"],
                    event_type="RUN_RESUMED",
                    payload={"previous_state": existing["state"]},
                    created_at=now,
                )
                selected_run_id = existing["run_id"]
            else:
                selected_run_id = run_id or (
                    "run-"
                    + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
                    + "-"
                    + campaign_key[:12]
                    + "-"
                    + uuid.uuid4().hex[:8]
                )
                deadline_at = (
                    None if wall_time_seconds is None else now + wall_time_seconds
                )
                soft_stop_at = (
                    None
                    if deadline_at is None
                    else max(now, deadline_at - soft_stop_seconds)
                )
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, campaign_key, state, template_sha256,
                        config_sha256, config_json, metadata_json, created_at,
                        updated_at, started_at, deadline_at, soft_stop_at,
                        owner_pid, owner_host
                    ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        selected_run_id,
                        campaign_key,
                        str(template_sha256).lower(),
                        str(actual_config_sha).lower(),
                        canonical_json(config_value),
                        canonical_json(dict(metadata or {})),
                        now,
                        now,
                        now,
                        deadline_at,
                        soft_stop_at,
                        os.getpid(),
                        socket.gethostname(),
                    ),
                )
                self._event_tx(
                    connection,
                    selected_run_id,
                    event_type="RUN_CREATED",
                    payload={
                        "deadline_at": deadline_at,
                        "soft_stop_at": soft_stop_at,
                    },
                    created_at=now,
                )
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (selected_run_id,)
            ).fetchone()
            return self._decode_row(row)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run, or ``None`` when it does not exist."""
        with self._reader() as connection:
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
            )

    def set_run_state(
        self,
        run_id: str,
        state: str,
        *,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set a run state and append the matching event transactionally."""
        new_state = str(state).upper()
        if new_state not in RUN_STATES:
            raise ValueError(f"unknown run state: {new_state}")
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            metadata_value = json.loads(row["metadata_json"])
            metadata_value.update(metadata or {})
            ended_at = now if new_state in TERMINAL_RUN_STATES | {"FAILED"} else None
            connection.execute(
                """
                UPDATE runs
                SET state = ?, updated_at = ?, ended_at = ?, last_error = ?,
                    metadata_json = ?
                WHERE run_id = ?
                """,
                (
                    new_state,
                    now,
                    ended_at,
                    error,
                    canonical_json(metadata_value),
                    run_id,
                ),
            )
            self._event_tx(
                connection,
                run_id,
                event_type="RUN_STATE_CHANGED",
                level="ERROR" if new_state == "FAILED" else "INFO",
                message=error,
                payload={"from": row["state"], "to": new_state},
                created_at=now,
            )
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
            )  # type: ignore[return-value]

    def deadline_status(
        self, run_id: str, *, now: float | None = None
    ) -> dict[str, Any]:
        """Return hard/soft deadline flags without mutating campaign state."""
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        current = time.time() if now is None else float(now)
        deadline = run["deadline_at"]
        soft_stop = run["soft_stop_at"]
        return {
            "now": current,
            "deadline_at": deadline,
            "soft_stop_at": soft_stop,
            "soft_stop_reached": soft_stop is not None and current >= soft_stop,
            "deadline_reached": deadline is not None and current >= deadline,
            "seconds_remaining": None if deadline is None else deadline - current,
        }

    @staticmethod
    def _upsert_job_tx(
        connection: sqlite3.Connection,
        run: sqlite3.Row,
        *,
        target: Any,
        product_uris: Mapping[str, Any] | Sequence[Any] | str,
        filters: Mapping[str, Any] | Sequence[Any] | str,
        program: Any = None,
        obsid: Any = None,
        payload: Mapping[str, Any] | None = None,
        priority: int = 0,
        queue_position: int | None = None,
        initial_state: str = "PENDING",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        initial = str(initial_state).upper()
        if initial not in JOB_STATES:
            raise ValueError(f"unknown job state: {initial}")
        computed_job_id = stable_job_id(
            program=program,
            obsid=obsid,
            target=target,
            product_uris=product_uris,
            filters=filters,
            template_sha256=run["template_sha256"],
            config_sha256=run["config_sha256"],
        )
        if job_id is not None and job_id != computed_job_id:
            raise ValueError("supplied job_id does not match stable job identity")
        now = time.time()
        normalised_uris = _normalise_named_values(product_uris)
        normalised_filters = _normalise_named_values(filters, upper_values=True)
        connection.execute(
            """
            INSERT INTO jobs (
                run_id, job_id, program, obsid, target, product_uris_json,
                filters_json, payload_json, state, priority, queue_position,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, job_id) DO UPDATE SET
                program = excluded.program,
                obsid = excluded.obsid,
                target = excluded.target,
                product_uris_json = excluded.product_uris_json,
                filters_json = excluded.filters_json,
                payload_json = excluded.payload_json,
                priority = excluded.priority,
                queue_position = excluded.queue_position,
                updated_at = excluded.updated_at
            """,
            (
                run["run_id"],
                computed_job_id,
                _normalise_scalar(program),
                _normalise_scalar(obsid),
                _normalise_scalar(target) or "",
                canonical_json(normalised_uris),
                canonical_json(normalised_filters),
                canonical_json(dict(payload or {})),
                initial,
                int(priority),
                queue_position,
                now,
                now,
            ),
        )
        return CampaignState._decode_row(
            connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                (run["run_id"], computed_job_id),
            ).fetchone()
        )  # type: ignore[return-value]

    def upsert_job(
        self,
        run_id: str,
        *,
        target: Any,
        product_uris: Mapping[str, Any] | Sequence[Any] | str,
        filters: Mapping[str, Any] | Sequence[Any] | str,
        program: Any = None,
        obsid: Any = None,
        payload: Mapping[str, Any] | None = None,
        priority: int = 0,
        queue_position: int | None = None,
        initial_state: str = "PENDING",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert a stable job or refresh its metadata without resetting state."""
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            job = self._upsert_job_tx(
                connection,
                run,
                target=target,
                product_uris=product_uris,
                filters=filters,
                program=program,
                obsid=obsid,
                payload=payload,
                priority=priority,
                queue_position=queue_position,
                initial_state=initial_state,
                job_id=job_id,
            )
            self._event_tx(
                connection,
                run_id,
                job_id=job["job_id"],
                event_type="JOB_UPSERTED",
                payload={"state": job["state"]},
            )
            return job

    def upsert_jobs(
        self, run_id: str, jobs: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Upsert an entire queue in one transaction.

        Each mapping accepts the keyword arguments of :meth:`upsert_job`.
        Missing ``queue_position`` values receive their iterable index.
        """
        result = []
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            for index, item in enumerate(jobs):
                values = dict(item)
                values.setdefault("queue_position", index)
                job = self._upsert_job_tx(connection, run, **values)
                result.append(job)
            self._event_tx(
                connection,
                run_id,
                event_type="QUEUE_UPSERTED",
                payload={"job_count": len(result)},
            )
        return result

    def get_job(self, run_id: str, job_id: str) -> dict[str, Any] | None:
        """Return a job row with decoded JSON fields."""
        with self._reader() as connection:
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                    (run_id, job_id),
                ).fetchone()
            )

    def list_jobs(
        self, run_id: str, *, states: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """List jobs in deterministic queue order, optionally by state."""
        parameters: list[Any] = [run_id]
        condition = ""
        if states is not None:
            values = sorted({str(state).upper() for state in states})
            if not values:
                return []
            condition = " AND state IN (" + ",".join("?" for _ in values) + ")"
            parameters.extend(values)
        with self._reader() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE run_id = ?
                """
                + condition
                + " ORDER BY priority DESC, queue_position, created_at, job_id",
                parameters,
            ).fetchall()
            return [self._decode_row(row) for row in rows]  # type: ignore[misc]

    def transition_job(
        self,
        run_id: str,
        job_id: str,
        new_state: str,
        *,
        expected_state: str | Iterable[str] | None = None,
        error: str | None = None,
        details: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Atomically transition a job and record an event.

        ``expected_state`` provides compare-and-swap semantics.  Illegal state
        edges raise ``ValueError`` unless ``force=True``.  Repeating the current
        state is idempotent.
        """
        desired = str(new_state).upper()
        if desired not in JOB_STATES:
            raise ValueError(f"unknown job state: {desired}")
        if expected_state is None:
            expected = None
        elif isinstance(expected_state, str):
            expected = {expected_state.upper()}
        else:
            expected = {str(value).upper() for value in expected_state}
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                (run_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job: {job_id}")
            current = row["state"]
            if expected is not None and current not in expected:
                raise RuntimeError(
                    f"job {job_id} is {current}; expected {sorted(expected)}"
                )
            if current != desired and not force and desired not in JOB_TRANSITIONS[current]:
                raise ValueError(f"illegal job transition: {current} -> {desired}")
            started_at = row["started_at"]
            if desired == "RUNNING" and started_at is None:
                started_at = now
            ended_at = now if desired in TERMINAL_JOB_STATES else None
            next_error = None if desired == "SUCCEEDED" else error
            connection.execute(
                """
                UPDATE jobs
                SET state = ?, updated_at = ?, started_at = ?, ended_at = ?,
                    last_error = ?
                WHERE run_id = ? AND job_id = ?
                """,
                (
                    desired,
                    now,
                    started_at,
                    ended_at,
                    next_error,
                    run_id,
                    job_id,
                ),
            )
            self._event_tx(
                connection,
                run_id,
                job_id=job_id,
                event_type="JOB_STATE_CHANGED",
                level="ERROR" if desired == "FAILED" else "INFO",
                message=error,
                payload={"from": current, "to": desired, **dict(details or {})},
                created_at=now,
            )
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                    (run_id, job_id),
                ).fetchone()
            )  # type: ignore[return-value]

    def claim_next_job(
        self,
        run_id: str,
        *,
        from_states: Iterable[str] = ("READY",),
        to_state: str = "RUNNING",
    ) -> dict[str, Any] | None:
        """Atomically claim the next queue item, or return ``None``."""
        source_states = sorted({str(value).upper() for value in from_states})
        if not source_states:
            return None
        desired = str(to_state).upper()
        if desired not in JOB_STATES:
            raise ValueError(f"unknown job state: {desired}")
        now = time.time()
        with self._transaction() as connection:
            placeholders = ",".join("?" for _ in source_states)
            row = connection.execute(
                f"""
                SELECT * FROM jobs
                WHERE run_id = ? AND state IN ({placeholders})
                ORDER BY priority DESC, queue_position, created_at, job_id
                LIMIT 1
                """,
                (run_id, *source_states),
            ).fetchone()
            if row is None:
                return None
            current = row["state"]
            if desired != current and desired not in JOB_TRANSITIONS[current]:
                raise ValueError(f"illegal job transition: {current} -> {desired}")
            started_at = now if desired == "RUNNING" and row["started_at"] is None else row["started_at"]
            connection.execute(
                """
                UPDATE jobs SET state = ?, updated_at = ?, started_at = ?
                WHERE run_id = ? AND job_id = ?
                """,
                (desired, now, started_at, run_id, row["job_id"]),
            )
            self._event_tx(
                connection,
                run_id,
                job_id=row["job_id"],
                event_type="JOB_CLAIMED",
                payload={"from": current, "to": desired},
                created_at=now,
            )
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                    (run_id, row["job_id"]),
                ).fetchone()
            )

    def record_attempt_start(
        self,
        run_id: str,
        job_id: str,
        *,
        command: Sequence[Any] | None = None,
        pid: int | None = None,
        log_path: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a numbered attempt and increment the job attempt counter.

        This method records execution only; callers should use
        :meth:`transition_job` or :meth:`claim_next_job` for job state changes.
        """
        now = time.time()
        with self._transaction() as connection:
            job = connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                (run_id, job_id),
            ).fetchone()
            if job is None:
                raise KeyError(f"unknown job: {job_id}")
            attempt_no = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_no), 0) + 1
                    FROM attempts WHERE run_id = ? AND job_id = ?
                    """,
                    (run_id, job_id),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO attempts (
                    run_id, job_id, attempt_no, state, command_json,
                    metadata_json, pid, log_path, started_at
                ) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    job_id,
                    attempt_no,
                    None if command is None else canonical_json(list(command)),
                    canonical_json(dict(metadata or {})),
                    pid,
                    None if log_path is None else str(Path(log_path)),
                    now,
                ),
            )
            attempt_id = int(cursor.lastrowid)
            connection.execute(
                """
                UPDATE jobs SET attempt_count = ?, updated_at = ?
                WHERE run_id = ? AND job_id = ?
                """,
                (attempt_no, now, run_id, job_id),
            )
            self._event_tx(
                connection,
                run_id,
                job_id=job_id,
                attempt_id=attempt_id,
                event_type="ATTEMPT_STARTED",
                payload={"attempt_no": attempt_no, "pid": pid},
                created_at=now,
            )
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
            )  # type: ignore[return-value]

    def record_attempt_end(
        self,
        attempt_id: int,
        *,
        state: str,
        exit_code: int | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish an attempt while preserving its original start metadata."""
        final_state = str(state).upper()
        if final_state == "RUNNING":
            raise ValueError("an ended attempt cannot remain RUNNING")
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            metadata_value = json.loads(row["metadata_json"])
            metadata_value.update(metadata or {})
            connection.execute(
                """
                UPDATE attempts
                SET state = ?, ended_at = ?, exit_code = ?, error = ?,
                    metadata_json = ?
                WHERE attempt_id = ?
                """,
                (
                    final_state,
                    now,
                    exit_code,
                    error,
                    canonical_json(metadata_value),
                    attempt_id,
                ),
            )
            self._event_tx(
                connection,
                row["run_id"],
                job_id=row["job_id"],
                attempt_id=attempt_id,
                event_type="ATTEMPT_ENDED",
                level="ERROR" if final_state == "FAILED" else "INFO",
                message=error,
                payload={"state": final_state, "exit_code": exit_code},
                created_at=now,
            )
            return self._decode_row(
                connection.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
            )  # type: ignore[return-value]

    def record_resource_sample(
        self,
        run_id: str,
        *,
        job_id: str | None = None,
        attempt_id: int | None = None,
        sampled_at: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        ram_total_bytes: int | None = None,
        ram_available_bytes: int | None = None,
        process_rss_bytes: int | None = None,
        children_rss_bytes: int | None = None,
        swap_used_bytes: int | None = None,
        disk_total_bytes: int | None = None,
        disk_free_bytes: int | None = None,
    ) -> int:
        """Record one resource sample and return its database id."""
        timestamp = time.time() if sampled_at is None else float(sampled_at)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO resource_samples (
                    run_id, job_id, attempt_id, sampled_at, ram_total_bytes,
                    ram_available_bytes, process_rss_bytes, children_rss_bytes,
                    swap_used_bytes, disk_total_bytes, disk_free_bytes,
                    metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    job_id,
                    attempt_id,
                    timestamp,
                    ram_total_bytes,
                    ram_available_bytes,
                    process_rss_bytes,
                    children_rss_bytes,
                    swap_used_bytes,
                    disk_total_bytes,
                    disk_free_bytes,
                    canonical_json(dict(metrics or {})),
                ),
            )
            return int(cursor.lastrowid)

    def record_artifact(
        self,
        run_id: str,
        job_id: str,
        *,
        kind: str,
        path: str | os.PathLike[str],
        attempt_id: int | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        verified: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert or update one produced artifact by its run/job/path key."""
        now = time.time()
        path_value = str(Path(path).resolve())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    run_id, job_id, attempt_id, kind, path, size_bytes,
                    sha256, verified, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, job_id, path) DO UPDATE SET
                    attempt_id = excluded.attempt_id,
                    kind = excluded.kind,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    verified = excluded.verified,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    job_id,
                    attempt_id,
                    str(kind),
                    path_value,
                    size_bytes,
                    sha256,
                    None if verified is None else int(verified),
                    canonical_json(dict(metadata or {})),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE run_id = ? AND job_id = ? AND path = ?
                """,
                (run_id, job_id, path_value),
            ).fetchone()
            artifact = self._decode_row(row)
            self._event_tx(
                connection,
                run_id,
                job_id=job_id,
                attempt_id=attempt_id,
                event_type="ARTIFACT_RECORDED",
                payload={
                    "artifact_id": artifact["artifact_id"],
                    "kind": kind,
                    "path": path_value,
                    "verified": verified,
                },
                created_at=now,
            )
            return artifact  # type: ignore[return-value]

    def record_artifacts(
        self,
        run_id: str,
        job_id: str,
        artifacts: Iterable[Mapping[str, Any]],
        *,
        attempt_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Record multiple artifacts through the single-artifact API."""
        result = []
        for artifact in artifacts:
            values = dict(artifact)
            values.setdefault("attempt_id", attempt_id)
            result.append(self.record_artifact(run_id, job_id, **values))
        return result

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        level: str = "INFO",
        message: str | None = None,
        job_id: str | None = None,
        attempt_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        """Append a structured event and return its monotonically increasing id."""
        with self._transaction() as connection:
            return self._event_tx(
                connection,
                run_id,
                event_type=event_type,
                level=level,
                message=message,
                job_id=job_id,
                attempt_id=attempt_id,
                payload=payload,
            )

    def recover_incomplete_work(self, run_id: str) -> list[str]:
        """Mark work left active by a dead parent as interrupted.

        Recovery is one SQLite transaction: active attempts and their jobs can
        therefore never disagree merely because the new parent was killed in
        the middle of recovery.  The scheduler may subsequently move these
        jobs back to ``READY`` or ``PENDING`` after checking local inputs.
        """
        now = time.time()
        active_states = ("DOWNLOADING", "RUNNING", "VERIFYING")
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT job_id, state FROM jobs
                WHERE run_id = ? AND state IN (?, ?, ?)
                ORDER BY queue_position, job_id
                """,
                (run_id, *active_states),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            connection.execute(
                """
                UPDATE attempts
                SET state = 'INTERRUPTED', ended_at = ?,
                    error = COALESCE(error, 'parent process restarted')
                WHERE run_id = ? AND state = 'RUNNING'
                """,
                (now, run_id),
            )
            if job_ids:
                placeholders = ",".join("?" for _ in job_ids)
                connection.execute(
                    f"""
                    UPDATE jobs
                    SET state = 'INTERRUPTED', updated_at = ?,
                        last_error = 'parent process restarted'
                    WHERE run_id = ? AND job_id IN ({placeholders})
                    """,
                    (now, run_id, *job_ids),
                )
                for row in rows:
                    self._event_tx(
                        connection,
                        run_id,
                        job_id=row["job_id"],
                        event_type="JOB_RECOVERED",
                        level="WARNING",
                        message="active work was interrupted by parent restart",
                        payload={"from": row["state"], "to": "INTERRUPTED"},
                        created_at=now,
                    )
            return job_ids

    def queue_snapshot(self, run_id: str) -> dict[str, Any]:
        """Return a consistent in-memory snapshot of run, jobs, and attempts."""
        with self._reader() as connection:
            connection.execute("BEGIN")
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                connection.rollback()
                raise KeyError(f"unknown run: {run_id}")
            jobs = connection.execute(
                """
                SELECT * FROM jobs WHERE run_id = ?
                ORDER BY priority DESC, queue_position, created_at, job_id
                """,
                (run_id,),
            ).fetchall()
            active_attempts = connection.execute(
                """
                SELECT * FROM attempts
                WHERE run_id = ? AND state = 'RUNNING'
                ORDER BY attempt_id
                """,
                (run_id,),
            ).fetchall()
            counts = connection.execute(
                """
                SELECT state, COUNT(*) AS count
                FROM jobs WHERE run_id = ? GROUP BY state
                """,
                (run_id,),
            ).fetchall()
            connection.commit()
        return {
            "generated_at": time.time(),
            "run": self._decode_row(run),
            "counts": {row["state"]: int(row["count"]) for row in counts},
            "jobs": [self._decode_row(row) for row in jobs],
            "active_attempts": [self._decode_row(row) for row in active_attempts],
        }

    def snapshot_queue(
        self,
        run_id: str,
        path: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Atomically write and return a queue snapshot.

        The default destination is ``run_root/queue_snapshot.json``.
        """
        snapshot = self.queue_snapshot(run_id)
        atomic_write_json(
            self.run_root / DEFAULT_SNAPSHOT_NAME if path is None else path,
            snapshot,
        )
        return snapshot


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sbf-campaign-state-") as directory:
        root = Path(directory)
        state = CampaignState(root)
        template_digest = "a" * 64
        config = {"wall_hours": 48, "prefetch": 1}
        run = state.create_or_resume_run(
            template_sha256=template_digest,
            config=config,
            wall_time_seconds=48 * 3600,
            soft_stop_seconds=30 * 60,
        )
        identity = {
            "program": "3055",
            "obsid": "1",
            "target": "NGC 1380",
            "product_uris": {
                "signal": "mast:JWST/product/signal.fits",
                "color": "mast:JWST/product/color.fits",
            },
            "filters": {"signal": "F150W", "color": "F090W"},
        }
        job = state.upsert_job(run["run_id"], **identity)
        assert job["job_id"] == stable_job_id(
            **identity,
            template_sha256=template_digest,
            config_sha256=canonical_sha256(config),
        )
        state.transition_job(run["run_id"], job["job_id"], "READY")
        state.transition_job(run["run_id"], job["job_id"], "RUNNING")
        attempt = state.record_attempt_start(
            run["run_id"], job["job_id"], command=["python", "worker.py"]
        )
        state.record_resource_sample(
            run["run_id"],
            job_id=job["job_id"],
            attempt_id=attempt["attempt_id"],
            ram_available_bytes=8 * 1024**3,
            disk_free_bytes=100 * 1024**3,
        )
        artifact_path = root / "result.fits"
        artifact_path.write_bytes(b"test")
        state.transition_job(run["run_id"], job["job_id"], "VERIFYING")
        state.record_artifact(
            run["run_id"],
            job["job_id"],
            attempt_id=attempt["attempt_id"],
            kind="fits",
            path=artifact_path,
            size_bytes=artifact_path.stat().st_size,
            sha256=sha256_file(artifact_path),
            verified=True,
        )
        state.record_attempt_end(
            attempt["attempt_id"], state="SUCCEEDED", exit_code=0
        )
        state.transition_job(run["run_id"], job["job_id"], "SUCCEEDED")
        snapshot = state.snapshot_queue(run["run_id"])
        assert snapshot["counts"] == {"SUCCEEDED": 1}
        assert (root / DEFAULT_DB_NAME).is_file()
        assert (root / DEFAULT_SNAPSHOT_NAME).is_file()
        with sqlite3.connect(root / DEFAULT_DB_NAME) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            table_count = connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name IN (
                    'runs', 'jobs', 'attempts', 'artifacts',
                    'resource_samples', 'events'
                )
                """
            ).fetchone()[0]
            assert table_count == 6
    print("sbf_campaign_state self-test: OK")


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("usage: python sbf_campaign_state.py --self-test")
    _self_test()
