#!/usr/bin/env python3
"""RU: быстрые проверки возобновления и безопасности. EN: restart/safety tests."""
from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits

import run_sbf_2_batch as batch
import sbf_campaign_runtime as runtime
from sbf_campaign_runtime import (
    Deadline, SignalController, build_artifact_manifest,
    launch_process_group, supervise_process, terminate_process_group,
)
from sbf_campaign_state import CampaignState, stable_job_id

class StableJobIdentityTests(unittest.TestCase):
    def test_stable_job_id_is_order_and_local_filesystem_independent(self):
        identity = {
            "program": "3055",
            "obsid": "jw03055-o003",
            "target": "NGC 1404",
            "product_uris": {
                "signal": "mast:JWST/product/signal_i2d.fits",
                "color": "mast:JWST/product/color_i2d.fits",
            },
            "filters": {"signal": "F150W", "color": "F090W"},
            "template_sha256": "a" * 64,
            "config_sha256": "b" * 64,
        }
        first = stable_job_id(**identity)
        reordered = stable_job_id(
            **{
                **identity,
                "target": "  ngc   1404 ",
                "product_uris": {
                    "color": identity["product_uris"]["color"],
                    "signal": identity["product_uris"]["signal"],
                },
                "filters": {"color": "f090w", "signal": "f150w"},
            }
        )

        self.assertEqual(first, reordered)
        self.assertRegex(first, r"^job-[0-9a-f]{64}$")
        self.assertEqual(
            first,
            stable_job_id(
                **{
                    **identity,
                    "template_sha256": "c" * 64,
                    "config_sha256": "d" * 64,
                }
            ),
        )
        self.assertNotEqual(
            first,
            stable_job_id(
                **{
                    **identity,
                    "product_uris": {
                        **identity["product_uris"],
                        "signal": "mast:JWST/product/another_signal_i2d.fits",
                    },
                }
            ),
        )



class SQLiteRecoveryTests(unittest.TestCase):
    def test_restart_recovers_running_job_and_attempt_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            state = CampaignState(directory)
            run = state.create_or_resume_run(
                template_sha256="a" * 64,
                config={"prefetch_targets": 1},
                wall_time_seconds=3600,
                soft_stop_seconds=60,
            )
            job = state.upsert_jobs(run["run_id"], [dict(
                target="NGC 1404",
                program="3055",
                obsid="jw03055-o003",
                product_uris={"signal": "mast:signal", "color": "mast:color"},
                filters={"signal": "F150W", "color": "F090W"},
                initial_state="READY",
            )])[0]
            state.transition_job(run["run_id"], job["job_id"], "RUNNING")
            attempt = state.record_attempt_start(
                run["run_id"], job["job_id"], command=["worker"], pid=12345
            )

            restarted = CampaignState(directory)
            recovered = restarted.recover_incomplete_work(run["run_id"])

            self.assertEqual(recovered, [job["job_id"]])
            self.assertEqual(
                restarted.get_job(run["run_id"], job["job_id"])["state"],
                "INTERRUPTED",
            )
            with closing(sqlite3.connect(restarted.db_path)) as connection:
                attempt_state, error = connection.execute(
                    "SELECT state, error FROM attempts WHERE attempt_id = ?",
                    (attempt["attempt_id"],),
                ).fetchone()
            self.assertEqual(attempt_state, "INTERRUPTED")
            self.assertIn("parent process restarted", error)
            self.assertEqual(
                restarted.queue_snapshot(run["run_id"])["active_attempts"], []
            )

            resumed = restarted.create_or_resume_run(
                run_id=run["run_id"],
                template_sha256="a" * 64,
                config={"prefetch_targets": 1},
            )
            self.assertEqual(resumed["run_id"], run["run_id"])
            self.assertEqual(resumed["resume_count"], 1)


class DeadlineAndSignalTests(unittest.TestCase):
    def test_deadline_has_distinct_soft_and_hard_boundaries(self):
        now = [100.0]
        deadline = Deadline(
            wall_time_seconds=100.0,
            soft_stop_seconds=20.0,
            started_monotonic=100.0,
            clock=lambda: now[0],
        )

        self.assertEqual(deadline.soft_at, 180.0)
        self.assertEqual(deadline.hard_at, 200.0)
        now[0] = 179.0
        self.assertTrue(deadline.may_start())
        self.assertFalse(deadline.soft_stop_reached)
        now[0] = 180.0
        self.assertFalse(deadline.may_start())
        self.assertTrue(deadline.soft_stop_reached)
        self.assertFalse(deadline.hard_expired)
        now[0] = 200.0
        self.assertTrue(deadline.hard_expired)
        self.assertEqual(deadline.remaining(), 0.0)

    def test_signal_controller_sets_flag_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        controller = SignalController(signals=(signal.SIGINT,))

        with controller:
            self.assertIsNot(signal.getsignal(signal.SIGINT), previous)
            controller.request_stop(signal.SIGTERM)
            self.assertTrue(controller.stop_requested)
            self.assertEqual(controller.signal_name, "SIGTERM")

        self.assertIs(signal.getsignal(signal.SIGINT), previous)

    def test_sleeping_process_is_killed_at_hard_deadline(self):
        process = launch_process_group(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            result = supervise_process(
                process,
                deadline=Deadline(wall_time_seconds=0.15),
                sample_interval_seconds=0.02,
                term_grace_seconds=0.2,
                kill_grace_seconds=0.1,
            )
        finally:
            if process.poll() is None:
                terminate_process_group(
                    process, term_grace_seconds=0.1, kill_grace_seconds=0.1
                )

        self.assertEqual(result.reason, "deadline")
        self.assertFalse(result.ok)
        self.assertIsNotNone(process.poll())
        self.assertGreaterEqual(result.sample_count, 1)
        self.assertLess(result.duration_seconds, 3.0)

    def test_resource_guard_stops_worker_after_persistent_samples(self):
        cases = {
            "emergency-ram": {
                "bad": {
                    "available_ram_bytes": 1,
                    "disk_free_bytes": 1_000,
                    "worker_total_rss_bytes": 0,
                },
                "limits": {
                    "min_available_ram_bytes": 100,
                    "emergency_available_ram_bytes": 10,
                },
                "detail": "emergency threshold",
            },
            "low-disk": {
                "bad": {
                    "available_ram_bytes": 1_000,
                    "disk_free_bytes": 1,
                    "worker_total_rss_bytes": 0,
                },
                "limits": {"min_free_disk_bytes": 10},
                "detail": "free disk",
            },
        }
        for label, case in cases.items():
            with self.subTest(label=label):
                process = launch_process_group(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                samples = []
                calls = [0]

                def sampler(_process, _disk_path):
                    calls[0] += 1
                    if calls[0] == 1:
                        return {
                            "available_ram_bytes": 1_000,
                            "disk_free_bytes": 1_000,
                            "worker_total_rss_bytes": 0,
                        }
                    return dict(case["bad"])

                try:
                    result = supervise_process(
                        process,
                        sample_interval_seconds=0.01,
                        resource_sampler=sampler,
                        callback=samples.append,
                        term_grace_seconds=0.2,
                        kill_grace_seconds=0.1,
                        **case["limits"],
                    )
                finally:
                    if process.poll() is None:
                        terminate_process_group(
                            process,
                            term_grace_seconds=0.1,
                            kill_grace_seconds=0.1,
                        )

                self.assertEqual(result.reason, "resource")
                self.assertIn(case["detail"], result.detail)
                self.assertGreaterEqual(len(samples), 2)
                self.assertIsNotNone(process.poll())


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _write_fits(path):
        fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(path)

    def test_manifest_accepts_exactly_five_fits_and_rejects_missing_or_bad(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                f"calibration_{index}": root / f"calibration_{index}.fits"
                for index in range(1, 6)
            }
            for path in artifacts.values():
                self._write_fits(path)

            valid = build_artifact_manifest(
                artifacts,
                base_dir=root,
                include_sha256=True,
                validate_fits=True,
            )
            self.assertTrue(valid["ok"])
            self.assertEqual(valid["count"], 5)
            self.assertTrue(all(entry["fits_valid"] for entry in valid["artifacts"]))
            self.assertTrue(
                all(len(entry["sha256"]) == 64 for entry in valid["artifacts"])
            )

            artifacts["calibration_5"].unlink()
            missing = build_artifact_manifest(
                artifacts, validate_fits=True
            )
            self.assertFalse(missing["ok"])
            self.assertEqual(missing["count"], 5)
            self.assertFalse(
                next(
                    entry
                    for entry in missing["artifacts"]
                    if entry["name"] == "calibration_5"
                )["exists"]
            )

            self._write_fits(artifacts["calibration_5"])
            artifacts["calibration_4"].write_bytes(b"not a FITS file")
            damaged = build_artifact_manifest(
                artifacts, validate_fits=True
            )
            self.assertFalse(damaged["ok"])
            bad_entry = next(
                entry
                for entry in damaged["artifacts"]
                if entry["name"] == "calibration_4"
            )
            self.assertFalse(bad_entry["fits_valid"])
            self.assertIn("FITS is not readable", bad_entry["fits_error"])



class AtomicAndCliTests(unittest.TestCase):
    def test_campaign_lock_rejects_second_parent_and_releases_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            first = batch.acquire_campaign_lock(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "active parent"):
                    batch.acquire_campaign_lock(directory)
            finally:
                batch.release_campaign_lock(first)
            second = batch.acquire_campaign_lock(directory)
            batch.release_campaign_lock(second)

    def test_atomic_json_keeps_previous_file_if_replace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "state.json"
            original = '{"generation": 1}\n'
            destination.write_text(original, encoding="utf-8")

            with patch.object(runtime.os, "replace", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    runtime.atomic_write_json(destination, {"generation": 2})

            self.assertEqual(destination.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(f".{destination.name}.*.tmp")), [])

    def test_invalid_resource_numbers_fail_before_any_worker_starts(self):
        for arguments in (
            ["--critical-free-gb", "-1"],
            ["--poll-seconds", "0"],
            ["--worker-timeout-hours", "-1"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                batch.run_parent(batch.parse_args(arguments))


class SafetyContractTests(unittest.TestCase):
    def test_resource_monitor_reports_system_and_process_fields(self):
        vm = SimpleNamespace(total=1000, available=700, used=300, percent=30.0)
        sm = SimpleNamespace(total=500, free=400, used=100, percent=20.0)
        process = Mock()
        process.memory_info.return_value.rss = 100
        process.children.return_value = []
        process.status.return_value = "running"
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runtime.psutil, "virtual_memory", return_value=vm), \
             patch.object(runtime.psutil, "swap_memory", return_value=sm), \
             patch.object(runtime.psutil, "Process", return_value=process):
            sample = runtime.collect_resource_sample(12345, directory)
        self.assertEqual(sample["collector"], "psutil")
        for name in ("available_ram_bytes", "disk_free_bytes", "worker_total_rss_bytes"):
            self.assertGreater(sample[name], 0)
        self.assertEqual(sample["available_ram_bytes"], 700)
        self.assertEqual(sample["worker_total_rss_bytes"], 100)

    def test_monitor_failure_is_reported_not_replaced_with_fake_values(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runtime.psutil, "virtual_memory", side_effect=OSError("denied")):
            with self.assertRaisesRegex(RuntimeError, "Resource monitoring failed"):
                runtime.collect_resource_sample(12345, directory)

    def test_missing_required_resource_measurement_stops_worker(self):
        process = launch_process_group(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            result = supervise_process(
                process, min_available_ram_bytes=1, sample_interval_seconds=0.01,
                resource_sampler=lambda *_: {},
                term_grace_seconds=0.2, kill_grace_seconds=0.1,
            )
            self.assertEqual(result.reason, "resource")
            self.assertIn("monitor unavailable", result.detail)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                terminate_process_group(process, term_grace_seconds=0.1)

    def test_missing_astropy_never_becomes_metadata_only_success(self):
        with patch.dict(sys.modules, {"astropy.io": None}):
            with self.assertRaises(ImportError):
                runtime.validate_fits_artifacts([])

    def test_sqlite_timeout_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CampaignState(directory, busy_timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
