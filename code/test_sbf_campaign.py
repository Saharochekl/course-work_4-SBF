#!/usr/bin/env python3
"""Fast failure-path tests for the durable SBF campaign runner."""

from __future__ import annotations

import csv
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_sbf_batch as batch
import sbf_campaign_runtime as runtime
from sbf_campaign_runtime import (
    Deadline,
    SignalController,
    build_artifact_manifest,
    launch_process_group,
    supervise_process,
    terminate_process_group,
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
            "filters": {"signal": "F356W", "color": "F277W"},
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
                "filters": {"color": "f277w", "signal": "f356w"},
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

        output_a = batch.target_output_dir(
            "NGC 1404",
            "/local/volume-a/signal.fits",
            "/local/volume-a/color.fits",
            products_root="/campaign-products",
            signal_filter="F356W",
            color_filter="F277W",
            signal_fingerprint={"inode": 1, "mtime_ns": 10},
            color_fingerprint={"inode": 2, "mtime_ns": 20},
            job_id=first,
        )
        output_b = batch.target_output_dir(
            "NGC 1404",
            "/different/mount/signal.fits",
            "/different/mount/color.fits",
            products_root="/campaign-products",
            signal_filter="F356W",
            color_filter="F277W",
            signal_fingerprint={"inode": 999, "mtime_ns": 9999},
            color_fingerprint={"inode": 888, "mtime_ns": 8888},
            job_id=first,
        )
        self.assertEqual(output_a, output_b)

        result_a = batch.result_json_path(
            "/batch",
            "NGC 1404",
            identity={
                "template_family": batch.SBF3_NOTEBOOK_FAMILY,
                "job_id": first,
                "input_pair_key": "local-a",
                "signal_filter": "F356W",
                "color_filter": "F277W",
            },
        )
        result_b = batch.result_json_path(
            "/batch",
            "NGC 1404",
            identity={
                "template_family": batch.SBF3_NOTEBOOK_FAMILY,
                "job_id": first,
                "input_pair_key": "local-b",
                "signal_filter": "F356W",
                "color_filter": "F277W",
            },
        )
        self.assertEqual(result_a, result_b)


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
            job = state.upsert_job(
                run["run_id"],
                target="NGC 1404",
                program="3055",
                obsid="jw03055-o003",
                product_uris={"signal": "mast:signal", "color": "mast:color"},
                filters={"signal": "F356W", "color": "F277W"},
                initial_state="READY",
            )
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
            self.assertEqual(controller.count, 1)

        self.assertIs(signal.getsignal(signal.SIGINT), previous)
        controller.clear()
        self.assertFalse(controller.stop_requested)

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


class ArtifactAndCleanupTests(unittest.TestCase):
    def _write_fits(self, path: Path) -> None:
        fits.PrimaryHDU(np.ones((3, 3), dtype=np.float32)).writeto(
            path, overwrite=True
        )

    def _verified_result(
        self, root: Path, signal_path: Path, color_path: Path
    ) -> dict:
        out_dir = root / "products"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            key: out_dir / f"{key}.fits"
            for key in batch.SBF3_REQUIRED_FITS_KEYS
        }
        for path in artifacts.values():
            self._write_fits(path)
        manifest = build_artifact_manifest(
            artifacts,
            include_sha256=True,
            validate_fits=True,
            require_astropy=True,
        )
        self.assertTrue(manifest["ok"])
        return {
            "galaxy": "NGC 1404",
            "status": "ok",
            "template_family": batch.SBF3_NOTEBOOK_FAMILY,
            "artifacts_verified": True,
            "artifact_manifest": manifest["artifacts"],
            "out_dir": str(out_dir),
            "signal_path": str(signal_path),
            "color_path": str(color_path),
            **{key: str(path) for key, path in artifacts.items()},
        }

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
                require_astropy=True,
            )
            self.assertTrue(valid["ok"])
            self.assertEqual(valid["count"], 5)
            self.assertTrue(all(entry["fits_valid"] for entry in valid["artifacts"]))
            self.assertTrue(
                all(len(entry["sha256"]) == 64 for entry in valid["artifacts"])
            )

            artifacts["calibration_5"].unlink()
            missing = build_artifact_manifest(
                artifacts, validate_fits=True, require_astropy=True
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
                artifacts, validate_fits=True, require_astropy=True
            )
            self.assertFalse(damaged["ok"])
            bad_entry = next(
                entry
                for entry in damaged["artifacts"]
                if entry["name"] == "calibration_4"
            )
            self.assertFalse(bad_entry["fits_valid"])
            self.assertIn("FITS is not readable", bad_entry["fits_error"])

    def test_verified_result_migrates_from_notebook_bound_job_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_root = root / "batch"
            batch_root.mkdir()
            signal_path = root / "signal.fits"
            color_path = root / "color.fits"
            signal_path.write_bytes(b"same signal input")
            color_path.write_bytes(b"same color input")
            template = root / "sbf-3.ipynb"
            template.write_text(
                json.dumps(
                    {
                        "metadata": {"sbf_pipeline": {"family": "sbf3"}},
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )
            target = {
                "name": "NGC 1404",
                "signal_filter": "F150W",
                "color_filter": "F115W",
            }
            old_job_id = "job-" + "1" * 64
            new_job_id = "job-" + "2" * 64
            current_identity = batch.expected_run_identity(
                template,
                target,
                signal_path,
                color_path,
                products_root=root / "new-products",
                job_id=new_job_id,
            )
            result = self._verified_result(root, signal_path, color_path)
            result.update(
                {
                    "job_id": old_job_id,
                    "template_sha256": "a" * 64,
                    "signal_filter": "F150W",
                    "color_filter": "F115W",
                    "signal_fingerprint": batch.input_fingerprint(signal_path),
                    "color_fingerprint": batch.input_fingerprint(color_path),
                    "signal_sha256": batch.sha256_file(signal_path),
                    "color_sha256": batch.sha256_file(color_path),
                }
            )
            old_path = batch.result_json_path(
                batch_root,
                target["name"],
                identity={
                    **current_identity,
                    "job_id": old_job_id,
                },
            )
            old_path.write_text(json.dumps(result), encoding="utf-8")

            adopted = batch.verified_campaign_result(
                target, batch_root, current_identity
            )

            self.assertIsNotNone(adopted)
            self.assertEqual(adopted["job_id"], new_job_id)
            self.assertEqual(adopted["producer_job_id"], old_job_id)
            self.assertEqual(adopted["template_sha256"], "a" * 64)
            self.assertEqual(
                adopted["reused_for_template_sha256"],
                current_identity["template_sha256"],
            )
            self.assertTrue(
                batch.result_json_path(
                    batch_root, target["name"], identity=current_identity
                ).exists()
            )

    def test_failed_legacy_result_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "sbf-3.ipynb"
            template.write_text(
                json.dumps(
                    {
                        "metadata": {"sbf_pipeline": {"family": "sbf3"}},
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )
            signal = root / "signal.fits"
            color = root / "color.fits"
            signal.write_bytes(b"signal")
            color.write_bytes(b"color")
            target = {
                "name": "FAILED TARGET",
                "signal_filter": "F150W",
                "color_filter": "F115W",
            }
            identity = batch.expected_run_identity(
                template,
                target,
                signal,
                color,
                products_root=root / "products",
                job_id="job-" + "2" * 64,
            )
            failed_path = batch.result_json_path(
                root,
                target["name"],
                identity={**identity, "job_id": "job-" + "1" * 64},
            )
            failed_path.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "galaxy": target["name"],
                        "signal_filter": "F150W",
                        "color_filter": "F115W",
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                batch.verified_campaign_result(target, root, identity)
            )

    def test_cleanup_preserves_inputs_when_artifacts_are_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal_path = root / "signal.fits"
            color_path = root / "color.fits"
            signal_path.write_bytes(b"signal")
            color_path.write_bytes(b"color")
            result = {
                "galaxy": "NGC 1404",
                "status": "ok",
                "artifacts_verified": False,
                "signal_path": str(signal_path),
                "color_path": str(color_path),
            }
            fake_disk = ({"free_gb": 0.0}, {"available_gb": 100.0})

            with patch.object(batch, "log_resources", return_value=fake_disk):
                enough_space = batch.ensure_disk_space_for_downloads(
                    root,
                    [result],
                    min_free_gb=1.0,
                    cleanup_enabled=True,
                )

            self.assertFalse(enough_space)
            self.assertTrue(signal_path.exists())
            self.assertTrue(color_path.exists())

    def test_cleanup_rechecks_five_fits_before_deleting_inputs(self):
        for damage in ("missing", "replaced"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fits_paths = {
                    key: root / f"{key}.fits"
                    for key in batch.SBF3_REQUIRED_FITS_KEYS
                }
                for path in fits_paths.values():
                    self._write_fits(path)
                manifest = build_artifact_manifest(
                    fits_paths,
                    include_sha256=True,
                    validate_fits=True,
                    require_astropy=True,
                )
                self.assertTrue(manifest["ok"])

                signal_path = root / "signal_input.fits"
                color_path = root / "color_input.fits"
                signal_path.write_bytes(b"signal")
                color_path.write_bytes(b"color")
                result = {
                    "galaxy": "NGC 1404",
                    "status": "ok",
                    "template_family": batch.SBF3_NOTEBOOK_FAMILY,
                    "artifacts_verified": True,
                    "artifact_manifest": manifest["artifacts"],
                    "signal_path": str(signal_path),
                    "color_path": str(color_path),
                    **{key: str(path) for key, path in fits_paths.items()},
                }

                damaged_path = fits_paths[batch.SBF3_REQUIRED_FITS_KEYS[2]]
                if damage == "missing":
                    damaged_path.unlink()
                else:
                    fits.PrimaryHDU(
                        np.full((3, 3), 17.0, dtype=np.float32)
                    ).writeto(damaged_path, overwrite=True)

                fake_disk = ({"free_gb": 0.0}, {"available_gb": 100.0})
                with patch.object(batch, "log_resources", return_value=fake_disk):
                    enough_space = batch.ensure_disk_space_for_downloads(
                        root,
                        [result],
                        min_free_gb=1.0,
                        cleanup_enabled=True,
                    )

                self.assertFalse(batch.result_artifacts_still_valid(result))
                self.assertFalse(enough_space)
                self.assertTrue(signal_path.exists())
                self.assertTrue(color_path.exists())

    def test_cleanup_refuses_verified_input_outside_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            outside_signal = root / "outside_signal.fits"
            inside_color = data_root / "inside_color.fits"
            outside_signal.write_bytes(b"signal")
            inside_color.write_bytes(b"color")
            result = self._verified_result(root, outside_signal, inside_color)
            self.assertTrue(batch.result_artifacts_still_valid(result))

            fake_disk = ({"free_gb": 0.0}, {"available_gb": 100.0})
            with patch.object(batch, "log_resources", return_value=fake_disk):
                enough_space = batch.ensure_disk_space_for_downloads(
                    data_root,
                    [result],
                    min_free_gb=1.0,
                    cleanup_enabled=True,
                )

            self.assertFalse(enough_space)
            self.assertTrue(outside_signal.exists())
            self.assertFalse(inside_color.exists())

    def test_cleanup_preserves_protected_verified_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            protected_signal = data_root / "shared_signal.fits"
            disposable_color = data_root / "finished_color.fits"
            protected_signal.write_bytes(b"signal")
            disposable_color.write_bytes(b"color")
            result = self._verified_result(root, protected_signal, disposable_color)
            self.assertTrue(batch.result_artifacts_still_valid(result))

            fake_disk = ({"free_gb": 0.0}, {"available_gb": 100.0})
            with patch.object(batch, "log_resources", return_value=fake_disk):
                enough_space = batch.ensure_disk_space_for_downloads(
                    data_root,
                    [result],
                    min_free_gb=1.0,
                    cleanup_enabled=True,
                    protected_input_paths={protected_signal},
                )

            self.assertFalse(enough_space)
            self.assertTrue(protected_signal.exists())
            self.assertFalse(disposable_color.exists())


class AtomicAndCliTests(unittest.TestCase):
    def test_external_consumer_ignores_part_and_records_wait_until_both_finals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            target = batch.normalize_target(
                {
                    "name": "WAIT GALAXY",
                    "program": "7763",
                    "obsid": "o001_t001",
                    "signal_filter": "F150W",
                    "color_filter": "F115W",
                    "signal_product": "signal.fits",
                    "color_product": "color.fits",
                }
            )
            target_dir = data_root / target["name"]
            target_dir.mkdir(parents=True)
            part = target_dir / "signal.fits.part"
            fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(part)
            self.assertFalse(batch.target_inputs_ready(target, data_root))

            status_path = data_root / "download_go3055_go7763_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "started_at": "now",
                        "updated_at": "now",
                        "programs": ["3055", "7763"],
                        "interrupted": False,
                        "counts": {"downloaded": 1},
                        "results": [],
                    }
                ),
                encoding="utf-8",
            )
            event_path = root / "events.jsonl"

            class CompletingController:
                stop_requested = False

                def wait(self, _seconds):
                    part.unlink()
                    for name in ("signal.fits", "color.fits"):
                        fits.PrimaryHDU(
                            np.ones((2, 2), dtype=np.float32)
                        ).writeto(target_dir / name, overwrite=True)

            fake_resources = (
                {"total": 2**40, "free": 2**39, "total_gb": 1024, "free_gb": 512},
                {"total": 2**35, "available": 2**34, "available_gb": 16},
            )
            with patch.object(batch, "log_resources", return_value=fake_resources):
                ready, reason = batch.wait_for_campaign_inputs(
                    target,
                    data_root,
                    process=None,
                    deadline=Deadline(wall_time_seconds=5),
                    signal_controller=CompletingController(),
                    poll_seconds=0.01,
                    timeout_seconds=2,
                    external_status_path=status_path,
                    event_log_path=event_path,
                )

            self.assertTrue(ready)
            self.assertEqual(reason, "ready")
            events = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["event_type"] for event in events],
                ["INPUT_WAIT_STARTED", "INPUT_WAIT_HEARTBEAT", "INPUT_READY"],
            )
            first_signal = events[0]["payload"]["inputs"]["signal"]
            self.assertFalse(first_signal["final_ready"])
            self.assertTrue(first_signal["part_exists"])
            self.assertEqual(
                events[-1]["payload"]["external_downloader"]["path"],
                str(status_path.resolve()),
            )

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

    def test_prefetch_cli_is_hard_capped_at_one_target(self):
        self.assertEqual(batch.parse_args([]).prefetch_targets, 1)
        self.assertEqual(batch.parse_args(["--prefetch-targets", "0"]).prefetch_targets, 0)
        self.assertEqual(batch.parse_args(["--prefetch-targets", "1"]).prefetch_targets, 1)
        for invalid in ("-1", "2", "99"):
            with self.subTest(invalid=invalid):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    batch.parse_args(["--prefetch-targets", invalid])

    def test_invalid_resource_numbers_fail_before_any_worker_starts(self):
        for arguments in (
            ["--critical-free-gb", "-1"],
            ["--poll-seconds", "0"],
            ["--worker-timeout-hours", "-1"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                batch.run_parent(batch.parse_args(arguments))


class ParentCampaignSmokeTests(unittest.TestCase):
    def test_one_target_sbf3_campaign_reaches_verified_success(self):
        notebook = {
            "metadata": {"sbf_pipeline": {"family": "sbf3"}},
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from pathlib import Path\n",
                        "import numpy as np\n",
                        "from astropy.io import fits\n",
                        "stem = Path(signal_path).stem\n",
                        "names = [\n",
                        "    '01_модель_чистая',\n",
                        "    '02_изофоты_чистые',\n",
                        "    '03_остатки_общие',\n",
                        "    '04_остатки_общие_рабочие',\n",
                        "    '05_остатки_общие_рабочие_два_кольца',\n",
                        "]\n",
                        "for name in names:\n",
                        "    fits.writeto(out_dir / f'{stem}_{name}.fits', np.zeros((3, 3)), overwrite=True)\n",
                        "recommended_sbf = {'mbar_weighted': 1.23, 'sigma_adopted': 0.05}\n",
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            target_dir = data_root / "SMOKE GALAXY"
            target_dir.mkdir(parents=True)
            signal_path = target_dir / "signal_i2d.fits"
            color_path = target_dir / "color_i2d.fits"
            fits.PrimaryHDU(np.ones((3, 3), dtype=np.float32)).writeto(signal_path)
            fits.PrimaryHDU(np.ones((3, 3), dtype=np.float32)).writeto(color_path)

            template = root / "synthetic-sbf3.ipynb"
            template.write_text(json.dumps(notebook), encoding="utf-8")
            manifest = root / "targets.csv"
            fields = [
                "target",
                "program",
                "obsid",
                "signal_filter",
                "color_filter",
                "signal_product",
                "color_product",
                "signal_product_uri",
                "color_product_uri",
                "download_enabled",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "target": "SMOKE GALAXY",
                        "program": "99999",
                        "obsid": "jw99999-o001",
                        "signal_filter": "F356W",
                        "color_filter": "F277W",
                        "signal_product": signal_path.name,
                        "color_product": color_path.name,
                        "signal_product_uri": "mast:JWST/product/signal_i2d.fits",
                        "color_product_uri": "mast:JWST/product/color_i2d.fits",
                        "download_enabled": "true",
                    }
                )

            batch_root = root / "batch"
            products_root = root / "products"
            campaign_root = root / "campaign"
            args = batch.parse_args(
                [
                    "--template",
                    str(template),
                    "--data-root",
                    str(data_root),
                    "--batch-root",
                    str(batch_root),
                    "--products-root",
                    str(products_root),
                    "--campaign-root",
                    str(campaign_root),
                    "--target-csv",
                    str(manifest),
                    "--no-download",
                    "--no-cleanup-inputs",
                    "--prefetch-targets",
                    "0",
                    "--new-run",
                    "--wall-time-hours",
                    "0",
                    "--soft-stop-minutes",
                    "0",
                    "--worker-timeout-hours",
                    "1",
                    "--resource-sample-seconds",
                    "0.01",
                    "--min-free-gb",
                    "0",
                    "--critical-free-gb",
                    "0",
                    "--min-available-ram-gb",
                    "0",
                    "--emergency-available-ram-gb",
                    "0",
                ]
            )

            def quiet_launch(command, **kwargs):
                return runtime.launch_process_group(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **kwargs,
                )

            def controlled_supervision(process, **kwargs):
                kwargs["resource_sampler"] = lambda _process, _path: {
                    "available_ram_bytes": 2**40,
                    "disk_free_bytes": 2**40,
                    "worker_total_rss_bytes": 0,
                    "system_ram": {
                        "total_bytes": 2**40,
                        "available_bytes": 2**40,
                    },
                    "swap": {"used_bytes": 0},
                    "disk": {"total_bytes": 2**41, "free_bytes": 2**40},
                    "worker": {"rss_bytes": 0, "children_rss_bytes": 0},
                }
                return runtime.supervise_process(process, **kwargs)

            with (
                patch.object(batch, "launch_process_group", side_effect=quiet_launch),
                patch.object(
                    batch, "supervise_process", side_effect=controlled_supervision
                ),
                redirect_stdout(io.StringIO()),
            ):
                returncode = batch.run_parent(args)

            self.assertEqual(returncode, 0)
            result_files = list(batch_root.glob("*_result.json"))
            self.assertEqual(len(result_files), 1)
            result = json.loads(result_files[0].read_text(encoding="utf-8"))
            self.assertTrue(result["artifacts_verified"])
            self.assertEqual(result["artifact_count"], 5)
            self.assertEqual(len(result["artifact_manifest"]), 5)
            self.assertEqual(len(result["signal_sha256"]), 64)
            self.assertEqual(len(result["color_sha256"]), 64)
            self.assertTrue(Path(result["cell_timings_path"]).exists())

            for path in (
                campaign_root / "campaign.log",
                campaign_root / "campaign_events.jsonl",
                campaign_root / "run_provenance.json",
                campaign_root / "invocations.jsonl",
                campaign_root / "campaign_report.txt",
                campaign_root / "queue_snapshot.json",
            ):
                self.assertTrue(path.exists(), path)
            event_types = {
                json.loads(line)["event_type"]
                for line in (campaign_root / "campaign_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            self.assertTrue(
                {
                    "PARENT_INVOCATION_STARTED",
                    "CAMPAIGN_INVOCATION",
                    "INPUT_READY_INITIAL",
                    "WORKER_START_REQUESTED",
                    "WORKER_SUPERVISION_ENDED",
                    "ARTIFACTS_VERIFIED",
                    "CAMPAIGN_FINISHED",
                    "PARENT_INVOCATION_ENDED",
                }.issubset(event_types)
            )
            provenance = json.loads(
                (campaign_root / "run_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["selected_target_count"], 1)
            self.assertTrue(provenance["template"]["snapshot"])

            with closing(
                sqlite3.connect(campaign_root / "campaign_state.sqlite")
            ) as connection:
                job_states = [
                    row[0] for row in connection.execute("SELECT state FROM jobs")
                ]
                artifact_count = connection.execute(
                    "SELECT COUNT(*) FROM artifacts"
                ).fetchone()[0]
                run_state = connection.execute(
                    "SELECT state FROM runs"
                ).fetchone()[0]
            self.assertEqual(job_states, ["SUCCEEDED"])
            self.assertEqual(artifact_count, 5)
            self.assertEqual(run_state, "COMPLETED")

    def test_resume_marks_unselected_old_job_skipped_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "synthetic-sbf3.ipynb"
            template.write_text(
                json.dumps(
                    {
                        "metadata": {"sbf_pipeline": {"family": "sbf3"}},
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "targets.csv"
            fields = [
                "target",
                "program",
                "obsid",
                "signal_filter",
                "color_filter",
                "signal_product",
                "color_product",
                "signal_product_uri",
                "color_product_uri",
                "download_enabled",
            ]
            selected_row = {
                "target": "A",
                "program": "99999",
                "obsid": "jw99999-o001",
                "signal_filter": "F356W",
                "color_filter": "F277W",
                "signal_product": "a_signal.fits",
                "color_product": "a_color.fits",
                "signal_product_uri": "mast:JWST/product/a_signal.fits",
                "color_product_uri": "mast:JWST/product/a_color.fits",
                "download_enabled": "true",
            }
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(selected_row)

            campaign_root = root / "campaign"
            campaign_config = {
                "schema_version": 2,
                "job_identity_version": 2,
                "notebook_family": batch.SBF3_NOTEBOOK_FAMILY,
                "required_fits": list(batch.SBF3_REQUIRED_FITS_KEYS),
            }
            state = CampaignState(campaign_root)
            old_run = state.create_or_resume_run(
                template_sha256=batch.sha256_file(template),
                config=campaign_config,
            )
            old_jobs = state.upsert_jobs(
                old_run["run_id"],
                [
                    {
                        "target": "A",
                        "program": "99999",
                        "obsid": "jw99999-o001",
                        "product_uris": {
                            "signal": selected_row["signal_product_uri"],
                            "color": selected_row["color_product_uri"],
                        },
                        "filters": {"signal": "F356W", "color": "F277W"},
                        "queue_position": 0,
                    },
                    {
                        "target": "B",
                        "program": "99999",
                        "obsid": "jw99999-o002",
                        "product_uris": {
                            "signal": "mast:JWST/product/b_signal.fits",
                            "color": "mast:JWST/product/b_color.fits",
                        },
                        "filters": {"signal": "F356W", "color": "F277W"},
                        "queue_position": 1,
                    },
                ],
            )
            old_b_job_id = old_jobs[1]["job_id"]
            args = batch.parse_args(
                [
                    "--template",
                    str(template),
                    "--data-root",
                    str(root / "data"),
                    "--batch-root",
                    str(root / "batch"),
                    "--products-root",
                    str(root / "products"),
                    "--campaign-root",
                    str(campaign_root),
                    "--target-csv",
                    str(manifest),
                    "--no-download",
                    "--no-cleanup-inputs",
                    "--prefetch-targets",
                    "0",
                    "--wall-time-hours",
                    "0",
                    "--soft-stop-minutes",
                    "0",
                    "--resource-sample-seconds",
                    "0.01",
                    "--min-free-gb",
                    "0",
                    "--critical-free-gb",
                    "0",
                    "--min-available-ram-gb",
                    "0",
                    "--emergency-available-ram-gb",
                    "0",
                ]
            )
            verification_calls = [0]

            def fake_verified(target, _batch_root, identity):
                verification_calls[0] += 1
                if verification_calls[0] == 1:
                    return None
                return {
                    "status": "ok",
                    "galaxy": target["name"],
                    "job_id": identity["job_id"],
                    "template_family": batch.SBF3_NOTEBOOK_FAMILY,
                    "artifacts_verified": True,
                    "artifact_manifest": [],
                }

            supervision = runtime.SupervisionResult(
                reason="completed",
                returncode=0,
                started_monotonic=0.0,
                ended_monotonic=1.0,
                sample_count=0,
            )
            process = Mock(pid=12345)
            with (
                patch.object(batch, "target_inputs_ready", return_value=True),
                patch.object(batch, "wait_for_worker_capacity", return_value=True),
                patch.object(batch, "launch_process_group", return_value=process),
                patch.object(batch, "supervise_process", return_value=supervision),
                patch.object(
                    batch, "verified_campaign_result", side_effect=fake_verified
                ),
                patch.object(
                    batch, "ensure_disk_space_for_downloads", return_value=True
                ),
                redirect_stdout(io.StringIO()),
            ):
                returncode = batch.run_parent(args)

            resumed = CampaignState(campaign_root)
            states = {
                job["target"]: job["state"]
                for job in resumed.list_jobs(old_run["run_id"])
            }
            self.assertEqual(returncode, 0)
            self.assertEqual(states, {"A": "SUCCEEDED", "B": "SKIPPED"})
            self.assertEqual(
                resumed.get_job(old_run["run_id"], old_b_job_id)["state"],
                "SKIPPED",
            )
            self.assertEqual(
                resumed.get_run(old_run["run_id"])["state"], "COMPLETED"
            )


class DownloadSelectionRegressionTests(unittest.TestCase):
    def test_duplicate_names_are_disambiguated_by_manifest_keys(self):
        targets = [
            batch.normalize_target(
                {
                    "name": "DUPLICATE",
                    "program": "1",
                    "obsid": "o001",
                    "signal_product": "first_signal.fits",
                    "color_product": "first_color.fits",
                }
            ),
            batch.normalize_target(
                {
                    "name": "DUPLICATE",
                    "program": "1",
                    "obsid": "o002",
                    "signal_product": "second_signal.fits",
                    "color_product": "second_color.fits",
                }
            ),
        ]
        args = Namespace(
            no_download=False,
            target_csv="targets.csv",
            data_root="data",
            batch_root="batch",
            download_retry_seconds=120,
            min_free_gb=30.0,
            no_cleanup_inputs=True,
        )
        with (
            patch.object(batch.subprocess, "Popen", return_value=Mock()) as _popen,
            patch.object(batch.atexit, "register") as _register,
        ):
            batch.start_download_manager(args, targets, [])

        command = _popen.call_args.args[0]
        key_start = command.index("--target-keys") + 1
        key_end = command.index("--galaxies")
        passed_keys = command[key_start:key_end]
        expected_keys = [batch.target_manifest_key(target) for target in targets]
        self.assertEqual(passed_keys, expected_keys)
        self.assertEqual(len(set(passed_keys)), 2)
        self.assertEqual(command[key_end + 1 :], ["DUPLICATE", "DUPLICATE"])

        selected = batch.select_targets(
            targets,
            galaxies=["DUPLICATE"],
            target_keys=[passed_keys[1]],
        )
        self.assertEqual(selected, [targets[1]])


if __name__ == "__main__":
    unittest.main()
