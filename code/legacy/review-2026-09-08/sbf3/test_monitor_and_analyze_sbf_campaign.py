import sqlite3
import tempfile
import unittest
from pathlib import Path

import monitor_and_analyze_sbf_campaign as monitor


class FirstPassStateTests(unittest.TestCase):
    def test_reused_success_and_finished_failure_are_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "campaign.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY, state TEXT, created_at REAL,
                    soft_stop_at REAL, deadline_at REAL
                );
                CREATE TABLE jobs (
                    run_id TEXT, job_id TEXT, target TEXT, state TEXT,
                    attempt_count INTEGER, queue_position INTEGER
                );
                CREATE TABLE attempts (
                    attempt_id INTEGER PRIMARY KEY, run_id TEXT, job_id TEXT,
                    attempt_no INTEGER, state TEXT, ended_at REAL
                );
                INSERT INTO runs VALUES ('run-1', 'RUNNING', 1, NULL, NULL);
                INSERT INTO jobs VALUES ('run-1', 'reused', 'old result', 'SUCCEEDED', 0, 1);
                INSERT INTO jobs VALUES ('run-1', 'failed', 'bad isophotes', 'RETRY_WAIT', 1, 2);
                INSERT INTO attempts VALUES (1, 'run-1', 'failed', 1, 'FAILED', 10);
                """
            )
            connection.commit()
            connection.close()
            state = monitor.first_pass_state(database)
            self.assertTrue(state["complete"])
            self.assertEqual(state["unfinished_count"], 0)

    def test_running_first_attempt_is_not_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "campaign.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY, state TEXT, created_at REAL,
                    soft_stop_at REAL, deadline_at REAL
                );
                CREATE TABLE jobs (
                    run_id TEXT, job_id TEXT, target TEXT, state TEXT,
                    attempt_count INTEGER, queue_position INTEGER
                );
                CREATE TABLE attempts (
                    attempt_id INTEGER PRIMARY KEY, run_id TEXT, job_id TEXT,
                    attempt_no INTEGER, state TEXT, ended_at REAL
                );
                INSERT INTO runs VALUES ('run-1', 'RUNNING', 1, NULL, NULL);
                INSERT INTO jobs VALUES ('run-1', 'active', 'active galaxy', 'RUNNING', 1, 1);
                INSERT INTO attempts VALUES (1, 'run-1', 'active', 1, 'RUNNING', NULL);
                """
            )
            connection.commit()
            connection.close()
            state = monitor.first_pass_state(database)
            self.assertFalse(state["complete"])
            self.assertEqual(state["unfinished_targets"], ["active galaxy"])


class ModelSeparationTests(unittest.TestCase):
    def test_models_use_distinct_program_samples(self):
        records = []
        for index in range(3):
            records.append(
                {
                    "program": "3055",
                    "galaxy": f"A{index}",
                    "status": "done",
                    "is_clean_effective": True,
                    "method_generation": "sbf2_two_annuli_trgb",
                    "color": 0.35 + 0.05 * index,
                    "absolute_mbar": -3.5 + 0.05 * index,
                    "sigma_absolute_mbar": 0.05,
                }
            )
            records.append(
                {
                    "program": "7763",
                    "galaxy": f"B{index}",
                    "status": "done",
                    "is_clean_effective": False,
                    "method_generation": "sbf3_single_annulus_qc_v1",
                    "qc_status": "PASS",
                    "color": 0.10 + 0.05 * index,
                    "mbar": 28.2 - 0.1 * index,
                    "sigma_mbar": 0.08,
                }
            )
        models = {model["model_id"]: model for model in monitor.build_models(records)}
        self.assertEqual(models["go3055_trgb_absolute_clean"]["n"], 3)
        self.assertEqual(models["go7763_virgo_single_annulus_all"]["n"], 3)
        self.assertNotIn("B0", models["go3055_trgb_absolute_clean"]["galaxies"])
        self.assertNotIn("A0", models["go7763_virgo_single_annulus_all"]["galaxies"])
        self.assertEqual(models["go3055_trgb_absolute_clean"]["pivot_color"], 0.4)
        self.assertEqual(models["go7763_virgo_single_annulus_all"]["pivot_color"], 0.15)


if __name__ == "__main__":
    unittest.main()
