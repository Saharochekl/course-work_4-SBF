"""CLI orchestration only: no science cells are executed."""

import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

import process


class ProcessCliTests(unittest.TestCase):
    def test_both_bands_run_in_dependency_order(self):
        args = process.parse_args(["--galaxies", "NGC 4636"])
        commands = process.build_commands(args)
        self.assertEqual([c[2] for c in commands], [
            "sbf.run_sbf_2_batch", "sbf.run_sbf_2_normalized_winsor",
            "sbf.run_sbf_f090w",
        ])
        self.assertTrue(all(c[-2:] == ["--galaxies", "NGC 4636"] for c in commands))

    def test_stage_and_check_are_passed_to_the_correct_runner(self):
        args = process.parse_args(["--filter", "F150W", "--stage", "spectra", "--check"])
        commands = process.build_commands(args)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][2], "sbf.run_sbf_2_normalized_winsor")
        self.assertEqual(commands[0][-1], "--dry-run")

    def test_f090w_rejects_unsupported_split_stage(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            process.parse_args(["--filter", "F090W", "--stage", "source"])

    def test_dry_run_never_starts_a_child_process(self):
        with patch("process.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(process.main(["--dry-run"]), 0)
            run.assert_not_called()

    def test_all_check_commands_are_accepted_by_their_parsers(self):
        args = process.parse_args(["--filter", "both", "--check"])
        for command in process.build_commands(args):
            with self.subTest(module=command[2]), patch("sys.argv", command[2:]):
                parsed = importlib.import_module(command[2]).parse_args()
                self.assertTrue(parsed.dry_run)


if __name__ == "__main__":
    unittest.main()
