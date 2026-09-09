"""Public CLI dispatch without downloads, workers or scientific notebooks."""
import argparse
import unittest
from unittest.mock import patch

import download
import process
from sbf.sbf_paths import CODE_DIR


class EntrypointTests(unittest.TestCase):
    def test_images_options_and_help_are_forwarded(self):
        with patch.object(download.subprocess, 'call', return_value=0) as call:
            self.assertEqual(download.main(['images', '--help']), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[1:], ['-m', 'sbf.download_go3055_go7763', '--help'])
        self.assertEqual(call.call_args.kwargs['cwd'], CODE_DIR)

    def test_opd_failure_is_not_hidden(self):
        with patch.object(download.subprocess, 'call', return_value=2):
            self.assertEqual(download.main(['opd', '--program', '3055']), 2)

    def test_both_filters_have_source_before_spectra(self):
        args = argparse.Namespace(filter='both', stage='all', galaxies=None, check=False)
        commands = process.build_commands(args)
        self.assertEqual([command[2] for command in commands], [
            'sbf.run_sbf_2_batch', 'sbf.run_sbf_2_normalized_winsor', 'sbf.run_sbf_f090w',
        ])

    def test_dry_run_does_not_launch_worker(self):
        with patch.object(process.subprocess, 'run') as run, patch('builtins.print'):
            self.assertEqual(process.main(['--filter', 'F090W', '--dry-run']), 0)
        run.assert_not_called()

    def test_worker_error_stops_next_stage(self):
        with patch.object(process.subprocess, 'run') as run, patch('builtins.print'):
            run.return_value.returncode = 3
            self.assertEqual(process.main(['--filter', 'both']), 3)
        self.assertEqual(run.call_count, 1)
