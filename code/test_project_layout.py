"""Проверка валидатора метаданных / Metadata validation with tiny local fixtures."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_project_layout as layout


class ProductLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'code').mkdir()
        (self.root / 'code/targets_go3055_manifest.csv').write_text('target\nNGC 3379\n')
        self.source = self.write('runs/sbf2_go3055/batch/NGC_3379_result.json',
                                 {'galaxy': 'NGC 3379', 'status': 'ok'})
        self.final = self.write('runs/sbf2_normalized_winsor/batch/results/NGC_3379_result.json',
                                {'galaxy': 'NGC 3379', 'status': 'ok'})
        self.published = self.write('runs/sbf_f090w_go3055/products/NGC_3379/products.json',
                                    {'galaxy': 'NGC 3379', 'source_result': str(self.source),
                                     'final_result': str(self.final)})

    def write(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def check(self):
        with patch.object(layout, 'PROJECT_ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            layout.check_products()

    def test_expected_targets_pass_without_any_fits_arrays(self):
        self.check()

    def test_same_count_wrong_galaxy_fails(self):
        self.write(self.source, {'galaxy': 'NGC 1399', 'status': 'ok'})
        with self.assertRaisesRegex(FileNotFoundError, 'wrong membership'):
            self.check()

    def test_failed_result_fails_even_when_files_exist(self):
        self.write(self.final, {'galaxy': 'NGC 3379', 'status': 'failed'})
        with self.assertRaisesRegex(FileNotFoundError, 'unsuccessful result'):
            self.check()

    def test_missing_product_fails(self):
        self.write(self.source, {'galaxy': 'NGC 3379', 'status': 'ok',
                                 'model': str(self.root / 'runs/missing.fits')})
        with self.assertRaisesRegex(FileNotFoundError, 'Missing product'):
            self.check()


if __name__ == '__main__':
    unittest.main()
