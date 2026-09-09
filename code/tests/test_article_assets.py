"""Lightweight publication-path tests; no notebooks, FITS, or plots are run."""

import json
from pathlib import Path
import re
import tempfile
import unittest

from figures.publish_article_assets import MANIFEST, ROOT, figure_sources, publish_article_assets, publish_figure


class ArticleAssetTests(unittest.TestCase):
    def test_manifest_matches_both_tex_documents_including_comments(self):
        assets = figure_sources()
        destinations = [asset["destination"] for asset in assets]
        self.assertEqual(len(destinations), len(set(destinations)))
        expected = set()
        for tex in (ROOT / "texts/paper_work").glob("*.tex"):
            for image in re.findall(r"\\paperfigure\{([^}]+)\}", tex.read_text()):
                self.assertTrue(image.startswith("materials/figures/"), image)
                expected.add(str((tex.parent / image).relative_to(ROOT)))
        self.assertEqual(expected, set(destinations))
        for asset in assets:
            self.assertFalse(Path(asset["source"]).is_absolute())
            self.assertFalse(Path(asset["destination"]).is_absolute())

    def test_publishing_keeps_source_and_refreshes_only_selected_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / MANIFEST
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{
                "source": "runs/selected.png",
                "destination": "texts/paper_work/materials/figures/selected.png",
            }]))
            source = root / "runs/selected.png"
            source.parent.mkdir()
            source.write_bytes(b"first figure")
            destination = publish_figure("runs/selected.png", root)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(publish_article_assets(root, check=True), 1)
            self.assertIsNone(publish_figure("runs/unselected.png", root))
            source.write_bytes(b"updated figure")
            with self.assertRaisesRegex(FileNotFoundError, "Outdated"):
                publish_article_assets(root, check=True)
            self.assertEqual(publish_article_assets(root), 1)
            self.assertEqual(destination.read_bytes(), b"updated figure")
            self.assertTrue(source.is_file())


if __name__ == "__main__":
    unittest.main()
