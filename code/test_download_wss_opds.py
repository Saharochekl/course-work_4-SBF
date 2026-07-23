from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import download_wss_opds as wss


class DownloadWssOpdTests(unittest.TestCase):
    def test_parse_fits_header_prefix(self):
        header = fits.Header()
        header["TELESCOP"] = "JWST"
        header["PROGRAM"] = "07763"
        header["DATE-OBS"] = "2026-06-18"
        header["TIME-OBS"] = "01:34:27"
        payload = header.tostring(padding=True).encode("ascii")
        restored = wss.parse_fits_header_prefix(payload)
        self.assertEqual(restored["PROGRAM"], "07763")
        self.assertEqual(restored["DATE-OBS"], "2026-06-18")

    def test_group_science_headers_deduplicates_sources(self):
        header = fits.Header()
        header["TELESCOP"] = "JWST"
        header["PROGRAM"] = "07763"
        header["DATE-OBS"] = "2026-06-18"
        header["TIME-OBS"] = "01:34:27"
        rows = wss.group_science_headers(
            [("same.fits", header), ("same.fits", header)]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file_count"], 1)
        self.assertEqual(rows[0]["programs"], ["7763"])

    def test_opd_time_rejects_truncated_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.fits"
            header = fits.Header()
            header["DATE-OBS"] = "2026-06-22"
            header["TIME-OBS"] = "05:08:17"
            fits.PrimaryHDU(
                data=np.ones((32, 32), dtype=np.float32), header=header
            ).writeto(valid)
            self.assertIsNotNone(wss.opd_time(valid))

            truncated = Path(directory) / "truncated.fits"
            payload = valid.read_bytes()
            truncated.write_bytes(payload[:-1024])
            self.assertIsNone(wss.opd_time(truncated))


if __name__ == "__main__":
    unittest.main()
