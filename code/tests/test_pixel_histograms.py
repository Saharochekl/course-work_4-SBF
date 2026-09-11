"""Равные интервалы и точные цвета / Uniform count bins and exact colour limits.

Synthetic data only; no FITS, scientific measurements or saved figures are used.
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from figures.build_f090w_appendix_diagnostics import draw_histogram, zone_colors


class PixelHistogramTests(unittest.TestCase):
    def setUp(self):
        style = matplotlib.rc_context({
            "axes.grid": False, "axes.facecolor": "white",
            "figure.facecolor": "white", "figure.dpi": 100,
        })
        style.__enter__()
        self.addCleanup(style.__exit__, None, None, None)
        self.fig, self.ax = plt.subplots()
        # Deliberately asymmetric thresholds, none aligned with the bin edges.
        self.thresholds = {3.0: (-2.83, 3.12), 3.5: (-3.5, 3.5), 4.0: (-4.13, 4.27)}
        self.edges = np.linspace(-6, 6, 151)
        self.centres = (self.edges[:-1] + self.edges[1:]) / 2

    def tearDown(self):
        plt.close(self.fig)

    def test_counts_use_equal_bins_without_threshold_dips(self):
        # A flat histogram must remain flat across all six colour boundaries.
        values = np.repeat(self.centres, 7)
        original = values.copy()
        draw_histogram(self.ax, values, self.thresholds, "F090W")

        self.assertEqual(len(self.ax.patches), 4)
        for layer in self.ax.patches:
            data = layer.get_data()
            np.testing.assert_array_equal(data.edges, self.edges)
            np.testing.assert_allclose(np.diff(data.edges), 0.08, rtol=0, atol=2e-15)
            np.testing.assert_array_equal(data.values, np.full(150, 7))
            self.assertEqual(data.values.sum(), values.size)
        np.testing.assert_array_equal(values, original)
        self.assertEqual(self.ax.get_ylabel(), "Number of pixels")

    def test_colours_change_at_exact_thresholds_without_rebinning(self):
        draw_histogram(self.ax, self.centres, self.thresholds, "F090W", pasa=True)
        self.fig.canvas.draw()

        self.assertEqual(self.ax.patches[0].get_facecolor(), to_rgba(zone_colors["orange"]))
        for layer, (sigma, colour) in zip(
            self.ax.patches[1:], ((4.0, "yellow"), (3.5, "red"), (3.0, "blue"))
        ):
            self.assertEqual(layer.get_facecolor(), to_rgba(zone_colors[colour]))
            clip = layer.get_clip_box().transformed(self.ax.get_xaxis_transform().inverted())
            np.testing.assert_allclose(clip.extents, [self.thresholds[sigma][0], 0,
                                                    self.thresholds[sigma][1], 1], atol=1e-12)
        # Also verify the in-memory rendering: all seven coloured regions exist.
        raster = np.asarray(self.fig.canvas.buffer_rgba())
        for x, colour in ((-5, "orange"), (-3.9, "yellow"), (-3.2, "red"),
                          (0, "blue"), (3.3, "red"), (3.8, "yellow"), (5, "orange")):
            pixel_x, pixel_y = self.ax.transData.transform((x, 0.5))
            expected = np.round(255 * np.array(to_rgba(zone_colors[colour]))).astype(np.uint8)
            np.testing.assert_array_equal(raster[int(raster.shape[0] - pixel_y), int(pixel_x)], expected)
        self.assertIn("Normalised residual", self.ax.get_xlabel())

    def test_far_tails_and_exact_boundaries_keep_every_pixel(self):
        values = np.r_[self.centres, [-9.0, 9.0, -6.0, 6.0], self.thresholds[3.5]]
        original = values.copy()
        draw_histogram(self.ax, values, self.thresholds, "F150W")

        counts = self.ax.patches[0].get_data().values
        expected, _ = np.histogram(np.clip(values, -6, 6), bins=self.edges)
        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(counts.sum(), values.size)
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[-1], 3)
        affected = 100 * np.mean((values < -3.5) | (values > 3.5))
        self.assertIn(f"{affected:.2f}%", self.ax.texts[0].get_text())
        np.testing.assert_array_equal(values, original)


if __name__ == "__main__":
    unittest.main()
