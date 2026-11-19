"""Unit tests for the image preprocessing variants."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fall_detection.preprocessing import VARIANTS, apply


class PreprocessingTests(unittest.TestCase):
    """Each variant keeps the image shape and does not modify its input."""
    def test_all_variants_preserve_shape_and_input(self):
        x = np.random.default_rng(17).integers(0, 256, (47, 63, 3), dtype=np.uint8)
        original = x.copy()
        for name in VARIANTS:
            a = apply(x, name)
            b = apply(x, name)
            self.assertEqual(a.shape, x.shape)
            self.assertEqual(a.dtype, np.uint8)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(x, original)

    def test_gamma_convention(self):
        x = np.full((20, 20, 3), 70, np.uint8)
        self.assertGreater(apply(x, "gamma_bright").mean(), x.mean())
        self.assertLess(apply(x, "gamma_dark").mean(), x.mean())

    def test_gray_keeps_three_equal_channels(self):
        x = np.random.default_rng(1).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        y = apply(x, "grayscale")
        np.testing.assert_array_equal(y[:, :, 0], y[:, :, 2])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            apply(np.zeros((8, 8, 3), np.uint8), "unknown")


if __name__ == "__main__":
    unittest.main()
