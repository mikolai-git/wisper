import unittest
import numpy as np
import cv2

from wisper.colour_processing import compute_lab_metrics

class TestColourMetrics(unittest.TestCase):

    def setUp(self):
        # Create synthetic images with known chroma and colourfulness properties
        self.gray_image = np.ones((100, 100, 3), dtype=np.uint8) * 127  # Neutral grey
        self.colour_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.colour_image[:, :, 0] = 255  # Pure blue in BGR
        self.vivid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.vivid_image[:, :, 1] = 255  # Pure green for vividness contrast

    def test_compute_lab_metrics_gray(self):
        avg_chroma, colorfulness = compute_lab_metrics(self.gray_image)
        self.assertAlmostEqual(avg_chroma, 0, places=1)
        self.assertAlmostEqual(colorfulness, 0, places=1)

    def test_compute_lab_metrics_coloured(self):
        avg_chroma, colorfulness = compute_lab_metrics(self.colour_image)
        self.assertGreater(avg_chroma, 0)
        self.assertGreater(colorfulness, 0)

    def test_compute_lab_metrics_vivid(self):
        avg_chroma, colorfulness = compute_lab_metrics(self.vivid_image)
        self.assertGreater(avg_chroma, 0)
        self.assertGreater(colorfulness, 0)

    def test_output_types(self):
        avg_chroma, colorfulness = compute_lab_metrics(self.colour_image)
        self.assertIsInstance(avg_chroma, float)
        self.assertIsInstance(colorfulness, float)

if __name__ == "__main__":
    unittest.main()
