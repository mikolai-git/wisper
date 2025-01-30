import os
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wisper.brightness_processing import *




class TestBrightnessFunctions(unittest.TestCase):
    
    def setUp(self):
        # Create sample frames for testing
        self.frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 150  # Average brightness = 150
        self.frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Average brightness = 200
        self.frames = [self.frame1, self.frame2]

    def test_calculate_frame_brightness(self):
        brightness = calculate_frame_brightness(self.frame1)
        self.assertAlmostEqual(brightness, 150, places=1)

    def test_calculate_frame_brightness_diff(self):
        brightness_diff = calculate_frame_brightness_diff(self.frame1, self.frame2)
        self.assertAlmostEqual(brightness_diff, -50, places=1)

    def test_calculate_average_brightness_of_frames(self):
        avg_brightness = calculate_average_brightness_of_frames(self.frames)
        self.assertAlmostEqual(avg_brightness, 175, places=1)
    
    def test_calculate_average_brightness_of_frames_empty(self):
        # Test with an empty frame list
        avg_brightness = calculate_average_brightness_of_frames([])
        self.assertEqual(avg_brightness, 0)

    @patch("os.listdir")
    @patch("cv2.imread")
    def test_get_brightness_list(self, mock_imread, mock_listdir):
        mock_listdir.return_value = ["frame1.jpg", "frame2.jpg"]
        mock_imread.side_effect = [self.frame1, self.frame2]
        
        with TemporaryDirectory() as folder_path:
            brightness_list = get_brightness_list(folder_path)
            self.assertEqual(len(brightness_list), 2)
            self.assertAlmostEqual(brightness_list[0], 150, places=1)
            self.assertAlmostEqual(brightness_list[1], 200, places=1)

    @patch("os.listdir")
    @patch("cv2.imread")
    def test_get_brightness_diff_list(self, mock_imread, mock_listdir):
        mock_listdir.return_value = ["frame1.jpg", "frame2.jpg"]
        mock_imread.side_effect = [self.frame1, self.frame2]

        with TemporaryDirectory() as folder_path:
            brightness_diff_list = get_brightness_diff_list(folder_path)
            self.assertEqual(len(brightness_diff_list), 1)
            self.assertAlmostEqual(brightness_diff_list[0], -50, places=1)

    @patch("os.listdir")
    @patch("cv2.imread")
    def test_get_sliding_window_brightness_list(self, mock_imread, mock_listdir):
        # Create 5 frames all with brightness 150 for a window size of 2
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 150 for _ in range(5)]
        mock_listdir.return_value = [f"frame{i}.jpg" for i in range(5)]
        mock_imread.side_effect = frames
        
        with TemporaryDirectory() as folder_path:
            sliding_window_list = get_sliding_window_brightness_list(folder_path, window_size=2)
            self.assertEqual(len(sliding_window_list), 4)
            for brightness in sliding_window_list:
                self.assertAlmostEqual(brightness, 150, places=1)

    @patch("os.listdir")
    @patch("cv2.imread")
    def test_get_sliding_window_brightness_list_empty(self, mock_imread, mock_listdir):
        # Test with fewer frames than window size
        mock_listdir.return_value = ["frame1.jpg"]
        mock_imread.side_effect = [self.frame1]

        with TemporaryDirectory() as folder_path:
            sliding_window_list = get_sliding_window_brightness_list(folder_path, window_size=2)
            self.assertEqual(sliding_window_list, [])

if __name__ == "__main__":
    unittest.main()
