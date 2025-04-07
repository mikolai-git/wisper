import os
import unittest
import numpy as np
import cv2
from tempfile import TemporaryDirectory
from unittest.mock import patch

from wisper.optical_flow import process_optical_flow, detect_objects_with_optical_flow, visualize_clusters


class TestOpticalFlowFunctions(unittest.TestCase):

    def setUp(self):
        # Create two simple synthetic frames with a moving object
        self.frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        self.frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(self.frame2, (10, 10), (20, 20), (255, 255, 255), -1)  # simulate motion

        self.gray1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)

    def test_detect_objects_with_optical_flow_typical_case(self):
        # Simulate motion
        flow = cv2.calcOpticalFlowFarneback(self.gray1, self.gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        labels, clustered_points = detect_objects_with_optical_flow(magnitude, angle, min_magnitude=0.5)
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(clustered_points, np.ndarray)

        if clustered_points.size > 0:
            self.assertEqual(clustered_points.shape[1], 4)

    def test_detect_objects_with_optical_flow_empty_input(self):
        magnitude = np.zeros((10, 10), dtype=np.float32)
        angle = np.zeros((10, 10), dtype=np.float32)
        labels, clustered_points = detect_objects_with_optical_flow(magnitude, angle)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(clustered_points), 0)

    def test_visualize_clusters(self):
        clustered_points = np.array([[10, 10, 2.0, 0.5], [15, 15, 2.5, 1.0]])
        labels = np.array([0, 0])
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        output_frame = visualize_clusters(frame, labels, clustered_points)
        self.assertEqual(output_frame.shape, frame.shape)
        self.assertTrue((output_frame != frame).any())  # Ensure drawing happened

    def test_process_optical_flow_returns_valid_output(self):
        with TemporaryDirectory() as input_dir, TemporaryDirectory() as output_dir:
            # Save two simple frames
            cv2.imwrite(os.path.join(input_dir, "frame_0001.jpg"), self.frame1)
            cv2.imwrite(os.path.join(input_dir, "frame_0002.jpg"), self.frame2)

            avg_magnitude_dict, motion_data_dict = process_optical_flow(input_dir, output_dir)

            self.assertIsInstance(avg_magnitude_dict, dict)
            self.assertIsInstance(motion_data_dict, dict)
            self.assertGreaterEqual(len(avg_magnitude_dict), 1)
            self.assertGreaterEqual(len(motion_data_dict), 1)

            for mag in avg_magnitude_dict.values():
                self.assertIsInstance(mag, float)

            for flow in motion_data_dict.values():
                self.assertIsInstance(flow, np.ndarray)
                self.assertEqual(flow.ndim, 3)  # HxWx2 for flow vectors

    def test_process_optical_flow_insufficient_frames(self):
        with TemporaryDirectory() as input_dir, TemporaryDirectory() as output_dir:
            # Only one frame
            cv2.imwrite(os.path.join(input_dir, "frame_0001.jpg"), self.frame1)
            with self.assertRaises(ValueError):
                process_optical_flow(input_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
