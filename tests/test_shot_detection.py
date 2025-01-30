import os
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
import sys
from transnetv2 import TransNetV2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wisper.shot_detection import *

class TestShotDetectionFunctions(unittest.TestCase):
    def setUp(self):
        # Mock the TransNetV2 model
        self.mock_model = MagicMock()
        # Patch the model globally in the module
        patcher = patch('video_processing.model', self.mock_model)
        self.addCleanup(patcher.stop)
        self.mock_model_patch = patcher.start()

    def test_shot_prediction(self):
        # Mock return values for predict_video and predictions_to_scenes
        self.mock_model.predict_video.return_value = (
            None,  # video_frames is not used in shot_prediction
            [0, 1, 0, 1],  # Example single frame predictions
            None,  # all_frame_predictions is not used in shot_prediction
        )
        self.mock_model.predictions_to_scenes.return_value = [(0, 10), (11, 20)]

        # Test shot_prediction
        video_path = "mock_video.mp4"
        predicted_scenes = shot_prediction(video_path)
        self.assertEqual(predicted_scenes, [(0, 10), (11, 20)])

        # Assert model methods were called with correct parameters
        self.mock_model.predict_video.assert_called_once_with(video_path)
        self.mock_model.predictions_to_scenes.assert_called_once_with([0, 1, 0, 1])

    def test_get_number_of_cuts(self):
        # Mock the shot_prediction function
        with patch('video_processing.shot_prediction', return_value=[(0, 10), (11, 20), (21, 30)]) as mock_shot_prediction:
            video_path = "mock_video.mp4"
            number_of_cuts = get_number_of_cuts(video_path)
            
            # Assert the number of cuts is correct
            self.assertEqual(number_of_cuts, 3)

            # Assert shot_prediction was called with the correct parameterasawwas
            mock_shot_prediction.assert_called_once_with(video_path)

if __name__ == "__main__":
    unittest.main()

