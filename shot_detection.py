from transnetv2 import TransNetV2
import cv2
import numpy as np
import pandas as pd
import os

model = TransNetV2()

def shot_prediction(video_path, save_path, hard_cut_threshold=0.5, soft_cut_threshold=0.5):
    """
    Predicts the shot boundaries (cuts) in a video using the TransNetV2 model and 
    saves the results into three CSV files.

    Parameters:
        video_path (str): Path to the input video file.
        save_path (str): Directory where CSV files should be saved.
        hard_cut_threshold (float): Probability threshold to classify a hard cut.
        soft_cut_threshold (float): Probability threshold to classify a soft cut.

    Returns:
        list of tuples: Start and end frame indices of predicted shot cuts.
    """
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)

    # Ensure all_frame_predictions is a 2D array (N, 2)
    if all_frame_predictions.ndim == 1:  
        all_frame_predictions = all_frame_predictions.reshape(-1, 1)  # Make it (N,1)
        all_frame_predictions = np.hstack((all_frame_predictions, np.zeros((all_frame_predictions.shape[0], 1))))  # Add soft cut column

    # Extract predicted cut scenes
    predicted_cut_scenes = model.predictions_to_scenes(single_frame_predictions)

    # Extract hard cuts (frames where hard cut probability > threshold)
    hard_cuts = np.where(all_frame_predictions[:, 0] > hard_cut_threshold)[0].tolist()

    # Extract soft cuts (frames where soft cut probability > threshold)
    soft_cuts = np.where(all_frame_predictions[:, 1] > soft_cut_threshold)[0].tolist()

    print(f"Predicted cut scenes: {predicted_cut_scenes}")  # Used for scene length analysis
    print(f"Predicted hard cuts: {hard_cuts}")
    print(f"Predicted soft cuts: {soft_cuts}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save predicted scenes to CSV
    scenes_df = pd.DataFrame(predicted_cut_scenes, columns=["Start Frame", "End Frame"])
    scenes_df.to_csv(os.path.join(save_path, "predicted_scenes.csv"), index=False)

    # Save hard cuts to CSV
    hard_cuts_df = pd.DataFrame(hard_cuts, columns=["Frame Index"])
    hard_cuts_df.to_csv(os.path.join(save_path, "hard_cuts.csv"), index=False)

    # Save soft cuts to CSV
    soft_cuts_df = pd.DataFrame(soft_cuts, columns=["Frame Index"])
    soft_cuts_df.to_csv(os.path.join(save_path, "soft_cuts.csv"), index=False)

    return predicted_cut_scenes

def get_number_of_cuts(video_path, save_path):
    """
    Computes the number of cuts (shot transitions) in a video.

    Parameters:
        video_path (str): Path to the input video file.
        save_path (str): Directory where CSV files should be saved.

    Returns:
        int: The total number of cuts detected in the video.
    """
    number_of_cuts = len(shot_prediction(video_path, save_path))
    return number_of_cuts

def get_pacing_data(video_path, save_path):
    """
    Computes pacing metrics for the given video.

    Parameters:
        video_path (str): Path to the input video file.
        save_path (str): Directory where CSV files should be saved.

    Returns:
        tuple: (num_cuts, cuts_per_min, frames_per_cut, inter_cut_time)
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the duration in seconds
    duration_in_secs = frame_count / fps
    duration_in_mins = duration_in_secs / 60

    # Close the video file
    cap.release()

    num_cuts = get_number_of_cuts(video_path, save_path)

    cuts_per_min = num_cuts / duration_in_mins
    frames_per_cut = frame_count / num_cuts
    inter_cut_time = duration_in_secs / num_cuts

    return num_cuts, cuts_per_min, frames_per_cut, inter_cut_time
