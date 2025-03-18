import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
from collections import OrderedDict

def calculate_frame_brightness_diff(frame1, frame2):
    """
    Calculate the difference in brightness between two frames.

    Args:
        frame1 (ndarray): The first frame.
        frame2 (ndarray): The second frame.

    Returns:
        float: The difference in average brightness between the two frames.
    """
    brightness1 = calculate_frame_brightness(frame1)
    brightness2 = calculate_frame_brightness(frame2)
    return brightness1 - brightness2

def calculate_frame_brightness(frame):
    """
    Calculate the brightness of a single frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_frame)

def normalize_brightness(brightness_values):
    """
    Normalize brightness values between 0 and 1 based on the theoretical min (0) and max (255).
    """
    min_brightness = 0
    max_brightness = 255
    
    return [round(b / max_brightness, 3) for b in brightness_values]

def calculate_average_brightness_of_frames(frames):
    """
    Calculate the average brightness over a series of frames.
    """
    if not frames:
        return 0
    
    total_brightness = sum(calculate_frame_brightness(frame) for frame in frames)
    return total_brightness / len(frames)

def get_brightness_list(folder_path):
    """
    Reads in frames from a folder and returns an OrderedDict of normalized brightness values for each frame,
    with the average brightness included at the end.
    """
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    
    brightness_values = []
    frame_to_brightness = OrderedDict()  # Maps filenames to brightness

    for frame_file in tqdm(frame_files, desc="Calculating brightness metrics"):
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: {frame_path} could not be read.")
            continue

        # Calculate brightness for the current frame
        brightness = calculate_frame_brightness(frame)
        brightness_values.append(brightness)
        frame_to_brightness[frame_file] = brightness  # Store brightness with filename

        del frame  # Release memory for the current frame
        cv2.waitKey(1)  # Allow OpenCV to process internal events

    # Normalize brightness values
    normalized_brightness = normalize_brightness(brightness_values)

    # Update brightness dictionary with normalized values
    for i, frame_file in enumerate(frame_to_brightness.keys()):
        frame_to_brightness[frame_file] = round(normalized_brightness[i], 3)

    # Calculate and store average brightness
    average_brightness = np.mean(normalized_brightness) if normalized_brightness else 0
    frame_to_brightness['average'] = round(average_brightness, 3)

    return frame_to_brightness

def get_sliding_window_brightness_list(folder_path, window_size=100):
    """
    Reads in frames from a folder and returns a list of average brightness values 
    over a sliding window of frames.

    Args:
        folder_path (str): Path to the folder containing frame images.
        window_size (int): Size of the sliding window. Defaults to 100.

    Returns:
        list: List of average brightness values for each sliding window of frames.
    """
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    frame_array = []

    for frame_file in tqdm(frame_files, desc="Calculating sliding window brightness"):
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        frame_array.append(frame)

    if len(frame_array) < window_size:
        return []

    sliding_window_avg_list = [
        calculate_average_brightness_of_frames(frame_array[i:i+window_size])
        for i in range(len(frame_array) - window_size + 1)
    ]
    return sliding_window_avg_list

def calculate_pixel_brightness_change_heatmaps(folder_path):
    """
    Calculate a pixel-wise "change heatmap" between consecutive frames in a folder.

    Args:
        folder_path (str): Path to the folder containing frame images.

    Returns:
        list of ndarray: List of change heatmaps (each is an ndarray of brightness changes for each pixel).
    """
    # Get sorted list of all frame files in the folder
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    frame_array = []

    # Read each frame and store it in the array
    for frame_file in tqdm(frame_files, desc="Generating change heatmaps"):
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_array.append(gray_frame)
    
    if len(frame_array) < 2:
        print("Not enough frames to generate change heatmaps.")
        return []

    # List to store heatmaps
    heatmaps = []

    # Calculate change heatmap between consecutive frames
    for i in range(len(frame_array) - 1):
        # Absolute difference between consecutive frames
        heatmap = cv2.absdiff(frame_array[i], frame_array[i + 1])
        heatmaps.append(heatmap)

    return heatmaps
