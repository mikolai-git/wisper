from wisper import pre_processing, brightness_processing, optical_flow
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os

# Auxiliary function defninitions (May be factored out eventually)
def save_change_heatmaps(change_heatmaps, output_folder_name):
    """
    Saves each heatmap in the change_heatmaps list as an image file in a specified folder.

    Args:
        change_heatmaps (list of ndarray): List of heatmaps (each as an ndarray).
        output_folder_name (str): Name of the output folder. Defaults to 'brightness_heatmaps'.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_name, exist_ok=True)

    # Save each heatmap as an image
    for i, heatmap in enumerate(tqdm(change_heatmaps, desc="Saving heatmaps")):
        output_path = os.path.join(output_folder_name, f"heatmap_{i + 1:04d}.jpg")
        cv2.imwrite(output_path, heatmap)
        
        
# Start of program proper

if len(sys.argv) != 2:
    print("Usage: python script.py <video_filepath>")
    sys.exit(1)

video_filepath = sys.argv[1]

if not os.path.exists(video_filepath):
    print(f"Error: File '{video_filepath}' not found.")
    sys.exit(1)
    
# Save the video name, set base folder name
video_name = os.path.basename(video_filepath)
folder_name = os.path.splitext(video_name)[0]  # Removes the file extension

# Create the main folder of the video, set base folder names
root_folder = os.getcwd() + "/processed_videos"
folder_path = os.path.join(root_folder, folder_name)
os.makedirs(folder_path, exist_ok=True)


# Export the video as individual frames, save to folder called "video_name_raw" inside folder_name
base_dir = folder_path + "/" + folder_name
raw_save_dir = base_dir + "_raw"
pre_processing.export_as_frames(video_filepath, raw_save_dir)

# Process brightness difference frames, save to folder called "video_name_brightness_diff" inside folder_name
save_dir = base_dir + "_brightness_diff"
change_heatmaps = brightness_processing.calculate_pixel_brightness_change_heatmaps(raw_save_dir)
save_change_heatmaps(change_heatmaps, save_dir)

# Process optical flow, save to folder called "video_name_optical_flow" inside folder_name
save_dir = base_dir + "_optical_flow"
optical_flow.process_optical_flow(raw_save_dir, save_dir)







