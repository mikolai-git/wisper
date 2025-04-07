from wisper import pre_processing, brightness_processing, optical_flow, colour_processing
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os
import csv
import scene_classifier
import scene_length_dist

# # Auxiliary function defninitions (May be factored out eventually)
# def save_change_heatmaps(change_heatmaps, output_folder_name):
#     """
#     Saves each heatmap in the change_heatmaps list as an image file in a specified folder.

#     Args:
#         change_heatmaps (list of ndarray): List of heatmaps (each as an ndarray).
#         output_folder_name (str): Name of the output folder. Defaults to 'brightness_heatmaps'.
#     """
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder_name, exist_ok=True)

#     # Save each heatmap as an image
#     for i, heatmap in enumerate(tqdm(change_heatmaps, desc="Saving heatmaps")):
#         output_path = os.path.join(output_folder_name, f"heatmap_{i + 1:04d}.jpg")
#         cv2.imwrite(output_path, heatmap)
        
# def run_optical_flow_object_detection(input_folder, clustered_output_folder, motion_data_dict):
#     """
#     Runs optical flow processing, object detection using DBSCAN, and saves cluster visualizations.

#     Parameters:
#         input_folder (str): Path to the folder containing video frames.
#         output_folder (str): Path to save optical flow visualizations.
#         clustered_output_folder (str): Path to save object detection visualizations.
#         motion_data_dict (dict): Dictionary containing motion data for clustering.
#     """
#     os.makedirs(clustered_output_folder, exist_ok=True)  # Ensure output directory exists

#     # Sort frames to process them sequentially
#     frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

#     # Initialize tqdm progress bar
#     with tqdm(total=len(frame_files) - 1, desc="Processing Object Detection", unit="frame") as pbar:
#         for i in range(1, len(frame_files)):
#             frame_id = f"frame_{i}"
#             motion_data = motion_data_dict.get(frame_id, np.empty((0, 4)))  # Retrieve motion data

#             if motion_data.size == 0:
#                 pbar.update(1)  # Update progress even if frame is skipped
#                 continue  # Skip if no motion data

#             # Get image dimensions from motion data
#             h, w = int(motion_data[:, 1].max()) + 1, int(motion_data[:, 0].max()) + 1  
#             magnitude_2d = np.zeros((h, w), dtype=np.float32)
#             angle_2d = np.zeros((h, w), dtype=np.float32)

#             # Convert motion data back to 2D format
#             for x, y, mag, ang in motion_data:
#                 magnitude_2d[int(y), int(x)] = mag
#                 angle_2d[int(y), int(x)] = ang

#             # Now pass 2D arrays to detect_objects_with_optical_flow
#             labels, clustered_points = optical_flow.detect_objects_with_optical_flow(magnitude_2d, angle_2d)


#             # Load original frame
#             frame_path = os.path.join(input_folder, frame_files[i])
#             frame = cv2.imread(frame_path)

#             # Step 4: Visualize detected objects
#             clustered_frame = optical_flow.visualize_clusters(frame, labels, clustered_points)

#             # Save visualization
#             output_path = os.path.join(clustered_output_folder, f"clustered_{i:04d}.jpg")
#             cv2.imwrite(output_path, clustered_frame)

#             # Update progress bar
#             pbar.update(1)

#     print("Optical flow and object detection processing completed.")
        

# import os
# import csv

# def save_metrics_to_csv(folder_path, save_path, optical_flow_mag_data):
#     # Ensure save_path exists
#     os.makedirs(save_path, exist_ok=True)

#     # Process the frames and get the metrics from the colour_processing function
#     all_metrics = colour_processing.process_colour(folder_path)
    
#     # Ensure all_metrics is a dictionary and contains lists
#     if not isinstance(all_metrics, dict):
#         raise TypeError(f"Expected all_metrics to be a dict, got {type(all_metrics)}: {all_metrics}")
    
#     for key, values in all_metrics.items():
#         if not isinstance(values, list):
#             raise TypeError(f"Expected {key} values to be a list, got {type(values)}: {values}")

#     # Get brightness values
#     brightness_data = brightness_processing.get_brightness_list(folder_path)
#     if not isinstance(brightness_data, dict):
#         raise TypeError(f"Expected brightness_data to be a dict, got {type(brightness_data)}: {brightness_data}")

#     # Get the list of frame filenames in sorted order
#     frame_files = sorted(
#         [filename for filename in os.listdir(folder_path) 
#          if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
#     )

#     frame_columns = [f'frame_{i+1}' for i in range(len(frame_files))]  # frame_1, frame_2, ..., frame_n
#     header = ['frame'] + frame_columns  # No 'average' here, as averages are computed separately

#     # Helper function to write a single metric to CSV
#     def write_metric_to_csv(metric_name, metric_values):
#         file_path = os.path.join(save_path, f"{metric_name}.csv")

#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(header)  # Write header
#             writer.writerow([metric_name] + metric_values)  # Write metric values

#     # Extract chroma and colorfulness from all_metrics
#     chroma_values = all_metrics.get('average_chroma', [])
#     colourfulness_values = all_metrics.get('colorfulness', [])

#     # Write chroma and colorfulness separately
#     write_metric_to_csv("chroma", chroma_values)
#     write_metric_to_csv("HS_colourfulness", colourfulness_values)

#     # Handle brightness (extract per-frame values, ignoring the average if present)
#     brightness_values = list(brightness_data.values())  # Extract only per-frame values
#     write_metric_to_csv("brightness", brightness_values)

#     # Ensure optical flow data is a dictionary
#     if not isinstance(optical_flow_mag_data, dict):
#         raise TypeError(f"Expected optical_flow_mag_data to be a dict, got {type(optical_flow_mag_data)}: {optical_flow_mag_data}")

#     optical_flow_values = [optical_flow_mag_data[f] for f in frame_files if f in optical_flow_mag_data]  # Extract per-frame values
#     write_metric_to_csv("optical_flow_magnitude", optical_flow_values)

#     print(f"Metrics saved in {save_path}")


    
# def save_pacing_to_csv(input_path, save_path):
    
#     print("Running shot detection")
    
#     num_cuts, cuts_per_min, frames_per_cut, inter_cut_time = shot_detection.get_pacing_data(input_path, save_path)
    
#     # Define the CSV file path
#     csv_file_path = f"{save_path}/pacing.csv"
    
#     # Write data to CSV
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write header
#         writer.writerow(['num_cuts', 'cuts_per_min', 'frames_per_cut', 'inter_cut_time'])
#         # Write data row
#         writer.writerow([num_cuts, cuts_per_min, frames_per_cut, inter_cut_time])

#     print(f"Pacing data saved to {csv_file_path}")
    



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


# # Export the video as individual frames, save to folder called "video_name_raw" inside folder_name
base_dir = folder_path + "/" + folder_name
raw_save_dir = base_dir + "_raw"
# pre_processing.export_as_frames(video_filepath, raw_save_dir)

# # Process optical flow, save to folder called "video_name_optical_flow" inside folder_name
# print("Processing optical flow")
# save_dir = base_dir + "_optical_flow"
# optical_flow_mag_data, optical_flow_movement = optical_flow.process_optical_flow(raw_save_dir, save_dir)


save_dir = folder_path
# save_pacing_to_csv(video_filepath, save_dir)

# # Optical flow DBSCAN
# save_dir = base_dir + "_optical_flow_clustering"
# run_optical_flow_object_detection(raw_save_dir, save_dir, optical_flow_movement)

# #Process all numeric metrics and save them to a csv file metrics.csv AND optical flow
# save_dir = folder_path
# optical_flow_output_folder = base_dir + "_optical_flow"
# save_metrics_to_csv(raw_save_dir, save_dir, optical_flow_mag_data)

# # Process brightness difference frames, save to folder called "video_name_brightness_diff" inside folder_name
# save_dir = base_dir + "_brightness_diff"
# change_heatmaps = brightness_processing.calculate_pixel_brightness_change_heatmaps(raw_save_dir)
# save_change_heatmaps(change_heatmaps, save_dir)


# csv_path = f"{folder_path}/predicted_scenes.csv"  # Replace with your actual file path
# csv_output_path = f"{folder_path}/all_cuts.csv"  # Replace with your actual file pathall_cuts.csv"

# scene_classifier.analyze_and_save_cuts(csv_path, csv_output_path)

csv_path = f"{folder_path}/predicted_scenes.csv"  # Replace with your actual file path
csv_output_path = f"{folder_path}/scene_lengths.csv"  # Replace with your actual file pathall_cuts.csv"

scene_length_dist.get_scene_length_dist(csv_path, csv_output_path)













