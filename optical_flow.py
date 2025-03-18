
from sklearn.cluster import DBSCAN

import os
import cv2
import numpy as np
import gc
from tqdm import tqdm

def process_optical_flow(input_folder, output_folder, subsample_factor=4):
    """
    Computes dense optical flow for a folder of video frames, saves visualizations, and returns motion data.

    Parameters:
        input_folder (str): Path to the folder containing video frames.
        output_folder (str): Path to the folder where optical flow visualizations will be saved.
        subsample_factor (int): Factor by which to subsample motion vectors for storage (default: 4).

    Returns:
        tuple: (dict, dict)
            - avg_magnitude_dict: Dictionary where keys are frame indices and values are the average optical flow magnitude.
            - motion_data_dict: Dictionary where keys are frame indices and values are sampled motion vectors.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Sort frames to process them sequentially
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(frame_files) < 2:
        raise ValueError("Input folder must contain at least two frames.")

    avg_magnitude_dict = {}
    motion_data_dict = {}

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Process frames with tqdm progress bar
    for i in tqdm(range(1, len(frame_files)), desc="Processing optical flow", unit="frame"):
        # Read the next frame
        next_frame = cv2.imread(os.path.join(input_folder, frame_files[i]))
        if next_frame is None:
            print(f"Warning: Frame {frame_files[i]} could not be read. Skipping.")
            continue
        
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)  # Compute the mean magnitude

        # Store average magnitude for each frame
        avg_magnitude_dict[frame_files[i]] = avg_magnitude

        # Subsample motion vectors to reduce memory footprint
        if subsample_factor > 1:
            motion_data = flow[::subsample_factor, ::subsample_factor]  # Downsample motion vectors
        else:
            motion_data = flow  # No subsampling

        motion_data_dict[f"frame_{i}"] = motion_data  # Save sampled motion features

        # Normalize magnitude for visualization
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Create HSV image for visualization
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255  # Set saturation to maximum
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue corresponds to flow direction
        hsv[..., 2] = magnitude_normalized.astype(np.uint8)  # Value corresponds to flow magnitude

        # Convert HSV to BGR for savinga
        flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Save visualization
        output_path = os.path.join(output_folder, f"flow_{i:04d}.jpg")
        cv2.imwrite(output_path, flow_visualization)

        # Explicitly release unused frames every 100 frames to reduce performance overhead
        if i % 100 == 0:
            gc.collect()  # Less frequent garbage collection for better performance

        # Update previous frame
        del prev_gray, prev_frame  # Free memory of the previous frame
        prev_gray = next_gray
        prev_frame = next_frame

    # Final cleanup
    cv2.destroyAllWindows()
    
    return avg_magnitude_dict, motion_data_dict




def detect_objects_with_optical_flow(magnitude, angle, min_magnitude=2.0, eps=5, min_samples=50):
    """
    Detect moving objects based on optical flow magnitude and angle using DBSCAN clustering.

    Parameters:
        magnitude (np.array): Optical flow magnitude (2D array).
        angle (np.array): Optical flow angle (2D array).
        min_magnitude (float): Minimum motion magnitude threshold.
        eps (float): DBSCAN neighborhood size.
        min_samples (int): Minimum number of points for a cluster.

    Returns:
        labels (np.array): Cluster labels (-1 for noise).
        clustered_points (np.array): Coordinates of clustered pixels.
    """
    if magnitude.ndim != 2 or angle.ndim != 2:
        raise ValueError("Magnitude and angle must be 2D arrays.")

    # Get pixels where motion is above a threshold
    motion_mask = magnitude > min_magnitude
    
    # Ensure motion_mask is 2D before extracting indices
    if motion_mask.any():  # Check if there are any valid motion pixels
        y_idx, x_idx = np.nonzero(motion_mask)  # Use np.nonzero instead of np.where
    else:
        return np.array([]), np.array([])  # Return empty arrays if no motion detected

    # Extract motion features: x, y, magnitude, and angle
    motion_features = np.column_stack((x_idx, y_idx, magnitude[motion_mask], angle[motion_mask]))

    # Apply DBSCAN clustering
    if len(motion_features) == 0:  # Prevent error if no features
        return np.array([]), np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(motion_features)
    labels = clustering.labels_

    return labels, motion_features


def visualize_clusters(frame, labels, clustered_points):
    """
    Draw clusters on a frame.
    
    Parameters:
        frame (np.array): Original frame.
        labels (np.array): Cluster labels.
        clustered_points (np.array): Pixel coordinates of clustered motion.

    Returns:
        output_frame (np.array): Frame with detected objects drawn.
    """
    output_frame = frame.copy()
    
    unique_labels = set(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)

    for label, (x, y, _, _) in zip(labels, clustered_points):
        if label == -1:
            continue  # Skip noise points
        color = tuple(map(int, colors[label]))
        cv2.circle(output_frame, (int(x), int(y)), 2, color, -1)

    return output_frame
