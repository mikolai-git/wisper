import os
import cv2
import numpy as np
from tqdm import tqdm

def mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def export_as_frames(video_filepath, save_dir=None, threshold=10):
    """
    Extracts frames from a video file and saves them as individual JPG images, 
    skipping duplicate frames.

    Parameters:
    - video_filepath (str): Path to the video file.
    - save_dir (str): Directory to save the extracted frames. If None, a folder 
                      named after the video file will be created in the current directory.
    - threshold (float): MSE threshold to consider a frame as a duplicate. Lower values = stricter filtering.

    Returns:
    - str: Path to the folder where the frames are saved.
    """
    
    # Get the video name (without extension) from the file path
    video_name = os.path.splitext(os.path.basename(video_filepath))[0]
    
    # Set default save location if save_dir is not provided
    if save_dir is None:
        save_dir = video_name  # Create a folder with the video name in the current directory
    
    # Create the folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created folder for video frames: {save_dir}")
    
    # Open the video file
    print(f"Opening video file: {video_filepath}...")
    cap = cv2.VideoCapture(video_filepath)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    prev_frame = None
    frame_idx = 0

    for frame_count in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break  # Stop if there are no more frames
        
        # Convert to grayscale for better comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if it's the first frame or different from the previous frame
        if prev_frame is None or mse(prev_frame, gray_frame) > threshold:
            # Save the frame as a JPG file in the specified folder
            frame_filename = f"{save_dir}/frame_{frame_idx:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1  # Increment frame index only for unique frames
            
            # Update previous frame
            prev_frame = gray_frame

    # Release the video capture object
    cap.release()
    print(f"Frames extracted (duplicates removed) and saved to '{save_dir}'.")
    
    return save_dir
