import os
import subprocess
from natsort import natsorted  # Natural sorting

# Define the folder containing the videos
video_folder = "videos/disney_songs_in_order"

# Get all files in the folder and sort them using natural order
video_files = natsorted(
    [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
)

print(video_files)

# # Iterate over sorted files
# for filename in video_files:
#     file_path = os.path.join(video_folder, filename)
    
#     print(f"Processing: {file_path}")
    
#     # Run the script with the video file as an argument
#     subprocess.run(["python", "run.py", file_path], check=True)
