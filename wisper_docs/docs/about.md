# Documentation

## Pre-processing

    def export_as_frames(video_filepath, save_dir=None):
    """
    Extracts frames from a video file and saves them as individual JPG images.

    Parameters:
    - video_filepath (str): Path to the video file.
    - save_dir (str): Directory to save the extracted frames. If None, a folder 
                      named after the video file will be created in the current directory.

    Returns:
    - str: Path to the folder where the frames are saved.
    """

## Brightness processing

    def calculate_frame_brightness(frame):
    """
    Calculate the average brightness of a frame by converting it to grayscale.

    Args:
        frame (ndarray): The frame image to calculate brightness for.

    Returns:
        float: The average brightness of the frame.
    """

#

    def calculate_frame_brightness_diff(frame1, frame2):
    """
    Calculate the difference in brightness between two frames.

    Args:
        frame1 (ndarray): The first frame.
        frame2 (ndarray): The second frame.

    Returns:
        float: The difference in average brightness between the two frames.
    """

#

    def calculate_average_brightness_of_frames(frames):
    """
    Calculate the average brightness over a series of frames.

    Args:
        frames (list of ndarray): List of frames for which to calculate brightness.

    Returns:
        float: The average brightness of all frames in the li

#

    def get_brightness_list(folder_path):
    """
    Reads in frames from a folder and returns a list of average brightness values for each frame.

    Args:
        folder_path (str): Path to the folder containing frame images.

    Returns:
        list: List of average brightness values for each frame.
    """

#

    def get_brightness_diff_list(folder_path):
    """
    Reads in frames from a folder and returns a list of brightness differences between consecutive frames.

    Args:
        folder_path (str): Path to the folder containing frame images.

    Returns:
        list: List of brightness differences between consecutive frames.
    """

#

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

## Shot Detection

    def shot_prediction(video_path):
    """
    Predicts the shot boundaries (cuts) in a video using the TransNetV2 model.

    Parameters:
        video_path (str): Path to the input video file.

    Returns:
        list of tuples: A list of tuples, where each tuple represents the start 
        and end frame indices of a predicted shot in the video.
    """
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    predicted_cut_scenes = model.predictions_to_scenes(single_frame_predictions)
    return predicted_cut_scenes
    """

#

    def get_number_of_cuts(video_path):
    """
    Computes the number of cuts (shot transitions) in a video.

    Parameters:
        video_path (str): Path to the input video file.

    Returns:
        int: The total number of cuts detected in the video.
    """
    number_of_cuts = len(shot_prediction(video_path))

## Optical Flow

    def process_optical_flow(input_folder, output_folder):
    """
    Computes dense optical flow for a folder of video frames and saves visualizations.

    Parameters:
        input_folder (str): Path to the folder containing video frames.
        output_folder (str): Path to the folder where optical flow visualizations will be saved.

    Returns:
        List[dict]: A list of dictionaries containing optical flow magnitude and angle for each frame.
    """

