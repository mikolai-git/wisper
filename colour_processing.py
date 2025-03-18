import cv2
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm  # Import tqdm for the progress bar

def compute_lab_metrics(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    chroma = np.sqrt(a**2 + b**2)
    avg_chroma = np.mean(chroma)
    sigma_a, sigma_b = np.std(a), np.std(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    colorfulness = np.sqrt(sigma_a**2 + sigma_b**2) + 0.3 * np.sqrt(mean_a**2 + mean_b**2)
    return avg_chroma, colorfulness

def compute_metrics(image):
    """
    Computes various color-related metrics for an image.
    """
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute HSV-based metrics
    avg_hue = np.mean(hsv[:, :, 0])

    # Compute Lab-based metrics
    avg_chroma, colorfulness = compute_lab_metrics(image)

    # Return a dictionary of all metrics
    return {
        'average_hue': avg_hue,
        'average_chroma': avg_chroma,  # CIELAB chroma (color vividness)
        'colorfulness': colorfulness   # Hasler & Süsstrunk colorfulness metric
    }

def process_frames(folder_path):
    all_metrics = OrderedDict({
        'average_chroma': [],
        'colorfulness': []
    })

    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))])

    for frame_file in tqdm(frame_files, desc="Processing colour metrics", unit="frame"):
        file_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Warning: {file_path} could not be read.")
            continue

        avg_chroma, colorfulness = compute_lab_metrics(frame)

        # Apply normalization before storing
        avg_chroma = avg_chroma / 180  # Normalize chroma (assuming range 0-180)
        colorfulness = colorfulness / 150  # Normalize colorfulness (assuming range 0-150)

        all_metrics['average_chroma'].append(round(avg_chroma, 3))
        all_metrics['colorfulness'].append(round(colorfulness, 3))

        del frame  # Release memory for the current frame
        cv2.waitKey(1)  # Allow OpenCV to process internal events

    return all_metrics

def compute_colourfulness(image):
    
    # Convert image to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Extract L, a, and b channels
    L, a, b = cv2.split(lab)
    
    # Hasler & Süsstrunk's colorfulness metric
    sigma_a, sigma_b = np.std(a), np.std(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    colorfulness = np.sqrt(sigma_a**2 + sigma_b**2) + 0.3 * np.sqrt(mean_a**2 + mean_b**2)
    
    return colorfulness

def normalize_colourfulness(colourfulness):
    
    min_chroma = 0
    max_chroma = 180
    
    return [round(b / max_chroma, 3) for b in colourfulness]

def normalize_chroma(chroma_values):
    
    min_chroma = 0
    max_chroma = 180
    
    return [round(b / max_chroma, 3) for b in chroma_values]


def compute_average_chroma(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Extract L, a, and b channels
    L, a, b = cv2.split(lab)
    
    # Compute mean chroma (color vividness)
    chroma = np.sqrt(a**2 + b**2)
    avg_chroma = np.mean(chroma)
    
    return avg_chroma

def normalize_metrics(all_metrics):
    """
    Normalizes the computed metrics to be within [0,1] range.
    """
    # Define known value ranges for fixed metrics
    value_ranges = {
        'colorfulness': (0, 150)  # Based on empirical values from Hasler & Süsstrunk
    }
    
    # Normalize all metrics
    normalized_metrics = OrderedDict()
    for metric, values in all_metrics.items():
        min_val, max_val = value_ranges.get(metric, (min(values), max(values)))
        if metric == 'average_hue':
            # Hue remains unchanged
            normalized_metrics[metric] = [round(v, 3) for v in values]
        else:
            normalized_metrics[metric] = [
                round((v - min_val) / (max_val - min_val), 3) if max_val > min_val else 0 for v in values
            ]
    
    return normalized_metrics

def compute_average_metrics(all_metrics):
    """
    Computes the average of all metrics across all processed frames.
    """
    average_metrics = OrderedDict()
    
    for metric, values in all_metrics.items():
        average_metrics[metric] = np.mean(values)
    
    return average_metrics

def process_colour(folder_path):
    """
    Main function to process color metrics for all images in a folder.
    """
    # Process frames and get the metrics for each frame
    all_metrics = process_frames(folder_path)
    
    # Normalize the metrics
    #normalized_metrics = normalize_metrics(all_metrics)
    
    # Compute average metrics from normalized values
    #average_metrics = compute_average_metrics(normalized_metrics)
    
    return all_metrics
