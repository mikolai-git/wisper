import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.cluster.hierarchy import set_link_color_palette
from sklearn.preprocessing import StandardScaler

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def read_flat_csv_row(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.shape[0] == 1:
            values = df.iloc[0].values[1:]  # Skip the frame column
            return np.array(values, dtype=float)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
    return None

def extract_scalar(csv_path, column_name):
    try:
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return float(df[column_name].iloc[0])
    except Exception as e:
        print(f"Failed to extract {column_name} from {csv_path}: {e}")
    return np.nan

def calculate_short_cut_percentage(csv_path, threshold=2.0):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or df.columns[0].lower() != "scene length (seconds)":
            return np.nan
        lengths = df[df.columns[0]]
        total_cuts = len(lengths)
        short_cuts = (lengths < threshold).sum()
        return (short_cuts / total_cuts) * 100 if total_cuts > 0 else np.nan
    except Exception:
        return np.nan

def main():
    base_folder = "processed_videos"
    subfolders = sorted(
        [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))],
        key=natural_sort_key
    )

    video_labels = []
    brightness_means = []
    colourfulness_means = []
    optical_flow_means = []
    cuts_per_min = []
    short_cut_percents = []

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        brightness_path = os.path.join(folder_path, "brightness.csv")
        colour_path = os.path.join(folder_path, "HS_colourfulness.csv")
        flow_path = os.path.join(folder_path, "optical_flow_magnitude.csv")
        pacing_path = os.path.join(folder_path, "pacing.csv")
        scene_lengths_path = os.path.join(folder_path, "scene_lengths.csv")

        if not (os.path.exists(brightness_path) and os.path.exists(colour_path) and os.path.exists(flow_path) and os.path.exists(pacing_path) and os.path.exists(scene_lengths_path)):
            print(f"⚠️ Skipping {subfolder}: missing one or more required CSVs.")
            continue

        brightness = read_flat_csv_row(brightness_path)
        colourfulness = read_flat_csv_row(colour_path)
        flow = read_flat_csv_row(flow_path)

        if brightness is None or colourfulness is None or flow is None:
            print(f"⚠️ Skipping {subfolder}: invalid brightness/colour/flow")
            continue

        flow = flow[np.isfinite(flow)]
        if len(flow) == 0:
            print(f"⚠️ Skipping {subfolder}: all optical flow values are non-finite.")
            continue

        b_mean = np.mean(brightness)
        c_mean = np.mean(colourfulness)
        f_mean = np.mean(flow)

        cpm = extract_scalar(pacing_path, "cuts_per_min")
        scp = calculate_short_cut_percentage(scene_lengths_path)

        if all(np.isfinite(x) for x in [b_mean, c_mean, f_mean, cpm, scp]):
            brightness_means.append(b_mean)
            colourfulness_means.append(c_mean)
            optical_flow_means.append(f_mean)
            cuts_per_min.append(cpm)
            short_cut_percents.append(scp)
            video_labels.append(subfolder)
        else:
            print(f"⚠️ Skipping {subfolder}: non-finite metric values.")

    if not video_labels:
        print("❌ No valid data found.")
        return

    # Normalize optical flow, cuts per minute, and short cuts %
    optical_flow_norm = np.array(optical_flow_means) / np.max(optical_flow_means)
    cuts_per_min_norm = np.array(cuts_per_min) / np.max(cuts_per_min)
    short_cut_percents_norm = np.array(short_cut_percents) / np.max(short_cut_percents)

    # Combine all features
    data = np.column_stack((
        brightness_means,
        colourfulness_means,
        optical_flow_norm,
        cuts_per_min_norm,
        short_cut_percents_norm
    ))

    # Standardize for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Hierarchical clustering
    linkage_matrix = linkage(data_scaled, method='ward')

    # Clean x-axis labels
    def clean_label(label):
        cleaned = re.sub(r'^\d+_', '', label)
        return cleaned.replace('_', ' ').title()

    cleaned_labels = [clean_label(label) for label in video_labels]

    # Okabe & Ito palette
    okabe_ito_colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000"
    ]

    num_clusters = 2  # or any t
    cluster_assignments = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    def color_func(link_id):
        return okabe_ito_colors[link_id % len(okabe_ito_colors)]
    
    set_link_color_palette(["#0072B2", "#D55E00", "#009E73"])  # Blue, Orange

    # Force exactly 2 clusters with color threshold
    last_merge_dist = linkage_matrix[-(num_clusters - 1), 2]
    color_threshold = last_merge_dist - 1e-5  # Slightly below the last merge

    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        labels=cleaned_labels,
        leaf_rotation=90,
        color_threshold=color_threshold,  # Ensures only 2 colors
    )
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Assign clusters (optional: choose t)
    cluster_assignments = fcluster(linkage_matrix, t=2, criterion='maxclust')

    # 3D Plot (first 3 metrics)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(cluster_assignments.astype(int) % 10)

    ax.scatter(
        brightness_means, colourfulness_means, optical_flow_norm,
        c=colors, s=80, edgecolor='k'
    )

    for i, label in enumerate(video_labels):
        ax.text(brightness_means[i], colourfulness_means[i], optical_flow_norm[i], label, fontsize=8)

    ax.set_xlabel("Avg Brightness")
    ax.set_ylabel("Avg Colourfulness")
    ax.set_zlabel("Norm Optical Flow")
    ax.set_title("Video Clusters in 3D Metric Space (Top 3 Features)")
    plt.tight_layout()
    plt.show()
    
    # Save clustering results to CSV
    output_df = pd.DataFrame({
        "video_label": video_labels,
        "cluster": cluster_assignments, 
        "avg_brightness": brightness_means,
        "avg_colourfulness": colourfulness_means,
        "norm_optical_flow": optical_flow_norm,
        "norm_cuts_per_min": cuts_per_min_norm,
        "norm_short_cut_percent": short_cut_percents_norm
    })

    output_df.to_csv("video_clusters.csv", index=False)
print("✅ Cluster results saved to video_clusters.csv")

if __name__ == "__main__":
    main()
