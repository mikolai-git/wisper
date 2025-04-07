import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cluster results
df = pd.read_csv("video_clusters.csv")

# Metrics and display names
metrics = [
    "avg_brightness",
    "avg_colourfulness",
    "norm_optical_flow",
    "norm_cuts_per_min",
    "norm_short_cut_percent"
]

metric_display_names = [
    "Brightness", "Colourfulness", "Optical Flow",
    "Cuts/Min", "Short Cuts %"
]

# Kei Ito / Okabe-Ito palette
okabe_ito_colors = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00"
]

# Group by cluster
grouped = df.groupby("cluster")[metrics]
means = grouped.mean()
stds = grouped.std()

clusters = sorted(means.index)
x = np.arange(len(clusters))  # clusters along x-axis
num_metrics = len(metrics)
bar_width = 0.13  # smaller bar width for spacing

# Setup figure
fig, ax = plt.subplots(figsize=(12, 6))

# Draw bars
for i, (metric, display_name) in enumerate(zip(metrics, metric_display_names)):
    offset = (i - num_metrics / 2) * (bar_width + 0.01) + bar_width / 2  # add spacing
    ax.bar(
        x + offset,
        means[metric],
        yerr=stds[metric],
        capsize=4,
        width=bar_width,
        label=display_name,
        color=okabe_ito_colors[i % len(okabe_ito_colors)]
    )

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels([f"Cluster {c}" for c in clusters], fontsize=11)
ax.set_ylabel("Metric Value", fontsize=12, fontweight='bold')

ax.legend(title="Metric", fontsize=10, title_fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Remove spines and borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
plt.savefig("metrics_by_cluster_beautified.pdf", format="pdf", bbox_inches='tight')
plt.show()
