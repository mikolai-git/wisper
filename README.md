# wisper

**A Python library for studying the visual perception of videos.**

wisper breaks a video down into frames and pulls out low-level visual features that shape how a viewer experiences it: **brightness**, **colourfulness**, **motion (optical flow)** and **pacing (shot cuts)**. You can then browse those metrics frame by frame next to the footage, or cluster many videos by their visual profile.

---

## Features

| Module | What it measures |
| --- | --- |
| `pre_processing` | Exports a video as individual JPG frames |
| `brightness_processing` | Per-frame mean brightness, frame-to-frame brightness change, sliding-window averages and pixel-level change heatmaps |
| `colour_processing` | CIELab chroma and Hasler & Süsstrunk colourfulness, normalised to `[0, 1]` |
| `optical_flow` | Dense optical flow magnitude/angle per frame, plus DBSCAN-based moving-object clustering |
| `shot_detection` | Shot boundary detection with [TransNetV2](https://github.com/soCzech/TransNetV2): number of cuts, cuts per minute, frames per cut, inter-cut time |

On top of the library:

- **`run.py`**: a pipeline that processes one video and writes its metrics to CSV files
- **`visualisation.py`**: a Tkinter frame player that plays processed videos side by side with a live plot of the selected metric
- **`clustering/`**: hierarchical (Ward) clustering of a whole set of processed videos by their visual metrics, with plotting scripts

---

## Project structure

```
wisper/
├── wisper/                # Core library (git submodule)
├── run.py                 # Process a single video
├── visualisation.py       # Interactive frame + metric viewer
├── clustering/
│   ├── metric_clustering.py   # Cluster videos → video_clusters.csv
│   └── cluster_anal.py        # Plot metric profiles per cluster
├── tests/                 # Unit tests for each processing module
└── wisper_docs/           # MkDocs documentation site
```

---

## Installation

The core library lives in the `wisper/` submodule, so clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd wisper
```

Create a Python 3.10 environment and install the dependencies:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install opencv-python numpy tqdm pandas matplotlib scipy scikit-learn pillow tensorflow
pip install git+https://github.com/soCzech/TransNetV2.git
```

> `visualisation.py` needs Tkinter. On Debian/Ubuntu: `sudo apt install python3-tk`.

---

## Usage

### 1. Process a video

```bash
python run.py path/to/video.mp4
```

Output goes to `processed_videos/<video_name>/`:

```
processed_videos/<video_name>/
├── <video_name>_raw/              # Extracted frames
├── <video_name>_optical_flow/     # Optical flow visualisations
├── <video_name>_brightness_diff/  # Brightness change heatmaps
├── brightness.csv
├── HS_colourfulness.csv
├── chroma.csv
├── optical_flow_magnitude.csv
├── pacing.csv                     # num_cuts, cuts_per_min, frames_per_cut, inter_cut_time
└── scene_lengths.csv
```

Each per-frame CSV is a single row: the metric name followed by one value per frame.

> Individual pipeline stages are toggled by commenting/uncommenting blocks in `run.py`.

### 2. Explore the results

```bash
python visualisation.py
```

Pick processed videos from the dropdown, choose a metric, and scrub or play through the frames while the metric plot follows along.

### 3. Cluster a collection of videos

Once several videos are in `processed_videos/`:

```bash
python clustering/metric_clustering.py   # → video_clusters.csv
python clustering/cluster_anal.py        # → metrics_by_cluster_beautified.pdf
```

Videos are compared on average brightness, colourfulness, optical flow, cuts per minute and the percentage of short cuts (< 2 s).

### Using the library directly

```python
from wisper import pre_processing, brightness_processing, colour_processing, optical_flow

frames_dir = pre_processing.export_as_frames("video.mp4", "frames/")

brightness = brightness_processing.get_brightness_list(frames_dir)
colour     = colour_processing.process_colour(frames_dir)      # {'average_chroma': [...], 'colorfulness': [...]}
flow_mag, flow_motion = optical_flow.process_optical_flow(frames_dir, "flow_vis/")
```

---

## Tests

```bash
python -m pytest tests/
```

---

## Documentation

Full API docs are built with [MkDocs](https://www.mkdocs.org/):

```bash
pip install mkdocs
cd wisper_docs
mkdocs serve
```

Then open <http://127.0.0.1:8000>.
