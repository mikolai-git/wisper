import tkinter as tk
from tkinter import filedialog
import csv
from PIL import Image, ImageTk
import os
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import re

def natural_sort_key(s):
        """Sorts strings in a human-friendly way, handling mixed numbers and text."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class FramePlayer:
        
    def __init__(self, root, parent_directory):
        
        self.num_videos_to_display = 2
        
        self.root = root
        self.root.title("Frame Player")
        self.parent_directory = parent_directory
        
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas1 = tk.Canvas(self.frame)
        self.canvas1.grid(row=0, column=0, sticky="nsew")
        self.canvas2 = tk.Canvas(self.frame)
        self.canvas2.grid(row=0, column=1, sticky="nsew")
        # self.canvas3 = tk.Canvas(self.frame)
        # self.canvas3.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        
        # Add matplotlib figure with padding and reduced size
        self.fig, self.ax = plt.subplots(figsize=(6, 1.5))  # Smaller figure size
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Metrics Over Time", fontsize=5)
        self.ax.set_xlabel("Frame", fontsize=4)
        self.ax.set_ylabel("Value", fontsize=4)
        self.ax.tick_params(axis='both', labelsize=4)
        self.line, = self.ax.plot([], [], "r-", linewidth=0.5)  # Thinner line
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_plot.get_tk_widget().pack(fill=tk.X, expand=False, padx=20, pady=20)
        
        self.btn_play_pause = tk.Button(root, text="Play", command=self.toggle_playback)
        self.btn_play_pause.pack()
        
        self.dropdown_label = tk.Label(root, text="Select Dataset Folder:")
        self.dropdown_label.pack()
        
        self.selected_metric = None  # Stores the currently selected metric

        # Dropdown for selecting metrics
        self.metric_var = tk.StringVar()
        self.metric_dropdown = ttk.Combobox(root, textvariable=self.metric_var, state="readonly")
        self.metric_dropdown.pack()
        self.metric_dropdown.bind("<<ComboboxSelected>>", self.on_metric_selected)
        
        self.folder_var = tk.StringVar()
        self.folder_dropdown = ttk.Combobox(root, textvariable=self.folder_var, state="readonly")
        self.folder_dropdown.pack()
        self.folder_dropdown.bind("<<ComboboxSelected>>", self.load_selected_folder)
        
        
        self.slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_frame, length=800)
        self.slider.pack(fill=tk.X)
        
        self.folders = [[], [], []]
        self.frame_indices = [0, 0, 0]
        self.playing = False
        self.metrics_data = {}
        self.loaded_images = [[], [], []]  # Store preloaded images for fast display
        
        self.root.bind("<Configure>", self.resize_canvases)
        self.populate_dropdown()

        # Fixed size for metrics container
        self.metrics_frame = tk.Frame(self.frame, width=250, height=200)
        self.metrics_frame.grid(row=0, column=2, sticky="n")
        self.metrics_frame.grid_propagate(False)  # Prevent resizing based on content
        
        # Frame for average metrics
        self.avg_metrics_frame = tk.Frame(self.frame, width=250, height=200)
        self.avg_metrics_frame.grid(row=1, column=2, sticky="n")
        self.avg_metrics_frame.grid_propagate(False)  # Prevent resizing based on content
        
        # Label for metrics display
        self.metrics_label = tk.Label(self.metrics_frame, font=("Helvetica", 12), anchor="w", justify="left", width=15)
        self.metrics_label.pack(side="left", fill=tk.BOTH, expand=True)
        
        self.values_label = tk.Label(self.metrics_frame, font=("Helvetica", 12), anchor="e", justify="right", width=7)
        self.values_label.pack(side="right", fill=tk.BOTH, expand=True)
        
        self.avg_metrics_label = tk.Label(self.avg_metrics_frame, font=("Helvetica", 12), anchor="w", justify="left", width=15)
        self.avg_metrics_label.pack(side="left", fill=tk.BOTH, expand=True)
        
        self.avg_values_label = tk.Label(self.avg_metrics_frame, font=("Helvetica", 12), anchor="e", justify="right", width=7)
        self.avg_values_label.pack(side="right", fill=tk.BOTH, expand=True)
        
        
    def update_graph(self, full_redraw=True):
        """Optimized: Only update the scatter marker when moving frames."""
        if not self.metrics_data or not self.selected_metric:
            return  # Exit if no data is loaded

        # Get the metric values (cache them for performance)
        if not hasattr(self, "cached_metric_values") or full_redraw:
            self.cached_metric_values = self.metrics_data.get(self.selected_metric, [])

        metric_values = self.cached_metric_values
        frames = list(range(len(metric_values)))

        # Ensure the current frame is within range
        current_frame = self.frame_indices[0]
        if current_frame >= len(metric_values):
            current_frame = len(metric_values) - 1

        if full_redraw:
            # **Redraw the entire graph only when switching metrics or reloading data**
            self.ax.clear()
            self.line, = self.ax.plot(frames, metric_values, "r-", linewidth=0.5, label=self.selected_metric, color="royalblue")

            # **Create the scatter marker ONCE and store the reference**
            self.current_marker = self.ax.scatter([current_frame], [metric_values[current_frame]], 
                                                color="orange", s=20, zorder=3, label="Current Frame")

            # Formatting
            self.ax.set_title(self.selected_metric, fontsize=5)
            self.ax.set_xlabel("Frames", fontsize=3)
            self.ax.set_ylabel("Value", fontsize=3)
            self.ax.tick_params(axis='both', labelsize=3)
            self.ax.set_xticks(frames[::100])  # Reduce x-ticks
            self.ax.legend(fontsize=6, loc="upper right")

        else:
            # **Only move the marker, instead of redrawing everything**
            self.current_marker.set_offsets([[current_frame, metric_values[current_frame]]])

        self.canvas_plot.draw_idle()  # Use draw_idle() instead of draw() for better performance


        
    def on_metric_selected(self, event):
        """Handles when the user selects a new metric."""
        self.selected_metric = self.metric_var.get()
        self.update_graph()
        
    def populate_dropdown(self):
        dataset_folders = [f for f in os.listdir(self.parent_directory) if os.path.isdir(os.path.join(self.parent_directory, f))]
        
        # Sort the folders naturally
        sorted_folders = sorted(dataset_folders, key=natural_sort_key)
        
        self.folder_dropdown["values"] = sorted_folders
        
    def load_selected_folder(self, event):
        """Loads images and metrics when a folder is selected from the dropdown."""
        
        # 1. Close previously opened images to prevent memory leaks
        for i in range(self.num_videos_to_display):
            for img in self.loaded_images[i]:
                img.close()
            self.loaded_images[i] = []  # Clear loaded images

        # 2. Reset state
        self.folders = [[], [], []]
        self.frame_indices = [0, 0, 0]
        self.metrics_data = {}

        # 3. Reset UI elements
        self.metrics_label.config(text="")
        self.values_label.config(text="")
        self.avg_metrics_label.config(text="")
        self.avg_values_label.config(text="")
        self.slider.config(to=0)

        # 4. Load the new folder
        selected_folder = self.folder_var.get()
        full_path = os.path.join(self.parent_directory, selected_folder)
        
        subfolders = sorted([os.path.join(full_path, d) for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))])
        if len(subfolders) >= self.num_videos_to_display:
            for i in range(self.num_videos_to_display):
                self.folders[i] = sorted([os.path.join(subfolders[i], f) for f in os.listdir(subfolders[i]) if f.endswith(('png', 'jpg', 'jpeg'))])
                self.frame_indices[i] = 0
                self.loaded_images[i] = [Image.open(f) for f in self.folders[i]]  # Pre-load images
                
            self.slider.config(to=len(self.folders[0]) - 1)
            self.show_frame()

        # 5. Load metrics
        print(f"Loading: {full_path}")
        metrics_file = os.path.join(full_path, "brightness.csv")
        if os.path.exists(metrics_file):
            self.metrics_data = self.load_metrics(full_path)
            
        self.update_graph()

        
    def load_folder(self, index):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folders[index] = sorted([os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.endswith(('png', 'jpg', 'jpeg'))])
            self.frame_indices[index] = 0
            self.loaded_images[index] = [Image.open(f) for f in self.folders[index]]  # Pre-load images
            if self.folders[index]:
                self.slider.config(to=len(self.folders[index]) - 1)
            self.show_frame()
    

    def load_metrics(self, full_path):
        # Define the CSV files to read from
        csv_files = {
            "brightness": os.path.join(full_path, "brightness.csv"),
            "optical_flow_magnitude": os.path.join(full_path, "optical_flow_magnitude.csv"),
            "HS_colourfulness": os.path.join(full_path, "HS_colourfulness.csv")
            
        }
        
        metrics_dict = {}

        for metric_name, file_name in csv_files.items():
            if os.path.exists(file_name):  # Ensure the file exists before reading
                with open(file_name, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # Read and ignore the header (frame numbers)
                    row = next(reader)  # Read the actual metric values
                    
                    metric = row[0]  # The first column contains the metric name
                    values = list(map(float, row[1:]))  # Convert the remaining columns to float
                    
                    metrics_dict[metric] = values  # Store values under the correct key
        
        self.metrics_data = metrics_dict  # Store loaded metrics

        # Populate the dropdown with metric names
        metric_names = list(metrics_dict.keys())
        self.metric_dropdown["values"] = metric_names
        if metric_names:
            self.metric_var.set(metric_names[0])  # Set default to the first metric
            self.selected_metric = metric_names[0]
            self.update_graph()

        return metrics_dict
        
    
    def load_pacing(self, pacing_file):
        # Load the metrics from the CSV file into a dictionary
        pacing_dict = {}
        
        with open(pacing_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the header
            for row in reader:
                metric = row[0]
                values = list(map(float, row[1:]))
                pacing_dict[metric] = values
                
            self.pacing_data = pacing_dict  # Store loaded metrics

            # Populate the dropdown with metric names
            metric_names = list(pacing_dict.keys())
            self.metric_dropdown["values"] = metric_names
            if metric_names:
                self.metric_var.set(metric_names[0])  # Set default to the first metric
                self.selected_metric = metric_names[0]
                self.update_graph()
                
        return pacing_dict

    
    def show_metrics(self):
        
        #Per frame metrics
        
        # Create a string displaying the metrics for the current frame
        frame_index = self.frame_indices[0]
        metrics_text = f"Frame {frame_index + 1}\n\n"
        values_text = f"     \n\n"
                
        # Create a fixed-size container for metrics
        for metric, values in self.metrics_data.items():
            value = f"{values[frame_index]:.3f}"
            
            # Adjust the spacing between metric and value based on the length of the metric
            metrics_text += f"{metric}:\n"
            values_text += f"{value}\n"
        
        self.metrics_label.config(text=metrics_text)
        self.values_label.config(text=values_text)
        
            
    def show_frame(self):
        canvases = [self.canvas1, self.canvas2]
        for i in range(self.num_videos_to_display):
            if self.folders[i]:
                # Get the image from pre-loaded list
                img = self.loaded_images[i][self.frame_indices[i]]
                width = canvases[i].winfo_width()
                height = canvases[i].winfo_height()
                if width > 0 and height > 0:
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    canvases[i].delete("all")  # Clear previous image
                    canvases[i].create_image(width // 2, height // 2, anchor=tk.CENTER, image=photo)
                    canvases[i].photo = photo  # Keep reference
        self.slider.set(self.frame_indices[0])
        self.show_metrics()  # Update metrics display when a frame is shown
    
    def toggle_playback(self):
        self.playing = not self.playing
        self.btn_play_pause.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.play_frames()

    def play_frames(self):
        if self.playing:
            for i in range(self.num_videos_to_display):
                if self.folders[i]:
                    self.frame_indices[i] = (self.frame_indices[i] + 1) % len(self.folders[i])
            
            self.show_frame()
            self.update_graph(full_redraw=False)  # Only move the marker
            self.root.after(33, self.play_frames)  # ~30 FPS
    
    def resize_canvases(self, event):
        self.show_frame()
    
    def set_frame(self, value):
        frame_number = int(value)
        for i in range(self.num_videos_to_display):
            if self.folders[i]:
                self.frame_indices[i] = frame_number % len(self.folders[i])
        self.show_frame()
        self.update_graph(full_redraw=False)  # **Only move the marker**

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x600")
    parent_directory = "processed_videos/"  # Change this to the directory containing dataset folders
    app = FramePlayer(root, parent_directory)
    root.mainloop()
