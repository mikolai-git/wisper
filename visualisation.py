import tkinter as tk
from tkinter import filedialog
import csv
from PIL import Image, ImageTk
import os

class FramePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Player")
        
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas1 = tk.Canvas(self.frame)
        self.canvas1.grid(row=0, column=0, sticky="nsew")
        self.canvas2 = tk.Canvas(self.frame)
        self.canvas2.grid(row=0, column=1, sticky="nsew")
        self.canvas3 = tk.Canvas(self.frame)
        self.canvas3.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        
        self.btn_select1 = tk.Button(root, text="Select Folder 1", command=lambda: self.load_folder(0))
        self.btn_select1.pack()
        self.btn_select2 = tk.Button(root, text="Select Folder 2", command=lambda: self.load_folder(1))
        self.btn_select2.pack()
        self.btn_select3 = tk.Button(root, text="Select Folder 3", command=lambda: self.load_folder(2))
        self.btn_select3.pack()
        
        self.btn_select_metrics = tk.Button(root, text="Select Metrics File", command=self.load_metrics_file)
        self.btn_select_metrics.pack()
        
        self.btn_play_pause = tk.Button(root, text="Play", command=self.toggle_playback)
        self.btn_play_pause.pack()
        
        self.slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_frame, length=800)
        self.slider.pack(fill=tk.X)
        
        self.folders = [[], [], []]
        self.frame_indices = [0, 0, 0]
        self.playing = False
        self.metrics_data = {}
        
        self.root.bind("<Configure>", self.resize_canvases)
    
    def load_folder(self, index):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folders[index] = sorted([os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.endswith(('png', 'jpg', 'jpeg'))])
            self.frame_indices[index] = 0
            if self.folders[index]:
                self.slider.config(to=len(self.folders[index]) - 1)
            self.show_frame()
    
    def load_metrics_file(self):
        # Open file dialog to select the metrics CSV file
        metrics_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if metrics_file:
            self.metrics_data = self.load_metrics(metrics_file)
            self.show_metrics()
    
    def load_metrics(self, metrics_file):
        # Load the metrics from the CSV file into a dictionary
        metrics_dict = {}
        with open(metrics_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the header
            for row in reader:
                metric = row[0]
                values = list(map(float, row[1:]))
                metrics_dict[metric] = values
        return metrics_dict
    
    def show_metrics(self):
        # Create a string displaying the metrics for the current frame
        frame_index = self.frame_indices[0]
        metrics_text = f"Frame {frame_index + 1}\n\n"
        
        for metric, values in self.metrics_data.items():
            metrics_text += f"{metric}: {values[frame_index]:.2f}\n"
        
        if hasattr(self, 'metrics_label'):
            self.metrics_label.config(text=metrics_text)
        else:
            self.metrics_label = tk.Label(self.frame, text=metrics_text, anchor="w", justify="left", font=("Helvetica", 10))
            self.metrics_label.grid(row=1, column=2, sticky="nsew")
    
    def show_frame(self):
        canvases = [self.canvas1, self.canvas2, self.canvas3]
        for i in range(3):
            if self.folders[i]:
                img = Image.open(self.folders[i][self.frame_indices[i]])
                width = canvases[i].winfo_width()
                height = canvases[i].winfo_height()
                if width > 0 and height > 0:
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
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
            for i in range(3):
                if self.folders[i]:
                    self.frame_indices[i] = (self.frame_indices[i] + 1) % len(self.folders[i])
            self.show_frame()
            self.root.after(33, self.play_frames)  # ~30 FPS
    
    def resize_canvases(self, event):
        self.show_frame()
    
    def set_frame(self, value):
        frame_number = int(value)
        for i in range(3):
            if self.folders[i]:
                self.frame_indices[i] = frame_number % len(self.folders[i])
        self.show_frame()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x600")
    app = FramePlayer(root)
    root.mainloop()
