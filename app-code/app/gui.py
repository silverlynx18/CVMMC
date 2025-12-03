"""Desktop GUI application for pedestrian counting analysis."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
from tkinter import scrolledtext
import threading
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from .video_processor import VideoProcessor, AnalysisResult
from .video_clip_processor import VideoClipProcessor
from .metadata_manager import MetadataManager, VideoClip, CameraMetadata
from .fruin_analysis import FruinAnalyzer
from .config import settings

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class PedestrianAnalysisGUI:
    """Main GUI application for pedestrian counting analysis."""
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = ctk.CTk()
        self.root.title("Pedestrian Counting & Service Level Analysis")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.video_processor = None
        self.video_clip_processor = VideoClipProcessor(settings.SAM2_MODEL_PATH, settings.DEVICE)
        self.metadata_manager = MetadataManager()
        self.fruin_analyzer = FruinAnalyzer()
        self.current_results = []
        self.zones = {}
        self.current_camera_id = None
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        
        # Left panel - Controls
        self.control_panel = ctk.CTkFrame(self.main_frame)
        self.control_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Right panel - Results and Visualization
        self.results_panel = ctk.CTkFrame(self.main_frame)
        self.results_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=2)
        
        # Control Panel Widgets
        self.create_control_panel()
        
        # Results Panel Widgets
        self.create_results_panel()
        
        # Camera Management Panel
        self.create_camera_management_panel()
        
        # Clip Testing Panel
        self.create_clip_testing_panel()
        
    def create_control_panel(self):
        """Create the control panel widgets."""
        # Title
        title_label = ctk.CTkLabel(
            self.control_panel, 
            text="Analysis Controls", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Model Configuration
        model_frame = ctk.CTkFrame(self.control_panel)
        model_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(model_frame, text="SAM2 Model Configuration", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Model path
        ctk.CTkLabel(model_frame, text="Model Path:").grid(row=1, column=0, sticky="w", padx=5)
        self.model_path_var = tk.StringVar(value="sam2_hiera_large.pt")
        self.model_path_entry = ctk.CTkEntry(model_frame, textvariable=self.model_path_var, width=300)
        self.model_path_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.browse_model_btn = ctk.CTkButton(
            model_frame, 
            text="Browse", 
            command=self.browse_model_file,
            width=80
        )
        self.browse_model_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Device selection
        ctk.CTkLabel(model_frame, text="Device:").grid(row=2, column=0, sticky="w", padx=5)
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ctk.CTkComboBox(
            model_frame, 
            values=["cuda", "cpu"], 
            variable=self.device_var,
            width=100
        )
        self.device_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Video Input
        video_frame = ctk.CTkFrame(self.control_panel)
        video_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(video_frame, text="Video Input", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Single video file
        self.single_video_btn = ctk.CTkButton(
            video_frame, 
            text="Select Single Video", 
            command=self.select_single_video,
            width=150
        )
        self.single_video_btn.grid(row=1, column=0, padx=5, pady=5)
        
        # Batch directory
        self.batch_dir_btn = ctk.CTkButton(
            video_frame, 
            text="Select Batch Directory", 
            command=self.select_batch_directory,
            width=150
        )
        self.batch_dir_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Selected path display
        self.selected_path_var = tk.StringVar(value="No file/directory selected")
        self.selected_path_label = ctk.CTkLabel(
            video_frame, 
            textvariable=self.selected_path_var,
            wraplength=400
        )
        self.selected_path_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Analysis Parameters
        params_frame = ctk.CTkFrame(self.control_panel)
        params_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(params_frame, text="Analysis Parameters", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Frame skip
        ctk.CTkLabel(params_frame, text="Frame Skip:").grid(row=1, column=0, sticky="w", padx=5)
        self.frame_skip_var = tk.IntVar(value=1)
        self.frame_skip_entry = ctk.CTkEntry(params_frame, textvariable=self.frame_skip_var, width=50)
        self.frame_skip_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Zone area
        ctk.CTkLabel(params_frame, text="Zone Area (sqm):").grid(row=2, column=0, sticky="w", padx=5)
        self.zone_area_var = tk.DoubleVar(value=25.0)
        self.zone_area_entry = ctk.CTkEntry(params_frame, textvariable=self.zone_area_var, width=50)
        self.zone_area_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Confidence threshold
        ctk.CTkLabel(params_frame, text="Confidence Threshold:").grid(row=3, column=0, sticky="w", padx=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ctk.CTkSlider(
            params_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var,
            width=200
        )
        self.confidence_scale.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Zone Configuration
        zone_frame = ctk.CTkFrame(self.control_panel)
        zone_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(zone_frame, text="Zone Configuration", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        self.zone_text = scrolledtext.ScrolledText(zone_frame, height=8, width=50)
        self.zone_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Load zones button
        self.load_zones_btn = ctk.CTkButton(
            zone_frame, 
            text="Load Zones from File", 
            command=self.load_zones_file,
            width=150
        )
        self.load_zones_btn.grid(row=2, column=0, padx=5, pady=5)
        
        # Save zones button
        self.save_zones_btn = ctk.CTkButton(
            zone_frame, 
            text="Save Zones to File", 
            command=self.save_zones_file,
            width=150
        )
        self.save_zones_btn.grid(row=2, column=1, padx=5, pady=5)
        
        # Analysis Controls
        analysis_frame = ctk.CTkFrame(self.control_panel)
        analysis_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(analysis_frame, text="Analysis Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Start analysis button
        self.start_analysis_btn = ctk.CTkButton(
            analysis_frame, 
            text="Start Analysis", 
            command=self.start_analysis,
            width=150,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.start_analysis_btn.grid(row=1, column=0, padx=5, pady=10)
        
        # Stop analysis button
        self.stop_analysis_btn = ctk.CTkButton(
            analysis_frame, 
            text="Stop Analysis", 
            command=self.stop_analysis,
            width=150,
            height=40,
            state="disabled"
        )
        self.stop_analysis_btn.grid(row=1, column=1, padx=5, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(analysis_frame, variable=self.progress_var)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(analysis_frame, text="Ready to start analysis")
        self.progress_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Output directory
        output_frame = ctk.CTkFrame(self.control_panel)
        output_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(output_frame, text="Output Directory", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        self.output_dir_var = tk.StringVar(value="./analysis_results")
        self.output_dir_entry = ctk.CTkEntry(output_frame, textvariable=self.output_dir_var, width=300)
        self.output_dir_entry.grid(row=1, column=0, padx=5, pady=5)
        
        self.browse_output_btn = ctk.CTkButton(
            output_frame, 
            text="Browse", 
            command=self.browse_output_directory,
            width=80
        )
        self.browse_output_btn.grid(row=1, column=1, padx=5, pady=5)
        
    def create_results_panel(self):
        """Create the results panel widgets."""
        # Title
        title_label = ctk.CTkLabel(
            self.results_panel, 
            text="Analysis Results", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_panel)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Summary tab
        self.summary_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Results table
        self.results_tree = ttk.Treeview(
            self.summary_frame, 
            columns=("Video", "Duration", "Detections", "Ingress", "Egress", "Avg LOS", "Peak LOS"),
            show="headings",
            height=15
        )
        
        # Configure columns
        self.results_tree.heading("Video", text="Video File")
        self.results_tree.heading("Duration", text="Duration (s)")
        self.results_tree.heading("Detections", text="Total Detections")
        self.results_tree.heading("Ingress", text="Ingress Count")
        self.results_tree.heading("Egress", text="Egress Count")
        self.results_tree.heading("Avg LOS", text="Avg LOS")
        self.results_tree.heading("Peak LOS", text="Peak LOS")
        
        # Scrollbar for results table
        results_scrollbar = ttk.Scrollbar(self.summary_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        results_scrollbar.grid(row=0, column=1, sticky="ns", pady=10)
        
        # Configure grid weights
        self.summary_frame.grid_rowconfigure(0, weight=1)
        self.summary_frame.grid_columnconfigure(0, weight=1)
        
        # Visualization tab
        self.viz_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Canvas for matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights
        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        
        # Log tab
        self.log_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20, width=80)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        # Configure main grid weights
        self.results_panel.grid_rowconfigure(1, weight=1)
        self.results_panel.grid_columnconfigure(0, weight=1)
    
    def create_camera_management_panel(self):
        """Create camera management panel."""
        # Camera Management Frame
        self.camera_frame = ctk.CTkFrame(self.main_frame)
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Camera Management Title
        camera_title = ctk.CTkLabel(
            self.camera_frame, 
            text="Camera Management", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        camera_title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Camera List
        ctk.CTkLabel(self.camera_frame, text="Cameras:").grid(row=1, column=0, sticky="w", padx=5)
        self.camera_listbox = tk.Listbox(self.camera_frame, height=6, width=40)
        self.camera_listbox.grid(row=1, column=1, padx=5, pady=5)
        self.camera_listbox.bind('<<ListboxSelect>>', self.on_camera_select)
        
        # Camera Controls
        camera_controls = ctk.CTkFrame(self.camera_frame)
        camera_controls.grid(row=1, column=2, sticky="nsew", padx=5)
        
        self.add_camera_btn = ctk.CTkButton(
            camera_controls, 
            text="Add Camera", 
            command=self.add_camera_dialog
        )
        self.add_camera_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.remove_camera_btn = ctk.CTkButton(
            camera_controls, 
            text="Remove Camera", 
            command=self.remove_camera
        )
        self.remove_camera_btn.grid(row=1, column=0, padx=5, pady=5)
        
        # Camera Details
        details_frame = ctk.CTkFrame(self.camera_frame)
        details_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(details_frame, text="Camera Details:", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Camera ID
        ctk.CTkLabel(details_frame, text="Camera ID:").grid(row=1, column=0, sticky="w", padx=5)
        self.camera_id_var = tk.StringVar()
        self.camera_id_entry = ctk.CTkEntry(details_frame, textvariable=self.camera_id_var, width=200)
        self.camera_id_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Camera Name
        ctk.CTkLabel(details_frame, text="Name:").grid(row=2, column=0, sticky="w", padx=5)
        self.camera_name_var = tk.StringVar()
        self.camera_name_entry = ctk.CTkEntry(details_frame, textvariable=self.camera_name_var, width=200)
        self.camera_name_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Location
        ctk.CTkLabel(details_frame, text="Location:").grid(row=3, column=0, sticky="w", padx=5)
        self.camera_location_var = tk.StringVar()
        self.camera_location_entry = ctk.CTkEntry(details_frame, textvariable=self.camera_location_var, width=200)
        self.camera_location_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Zone Area
        ctk.CTkLabel(details_frame, text="Zone Area (sqm):").grid(row=4, column=0, sticky="w", padx=5)
        self.zone_area_var = tk.StringVar(value="25.0")
        self.zone_area_entry = ctk.CTkEntry(details_frame, textvariable=self.zone_area_var, width=200)
        self.zone_area_entry.grid(row=4, column=1, padx=5, pady=2)
        
        # Update Camera Button
        self.update_camera_btn = ctk.CTkButton(
            details_frame, 
            text="Update Camera", 
            command=self.update_camera
        )
        self.update_camera_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # Load cameras
        self.refresh_camera_list()
    
    def create_clip_testing_panel(self):
        """Create clip testing panel."""
        # Clip Testing Frame
        self.clip_frame = ctk.CTkFrame(self.main_frame)
        self.clip_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)
        
        # Clip Testing Title
        clip_title = ctk.CTkLabel(
            self.clip_frame, 
            text="Clip Testing", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        clip_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Test Duration
        ctk.CTkLabel(self.clip_frame, text="Test Duration (seconds):").grid(row=1, column=0, sticky="w", padx=5)
        self.test_duration_var = tk.StringVar(value="60")
        self.test_duration_entry = ctk.CTkEntry(self.clip_frame, textvariable=self.test_duration_var, width=100)
        self.test_duration_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Test Button
        self.test_camera_btn = ctk.CTkButton(
            self.clip_frame, 
            text="Test Camera", 
            command=self.test_camera,
            state="disabled"
        )
        self.test_camera_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Test Results
        ctk.CTkLabel(self.clip_frame, text="Test Results:", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(row=3, column=0, columnspan=2, pady=5)
        
        self.test_results_text = ctk.CTkTextbox(self.clip_frame, height=200, width=400)
        self.test_results_text.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Progress Bar
        self.test_progress = ctk.CTkProgressBar(self.clip_frame)
        self.test_progress.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.test_progress.set(0)
        
    def setup_layout(self):
        """Setup the main layout."""
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize default zones
        self.load_default_zones()
        
    def load_default_zones(self):
        """Load default zone configuration."""
        default_zones = {
            "ingress_1": {
                "polygon": [[100, 100], [400, 100], [400, 300], [100, 300]],
                "type": "ingress"
            },
            "egress_1": {
                "polygon": [[500, 100], [800, 100], [800, 300], [500, 300]],
                "type": "egress"
            }
        }
        
        self.zones = default_zones
        self.update_zone_display()
        
    def update_zone_display(self):
        """Update the zone configuration display."""
        self.zone_text.delete(1.0, tk.END)
        self.zone_text.insert(1.0, json.dumps(self.zones, indent=2))
        
    def browse_model_file(self):
        """Browse for SAM2 model file."""
        filename = filedialog.askopenfilename(
            title="Select SAM2 Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
            
    def select_single_video(self):
        """Select a single video file for analysis."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.selected_path_var.set(filename)
            self.analysis_mode = "single"
            
    def select_batch_directory(self):
        """Select a directory for batch analysis."""
        directory = filedialog.askdirectory(title="Select Video Directory")
        if directory:
            self.selected_path_var.set(directory)
            self.analysis_mode = "batch"
            
    def browse_output_directory(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
            
    def load_zones_file(self):
        """Load zone configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Zone Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.zones = json.load(f)
                self.update_zone_display()
                self.log_message(f"Loaded zones from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load zones: {e}")
                
    def save_zones_file(self):
        """Save zone configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Zone Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.zones, f, indent=2)
                self.log_message(f"Saved zones to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save zones: {e}")
                
    def log_message(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_analysis(self):
        """Start the analysis process."""
        # Validate inputs
        if not self.model_path_var.get():
            messagebox.showerror("Error", "Please select a SAM2 model file")
            return
            
        if not self.selected_path_var.get() or self.selected_path_var.get() == "No file/directory selected":
            messagebox.showerror("Error", "Please select a video file or directory")
            return
            
        # Update UI
        self.start_analysis_btn.configure(state="disabled")
        self.stop_analysis_btn.configure(state="normal")
        self.progress_var.set(0)
        self.progress_label.configure(text="Initializing analysis...")
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
    def run_analysis(self):
        """Run the analysis in a separate thread."""
        try:
            # Initialize video processor
            self.log_message("Initializing SAM2 detector...")
            self.video_processor = VideoProcessor(
                self.model_path_var.get(),
                self.device_var.get()
            )
            
            # Update parameters
            self.video_processor.set_analysis_parameters(
                frame_skip=self.frame_skip_var.get(),
                analysis_window_minutes=5,
                zone_area_sqm=self.zone_area_var.get()
            )
            
            # Update confidence threshold
            self.video_processor.sam2_detector.update_confidence_threshold(
                self.confidence_var.get()
            )
            
            self.log_message("Analysis started")
            
            # Run analysis based on mode
            if self.analysis_mode == "single":
                self.run_single_analysis()
            else:
                self.run_batch_analysis()
                
        except Exception as e:
            self.log_message(f"Analysis failed: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
        finally:
            # Reset UI
            self.root.after(0, self.analysis_completed)
            
    def run_single_analysis(self):
        """Run analysis on a single video file."""
        video_path = self.selected_path_var.get()
        output_dir = self.output_dir_var.get()
        
        self.log_message(f"Processing single video: {os.path.basename(video_path)}")
        
        def progress_callback(progress):
            self.root.after(0, lambda: self.progress_var.set(progress / 100))
            self.root.after(0, lambda: self.progress_label.configure(text=f"Processing... {progress:.1f}%"))
        
        # Process video
        result = self.video_processor.process_video(video_path, self.zones, progress_callback)
        
        if result:
            self.current_results = [result]
            self.root.after(0, self.update_results_display)
            self.log_message("Single video analysis completed")
        else:
            self.log_message("Single video analysis failed")
            
    def run_batch_analysis(self):
        """Run batch analysis on a directory."""
        video_dir = self.selected_path_var.get()
        output_dir = self.output_dir_var.get()
        
        self.log_message(f"Processing batch directory: {video_dir}")
        
        def progress_callback(progress):
            self.root.after(0, lambda: self.progress_var.set(progress / 100))
            self.root.after(0, lambda: self.progress_label.configure(text=f"Processing batch... {progress:.1f}%"))
        
        # Process batch
        results = self.video_processor.process_batch(video_dir, output_dir, self.zones, progress_callback)
        
        if results:
            self.current_results = results
            self.root.after(0, self.update_results_display)
            self.log_message(f"Batch analysis completed: {len(results)} videos processed")
        else:
            self.log_message("Batch analysis failed")
            
    def stop_analysis(self):
        """Stop the analysis process."""
        # This would need to be implemented with proper thread stopping
        self.log_message("Analysis stop requested")
        
    def analysis_completed(self):
        """Called when analysis is completed."""
        self.start_analysis_btn.configure(state="normal")
        self.stop_analysis_btn.configure(state="disabled")
        self.progress_var.set(100)
        self.progress_label.configure(text="Analysis completed")
        
    def update_results_display(self):
        """Update the results display."""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add new results
        for result in self.current_results:
            self.results_tree.insert("", "end", values=(
                result.video_file.name,
                f"{result.video_file.duration:.1f}",
                result.total_detections,
                result.total_ingress,
                result.total_egress,
                result.avg_los,
                result.peak_los
            ))
            
        # Update visualizations
        self.update_visualizations()
        
    def update_visualizations(self):
        """Update the visualization plots."""
        if not self.current_results:
            return
            
        # Clear existing plots
        for ax in self.axes.flat:
            ax.clear()
            
        # Plot 1: Density over time for first video
        if self.current_results:
            result = self.current_results[0]
            frame_data = result.frame_results
            
            if frame_data:
                times = [i * 5 for i in range(len(frame_data))]  # 5-second intervals
                densities = [fr['density'] for fr in frame_data]
                
                self.axes[0, 0].plot(times, densities)
                self.axes[0, 0].set_title("Pedestrian Density Over Time")
                self.axes[0, 0].set_xlabel("Time (seconds)")
                self.axes[0, 0].set_ylabel("Density (peds/m²)")
                self.axes[0, 0].grid(True)
        
        # Plot 2: LOS distribution
        if self.current_results:
            los_levels = [result.avg_los for result in self.current_results]
            los_counts = {}
            for los in los_levels:
                los_counts[los] = los_counts.get(los, 0) + 1
                
            self.axes[0, 1].bar(los_counts.keys(), los_counts.values())
            self.axes[0, 1].set_title("Level of Service Distribution")
            self.axes[0, 1].set_xlabel("LOS Level")
            self.axes[0, 1].set_ylabel("Number of Videos")
            
        # Plot 3: Ingress vs Egress
        if self.current_results:
            ingress_counts = [result.total_ingress for result in self.current_results]
            egress_counts = [result.total_egress for result in self.current_results]
            video_names = [result.video_file.name[:20] + "..." if len(result.video_file.name) > 20 
                          else result.video_file.name for result in self.current_results]
            
            x = np.arange(len(video_names))
            width = 0.35
            
            self.axes[1, 0].bar(x - width/2, ingress_counts, width, label='Ingress')
            self.axes[1, 0].bar(x + width/2, egress_counts, width, label='Egress')
            self.axes[1, 0].set_title("Ingress vs Egress Counts")
            self.axes[1, 0].set_xlabel("Video")
            self.axes[1, 0].set_ylabel("Count")
            self.axes[1, 0].set_xticks(x)
            self.axes[1, 0].set_xticklabels(video_names, rotation=45)
            self.axes[1, 0].legend()
            
        # Plot 4: Average density comparison
        if self.current_results:
            avg_densities = [result.avg_density for result in self.current_results]
            peak_densities = [result.peak_density for result in self.current_results]
            
            x = np.arange(len(video_names))
            width = 0.35
            
            self.axes[1, 1].bar(x - width/2, avg_densities, width, label='Average')
            self.axes[1, 1].bar(x + width/2, peak_densities, width, label='Peak')
            self.axes[1, 1].set_title("Average vs Peak Density")
            self.axes[1, 1].set_xlabel("Video")
            self.axes[1, 1].set_ylabel("Density (peds/m²)")
            self.axes[1, 1].set_xticks(x)
            self.axes[1, 1].set_xticklabels(video_names, rotation=45)
            self.axes[1, 1].legend()
            
        # Refresh canvas
        self.canvas.draw()
        
    def refresh_camera_list(self):
        """Refresh the camera list display."""
        self.camera_listbox.delete(0, tk.END)
        cameras = self.metadata_manager.get_all_cameras()
        for camera in cameras:
            self.camera_listbox.insert(tk.END, f"{camera.camera_id}: {camera.name}")
    
    def on_camera_select(self, event):
        """Handle camera selection."""
        selection = self.camera_listbox.curselection()
        if selection:
            index = selection[0]
            cameras = self.metadata_manager.get_all_cameras()
            if index < len(cameras):
                camera = cameras[index]
                self.current_camera_id = camera.camera_id
                self.camera_id_var.set(camera.camera_id)
                self.camera_name_var.set(camera.name)
                self.camera_location_var.set(camera.location)
                self.zone_area_var.set(str(camera.zone_area_sqm))
                self.test_camera_btn.configure(state="normal")
    
    def add_camera_dialog(self):
        """Open dialog to add a new camera."""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add Camera")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Camera ID
        ctk.CTkLabel(dialog, text="Camera ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        camera_id_var = tk.StringVar()
        ctk.CTkEntry(dialog, textvariable=camera_id_var, width=200).grid(row=0, column=1, padx=5, pady=5)
        
        # Camera Name
        ctk.CTkLabel(dialog, text="Name:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        camera_name_var = tk.StringVar()
        ctk.CTkEntry(dialog, textvariable=camera_name_var, width=200).grid(row=1, column=1, padx=5, pady=5)
        
        # Location
        ctk.CTkLabel(dialog, text="Location:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        camera_location_var = tk.StringVar()
        ctk.CTkEntry(dialog, textvariable=camera_location_var, width=200).grid(row=2, column=1, padx=5, pady=5)
        
        # Description
        ctk.CTkLabel(dialog, text="Description:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        camera_desc_var = tk.StringVar()
        ctk.CTkEntry(dialog, textvariable=camera_desc_var, width=200).grid(row=3, column=1, padx=5, pady=5)
        
        # Zone Area
        ctk.CTkLabel(dialog, text="Zone Area (sqm):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        zone_area_var = tk.StringVar(value="25.0")
        ctk.CTkEntry(dialog, textvariable=zone_area_var, width=200).grid(row=4, column=1, padx=5, pady=5)
        
        def add_camera():
            camera_id = camera_id_var.get().strip()
            name = camera_name_var.get().strip()
            location = camera_location_var.get().strip()
            description = camera_desc_var.get().strip()
            zone_area = float(zone_area_var.get())
            
            if not camera_id or not name or not location:
                tk.messagebox.showerror("Error", "Please fill in all required fields")
                return
            
            success = self.metadata_manager.add_camera(
                camera_id, name, location, description, zone_area
            )
            
            if success:
                self.refresh_camera_list()
                dialog.destroy()
                tk.messagebox.showinfo("Success", f"Camera {camera_id} added successfully")
            else:
                tk.messagebox.showerror("Error", f"Failed to add camera {camera_id}")
        
        # Buttons
        ctk.CTkButton(dialog, text="Add Camera", command=add_camera).grid(row=5, column=0, padx=5, pady=10)
        ctk.CTkButton(dialog, text="Cancel", command=dialog.destroy).grid(row=5, column=1, padx=5, pady=10)
    
    def remove_camera(self):
        """Remove selected camera."""
        if not self.current_camera_id:
            tk.messagebox.showwarning("Warning", "Please select a camera to remove")
            return
        
        result = tk.messagebox.askyesno("Confirm", f"Remove camera {self.current_camera_id}?")
        if result:
            success = self.metadata_manager.remove_camera(self.current_camera_id)
            if success:
                self.refresh_camera_list()
                self.current_camera_id = None
                self.camera_id_var.set("")
                self.camera_name_var.set("")
                self.camera_location_var.set("")
                self.zone_area_var.set("25.0")
                self.test_camera_btn.configure(state="disabled")
                tk.messagebox.showinfo("Success", "Camera removed successfully")
            else:
                tk.messagebox.showerror("Error", "Failed to remove camera")
    
    def update_camera(self):
        """Update selected camera."""
        if not self.current_camera_id:
            tk.messagebox.showwarning("Warning", "Please select a camera to update")
            return
        
        camera_id = self.camera_id_var.get().strip()
        name = self.camera_name_var.get().strip()
        location = self.camera_location_var.get().strip()
        zone_area = float(self.zone_area_var.get())
        
        if not camera_id or not name or not location:
            tk.messagebox.showerror("Error", "Please fill in all required fields")
            return
        
        # Update zone area
        success = self.metadata_manager.update_zone_area(camera_id, zone_area)
        
        if success:
            self.refresh_camera_list()
            tk.messagebox.showinfo("Success", "Camera updated successfully")
        else:
            tk.messagebox.showerror("Error", "Failed to update camera")
    
    def test_camera(self):
        """Test selected camera with short clips."""
        if not self.current_camera_id:
            tk.messagebox.showwarning("Warning", "Please select a camera to test")
            return
        
        try:
            test_duration = float(self.test_duration_var.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter a valid test duration")
            return
        
        self.test_results_text.delete("1.0", tk.END)
        self.test_results_text.insert("1.0", "Starting camera test...\n")
        self.test_progress.set(0)
        
        def progress_callback(progress):
            self.root.after(0, lambda: self.test_progress.set(progress / 100))
        
        def run_test():
            try:
                # Get camera metadata
                camera = self.metadata_manager.get_camera(self.current_camera_id)
                if not camera:
                    self.root.after(0, lambda: self.test_results_text.insert(tk.END, f"Camera {self.current_camera_id} not found\n"))
                    return
                
                # Run test
                result = self.video_clip_processor.test_camera_with_short_clips(
                    self.current_camera_id, test_duration, self.zones
                )
                
                # Display results
                if 'error' in result:
                    self.root.after(0, lambda: self.test_results_text.insert(tk.END, f"Error: {result['error']}\n"))
                else:
                    summary = f"""
Test Results for Camera {result['camera_id']} ({result['camera_name']}):
Test Duration: {result['test_duration']}s
Clips Tested: {result['clips_tested']}

Summary:
- Total Detections: {result['total_detections']}
- Total Ingress: {result['total_ingress']}
- Total Egress: {result['total_egress']}
- Average Density: {result['avg_density']:.2f} peds/sqm
- Peak Density: {result['peak_density']:.2f} peds/sqm
- Average LOS: {result['avg_los']}
- Average Processing Time: {result['avg_processing_time']:.1f}s

Detailed Results:
"""
                    for i, clip_result in enumerate(result['results']):
                        summary += f"""
Clip {i+1}: {clip_result['clip_name']}
  Duration: {clip_result['duration']:.1f}s
  Detections: {clip_result['total_detections']}
  Ingress: {clip_result['total_ingress']}
  Egress: {clip_result['total_egress']}
  Avg Density: {clip_result['avg_density']:.2f} peds/sqm
  Peak Density: {clip_result['peak_density']:.2f} peds/sqm
  Avg LOS: {clip_result['avg_los']}
  Processing Time: {clip_result['processing_time']:.1f}s
"""
                    
                    self.root.after(0, lambda: self.test_results_text.insert(tk.END, summary))
                
                self.root.after(0, lambda: self.test_progress.set(1.0))
                
            except Exception as e:
                self.root.after(0, lambda: self.test_results_text.insert(tk.END, f"Test failed: {str(e)}\n"))
                self.root.after(0, lambda: self.test_progress.set(0))
        
        # Run test in thread
        import threading
        thread = threading.Thread(target=run_test)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the GUI application."""
    app = PedestrianAnalysisGUI()
    app.run()


if __name__ == "__main__":
    main()