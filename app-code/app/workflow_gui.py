"""Modern GUI for Two-Stage Pedestrian Detection Workflow with Detection Tuning."""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from tkinter import scrolledtext
import threading
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import sys
import time
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import get_config
from scripts.master_workflow import MasterWorkflow
from app.tuning_helpers import create_stage_tuner, get_stage_from_fps

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class WorkflowGUI:
    """GUI for two-stage pedestrian detection workflow."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = ctk.CTk()
        self.root.title("Pedestrian Detection Workflow - Stage 1 & Stage 2")
        self.root.geometry("1600x1000")
        
        # Initialize workflow
        self.config_manager = get_config()
        self.workflow = MasterWorkflow()
        self.processing = False
        
        # State variables
        self.video_path = None
        self.output_path = None
        self.camera_id = None
        self.current_stage = None
        
        # Resource monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Progress tracking
        self.current_frame = 0
        self.total_frames = 0
        self.processing_speed = 0.0  # fps
        
        # Create interface
        self.create_interface()
        
        # Start resource monitoring
        self.start_resource_monitoring()
        
    def create_interface(self):
        """Create the main interface."""
        # Top menu bar
        self.create_menu_bar()
        
        # Main container with tabs
        self.notebook = ctk.CTkTabview(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_stage1_tab()
        self.create_stage2_tab()
        self.create_detection_tuning_tab()
        self.create_results_tab()
        
    def create_menu_bar(self):
        """Create menu bar."""
        menu_frame = ctk.CTkFrame(self.root, height=40)
        menu_frame.pack(fill="x", padx=5, pady=5)
        
        # File menu
        file_btn = ctk.CTkButton(
            menu_frame, 
            text="File", 
            command=self.file_menu,
            width=80
        )
        file_btn.pack(side="left", padx=5)
        
        # Settings
        settings_btn = ctk.CTkButton(
            menu_frame,
            text="Settings",
            command=self.open_settings,
            width=80
        )
        settings_btn.pack(side="left", padx=5)
        
        # Help
        help_btn = ctk.CTkButton(
            menu_frame,
            text="Help",
            command=self.show_help,
            width=80
        )
        help_btn.pack(side="left", padx=5)
        
    def create_stage1_tab(self):
        """Create Stage 1 (Peak Detection) tab."""
        tab = self.notebook.add("Stage 1: Peak Detection")
        
        # Left panel - Configuration
        left_panel = ctk.CTkFrame(tab)
        left_panel.pack(side="left", fill="both", padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            left_panel,
            text="Stage 1: Peak Period Detection",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=10)
        
        # Description
        desc = ctk.CTkLabel(
            left_panel,
            text="Fast scanning to identify peak activity periods.\nRecommended: 15fps processing",
            font=ctk.CTkFont(size=12)
        )
        desc.pack(pady=5)
        
        # Video selection
        video_frame = ctk.CTkFrame(left_panel)
        video_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(video_frame, text="Source Video:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        self.stage1_video_var = tk.StringVar(value="No video selected")
        video_entry = ctk.CTkEntry(video_frame, textvariable=self.stage1_video_var, width=400)
        video_entry.pack(fill="x", padx=5, pady=5)
        
        browse_btn = ctk.CTkButton(
            video_frame,
            text="Browse Video",
            command=lambda: self.select_video("stage1"),
            width=120
        )
        browse_btn.pack(padx=5, pady=5)
        
        # Camera configuration
        camera_frame = ctk.CTkFrame(left_panel)
        camera_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(camera_frame, text="Camera ID:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_camera_var = tk.StringVar()
        camera_entry = ctk.CTkEntry(camera_frame, textvariable=self.stage1_camera_var, width=200)
        camera_entry.pack(fill="x", padx=5, pady=5)
        
        # Output directory
        output_frame = ctk.CTkFrame(left_panel)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(output_frame, text="Output Directory:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_output_var = tk.StringVar(value=str(Path.cwd() / "output"))
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.stage1_output_var, width=400)
        output_entry.pack(fill="x", padx=5, pady=5)
        
        browse_output_btn = ctk.CTkButton(
            output_frame,
            text="Browse",
            command=lambda: self.select_directory("stage1"),
            width=120
        )
        browse_output_btn.pack(padx=5, pady=5)
        
        # Settings
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Detection Method:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_method_var = tk.StringVar(value="pedestrian")
        method_combo = ctk.CTkComboBox(
            settings_frame,
            values=["pedestrian", "motion"],
            variable=self.stage1_method_var,
            width=200
        )
        method_combo.pack(fill="x", padx=5, pady=5)
        
        # FPS selection
        ctk.CTkLabel(settings_frame, text="Processing FPS:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_fps_var = tk.StringVar(value="15")
        fps_combo = ctk.CTkComboBox(
            settings_frame,
            values=["1", "15"],
            variable=self.stage1_fps_var,
            width=200
        )
        fps_combo.pack(fill="x", padx=5, pady=5)
        
        # Peak window size
        ctk.CTkLabel(settings_frame, text="Peak Window (minutes):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_window_var = tk.StringVar(value="15")
        window_entry = ctk.CTkEntry(settings_frame, textvariable=self.stage1_window_var, width=100)
        window_entry.pack(fill="x", padx=5, pady=5)
        
        # Run button - Make it prominent
        run_frame = ctk.CTkFrame(left_panel)
        run_frame.pack(fill="x", padx=10, pady=20)
        
        self.stage1_run_btn = ctk.CTkButton(
            run_frame,
            text="▶ START STAGE 1 ANALYSIS",
            command=self.run_stage1,
            width=350,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.stage1_run_btn.pack(pady=10)
        
        # Right panel - Progress and Log
        right_panel = ctk.CTkFrame(tab)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Progress
        progress_frame = ctk.CTkFrame(right_panel)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(progress_frame, text="Progress:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # Progress bar
        self.stage1_progress = ctk.CTkProgressBar(progress_frame, width=500)
        self.stage1_progress.pack(fill="x", padx=5, pady=5)
        self.stage1_progress.set(0)
        
        # Progress details
        progress_details_frame = ctk.CTkFrame(progress_frame)
        progress_details_frame.pack(fill="x", padx=5, pady=5)
        
        self.stage1_progress_text_var = tk.StringVar(value="0 / 0 frames (0%)")
        progress_text_label = ctk.CTkLabel(progress_details_frame, textvariable=self.stage1_progress_text_var, font=ctk.CTkFont(size=11))
        progress_text_label.pack(side="left", padx=5)
        
        self.stage1_speed_var = tk.StringVar(value="Speed: 0.0 fps")
        speed_label = ctk.CTkLabel(progress_details_frame, textvariable=self.stage1_speed_var, font=ctk.CTkFont(size=11))
        speed_label.pack(side="left", padx=10)
        
        # Status
        self.stage1_status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(progress_frame, textvariable=self.stage1_status_var, font=ctk.CTkFont(size=12))
        status_label.pack(pady=5)
        
        # Resource usage
        resource_frame = ctk.CTkFrame(right_panel)
        resource_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(resource_frame, text="Resource Usage:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # CPU
        cpu_frame = ctk.CTkFrame(resource_frame)
        cpu_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(cpu_frame, text="CPU:", width=50).pack(side="left", padx=5)
        self.stage1_cpu_var = tk.StringVar(value="0%")
        ctk.CTkLabel(cpu_frame, textvariable=self.stage1_cpu_var, width=60).pack(side="left", padx=5)
        self.stage1_cpu_bar = ctk.CTkProgressBar(cpu_frame, width=200)
        self.stage1_cpu_bar.pack(side="left", padx=5, fill="x", expand=True)
        self.stage1_cpu_bar.set(0)
        
        # RAM
        ram_frame = ctk.CTkFrame(resource_frame)
        ram_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(ram_frame, text="RAM:", width=50).pack(side="left", padx=5)
        self.stage1_ram_var = tk.StringVar(value="0%")
        ctk.CTkLabel(ram_frame, textvariable=self.stage1_ram_var, width=60).pack(side="left", padx=5)
        self.stage1_ram_bar = ctk.CTkProgressBar(ram_frame, width=200)
        self.stage1_ram_bar.pack(side="left", padx=5, fill="x", expand=True)
        self.stage1_ram_bar.set(0)
        
        # GPU (if available)
        self.stage1_gpu_frame = None
        if GPU_AVAILABLE:
            gpu_frame = ctk.CTkFrame(resource_frame)
            gpu_frame.pack(fill="x", padx=5, pady=2)
            ctk.CTkLabel(gpu_frame, text="GPU:", width=50).pack(side="left", padx=5)
            self.stage1_gpu_var = tk.StringVar(value="N/A")
            ctk.CTkLabel(gpu_frame, textvariable=self.stage1_gpu_var, width=60).pack(side="left", padx=5)
            self.stage1_gpu_bar = ctk.CTkProgressBar(gpu_frame, width=200)
            self.stage1_gpu_bar.pack(side="left", padx=5, fill="x", expand=True)
            self.stage1_gpu_bar.set(0)
            self.stage1_gpu_frame = gpu_frame
        
        # Log output
        log_frame = ctk.CTkFrame(right_panel)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(log_frame, text="Log Output:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage1_log = scrolledtext.ScrolledText(
            log_frame,
            width=60,
            height=30,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 10)
        )
        self.stage1_log.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_stage2_tab(self):
        """Create Stage 2 (Detailed Analysis) tab."""
        tab = self.notebook.add("Stage 2: Detailed Analysis")
        
        # Left panel
        left_panel = ctk.CTkFrame(tab)
        left_panel.pack(side="left", fill="both", padx=10, pady=10)
        
        title = ctk.CTkLabel(
            left_panel,
            text="Stage 2: MassMotion Preparation",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=10)
        
        desc = ctk.CTkLabel(
            left_panel,
            text="High-accuracy analysis for MassMotion simulation.\nRequired: 30fps processing",
            font=ctk.CTkFont(size=12)
        )
        desc.pack(pady=5)
        
        # Peak segment selection
        segment_frame = ctk.CTkFrame(left_panel)
        segment_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(segment_frame, text="Peak Segment:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage2_segment_var = tk.StringVar(value="No segment selected")
        segment_entry = ctk.CTkEntry(segment_frame, textvariable=self.stage2_segment_var, width=400)
        segment_entry.pack(fill="x", padx=5, pady=5)
        
        browse_segment_btn = ctk.CTkButton(
            segment_frame,
            text="Browse Segment",
            command=lambda: self.select_video("stage2"),
            width=120
        )
        browse_segment_btn.pack(padx=5, pady=5)
        
        # Or use Stage 1 output
        use_stage1_frame = ctk.CTkFrame(left_panel)
        use_stage1_frame.pack(fill="x", padx=10, pady=10)
        
        self.use_stage1_output_var = tk.BooleanVar(value=True)
        use_stage1_check = ctk.CTkCheckBox(
            use_stage1_frame,
            text="Use Stage 1 output automatically",
            variable=self.use_stage1_output_var
        )
        use_stage1_check.pack(padx=5, pady=5)
        
        # Configuration files
        config_frame = ctk.CTkFrame(left_panel)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(config_frame, text="Configuration:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        ctk.CTkLabel(config_frame, text="Lines Config:").pack(anchor="w", padx=5)
        self.stage2_lines_var = tk.StringVar()
        lines_entry = ctk.CTkEntry(config_frame, textvariable=self.stage2_lines_var, width=300)
        lines_entry.pack(fill="x", padx=5, pady=2)
        
        lines_button_frame = ctk.CTkFrame(config_frame)
        lines_button_frame.pack(fill="x", padx=5, pady=2)
        
        lines_browse = ctk.CTkButton(
            lines_button_frame,
            text="Browse",
            command=lambda: self.select_file("lines"),
            width=80
        )
        lines_browse.pack(side="left", padx=2)
        
        lines_create = ctk.CTkButton(
            lines_button_frame,
            text="Create New",
            command=lambda: self.create_lines_config(),
            width=100,
            fg_color="green"
        )
        lines_create.pack(side="left", padx=2)
        
        ctk.CTkLabel(config_frame, text="Zones Config:").pack(anchor="w", padx=5)
        self.stage2_zones_var = tk.StringVar()
        zones_entry = ctk.CTkEntry(config_frame, textvariable=self.stage2_zones_var, width=300)
        zones_entry.pack(fill="x", padx=5, pady=2)
        
        zones_button_frame = ctk.CTkFrame(config_frame)
        zones_button_frame.pack(fill="x", padx=5, pady=2)
        
        zones_browse = ctk.CTkButton(
            zones_button_frame,
            text="Browse",
            command=lambda: self.select_file("zones"),
            width=80
        )
        zones_browse.pack(side="left", padx=2)
        
        zones_create = ctk.CTkButton(
            zones_button_frame,
            text="Create New",
            command=lambda: self.create_zones_config(),
            width=100,
            fg_color="green"
        )
        zones_create.pack(side="left", padx=2)
        
        # Output directory
        output_frame = ctk.CTkFrame(left_panel)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(output_frame, text="Output Directory:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage2_output_var = tk.StringVar(value=str(Path.cwd() / "output" / "stage2"))
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.stage2_output_var, width=400)
        output_entry.pack(fill="x", padx=5, pady=5)
        
        # Run button - Make it prominent
        run_frame = ctk.CTkFrame(left_panel)
        run_frame.pack(fill="x", padx=10, pady=20)
        
        self.stage2_run_btn = ctk.CTkButton(
            run_frame,
            text="▶ START STAGE 2 ANALYSIS",
            command=self.run_stage2,
            width=350,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.stage2_run_btn.pack(pady=10)
        
        # Right panel - Progress, Resources, Configuration, and Log
        right_panel = ctk.CTkFrame(tab)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Progress
        progress_frame = ctk.CTkFrame(right_panel)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(progress_frame, text="Progress:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # Progress bar
        self.stage2_progress = ctk.CTkProgressBar(progress_frame, width=500)
        self.stage2_progress.pack(fill="x", padx=5, pady=5)
        self.stage2_progress.set(0)
        
        # Progress details
        progress_details_frame = ctk.CTkFrame(progress_frame)
        progress_details_frame.pack(fill="x", padx=5, pady=5)
        
        self.stage2_progress_text_var = tk.StringVar(value="0 / 0 frames (0%)")
        progress_text_label = ctk.CTkLabel(progress_details_frame, textvariable=self.stage2_progress_text_var, font=ctk.CTkFont(size=11))
        progress_text_label.pack(side="left", padx=5)
        
        self.stage2_speed_var = tk.StringVar(value="Speed: 0.0 fps")
        speed_label = ctk.CTkLabel(progress_details_frame, textvariable=self.stage2_speed_var, font=ctk.CTkFont(size=11))
        speed_label.pack(side="left", padx=10)
        
        # Status
        self.stage2_status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(progress_frame, textvariable=self.stage2_status_var, font=ctk.CTkFont(size=12))
        status_label.pack(pady=5)
        
        # Resource usage
        resource_frame = ctk.CTkFrame(right_panel)
        resource_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(resource_frame, text="Resource Usage:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # CPU
        cpu_frame = ctk.CTkFrame(resource_frame)
        cpu_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(cpu_frame, text="CPU:", width=50).pack(side="left", padx=5)
        self.stage2_cpu_var = tk.StringVar(value="0%")
        ctk.CTkLabel(cpu_frame, textvariable=self.stage2_cpu_var, width=60).pack(side="left", padx=5)
        self.stage2_cpu_bar = ctk.CTkProgressBar(cpu_frame, width=200)
        self.stage2_cpu_bar.pack(side="left", padx=5, fill="x", expand=True)
        self.stage2_cpu_bar.set(0)
        
        # RAM
        ram_frame = ctk.CTkFrame(resource_frame)
        ram_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(ram_frame, text="RAM:", width=50).pack(side="left", padx=5)
        self.stage2_ram_var = tk.StringVar(value="0%")
        ctk.CTkLabel(ram_frame, textvariable=self.stage2_ram_var, width=60).pack(side="left", padx=5)
        self.stage2_ram_bar = ctk.CTkProgressBar(ram_frame, width=200)
        self.stage2_ram_bar.pack(side="left", padx=5, fill="x", expand=True)
        self.stage2_ram_bar.set(0)
        
        # GPU (if available)
        self.stage2_gpu_frame = None
        if GPU_AVAILABLE:
            gpu_frame = ctk.CTkFrame(resource_frame)
            gpu_frame.pack(fill="x", padx=5, pady=2)
            ctk.CTkLabel(gpu_frame, text="GPU:", width=50).pack(side="left", padx=5)
            self.stage2_gpu_var = tk.StringVar(value="N/A")
            ctk.CTkLabel(gpu_frame, textvariable=self.stage2_gpu_var, width=60).pack(side="left", padx=5)
            self.stage2_gpu_bar = ctk.CTkProgressBar(gpu_frame, width=200)
            self.stage2_gpu_bar.pack(side="left", padx=5, fill="x", expand=True)
            self.stage2_gpu_bar.set(0)
            self.stage2_gpu_frame = gpu_frame
        
        # Analysis Configuration (moved here, between resources and log)
        config_frame = ctk.CTkFrame(right_panel)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(config_frame, text="Analysis Configuration:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # Mode selection: Production vs Testing
        mode_frame = ctk.CTkFrame(config_frame)
        mode_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(mode_frame, text="Mode:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.stage2_mode_var = tk.StringVar(value="production")
        production_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Production (Single Analysis)",
            variable=self.stage2_mode_var,
            value="production"
        )
        production_radio.pack(side="left", padx=10)
        
        testing_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Testing (Multiple Methods)",
            variable=self.stage2_mode_var,
            value="testing"
        )
        testing_radio.pack(side="left", padx=10)
        
        # FPS Selection
        fps_frame = ctk.CTkFrame(config_frame)
        fps_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(fps_frame, text="FPS:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.stage2_fps_vars = {
            "1": tk.BooleanVar(value=False),
            "15": tk.BooleanVar(value=False),
            "30": tk.BooleanVar(value=True)  # Default to 30fps
        }
        
        fps_1_check = ctk.CTkCheckBox(fps_frame, text="1fps", variable=self.stage2_fps_vars["1"])
        fps_1_check.pack(side="left", padx=5)
        fps_15_check = ctk.CTkCheckBox(fps_frame, text="15fps", variable=self.stage2_fps_vars["15"])
        fps_15_check.pack(side="left", padx=5)
        fps_30_check = ctk.CTkCheckBox(fps_frame, text="30fps", variable=self.stage2_fps_vars["30"])
        fps_30_check.pack(side="left", padx=5)
        
        # Analysis Type Selection (Whole Frame vs Zone-only)
        analysis_type_frame = ctk.CTkFrame(config_frame)
        analysis_type_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(analysis_type_frame, text="Analysis Type:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.stage2_analysis_type_vars = {
            "whole": tk.BooleanVar(value=True),  # Default to whole frame
            "zone": tk.BooleanVar(value=False)
        }
        
        whole_check = ctk.CTkCheckBox(analysis_type_frame, text="Whole Frame", variable=self.stage2_analysis_type_vars["whole"])
        whole_check.pack(side="left", padx=5)
        zone_check = ctk.CTkCheckBox(analysis_type_frame, text="Zone Only", variable=self.stage2_analysis_type_vars["zone"])
        zone_check.pack(side="left", padx=5)
        
        # Detection Method Selection
        method_frame = ctk.CTkFrame(config_frame)
        method_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(method_frame, text="Method:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.stage2_method_vars = {
            "pedestrian": tk.BooleanVar(value=True),  # Default to pedestrian
            "motion": tk.BooleanVar(value=False)
        }
        
        pedestrian_check = ctk.CTkCheckBox(method_frame, text="Pedestrian (YOLO)", variable=self.stage2_method_vars["pedestrian"])
        pedestrian_check.pack(side="left", padx=5)
        motion_method_check = ctk.CTkCheckBox(method_frame, text="Motion", variable=self.stage2_method_vars["motion"])
        motion_method_check.pack(side="left", padx=5)
        
        # Info label
        info_label = ctk.CTkLabel(
            config_frame,
            text="💡 Production: One optimized method | Testing: Multiple methods for comparison",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        info_label.pack(padx=5, pady=5)
        
        # Update mode behavior when mode changes
        def update_mode_behavior(*args):
            mode = self.stage2_mode_var.get()
            if mode == "production":
                # Production: Only allow single selections
                # Ensure at least one is selected, but only allow one FPS, one frame type, one method
                pass  # Will enforce in validation
            else:
                # Testing: Allow multiple selections
                pass  # Already supported by checkboxes
        
        self.stage2_mode_var.trace_add("write", update_mode_behavior)
        
        # Log output
        log_frame = ctk.CTkFrame(right_panel)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(log_frame, text="Log Output:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        self.stage2_log = scrolledtext.ScrolledText(
            log_frame,
            width=60,
            height=30,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 10)
        )
        self.stage2_log.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_detection_tuning_tab(self):
        """Create Detection Tuning configuration tab."""
        tab = self.notebook.add("Detection Tuning")
        
        # Create scrollable frame
        main_frame = ctk.CTkScrollableFrame(tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        title = ctk.CTkLabel(
            main_frame,
            text="Adaptive Detection Tuning Configuration",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=20)
        
        # Global settings
        global_frame = ctk.CTkFrame(main_frame)
        global_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            global_frame,
            text="Global Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.tuning_enabled_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            global_frame,
            text="Enable Adaptive Detection Tuning",
            variable=self.tuning_enabled_var,
            command=self.update_tuning_state
        ).pack(anchor="w", padx=10, pady=5)
        
        # Confidence thresholds
        conf_frame = ctk.CTkFrame(global_frame)
        conf_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(conf_frame, text="Confidence Floor:").grid(row=0, column=0, padx=5, pady=5)
        self.conf_floor_var = tk.DoubleVar(value=0.3)
        conf_floor_entry = ctk.CTkEntry(conf_frame, textvariable=self.conf_floor_var, width=100)
        conf_floor_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(conf_frame, text="Confidence Ceiling:").grid(row=0, column=2, padx=5, pady=5)
        self.conf_ceiling_var = tk.DoubleVar(value=0.7)
        conf_ceiling_entry = ctk.CTkEntry(conf_frame, textvariable=self.conf_ceiling_var, width=100)
        conf_ceiling_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # GoPro settings
        gopro_frame = ctk.CTkFrame(global_frame)
        gopro_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(gopro_frame, text="GoPro Wide-Angle Settings", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        self.edge_compensation_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            gopro_frame,
            text="Enable Edge Compensation (for GoPro distortion)",
            variable=self.edge_compensation_var
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(gopro_frame, text="FOV (degrees):").pack(anchor="w", padx=10, pady=2)
        self.fov_var = tk.DoubleVar(value=120.0)
        fov_entry = ctk.CTkEntry(gopro_frame, textvariable=self.fov_var, width=100)
        fov_entry.pack(anchor="w", padx=10, pady=2)
        
        # Stage 1 settings
        stage1_tuning_frame = ctk.CTkFrame(main_frame)
        stage1_tuning_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            stage1_tuning_frame,
            text="Stage 1 Settings (Lightweight - Speed Optimized)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.stage1_tuning_enabled_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            stage1_tuning_frame,
            text="Enable tuning for Stage 1",
            variable=self.stage1_tuning_enabled_var
        ).pack(anchor="w", padx=10, pady=5)
        
        # Stage 1 temporal smoothing
        ctk.CTkLabel(stage1_tuning_frame, text="Temporal Smoothing:").pack(anchor="w", padx=10, pady=2)
        self.stage1_smoothing_var = tk.DoubleVar(value=0.15)
        stage1_smoothing_entry = ctk.CTkEntry(stage1_tuning_frame, textvariable=self.stage1_smoothing_var, width=100)
        stage1_smoothing_entry.pack(anchor="w", padx=10, pady=2)
        
        # Stage 1 analysis window
        ctk.CTkLabel(stage1_tuning_frame, text="Scene Analysis Window (frames):").pack(anchor="w", padx=10, pady=2)
        self.stage1_window_frames_var = tk.IntVar(value=30)
        stage1_window_entry = ctk.CTkEntry(stage1_tuning_frame, textvariable=self.stage1_window_frames_var, width=100)
        stage1_window_entry.pack(anchor="w", padx=10, pady=2)
        
        # Stage 2 settings
        stage2_tuning_frame = ctk.CTkFrame(main_frame)
        stage2_tuning_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            stage2_tuning_frame,
            text="Stage 2 Settings (Full - Accuracy Optimized)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.stage2_tuning_enabled_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            stage2_tuning_frame,
            text="Enable tuning for Stage 2",
            variable=self.stage2_tuning_enabled_var
        ).pack(anchor="w", padx=10, pady=5)
        
        # Stage 2 temporal smoothing
        ctk.CTkLabel(stage2_tuning_frame, text="Temporal Smoothing:").pack(anchor="w", padx=10, pady=2)
        self.stage2_smoothing_var = tk.DoubleVar(value=0.08)
        stage2_smoothing_entry = ctk.CTkEntry(stage2_tuning_frame, textvariable=self.stage2_smoothing_var, width=100)
        stage2_smoothing_entry.pack(anchor="w", padx=10, pady=2)
        
        # Stage 2 analysis window
        ctk.CTkLabel(stage2_tuning_frame, text="Scene Analysis Window (frames):").pack(anchor="w", padx=10, pady=2)
        self.stage2_window_frames_var = tk.IntVar(value=90)
        stage2_window_entry = ctk.CTkEntry(stage2_tuning_frame, textvariable=self.stage2_window_frames_var, width=100)
        stage2_window_entry.pack(anchor="w", padx=10, pady=2)
        
        # MassMotion quality
        ctk.CTkLabel(stage2_tuning_frame, text="MassMotion Min Quality:").pack(anchor="w", padx=10, pady=2)
        self.massmotion_quality_var = tk.DoubleVar(value=0.4)
        quality_entry = ctk.CTkEntry(stage2_tuning_frame, textvariable=self.massmotion_quality_var, width=100)
        quality_entry.pack(anchor="w", padx=10, pady=2)
        
        # Save button
        save_btn = ctk.CTkButton(
            main_frame,
            text="Save Tuning Configuration",
            command=self.save_tuning_config,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        save_btn.pack(pady=20)
        
    def create_results_tab(self):
        """Create Results viewing tab."""
        tab = self.notebook.add("Results")
        
        # Results display area
        results_frame = ctk.CTkFrame(tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            width=80,
            height=40,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 11)
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def select_video(self, stage):
        """Select video file."""
        filename = filedialog.askopenfilename(
            title=f"Select Video File for Stage {stage[-1]}",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4"), ("All files", "*.*")]
        )
        if filename:
            if stage == "stage1":
                self.stage1_video_var.set(filename)
            else:
                self.stage2_segment_var.set(filename)
                
    def select_directory(self, stage):
        """Select output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            if stage == "stage1":
                self.stage1_output_var.set(dirname)
            else:
                self.stage2_output_var.set(dirname)
                
    def select_file(self, file_type):
        """Select configuration file."""
        filename = filedialog.askopenfilename(
            title=f"Select {file_type.title()} Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            if file_type == "lines":
                self.stage2_lines_var.set(filename)
            else:
                self.stage2_zones_var.set(filename)
    
    def create_lines_config(self):
        """Create lines configuration interactively using OpenCV."""
        # Get video path - try segment first, then prompt user
        video_path = None
        
        # Try to get from segment selection
        segment_path = self.stage2_segment_var.get()
        if segment_path and segment_path != "No segment selected" and Path(segment_path).exists():
            video_path = segment_path
        
        # If no segment, try Stage 1 video
        if not video_path:
            stage1_video = self.stage1_video_var.get()
            if stage1_video and stage1_video != "No video selected" and Path(stage1_video).exists():
                video_path = stage1_video
        
        # If still no video, ask user
        if not video_path:
            video_path = filedialog.askopenfilename(
                title="Select video file for line editing",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if not video_path:
                messagebox.showwarning("Warning", "Please select a video file to create lines configuration")
                return
        
        if not Path(video_path).exists():
            messagebox.showerror("Error", f"Video file does not exist: {video_path}")
            return
        
        # Ask for output path
        default_name = Path(video_path).stem + "_counting_lines.json"
        default_dir = Path(video_path).parent
        output_path = filedialog.asksaveasfilename(
            title="Save Lines Configuration",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        try:
            # Import and use LineEditor
            from line_editor import LineEditor
            
            self.log_message("stage2", f"Launching Line Editor with video: {Path(video_path).name}")
            self.log_message("stage2", f"Output will be saved to: {output_path}")
            
            # Show instructions in a persistent window (non-modal)
            instructions_window = ctk.CTkToplevel(self.root)
            instructions_window.title("Line Editor - Instructions")
            instructions_window.geometry("500x600")
            instructions_window.attributes('-topmost', True)  # Keep on top
            
            # Instructions text
            instructions_text = """Line Editor - Instructions

STEP 1 - SELECT A FRAME (Current Step):
• SPACE: Pause/Play video
• LEFT/RIGHT arrows: Navigate frames
• ENTER: Select this frame → Switches to DRAW mode

STEP 2 - DRAW LINES (After pressing ENTER):
• LEFT CLICK anywhere on frame: Place point (need 2 points)
• RIGHT CLICK: Remove last point
• ENTER: Complete line (after placing 2 points)
• 'N': Start a new line
• 'T': Edit direction names  
• 'S': Save and quit
• 'Q': Quit without saving

IMPORTANT: 
1. Press ENTER in frame selection mode
2. The mode will change to "DRAW LINE"
3. LEFT CLICK on the video frame to place points

The OpenCV editor window will open now.
Keep this window open for reference."""
            
            text_widget = scrolledtext.ScrolledText(
                instructions_window,
                wrap=tk.WORD,
                width=60,
                height=25,
                bg="#2b2b2b",
                fg="white",
                font=("Consolas", 10)
            )
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)
            text_widget.insert("1.0", instructions_text)
            text_widget.configure(state="disabled")
            
            # Close button
            close_btn = ctk.CTkButton(
                instructions_window,
                text="Close Instructions",
                command=instructions_window.destroy,
                width=200
            )
            close_btn.pack(pady=10)
            
            # Run line editor in a separate thread to avoid blocking GUI
            def run_editor():
                try:
                    editor = LineEditor(video_path, output_path)
                    success = editor.run()
                    # Update GUI with created file
                    if success and Path(output_path).exists():
                        self.root.after(0, lambda: self.stage2_lines_var.set(output_path))
                        self.root.after(0, lambda: self.log_message("stage2", f"✅ Lines configuration saved: {output_path}"))
                        self.root.after(0, lambda: messagebox.showinfo("Success", f"Lines configuration saved to:\n{output_path}"))
                    else:
                        self.root.after(0, lambda: messagebox.showwarning("Warning", "Line editor closed without saving"))
                except Exception as e:
                    import traceback
                    error_msg = f"Line editor failed: {str(e)}\n{traceback.format_exc()}"
                    self.root.after(0, lambda: self.log_message("stage2", f"❌ Error: {error_msg}"))
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Line editor failed:\n{str(e)}"))
            
            thread = threading.Thread(target=run_editor, daemon=True)
            thread.start()
            
        except ImportError as e:
            messagebox.showerror("Error", f"Line editor module not found: {e}\n\nPlease ensure line_editor.py is in the project root.")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to launch line editor:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def create_zones_config(self):
        """Create zones configuration interactively using OpenCV."""
        # Get video path - try segment first, then prompt user
        video_path = None
        
        # Try to get from segment selection
        segment_path = self.stage2_segment_var.get()
        if segment_path and segment_path != "No segment selected" and Path(segment_path).exists():
            video_path = segment_path
        
        # If no segment, try Stage 1 video
        if not video_path:
            stage1_video = self.stage1_video_var.get()
            if stage1_video and stage1_video != "No video selected" and Path(stage1_video).exists():
                video_path = stage1_video
        
        # If still no video, ask user
        if not video_path:
            video_path = filedialog.askopenfilename(
                title="Select video file for zone editing",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if not video_path:
                messagebox.showwarning("Warning", "Please select a video file to create zones configuration")
                return
        
        if not Path(video_path).exists():
            messagebox.showerror("Error", f"Video file does not exist: {video_path}")
            return
        
        # Ask for output path
        default_name = Path(video_path).stem + "_zones.json"
        default_dir = Path(video_path).parent
        output_path = filedialog.asksaveasfilename(
            title="Save Zones Configuration",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        try:
            # Import and use ZoneEditor
            from zone_editor import ZoneEditor
            
            self.log_message("stage2", f"Launching Zone Editor with video: {Path(video_path).name}")
            self.log_message("stage2", f"Output will be saved to: {output_path}")
            
            # Show instructions in a persistent window (non-modal)
            instructions_window = ctk.CTkToplevel(self.root)
            instructions_window.title("Zone Editor - Instructions")
            instructions_window.geometry("500x650")
            instructions_window.attributes('-topmost', True)  # Keep on top
            
            # Instructions text
            instructions_text = """Zone Editor - Instructions

STEP 1 - SELECT A FRAME (Current Step):
• SPACE: Pause/Play video
• LEFT/RIGHT arrows: Navigate frames
• ENTER: Select this frame → Switches to DRAW mode

STEP 2 - DRAW ZONES (After pressing ENTER):
• Press 'N' or 'D' to start a new zone (do this first!)
• LEFT CLICK on video frame: Add point to polygon
• RIGHT CLICK: Remove last point
• ENTER: Complete polygon (need min 3 points)
• 'T': Toggle zone type (ingress/egress/counting)
• 'S': Save and quit
• 'Q': Quit without saving

IMPORTANT: 
1. Press ENTER in frame selection mode
2. The mode will change to "DRAW ZONE"
3. Press 'N' to start drawing a zone
4. LEFT CLICK on the video frame to add points

The OpenCV editor window will open now.
Keep this window open for reference."""
            
            text_widget = scrolledtext.ScrolledText(
                instructions_window,
                wrap=tk.WORD,
                width=60,
                height=27,
                bg="#2b2b2b",
                fg="white",
                font=("Consolas", 10)
            )
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)
            text_widget.insert("1.0", instructions_text)
            text_widget.configure(state="disabled")
            
            # Close button
            close_btn = ctk.CTkButton(
                instructions_window,
                text="Close Instructions",
                command=instructions_window.destroy,
                width=200
            )
            close_btn.pack(pady=10)
            
            # Run zone editor in a separate thread to avoid blocking GUI
            def run_editor():
                try:
                    editor = ZoneEditor(video_path, output_path)
                    success = editor.run()
                    # Update GUI with created file
                    if success and Path(output_path).exists():
                        self.root.after(0, lambda: self.stage2_zones_var.set(output_path))
                        self.root.after(0, lambda: self.log_message("stage2", f"✅ Zones configuration saved: {output_path}"))
                        self.root.after(0, lambda: messagebox.showinfo("Success", f"Zones configuration saved to:\n{output_path}"))
                    else:
                        self.root.after(0, lambda: messagebox.showwarning("Warning", "Zone editor closed without saving"))
                except Exception as e:
                    import traceback
                    error_msg = f"Zone editor failed: {str(e)}\n{traceback.format_exc()}"
                    self.root.after(0, lambda: self.log_message("stage2", f"❌ Error: {error_msg}"))
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Zone editor failed:\n{str(e)}"))
            
            thread = threading.Thread(target=run_editor, daemon=True)
            thread.start()
            
        except ImportError as e:
            messagebox.showerror("Error", f"Zone editor module not found: {e}\n\nPlease ensure zone_editor.py is in the project root.")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to launch zone editor:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def update_tuning_state(self):
        """Update tuning configuration state."""
        enabled = self.tuning_enabled_var.get()
        # Enable/disable tuning controls based on state
        
    def save_tuning_config(self):
        """Save tuning configuration to config file."""
        try:
            config_data = {
                "detection_tuning": {
                    "adaptive_confidence": self.tuning_enabled_var.get(),
                    "confidence_floor": self.conf_floor_var.get(),
                    "confidence_ceiling": self.conf_ceiling_var.get(),
                    "enable_edge_compensation": self.edge_compensation_var.get(),
                    "gopro_fov_degrees": self.fov_var.get(),
                    "mass_motion_min_quality": self.massmotion_quality_var.get(),
                    "stage1": {
                        "enabled": self.stage1_tuning_enabled_var.get(),
                        "temporal_smoothing": self.stage1_smoothing_var.get(),
                        "scene_analysis_window": self.stage1_window_frames_var.get(),
                        "lightweight_mode": True
                    },
                    "stage2": {
                        "enabled": self.stage2_tuning_enabled_var.get(),
                        "temporal_smoothing": self.stage2_smoothing_var.get(),
                        "scene_analysis_window": self.stage2_window_frames_var.get(),
                        "lightweight_mode": False,
                        "mass_motion_quality_enforcement": True
                    }
                }
            }
            
            # Save to config file
            config_path = Path("config") / "default_config.json"
            with open(config_path, 'r') as f:
                full_config = json.load(f)
            
            full_config.update(config_data)
            
            with open(config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            
            messagebox.showinfo("Success", "Tuning configuration saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def log_message(self, stage, message):
        """Log message to appropriate log window."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if stage == "stage1":
            self.stage1_log.insert(tk.END, log_entry)
            self.stage1_log.see(tk.END)
        else:
            self.stage2_log.insert(tk.END, log_entry)
            self.stage2_log.see(tk.END)
    
    def run_stage1(self):
        """Run Stage 1 analysis."""
        if self.processing:
            messagebox.showwarning("Warning", "Another analysis is already running")
            return
        
        # Validate inputs
        video_path = self.stage1_video_var.get()
        if not video_path or video_path == "No video selected":
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not Path(video_path).exists():
            messagebox.showerror("Error", "Video file does not exist")
            return
        
        camera_id = self.stage1_camera_var.get().strip()
        if not camera_id:
            messagebox.showerror("Error", "Please enter a camera ID")
            return
        
        output_path = self.stage1_output_var.get()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output directory")
            return
        
        # Get video info for progress tracking
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except:
            total_frames_video = 0
            fps_video = 0
        
        # Run in thread
        def run_analysis():
            try:
                self.processing = True
                start_time = time.time()
                self.root.after(0, lambda: self.stage1_run_btn.configure(state="disabled"))
                self.root.after(0, lambda: self.stage1_status_var.set("Running..."))
                self.root.after(0, lambda: self.stage1_progress.set(0.1))
                
                self.log_message("stage1", f"Starting Stage 1 analysis for camera: {camera_id}")
                self.log_message("stage1", f"Video: {video_path}")
                if total_frames_video > 0:
                    self.log_message("stage1", f"Video info: {total_frames_video:,} frames @ {fps_video:.2f} fps")
                
                # Extract source directory from video path
                source_path = str(Path(video_path).parent.parent)
                
                # Progress tracking helper
                last_update_time = time.time()
                frames_processed = 0
                
                # Simulate progress updates (since workflow doesn't provide direct callbacks)
                def update_progress_simulation():
                    nonlocal frames_processed, last_update_time
                    if not self.processing:
                        return
                    
                    elapsed = time.time() - start_time
                    if total_frames_video > 0 and elapsed > 0:
                        # Estimate based on elapsed time
                        estimated_frames = int(min(total_frames_video * 0.8, elapsed * fps_video * 0.5))
                        if estimated_frames > frames_processed:
                            frames_processed = estimated_frames
                            speed = frames_processed / elapsed if elapsed > 0 else 0
                            self.root.after(0, lambda: self.update_progress("stage1", frames_processed, total_frames_video, speed))
                    
                    if self.processing:
                        self.root.after(500, update_progress_simulation)
                
                # Start progress simulation
                self.root.after(500, update_progress_simulation)
                
                # Run Stage 1
                results = self.workflow.stage1_peak_detection(
                    camera_id=camera_id,
                    source_path=source_path,
                    output_path=output_path
                )
                
                elapsed_time = time.time() - start_time
                if total_frames_video > 0:
                    final_speed = total_frames_video / elapsed_time if elapsed_time > 0 else 0
                    self.root.after(0, lambda: self.update_progress("stage1", total_frames_video, total_frames_video, final_speed))
                else:
                    self.root.after(0, lambda: self.stage1_progress.set(1.0))
                
                self.root.after(0, lambda: self.stage1_status_var.set("Complete"))
                
                if results.get('status') == 'failed':
                    self.log_message("stage1", f"Error: {results.get('error', 'Unknown error')}")
                    messagebox.showerror("Error", f"Stage 1 failed: {results.get('error')}")
                else:
                    self.log_message("stage1", f"Stage 1 analysis completed in {elapsed_time:.1f} seconds!")
                    if 'peak_window' in results:
                        peak = results['peak_window']
                        self.log_message("stage1", f"Peak period: {peak['start_seconds']}s - {peak['end_seconds']}s")
                        messagebox.showinfo("Success", f"Stage 1 complete!\nPeak period identified: {peak['start_seconds']}s - {peak['end_seconds']}s")
                
            except Exception as e:
                self.log_message("stage1", f"Error: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            finally:
                self.processing = False
                self.root.after(0, lambda: self.stage1_run_btn.configure(state="normal"))
                if total_frames_video == 0:
                    self.root.after(0, lambda: self.stage1_progress.set(0))
        
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
    
    def run_stage2(self):
        """Run Stage 2 analysis."""
        if self.processing:
            messagebox.showwarning("Warning", "Another analysis is already running")
            return
        
        # Determine segment path
        if self.use_stage1_output_var.get():
            # Use Stage 1 output automatically
            stage1_output = self.stage1_output_var.get()
            segments_path = str(Path(stage1_output) / "stage1" / "videos")
        else:
            segment_path = self.stage2_segment_var.get()
            if not segment_path or segment_path == "No segment selected":
                messagebox.showerror("Error", "Please select a peak segment or enable 'Use Stage 1 output'")
                return
            segments_path = str(Path(segment_path).parent) if Path(segment_path).is_file() else segment_path
        
        output_path = self.stage2_output_var.get()
        config_path = str(Path.cwd() / "config")
        
        lines_config = self.stage2_lines_var.get() if self.stage2_lines_var.get() else None
        zones_config = self.stage2_zones_var.get() if self.stage2_zones_var.get() else None
        
        # Get analysis configuration
        mode = self.stage2_mode_var.get()
        
        # Get selected FPS options
        selected_fps = [fps for fps, var in self.stage2_fps_vars.items() if var.get()]
        if not selected_fps:
            messagebox.showerror("Error", "Please select at least one FPS option")
            return
        
        # Get selected analysis types (whole frame vs zone-only)
        selected_analysis_types = [atype for atype, var in self.stage2_analysis_type_vars.items() if var.get()]
        if not selected_analysis_types:
            messagebox.showerror("Error", "Please select at least one analysis type (Whole Frame or Zone Only)")
            return
        
        # Get selected methods
        selected_methods = [method for method, var in self.stage2_method_vars.items() if var.get()]
        if not selected_methods:
            messagebox.showerror("Error", "Please select at least one detection method")
            return
        
        # Validate production mode (single selection)
        if mode == "production":
            if len(selected_fps) > 1:
                messagebox.showwarning("Warning", "Production mode: Only one FPS should be selected. Using the first selection.")
                selected_fps = [selected_fps[0]]
            if len(selected_analysis_types) > 1:
                messagebox.showwarning("Warning", "Production mode: Only one analysis type should be selected. Using 'whole' if available.")
                if "whole" in selected_analysis_types:
                    selected_analysis_types = ["whole"]
                else:
                    selected_analysis_types = [selected_analysis_types[0]]
            if len(selected_methods) > 1:
                messagebox.showwarning("Warning", "Production mode: Only one method should be selected. Using 'pedestrian'.")
                selected_methods = ["pedestrian"]
        
        # Generate all combinations
        from itertools import product
        analysis_configs = list(product(selected_fps, selected_analysis_types, selected_methods))
        
        # Calculate total analysis runs
        total_runs = len(analysis_configs)
        
        if mode == "testing" and total_runs > 1:
            config_summary = "\n".join([
                f"  • {fps}fps, {analysis_type}, {method}"
                for fps, analysis_type, method in analysis_configs
            ])
            confirm_msg = (
                f"Testing Mode: {total_runs} analysis runs will be performed:\n\n"
                f"{config_summary}\n\n"
                f"This will take significantly longer. Continue?"
            )
            if not messagebox.askyesno("Confirm Testing Mode", confirm_msg):
                return
        
        # Log configuration
        self.log_message("stage2", f"Mode: {mode.upper()}")
        self.log_message("stage2", f"Total analysis runs: {total_runs}")
        for i, (fps, analysis_type, method) in enumerate(analysis_configs, 1):
            self.log_message("stage2", f"  Run {i}: {fps}fps, {analysis_type} frame, {method} method")
        
        # Try to get segment info for progress tracking
        total_frames_stage2 = 0
        fps_stage2 = 30.0  # Stage 2 typically uses 30fps
        
        if Path(segments_path).is_file():
            try:
                import cv2
                cap = cv2.VideoCapture(segments_path)
                total_frames_stage2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_stage2 = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            except:
                pass
        elif Path(segments_path).is_dir():
            # Estimate from directory
            video_files = list(Path(segments_path).glob("*.mp4"))
            if video_files:
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(video_files[0]))
                    frames_per_segment = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_stage2 = cap.get(cv2.CAP_PROP_FPS)
                    total_frames_stage2 = frames_per_segment * len(video_files)
                    cap.release()
                except:
                    pass
        
        # Run in thread
        def run_analysis():
            try:
                self.processing = True
                start_time = time.time()
                self.root.after(0, lambda: self.stage2_run_btn.configure(state="disabled"))
                self.root.after(0, lambda: self.stage2_status_var.set("Running..."))
                self.root.after(0, lambda: self.stage2_progress.set(0.1))
                
                self.log_message("stage2", "Starting Stage 2 analysis")
                self.log_message("stage2", f"Segments path: {segments_path}")
                if total_frames_stage2 > 0:
                    self.log_message("stage2", f"Estimated: {total_frames_stage2:,} frames @ {fps_stage2:.2f} fps")
                
                # Progress tracking helper
                last_update_time = time.time()
                frames_processed = 0
                
                # Simulate progress updates
                def update_progress_simulation():
                    nonlocal frames_processed, last_update_time
                    if not self.processing:
                        return
                    
                    elapsed = time.time() - start_time
                    if total_frames_stage2 > 0 and elapsed > 0:
                        # Estimate based on elapsed time (Stage 2 is slower)
                        estimated_frames = int(min(total_frames_stage2 * 0.8, elapsed * fps_stage2 * 0.3))
                        if estimated_frames > frames_processed:
                            frames_processed = estimated_frames
                            speed = frames_processed / elapsed if elapsed > 0 else 0
                            self.root.after(0, lambda: self.update_progress("stage2", frames_processed, total_frames_stage2, speed))
                    
                    if self.processing:
                        self.root.after(500, update_progress_simulation)
                
                # Start progress simulation
                self.root.after(500, update_progress_simulation)
                
                # Import PeakWindowAnalyzer for direct control
                from analyze_peak_window import PeakWindowAnalyzer, load_counting_zone
                
                # Get device from hardware detector
                from app.workflow_estimator import HardwareDetector
                hardware_detector = HardwareDetector()
                device = hardware_detector.device
                
                all_results = {}
                output_base = Path(output_path)
                output_base.mkdir(parents=True, exist_ok=True)
                
                # Find the video file(s) to analyze
                if Path(segments_path).is_file():
                    video_files = {segments_path: Path(segments_path)}
                elif Path(segments_path).is_dir():
                    video_files = {str(f): f for f in Path(segments_path).glob("*.mp4")}
                else:
                    raise FileNotFoundError(f"Segment path not found: {segments_path}")
                
                if not video_files:
                    raise FileNotFoundError(f"No video files found in: {segments_path}")
                
                # Load zone config if zone analysis is needed
                zone_polygon = None
                if "zone" in selected_analysis_types and zones_config and Path(zones_config).exists():
                    zone_data = load_counting_zone(zones_config)
                    if zone_data:
                        zone_polygon = zone_data
                        self.log_message("stage2", f"Loaded zone from: {zones_config}")
                
                # Run each configuration
                for run_num, (fps_str, analysis_type, method) in enumerate(analysis_configs, 1):
                    fps_value = float(fps_str)
                    zone_only = (analysis_type == "zone")
                    
                    run_name = f"{fps_str}fps_{analysis_type}_{method}"
                    self.log_message("stage2", f"\n{'='*60}")
                    self.log_message("stage2", f"Run {run_num}/{total_runs}: {run_name}")
                    self.log_message("stage2", f"{'='*60}")
                    
                    # Find appropriate video file (match FPS if possible)
                    video_to_use = None
                    for video_path_str, video_path_obj in video_files.items():
                        # Try to match FPS in filename or use first available
                        if fps_str in video_path_str.lower() or len(video_files) == 1:
                            video_to_use = video_path_obj
                            break
                    
                    if not video_to_use:
                        video_to_use = list(video_files.values())[0]
                    
                    # Create analyzer for this run
                    analyzer = PeakWindowAnalyzer(
                        device=device,
                        use_cropped=False,  # Can add crop support later
                        crop_region=None
                    )
                    
                    # Set zone if needed
                    if zone_only:
                        if not zone_polygon:
                            self.log_message("stage2", f"⚠️ Zone needed for {run_name} but not configured. Skipping.")
                            all_results[run_name] = {'success': False, 'error': 'Zone not configured'}
                            continue
                        analyzer.counting_zone = zone_polygon
                    
                    try:
                        # Run analysis on the video segment
                        result = analyzer.analyze_segment(
                            str(video_to_use),
                            detection_method=method,
                            zone_only=zone_only,
                            fps_label=fps_str
                        )
                        
                        if result:
                            # Save individual result
                            run_output_dir = output_base / run_name
                            run_output_dir.mkdir(parents=True, exist_ok=True)
                            
                            result_file = run_output_dir / "analysis_results.json"
                            import json
                            from datetime import datetime
                            with open(result_file, 'w') as f:
                                json.dump({
                                    'run_name': run_name,
                                    'config': {
                                        'fps': fps_str,
                                        'analysis_type': analysis_type,
                                        'method': method,
                                        'zone_only': zone_only
                                    },
                                    'timestamp': datetime.now().isoformat(),
                                    'results': result
                                }, f, indent=2)
                            
                            all_results[run_name] = {
                                'success': True,
                                'result': result,
                                'output_dir': str(run_output_dir)
                            }
                            self.log_message("stage2", f"✅ {run_name} completed successfully")
                        else:
                            all_results[run_name] = {'success': False, 'error': 'Analysis returned None'}
                            
                    except Exception as e:
                        self.log_message("stage2", f"❌ Error in {run_name}: {str(e)}")
                        all_results[run_name] = {'success': False, 'error': str(e)}
                    
                    # Update overall progress
                    run_progress = run_num / total_runs
                    self.root.after(0, lambda p=run_progress: self.stage2_progress.set(p))
                
                # Create summary
                successful_runs = sum(1 for r in all_results.values() if r.get('success', False))
                results = {
                    'status': 'complete' if successful_runs == total_runs else 'partial',
                    'total_runs': total_runs,
                    'successful_runs': successful_runs,
                    'failed_runs': total_runs - successful_runs,
                    'results': all_results,
                    'output_path': str(output_base)
                }
                
                elapsed_time = time.time() - start_time
                if total_frames_stage2 > 0:
                    final_speed = total_frames_stage2 / elapsed_time if elapsed_time > 0 else 0
                    self.root.after(0, lambda: self.update_progress("stage2", total_frames_stage2, total_frames_stage2, final_speed))
                else:
                    self.root.after(0, lambda: self.stage2_progress.set(1.0))
                
                self.root.after(0, lambda: self.stage2_status_var.set("Complete"))
                
                if results.get('status') == 'failed':
                    self.log_message("stage2", f"Error: {results.get('error', 'Unknown error')}")
                    messagebox.showerror("Error", f"Stage 2 failed: {results.get('error')}")
                else:
                    self.log_message("stage2", f"Stage 2 analysis completed in {elapsed_time:.1f} seconds!")
                    self.log_message("stage2", f"Segments processed: {results.get('segments_processed', 0)}")
                    messagebox.showinfo("Success", f"Stage 2 complete!\nProcessed {results.get('segments_processed', 0)} segments")
                
            except Exception as e:
                self.log_message("stage2", f"Error: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            finally:
                self.processing = False
                self.root.after(0, lambda: self.stage2_run_btn.configure(state="normal"))
                if total_frames_stage2 == 0:
                    self.root.after(0, lambda: self.stage2_progress.set(0))
        
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
    
    def file_menu(self):
        """File menu actions."""
        pass
    
    def open_settings(self):
        """Open settings dialog."""
        messagebox.showinfo("Settings", "Settings dialog coming soon!")
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
Two-Stage Pedestrian Detection Workflow

STAGE 1: Peak Detection
- Fast scanning to identify peak activity periods
- Recommended: 15fps processing
- Output: Peak 15-minute windows

STAGE 2: Detailed Analysis
- High-accuracy analysis for MassMotion simulation
- Required: 30fps processing
- Output: Detailed trajectories and flow rates

DETECTION TUNING
- Adaptive confidence thresholds based on scene conditions
- GoPro wide-angle camera support with edge compensation
- Stage-specific optimization (speed vs accuracy)

PROGRESS & RESOURCES
- Real-time frame progress and processing speed
- CPU, RAM, and GPU usage monitoring
- Visual progress bars and resource indicators
        """
        messagebox.showinfo("Help", help_text)
    
    def start_resource_monitoring(self):
        """Start background resource monitoring."""
        self.monitoring_active = True
        
        def monitor_resources():
            """Monitor system resources in background thread."""
            process = psutil.Process()
            
            while self.monitoring_active:
                try:
                    # CPU usage (per core average)
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    
                    # RAM usage
                    ram_info = psutil.virtual_memory()
                    ram_percent = ram_info.percent
                    ram_used_gb = ram_info.used / (1024**3)
                    ram_total_gb = ram_info.total / (1024**3)
                    
                    # GPU usage (if available)
                    gpu_percent = None
                    gpu_memory_percent = None
                    if GPU_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0]
                                gpu_percent = gpu.load * 100
                                gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                        except:
                            pass
                    
                    # Update Stage 1 indicators
                    self.root.after(0, lambda: self.update_resource_display(
                        "stage1", cpu_percent, ram_percent, ram_used_gb, ram_total_gb,
                        gpu_percent, gpu_memory_percent
                    ))
                    
                    # Update Stage 2 indicators
                    self.root.after(0, lambda: self.update_resource_display(
                        "stage2", cpu_percent, ram_percent, ram_used_gb, ram_total_gb,
                        gpu_percent, gpu_memory_percent
                    ))
                    
                    time.sleep(0.5)  # Update every 500ms
                    
                except Exception as e:
                    # Silently handle errors in monitoring
                    time.sleep(1)
        
        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
    
    def update_resource_display(self, stage, cpu_percent, ram_percent, ram_used_gb, ram_total_gb,
                                gpu_percent=None, gpu_memory_percent=None):
        """Update resource usage displays."""
        if stage == "stage1":
            # CPU
            self.stage1_cpu_var.set(f"{cpu_percent:.1f}%")
            self.stage1_cpu_bar.set(cpu_percent / 100)
            
            # RAM
            self.stage1_ram_var.set(f"{ram_percent:.1f}% ({ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB)")
            self.stage1_ram_bar.set(ram_percent / 100)
            
            # GPU
            if self.stage1_gpu_frame is not None:
                if gpu_percent is not None:
                    self.stage1_gpu_var.set(f"{gpu_percent:.1f}%")
                    self.stage1_gpu_bar.set(gpu_percent / 100)
                else:
                    self.stage1_gpu_var.set("N/A")
                    self.stage1_gpu_bar.set(0)
        else:  # stage2
            # CPU
            self.stage2_cpu_var.set(f"{cpu_percent:.1f}%")
            self.stage2_cpu_bar.set(cpu_percent / 100)
            
            # RAM
            self.stage2_ram_var.set(f"{ram_percent:.1f}% ({ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB)")
            self.stage2_ram_bar.set(ram_percent / 100)
            
            # GPU
            if self.stage2_gpu_frame is not None:
                if gpu_percent is not None:
                    self.stage2_gpu_var.set(f"{gpu_percent:.1f}%")
                    self.stage2_gpu_bar.set(gpu_percent / 100)
                else:
                    self.stage2_gpu_var.set("N/A")
                    self.stage2_gpu_bar.set(0)
    
    def update_progress(self, stage, current_frame, total_frames, speed=None):
        """Update progress indicators."""
        if total_frames > 0:
            progress = current_frame / total_frames
            percent = progress * 100
            
            if stage == "stage1":
                self.stage1_progress.set(progress)
                self.stage1_progress_text_var.set(
                    f"{current_frame:,} / {total_frames:,} frames ({percent:.1f}%)"
                )
                if speed is not None:
                    self.stage1_speed_var.set(f"Speed: {speed:.1f} fps")
                    self.processing_speed = speed
            else:  # stage2
                self.stage2_progress.set(progress)
                self.stage2_progress_text_var.set(
                    f"{current_frame:,} / {total_frames:,} frames ({percent:.1f}%)"
                )
                if speed is not None:
                    self.stage2_speed_var.set(f"Speed: {speed:.1f} fps")
                    self.processing_speed = speed
    
    def run(self):
        """Run the GUI."""
        try:
            self.root.mainloop()
        finally:
            self.monitoring_active = False


def main():
    """Main entry point."""
    app = WorkflowGUI()
    app.run()


if __name__ == "__main__":
    main()

