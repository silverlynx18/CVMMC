#!/usr/bin/env python3
"""
Unified Video Preprocessing Module

Handles both scenarios:
1. Multiple GoPro segments → Concatenate + create FPS versions
2. Concatenated video → Create FPS versions only
"""

import os
import subprocess
import cv2
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Unified video preprocessing that handles both scenarios:
    - Multiple GoPro segments → Concatenate + create FPS versions
    - Concatenated video → Create FPS versions only
    """
    
    def __init__(self, output_dir: str = "processed_videos"):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory for processed videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess(self, 
                   input_type: str,
                   source_path: str,
                   create_versions: Dict[str, bool] = None) -> Dict[str, Path]:
        """
        Process input videos based on scenario.
        
        Args:
            input_type: "multiple_segments" or "concatenated_video"
            source_path: Folder path (segments) or video file path (concatenated)
            create_versions: Dict of {fps: bool} for which versions to create
                           e.g., {"1fps": True, "15fps": False, "30fps": True}
        
        Returns:
            dict: Paths to created videos
                - full_fps: Path to full-FPS video (always)
                - fps_1: Path to 1fps video (if created)
                - fps_15: Path to 15fps video (if created)
        """
        create_versions = create_versions or {"1fps": True, "30fps": True}
        output_paths = {}
        
        logger.info(f"Preprocessing: {input_type}")
        logger.info(f"Source: {source_path}")
        
        # Step 1: Get full-FPS video (concatenate if needed)
        if input_type == "multiple_segments":
            logger.info("Concatenating GoPro segments...")
            full_fps_path = self._concatenate_segments(
                source_path,
                self.output_dir / "concatenated_full_fps.mp4"
            )
            output_paths['full_fps'] = full_fps_path
            
        elif input_type == "concatenated_video":
            logger.info("Using existing concatenated video...")
            full_fps_path = Path(source_path)
            if not full_fps_path.exists():
                raise FileNotFoundError(f"Video not found: {source_path}")
            output_paths['full_fps'] = full_fps_path
        
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        # Step 2: Create FPS versions as requested
        base_name = full_fps_path.stem
        
        if create_versions.get("1fps", False):
            logger.info("Creating 1fps video for Stage 1 analysis...")
            fps_1_path = self._create_fps_version(
                full_fps_path,
                self.output_dir / f"{base_name}_1fps.mp4",
                target_fps=1.0
            )
            output_paths['fps_1'] = fps_1_path
        
        if create_versions.get("15fps", False):
            logger.info("Creating 15fps video...")
            fps_15_path = self._create_fps_version(
                full_fps_path,
                self.output_dir / f"{base_name}_15fps.mp4",
                target_fps=15.0
            )
            output_paths['fps_15'] = fps_15_path
        
        if create_versions.get("30fps", False) and input_type == "multiple_segments":
            # Only create 30fps if we concatenated (otherwise use original)
            # For concatenated input, 30fps version is the original
            logger.info("30fps version available (concatenated video)")
            output_paths['fps_30'] = full_fps_path
        
        logger.info("✅ Preprocessing complete!")
        return output_paths
    
    def _concatenate_segments(self, folder_path: str, output_path: Path) -> Path:
        """
        Concatenate multiple GoPro video segments.
        
        Uses FFmpeg concat demuxer for fast, no-re-encode concatenation.
        """
        folder = Path(folder_path)
        
        # Find all GoPro video files
        video_files = sorted([
            f for f in folder.iterdir()
            if f.name.startswith('GX') and f.suffix.upper() == '.MP4'
        ])
        
        if not video_files:
            raise ValueError(f"No GX*.MP4 files found in {folder_path}")
        
        logger.info(f"Found {len(video_files)} video segments")
        
        # Create concat filelist
        filelist_path = folder / "concat_filelist.txt"
        with open(filelist_path, 'w') as f:
            for video_file in video_files:
                abs_path = os.path.abspath(str(video_file))
                f.write(f"file '{abs_path}'\n")
        
        try:
            # Use FFmpeg concat demuxer (fast, no re-encoding)
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(filelist_path),
                '-c', 'copy',  # Copy streams without re-encoding
                '-avoid_negative_ts', 'make_zero',
                '-y',
                str(output_path)
            ]
            
            logger.info("Concatenating segments (no re-encoding)...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output
            if not output_path.exists():
                raise RuntimeError("Concatenation completed but output file not found")
            
            # Get video info
            cap = cv2.VideoCapture(str(output_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open output video: {output_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            logger.info(f"✅ Concatenated {len(video_files)} segments")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   Duration: {duration/60:.1f} minutes @ {fps:.2f} fps")
            logger.info(f"   Resolution: {width}x{height}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
        finally:
            # Clean up temp file
            if filelist_path.exists():
                filelist_path.unlink()
    
    def _create_fps_version(self, 
                           input_video: Path,
                           output_path: Path,
                           target_fps: float) -> Path:
        """
        Create a version of video at specific FPS.
        
        Args:
            input_video: Input video path
            output_path: Output video path
            target_fps: Target frame rate
        
        Returns:
            Path to created video
        """
        logger.info(f"Creating {target_fps}fps version from {input_video.name}...")
        
        # Use FFmpeg to extract frames at target FPS
        cmd = [
            'ffmpeg',
            '-i', str(input_video),
            '-vf', f'fps={target_fps}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if not output_path.exists():
                raise RuntimeError("FPS conversion completed but output file not found")
            
            # Verify output
            cap = cv2.VideoCapture(str(output_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open output video: {output_path}")
            
            output_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            logger.info(f"✅ Created {target_fps}fps video: {output_path}")
            logger.info(f"   Output FPS: {output_fps:.2f}")
            logger.info(f"   Frames: {frame_count}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error creating {target_fps}fps version: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
    
    def get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = frame_count / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60
        
        cap.release()
        
        return {
            'path': str(video_path),
            'fps': float(fps),
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_minutes,
            'resolution': f"{width}x{height}"
        }
    
    def detect_input_type(self, source_path: str) -> str:
        """
        Automatically detect if input is segments or concatenated video.
        
        Returns:
            "multiple_segments" or "concatenated_video"
        """
        path = Path(source_path)
        
        if path.is_file():
            # Single file - concatenated video
            return "concatenated_video"
        elif path.is_dir():
            # Directory - check for GoPro segments
            video_files = list(path.glob("GX*.MP4")) + list(path.glob("GX*.mp4"))
            if len(video_files) > 1:
                return "multiple_segments"
            else:
                # Could be directory with single file - treat as concatenated
                return "concatenated_video"
        else:
            raise ValueError(f"Path not found: {source_path}")

