"""Video file processing for batch analysis."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm

# SAM detector removed - using YOLO-only detection
# Detection class imported from detection_utils if needed
from .pedestrian_tracker import PedestrianTracker
from .fruin_analysis import FruinAnalyzer, FruinLOSResult

logger = logging.getLogger(__name__)


@dataclass
class VideoFile:
    """Video file information."""
    path: str
    name: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
class AnalysisResult:
    """Result of video analysis."""
    video_file: VideoFile
    start_time: datetime
    end_time: datetime
    total_detections: int
    total_ingress: int
    total_egress: int
    avg_density: float
    peak_density: float
    avg_los: str
    peak_los: str
    frame_results: List[Dict]


class VideoProcessor:
    """Process video files for pedestrian counting and analysis."""
    
    def __init__(self, sam2_model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize video processor.
        
        Args:
            sam2_model_path: Path to SAM2 model file (deprecated, detector set separately)
            device: Device to run inference on
        """
        # Detector will be set by subclass (e.g., YOLO detector)
        self.sam2_detector = None
        self.fruin_analyzer = FruinAnalyzer()
        self.device = device
        
        # Analysis parameters
        self.frame_skip = 1  # Process every Nth frame
        self.analysis_window_minutes = 5
        self.zone_area_sqm = 25.0  # Default area in square meters
        
    def load_video(self, video_path: str) -> Optional[VideoFile]:
        """
        Load video file and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoFile object or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return VideoFile(
                path=video_path,
                name=os.path.basename(video_path),
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=frame_count
            )
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None
    
    def process_video(self, video_path: str, 
                     zones: Optional[Dict] = None,
                     progress_callback: Optional[Callable] = None) -> Optional[AnalysisResult]:
        """
        Process a video file for pedestrian counting and analysis.
        
        Args:
            video_path: Path to video file
            zones: Zone definitions for ingress/egress detection
            progress_callback: Callback function for progress updates
            
        Returns:
            AnalysisResult object or None if failed
        """
        # Load video
        video_file = self.load_video(video_path)
        if video_file is None:
            return None
        
        logger.info(f"Processing video: {video_file.name}")
        logger.info(f"Duration: {video_file.duration:.2f}s, Frames: {video_file.total_frames}")
        
        # Initialize tracker
        tracker = PedestrianTracker()
        if zones:
            ingress_zones = [k for k, v in zones.items() if v.get('type') == 'ingress']
            egress_zones = [k for k, v in zones.items() if v.get('type') == 'egress']
            tracker.set_zones(zones, ingress_zones, egress_zones)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_frames = 0
        start_time = datetime.utcnow()
        
        # Calculate total frames to process
        total_frames_to_process = video_file.total_frames // self.frame_skip
        
        try:
            with tqdm(total=total_frames_to_process, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if needed
                    if frame_count % self.frame_skip != 0:
                        frame_count += 1
                        continue
                    
                    # Process frame
                    frame_timestamp = start_time + timedelta(seconds=frame_count / video_file.fps)
                    
                    # Detect pedestrians
                    detections = self.sam2_detector.detect_pedestrians(frame)
                    
                    # Update tracker
                    tracked_pedestrians = tracker.update(detections, frame_timestamp)
                    
                    # Get counts
                    counts = tracker.get_counts()
                    
                    # Calculate density and LOS
                    density = 0.0
                    los_result = None
                    if counts and 'current_pedestrians' in counts:
                        density = self.fruin_analyzer.calculate_density(
                            counts['current_pedestrians'], 
                            self.zone_area_sqm
                        )
                        los_result = self.fruin_analyzer.determine_los(density)
                    
                    # Store frame result
                    frame_result = {
                        'frame_number': frame_count,
                        'timestamp': frame_timestamp.isoformat(),
                        'detections': len(detections),
                        'tracked_pedestrians': len(tracked_pedestrians),
                        'ingress_count': counts.get('ingress_count', 0),
                        'egress_count': counts.get('egress_count', 0),
                        'current_pedestrians': counts.get('current_pedestrians', 0),
                        'density': density,
                        'los_level': los_result.level if los_result else 'A'
                    }
                    frame_results.append(frame_result)
                    
                    # Update progress
                    processed_frames += 1
                    pbar.update(1)
                    
                    if progress_callback:
                        progress = (processed_frames / total_frames_to_process) * 100
                        progress_callback(progress)
                    
                    frame_count += 1
        
        finally:
            cap.release()
        
        # Calculate summary statistics
        end_time = datetime.utcnow()
        
        # Calculate averages and peaks
        densities = [fr['density'] for fr in frame_results]
        avg_density = np.mean(densities) if densities else 0.0
        peak_density = np.max(densities) if densities else 0.0
        
        # Calculate LOS
        avg_los_result = self.fruin_analyzer.determine_los(avg_density)
        peak_los_result = self.fruin_analyzer.determine_los(peak_density)
        
        # Calculate total movements
        total_ingress = sum(fr['ingress_count'] for fr in frame_results)
        total_egress = sum(fr['egress_count'] for fr in frame_results)
        
        # Calculate total detections
        total_detections = sum(fr['detections'] for fr in frame_results)
        
        result = AnalysisResult(
            video_file=video_file,
            start_time=start_time,
            end_time=end_time,
            total_detections=total_detections,
            total_ingress=total_ingress,
            total_egress=total_egress,
            avg_density=avg_density,
            peak_density=peak_density,
            avg_los=avg_los_result.level,
            peak_los=peak_los_result.level,
            frame_results=frame_results
        )
        
        logger.info(f"Analysis completed: {total_detections} detections, "
                   f"LOS: {avg_los_result.level} (avg), {peak_los_result.level} (peak)")
        
        return result
    
    def process_batch(self, video_directory: str, 
                     output_directory: str,
                     zones: Optional[Dict] = None,
                     progress_callback: Optional[Callable] = None) -> List[AnalysisResult]:
        """
        Process multiple video files in a directory.
        
        Args:
            video_directory: Directory containing video files
            output_directory: Directory to save results
            zones: Zone definitions for all videos
            progress_callback: Callback function for progress updates
            
        Returns:
            List of AnalysisResult objects
        """
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_directory).glob(f"*{ext}"))
            video_files.extend(Path(video_directory).glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"No video files found in {video_directory}")
            return []
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Process each video
        results = []
        for i, video_path in enumerate(video_files):
            logger.info(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
            
            # Process video
            result = self.process_video(str(video_path), zones, progress_callback)
            if result:
                results.append(result)
                
                # Save individual result
                output_file = os.path.join(output_directory, f"{video_path.stem}_analysis.json")
                self.save_result(result, output_file)
        
        # Save batch summary
        self.save_batch_summary(results, output_directory)
        
        return results
    
    def save_result(self, result: AnalysisResult, output_file: str):
        """Save analysis result to JSON file."""
        try:
            data = {
                'video_file': {
                    'path': result.video_file.path,
                    'name': result.video_file.name,
                    'duration': result.video_file.duration,
                    'fps': result.video_file.fps,
                    'width': result.video_file.width,
                    'height': result.video_file.height,
                    'total_frames': result.video_file.total_frames
                },
                'analysis': {
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'total_detections': result.total_detections,
                    'total_ingress': result.total_ingress,
                    'total_egress': result.total_egress,
                    'avg_density': result.avg_density,
                    'peak_density': result.peak_density,
                    'avg_los': result.avg_los,
                    'peak_los': result.peak_los
                },
                'frame_results': result.frame_results
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
    
    def save_batch_summary(self, results: List[AnalysisResult], output_directory: str):
        """Save batch analysis summary."""
        try:
            summary = {
                'batch_info': {
                    'total_videos': len(results),
                    'analysis_date': datetime.utcnow().isoformat(),
                    'total_detections': sum(r.total_detections for r in results),
                    'total_ingress': sum(r.total_ingress for r in results),
                    'total_egress': sum(r.total_egress for r in results)
                },
                'videos': []
            }
            
            for result in results:
                summary['videos'].append({
                    'name': result.video_file.name,
                    'duration': result.video_file.duration,
                    'total_detections': result.total_detections,
                    'total_ingress': result.total_ingress,
                    'total_egress': result.total_egress,
                    'avg_density': result.avg_density,
                    'peak_density': result.peak_density,
                    'avg_los': result.avg_los,
                    'peak_los': result.peak_los
                })
            
            summary_file = os.path.join(output_directory, "batch_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Batch summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving batch summary: {e}")
    
    def set_analysis_parameters(self, frame_skip: int = 1, 
                               analysis_window_minutes: int = 5,
                               zone_area_sqm: float = 25.0):
        """Update analysis parameters."""
        self.frame_skip = frame_skip
        self.analysis_window_minutes = analysis_window_minutes
        self.zone_area_sqm = zone_area_sqm
        logger.info(f"Updated parameters: frame_skip={frame_skip}, "
                   f"window={analysis_window_minutes}min, area={zone_area_sqm}sqm")