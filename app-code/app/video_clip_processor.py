"""Video clip processing with time segment selection."""

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

from .detection_utils import Detection
# SAM2PedestrianDetector deprecated - using YOLO-only detection
from .pedestrian_tracker import PedestrianTracker
from .fruin_analysis import FruinAnalyzer, FruinLOSResult
from .transit_analyzer import TransitAnalyzer, TransitEntrance, TransitMetrics
from .metadata_manager import MetadataManager, VideoClip, CameraMetadata

logger = logging.getLogger(__name__)


@dataclass
class ClipAnalysisResult:
    """Result of video clip analysis."""
    clip: VideoClip
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
    processing_time: float


class VideoClipProcessor:
    """Process video clips with time segment selection for testing."""
    
    def __init__(self, sam2_model_path: str, device: str = "cuda"):
        """
        Initialize video clip processor.
        
        Args:
            sam2_model_path: Path to SAM2 model file
            device: Device to run inference on
        """
        self.sam2_detector = SAM2PedestrianDetector(sam2_model_path, device)
        self.fruin_analyzer = FruinAnalyzer()
        self.transit_analyzer = TransitAnalyzer()
        self.metadata_manager = MetadataManager()
        self.device = device
        
        # Analysis parameters
        self.frame_skip = 1
        self.analysis_window_minutes = 5
        self.zone_area_sqm = 25.0
        
    def process_clip(self, clip: VideoClip, 
                    zones: Optional[Dict] = None,
                    progress_callback: Optional[Callable] = None) -> Optional[ClipAnalysisResult]:
        """
        Process a single video clip.
        
        Args:
            clip: VideoClip object to process
            zones: Zone definitions for ingress/egress detection
            progress_callback: Callback function for progress updates
            
        Returns:
            ClipAnalysisResult object or None if failed
        """
        logger.info(f"Processing clip: {clip.file_path}")
        logger.info(f"Duration: {clip.duration:.2f}s, Resolution: {clip.resolution}")
        
        # Initialize tracker
        tracker = PedestrianTracker()
        if zones:
            ingress_zones = [k for k, v in zones.items() if v.get('type') == 'ingress']
            egress_zones = [k for k, v in zones.items() if v.get('type') == 'egress']
            tracker.set_zones(zones, ingress_zones, egress_zones)
        
        # Open video
        cap = cv2.VideoCapture(clip.file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {clip.file_path}")
            return None
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_frames = 0
        start_time = datetime.utcnow()
        
        # Calculate total frames to process
        total_frames = int(clip.duration * clip.fps)
        total_frames_to_process = total_frames // self.frame_skip
        
        try:
            with tqdm(total=total_frames_to_process, desc="Processing clip") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if needed
                    if frame_count % self.frame_skip != 0:
                        frame_count += 1
                        continue
                    
                    # Process frame
                    frame_timestamp = clip.start_time + timedelta(seconds=frame_count / clip.fps)
                    
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
        processing_time = (end_time - start_time).total_seconds()
        
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
        
        result = ClipAnalysisResult(
            clip=clip,
            start_time=start_time,
            end_time=end_time,
            total_detections=total_detections,
            total_ingress=total_ingress,
            total_egress=total_egress,
            avg_density=avg_density,
            peak_density=peak_density,
            avg_los=avg_los_result.level,
            peak_los=peak_los_result.level,
            frame_results=frame_results,
            processing_time=processing_time
        )
        
        logger.info(f"Clip analysis completed: {total_detections} detections, "
                   f"LOS: {avg_los_result.level} (avg), {peak_los_result.level} (peak)")
        
        return result
    
    def process_clip_segment(self, clip: VideoClip, 
                           start_seconds: float, 
                           end_seconds: float,
                           zones: Optional[Dict] = None,
                           progress_callback: Optional[Callable] = None) -> Optional[ClipAnalysisResult]:
        """
        Process a specific time segment of a video clip.
        
        Args:
            clip: VideoClip object
            start_seconds: Start time in seconds from beginning of clip
            end_seconds: End time in seconds from beginning of clip
            zones: Zone definitions
            progress_callback: Progress callback
            
        Returns:
            ClipAnalysisResult object or None if failed
        """
        logger.info(f"Processing clip segment: {start_seconds:.1f}s - {end_seconds:.1f}s")
        
        # Create a modified clip for the segment
        segment_duration = end_seconds - start_seconds
        segment_start_time = clip.start_time + timedelta(seconds=start_seconds)
        segment_end_time = clip.start_time + timedelta(seconds=end_seconds)
        
        segment_clip = VideoClip(
            file_path=clip.file_path,
            camera_id=clip.camera_id,
            start_time=segment_start_time,
            end_time=segment_end_time,
            duration=segment_duration,
            file_size=clip.file_size,
            resolution=clip.resolution,
            fps=clip.fps,
            codec=clip.codec,
            tags=clip.tags + ['segment'],
            notes=f"Segment: {start_seconds:.1f}s - {end_seconds:.1f}s"
        )
        
        # Process the segment
        return self.process_clip(segment_clip, zones, progress_callback)
    
    def process_camera_clips(self, camera_id: str, 
                           zones: Optional[Dict] = None,
                           max_duration: Optional[float] = None,
                           progress_callback: Optional[Callable] = None) -> List[ClipAnalysisResult]:
        """
        Process all clips for a specific camera.
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions
            max_duration: Maximum clip duration to process (for testing)
            progress_callback: Progress callback
            
        Returns:
            List of ClipAnalysisResult objects
        """
        # Get camera metadata
        camera = self.metadata_manager.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return []
        
        # Get clips
        clips = self.metadata_manager.get_camera_clips(camera_id)
        if max_duration:
            clips = [clip for clip in clips if clip.duration <= max_duration]
        
        if not clips:
            logger.warning(f"No clips found for camera {camera_id}")
            return []
        
        logger.info(f"Processing {len(clips)} clips for camera {camera_id}")
        
        # Process each clip
        results = []
        for i, clip in enumerate(clips):
            logger.info(f"Processing clip {i+1}/{len(clips)}: {os.path.basename(clip.file_path)}")
            
            result = self.process_clip(clip, zones, progress_callback)
            if result:
                results.append(result)
        
        return results
    
    def test_camera_with_short_clips(self, camera_id: str, 
                                   test_duration: float = 60.0,
                                   zones: Optional[Dict] = None) -> Dict:
        """
        Test a camera with short clips to verify pedestrian detection.
        
        Args:
            camera_id: Camera identifier
            test_duration: Duration of test clips in seconds
            zones: Zone definitions
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing camera {camera_id} with {test_duration}s clips")
        
        # Get camera metadata
        camera = self.metadata_manager.get_camera(camera_id)
        if not camera:
            return {'error': f'Camera {camera_id} not found'}
        
        # Get clips suitable for testing
        test_clips = self.metadata_manager.get_clips_by_duration(
            camera_id, min_duration=test_duration, max_duration=test_duration * 2
        )
        
        if not test_clips:
            # Try to find longer clips and use segments
            all_clips = self.metadata_manager.get_camera_clips(camera_id)
            test_clips = [clip for clip in all_clips if clip.duration >= test_duration]
        
        if not test_clips:
            return {'error': f'No suitable clips found for testing camera {camera_id}'}
        
        # Process test clips
        test_results = []
        for i, clip in enumerate(test_clips[:3]):  # Test up to 3 clips
            logger.info(f"Testing with clip {i+1}: {os.path.basename(clip.file_path)}")
            
            if clip.duration > test_duration * 1.5:
                # Use a segment of the clip
                start_time = clip.duration / 4  # Start at 25% into the clip
                end_time = start_time + test_duration
                result = self.process_clip_segment(clip, start_time, end_time, zones)
            else:
                # Use the entire clip
                result = self.process_clip(clip, zones)
            
            if result:
                test_results.append({
                    'clip_name': os.path.basename(clip.file_path),
                    'duration': result.clip.duration,
                    'total_detections': result.total_detections,
                    'total_ingress': result.total_ingress,
                    'total_egress': result.total_egress,
                    'avg_density': result.avg_density,
                    'peak_density': result.peak_density,
                    'avg_los': result.avg_los,
                    'peak_los': result.peak_los,
                    'processing_time': result.processing_time
                })
        
        # Calculate summary statistics
        if test_results:
            summary = {
                'camera_id': camera_id,
                'camera_name': camera.name,
                'test_duration': test_duration,
                'clips_tested': len(test_results),
                'total_detections': sum(r['total_detections'] for r in test_results),
                'total_ingress': sum(r['total_ingress'] for r in test_results),
                'total_egress': sum(r['total_egress'] for r in test_results),
                'avg_density': np.mean([r['avg_density'] for r in test_results]),
                'peak_density': max(r['peak_density'] for r in test_results),
                'avg_los': max(r['avg_los'] for r in test_results),  # Use worst LOS
                'avg_processing_time': np.mean([r['processing_time'] for r in test_results]),
                'results': test_results
            }
        else:
            summary = {'error': 'No successful test results'}
        
        return summary
    
    def save_clip_result(self, result: ClipAnalysisResult, output_file: str):
        """Save clip analysis result to JSON file."""
        try:
            data = {
                'clip_info': {
                    'file_path': result.clip.file_path,
                    'camera_id': result.clip.camera_id,
                    'start_time': result.clip.start_time.isoformat(),
                    'end_time': result.clip.end_time.isoformat(),
                    'duration': result.clip.duration,
                    'resolution': list(result.clip.resolution),
                    'fps': result.clip.fps,
                    'codec': result.clip.codec,
                    'tags': result.clip.tags,
                    'notes': result.clip.notes
                },
                'analysis': {
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'processing_time': result.processing_time,
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
            
            logger.info(f"Clip result saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving clip result to {output_file}: {e}")
    
    def set_analysis_parameters(self, frame_skip: int = 1, 
                               analysis_window_minutes: int = 5,
                               zone_area_sqm: float = 25.0):
        """Update analysis parameters."""
        self.frame_skip = frame_skip
        self.analysis_window_minutes = analysis_window_minutes
        self.zone_area_sqm = zone_area_sqm
        logger.info(f"Updated parameters: frame_skip={frame_skip}, "
                   f"window={analysis_window_minutes}min, area={zone_area_sqm}sqm")
    
    def process_clip_with_transit_analysis(self, clip: VideoClip, 
                                         transit_entrances: List[TransitEntrance],
                                         progress_callback: Optional[Callable] = None) -> Optional[ClipAnalysisResult]:
        """
        Process a video clip with transit station analysis.
        
        Args:
            clip: VideoClip object to process
            transit_entrances: List of transit entrance/exit definitions
            progress_callback: Callback function for progress updates
            
        Returns:
            ClipAnalysisResult object with transit analysis or None if failed
        """
        logger.info(f"Processing clip with transit analysis: {clip.file_path}")
        
        # Set up transit entrances
        for entrance in transit_entrances:
            self.transit_analyzer.add_entrance(entrance)
        
        # Initialize tracker
        tracker = PedestrianTracker()
        
        # Open video
        cap = cv2.VideoCapture(clip.file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {clip.file_path}")
            return None
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_frames = 0
        start_time = datetime.utcnow()
        
        # Calculate total frames to process
        total_frames = int(clip.duration * clip.fps)
        total_frames_to_process = total_frames // self.frame_skip
        
        try:
            with tqdm(total=total_frames_to_process, desc="Processing clip with transit analysis") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if needed
                    if frame_count % self.frame_skip != 0:
                        frame_count += 1
                        continue
                    
                    # Process frame
                    frame_timestamp = clip.start_time + timedelta(seconds=frame_count / clip.fps)
                    
                    # Detect pedestrians
                    detections = self.sam2_detector.detect_pedestrians(frame)
                    
                    # Update tracker
                    tracked_pedestrians = tracker.update(detections, frame_timestamp)
                    
                    # Update transit analysis
                    transit_movements = self.transit_analyzer.update_pedestrian_tracking(
                        tracked_pedestrians, frame_timestamp
                    )
                    
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
                        'los_level': los_result.level if los_result else 'A',
                        'transit_movements': len(transit_movements)
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
        processing_time = (end_time - start_time).total_seconds()
        
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
        
        # Get transit metrics
        transit_metrics = self.transit_analyzer.calculate_transit_metrics()
        
        result = ClipAnalysisResult(
            clip=clip,
            start_time=start_time,
            end_time=end_time,
            total_detections=total_detections,
            total_ingress=total_ingress,
            total_egress=total_egress,
            avg_density=avg_density,
            peak_density=peak_density,
            avg_los=avg_los_result.level,
            peak_los=peak_los_result.level,
            frame_results=frame_results,
            processing_time=processing_time
        )
        
        # Add transit metrics to result
        result.transit_metrics = transit_metrics
        
        logger.info(f"Transit analysis completed: {total_detections} detections, "
                   f"LOS: {avg_los_result.level} (avg), {peak_los_result.level} (peak)")
        logger.info(f"Transit movements: {transit_metrics.total_boardings} boardings, "
                   f"{transit_metrics.total_alightings} alightings, "
                   f"{transit_metrics.total_transfers} transfers")
        
        return result
    
    def export_transit_data(self, output_file: str, 
                          time_window: Optional[timedelta] = None) -> bool:
        """
        Export transit analysis data in Mass Motion compatible format.
        
        Args:
            output_file: Output file path
            time_window: Time window for export
            
        Returns:
            True if successful, False otherwise
        """
        return self.transit_analyzer.export_mass_motion_data(output_file, time_window)