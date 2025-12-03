"""
Enhanced video processor with trajectory-based bidirectional counting.
Extends the base VideoProcessor with line-crossing detection.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from .video_processor import VideoProcessor, VideoFile, AnalysisResult
from .trajectory_analyzer import TrajectoryAnalyzer, CountingLine, LineCrossingEvent
from .pedestrian_tracker import PedestrianTracker, TrackedPedestrian
from .yolo_detector import YOLOPedestrianDetector
from .fruin_analysis import FruinAnalyzer
from .adobe_export import AdobeExporter, create_adobe_compatible_video

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnalysisResult(AnalysisResult):
    """Analysis result with trajectory-based counting and flow rate analysis."""
    # Line crossing counts
    line_crossing_counts: Dict[str, Dict[str, int]] = None  # {line_name: {direction: count}}
    net_flows: Dict[str, int] = None  # {line_name: net_flow}
    crossing_events: List[LineCrossingEvent] = None
    
    # Flow rate metrics (Arup/MassMotion standards)
    flow_rates: Dict[str, Dict[str, float]] = None  # {line_name: {forward_rate, backward_rate, ...}}
    peak_flow_rates: Dict[str, Dict[str, float]] = None  # {line_name: {peak_forward_rate, ...}}
    hourly_flow_rates: Dict[str, Dict[str, Dict[str, float]]] = None  # {line_name: {hour_X: {...}}}
    
    # Additional trajectory metrics
    avg_pedestrian_speed: float = 0.0
    movement_patterns: Dict[str, int] = None  # {direction_type: count}


class VideoProcessorWithTrajectories(VideoProcessor):
    """
    Video processor with trajectory-based analysis.
    Supports both zone-based and line-based counting.
    """
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt", device: str = "cpu",
                 stage: str = "stage2", image_size: tuple = (1920, 1080)):
        """
        Initialize the enhanced video processor with YOLO-only detection.
        
        Args:
            yolo_model_path: Path to YOLO model
            device: Device for inference
            stage: Workflow stage ("stage1" or "stage2") - affects tuning
            image_size: (width, height) for tuning initialization
        """
        # Create stage-appropriate detection tuner
        from .tuning_helpers import create_stage_tuner
        from config.config_manager import get_config
        
        config_manager = get_config()
        detection_tuner = create_stage_tuner(
            stage=stage,
            image_size=image_size,
            config_manager=config_manager
        )
        
        if detection_tuner:
            logger.info(f"Initialized {stage} video processor with detection tuning")
        
        # Initialize YOLO detector with tuner
        self.yolo_detector = YOLOPedestrianDetector(
            yolo_model_path=yolo_model_path,
            device=device,
            detection_tuner=detection_tuner
        )
        # Call parent init with None for sam2 (not used)
        super().__init__(None, device)
        # Override the detector
        self.sam2_detector = self.yolo_detector
        self.detection_tuner = detection_tuner
        
        # Trajectory analyzer
        self.trajectory_analyzer = TrajectoryAnalyzer(
            min_trajectory_length=5,
            smoothing_window=3,
            stationary_threshold=10.0
        )
        
        # Track completed trajectories
        self.completed_trajectories: Dict[int, List] = {}
        
        # Tracking parameters
        self.tracking_max_disappeared = 30
        self.tracking_max_distance = 100.0
        
    def load_counting_lines(self, lines_file: str) -> bool:
        """
        Load counting lines from JSON file.
        
        Args:
            lines_file: Path to counting lines JSON file
            
        Returns:
            True if successful
        """
        try:
            with open(lines_file, 'r') as f:
                lines_config = json.load(f)
            
            for line_name, line_data in lines_config.items():
                line = CountingLine(
                    name=line_name,
                    start_point=tuple(line_data['start_point']),
                    end_point=tuple(line_data['end_point']),
                    direction_names=tuple(line_data.get('direction_names', ['forward', 'backward'])),
                    width_meters=line_data.get('width_meters')  # Optional: effective width in meters
                )
                self.trajectory_analyzer.add_counting_line(line)
            
            logger.info(f"Loaded {len(lines_config)} counting lines from {lines_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load counting lines: {e}")
            return False
    
    def process_video_with_lines(self,
                                 video_path: str,
                                 lines_file: str,
                                 output_dir: Optional[str] = None,
                                 visualize: bool = True,
                                 save_events: bool = True,
                                 adobe_export: bool = False,
                                 export_format: str = 'h264') -> Optional[EnhancedAnalysisResult]:
        """
        Process video with line-based counting.
        
        Args:
            video_path: Path to video file
            lines_file: Path to counting lines JSON file
            output_dir: Directory to save results
            visualize: Whether to save visualization video
            save_events: Whether to save crossing events to CSV
            
        Returns:
            EnhancedAnalysisResult or None
        """
        # Load counting lines
        if not self.load_counting_lines(lines_file):
            return None
        
        # Load video
        video_file = self.load_video(video_path)
        if not video_file:
            return None
        
        # Create output directory
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize tracker with detection tuner for quality feedback
        tracker = PedestrianTracker(
            max_disappeared=self.tracking_max_disappeared,
            max_distance=self.tracking_max_distance,
            detection_tuner=self.detection_tuner
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Prepare video writer for visualization
        video_writer = None
        if visualize and output_dir:
            output_video_path = Path(output_dir) / f"{Path(video_path).stem}_analyzed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                video_file.fps,
                (video_file.width, video_file.height)
            )
        
        # Processing
        frame_count = 0
        start_time = datetime.now()
        
        print(f"\n📹 Processing video: {video_path}", flush=True)
        print(f"   Resolution: {video_file.width}x{video_file.height}", flush=True)
        print(f"   FPS: {video_file.fps:.2f}", flush=True)
        print(f"   Total frames: {video_file.total_frames}", flush=True)
        print(f"   Duration: {video_file.duration:.1f} seconds\n", flush=True)
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {video_file.total_frames}")
        
        # Load counting lines for visualization
        with open(lines_file, 'r') as f:
            lines_config = json.load(f)
        
        # Determine crop region based on counting zone if available
        crop_region = None
        crop_x, crop_y, crop_w, crop_h = 0, 0, video_file.width, video_file.height
        
        # Try to load counting zone to determine crop
        zone_file = Path(video_path).parent / "counting_zone.json"
        if zone_file.exists():
            try:
                with open(zone_file, 'r') as f:
                    zone_config = json.load(f)
                    # Calculate bounds from polygon if not explicitly provided
                    if 'bounds' in zone_config:
                        bounds = zone_config['bounds']
                        if isinstance(bounds, list) and len(bounds) == 4:
                            x1, y1, x2, y2 = bounds
                        else:
                            raise ValueError(f"Invalid bounds format: {bounds}")
                    elif 'polygon' in zone_config and len(zone_config['polygon']) > 0:
                        polygon = zone_config['polygon']
                        # Handle list of lists format
                        if isinstance(polygon[0], (list, tuple)):
                            xs = [p[0] for p in polygon]
                            ys = [p[1] for p in polygon]
                        else:
                            # Single list format [x1, y1, x2, y2, ...]
                            xs = [polygon[i] for i in range(0, len(polygon), 2)]
                            ys = [polygon[i+1] for i in range(0, len(polygon), 2)]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                    else:
                        logger.warning("No bounds or polygon in zone config, using default crop")
                        raise ValueError("No bounds or polygon in zone config")
                    
                    padding = 100
                    crop_x = max(0, x1 - padding)
                    crop_y = max(0, y1 - padding)
                    crop_w = min(video_file.width, x2 + padding) - crop_x
                    crop_h = min(video_file.height, y2 + padding) - crop_y
                    crop_region = (crop_x, crop_y, crop_w, crop_h)
                    logger.info(f"Using crop region from zone: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
            except Exception as e:
                logger.warning(f"Could not load crop from zone file: {e}")
        
        # If no zone-based crop, use default: bottom 70% of frame (pedestrian area)
        if crop_region is None:
            crop_y = int(video_file.height * 0.3)  # Start from 30% down
            crop_h = video_file.height - crop_y
            crop_x = 0
            crop_w = video_file.width
            crop_region = (crop_x, crop_y, crop_w, crop_h)
            logger.info(f"Using default crop: bottom 70% ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if frame_count % self.frame_skip != 0:
                continue
            
            # Apply crop to frame
            cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Detect pedestrians on cropped frame
            detections = self.sam2_detector.detect_pedestrians(cropped_frame)
            
            # Adjust detection bboxes back to original frame coordinates
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (x1 + crop_x, y1 + crop_y, x2 + crop_x, y2 + crop_y)
                # Adjust center too
                cx, cy = det.center
                det.center = (cx + crop_x, cy + crop_y)
            
            # Update tracker
            frame_timestamp = start_time + timedelta(seconds=frame_count / video_file.fps)
            tracked_peds = tracker.update(detections, frame_timestamp)
            
            # Check for line crossings on active tracks
            for ped in tracked_peds:
                if len(ped.detections) >= self.trajectory_analyzer.min_trajectory_length:
                    # Extract trajectory
                    trajectory = self.trajectory_analyzer.extract_trajectory(
                        ped.detections,
                        fps=video_file.fps
                    )
                    
                    # Smooth trajectory
                    trajectory = self.trajectory_analyzer.smooth_trajectory(trajectory)
                    
                    # Check for crossings
                    crossing_events = self.trajectory_analyzer.check_line_crossing(
                        ped.track_id,
                        trajectory
                    )
                    
                    # Log new crossings
                    for event in crossing_events:
                        logger.info(f"Frame {frame_count}: Track {event.track_id} crossed "
                                  f"{event.line_name} going {event.direction} "
                                  f"(confidence: {event.confidence:.2f})")
            
            # Visualize
            if video_writer:
                vis_frame = self._visualize_frame(
                    frame,
                    tracked_peds,
                    lines_config,
                    frame_count
                )
                video_writer.write(vis_frame)
            
            # Progress (print every 50 frames for better visibility)
            if frame_count % 50 == 0:
                progress = (frame_count / video_file.total_frames) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                progress_msg = f"  [{progress:5.1f}%] Frame {frame_count:5d}/{video_file.total_frames} | Tracking: {len(tracked_peds):2d} peds | Speed: {fps_processing:.1f} fps"
                print(progress_msg, flush=True)
                logger.info(progress_msg)
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        end_time = datetime.now()
        
        # Get results
        crossing_counts = self.trajectory_analyzer.get_crossing_counts()
        net_flows = {
            line_name: self.trajectory_analyzer.get_net_flow(line_name)
            for line_name in crossing_counts.keys()
        }
        
        # Calculate comprehensive flow rates (Arup/MassMotion standards)
        # Get pixels_per_meter from camera calibration if available
        pixels_per_meter = getattr(self, 'pixels_per_meter', None)
        
        flow_rates = {}
        peak_flow_rates = {}
        hourly_flow_rates = {}
        
        for line_name in crossing_counts.keys():
            # Standard flow rates (1-minute window)
            flow_rates[line_name] = self.trajectory_analyzer.calculate_flow_rate(
                line_name, time_window_seconds=60.0, pixels_per_meter=pixels_per_meter
            )
            
            # Peak 15-minute flow rates
            peak_flow_rates[line_name] = self.trajectory_analyzer.calculate_peak_flow_rates(
                line_name, window_minutes=15, pixels_per_meter=pixels_per_meter
            )
            
            # Hourly flow rates
            hourly_flow_rates[line_name] = self.trajectory_analyzer.calculate_hourly_flow_rates(
                line_name, pixels_per_meter=pixels_per_meter
            )
        
        # Calculate metrics
        total_crossings = sum(sum(dirs.values()) for dirs in crossing_counts.values())
        
        # Create result
        result = EnhancedAnalysisResult(
            video_file=video_file,
            start_time=start_time,
            end_time=end_time,
            total_detections=len(tracker.tracked_pedestrians),
            total_ingress=0,  # Not used in line-based counting
            total_egress=0,   # Not used in line-based counting
            avg_density=0.0,  # Could calculate from area
            peak_density=0.0,
            avg_los="N/A",
            peak_los="N/A",
            frame_results=[],
            line_crossing_counts=crossing_counts,
            net_flows=net_flows,
            crossing_events=self.trajectory_analyzer.crossing_events,
            flow_rates=flow_rates,
            peak_flow_rates=peak_flow_rates,
            hourly_flow_rates=hourly_flow_rates
        )
        
        # Save results
        if output_dir:
            self._save_results(result, output_dir, save_events)
            
            # Create Adobe-compatible export if requested
            if adobe_export:
                self._create_adobe_export(video_path, result, output_dir, export_format)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _visualize_frame(self,
                        frame: np.ndarray,
                        tracked_pedestrians: List[TrackedPedestrian],
                        lines_config: Dict,
                        frame_number: int) -> np.ndarray:
        """
        Visualize frame with detections and counting lines.
        
        Args:
            frame: Input frame
            tracked_pedestrians: List of tracked pedestrians
            lines_config: Counting lines configuration
            frame_number: Current frame number
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw counting lines
        for line_name, line_data in lines_config.items():
            start = tuple(line_data['start_point'])
            end = tuple(line_data['end_point'])
            
            # Draw line
            cv2.line(vis_frame, start, end, (0, 255, 255), 2)
            
            # Draw endpoints
            cv2.circle(vis_frame, start, 5, (255, 255, 255), -1)
            cv2.circle(vis_frame, end, 5, (255, 255, 255), -1)
            
            # Draw label
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(vis_frame, line_name, mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tracked pedestrians
        for ped in tracked_pedestrians:
            if ped.detections:
                # Get latest detection
                detection = ped.detections[-1]
                # bbox is (x1, y1, x2, y2) format
                x1, y1, x2, y2 = detection.bbox
                
                # Draw bounding box
                color = (0, 255, 0) if ped.is_ingress is None else \
                        (0, 255, 0) if ped.is_ingress else (0, 0, 255)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID - positioned at bottom of bounding box with background
                label = f"ID: {ped.track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_w, label_h = label_size
                
                # Position label at bottom of box (y2 + 15), but clip to frame if needed
                h, w = vis_frame.shape[:2]
                label_y = min(y2 + 15, h - 5)  # Position below box, but stay in frame
                label_x = max(0, min(x1, w - label_w - 5))  # Stay within frame bounds
                
                # Draw background rectangle for label readability
                cv2.rectangle(vis_frame, 
                            (label_x - 2, label_y - label_h - 2),
                            (label_x + label_w + 2, label_y + 2),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(vis_frame, label,
                           (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trajectory (last N points) - only draw if we have valid positions
                if len(ped.detections) > 1:
                    points = []
                    for det in ped.detections[-20:]:  # Last 20 points
                        # bbox is (x1, y1, x2, y2) format
                        dx1, dy1, dx2, dy2 = det.bbox
                        center_x = int((dx1 + dx2) / 2)
                        center_y = int((dy1 + dy2) / 2)
                        # Only draw trajectories above a minimum height (pedestrian level, not floor)
                        # Filter out points that are likely on the floor (near bottom of frame)
                        h, w = vis_frame.shape[:2]
                        if center_y < h * 0.95:  # Don't draw if too close to bottom (floor)
                            points.append((center_x, center_y))
                    
                    # Draw trajectory line (only if we have valid points)
                    if len(points) > 1:
                        for i in range(len(points) - 1):
                            cv2.line(vis_frame, points[i], points[i+1], color, 1)
        
        # Draw current counts
        y_offset = 30
        counts = self.trajectory_analyzer.get_crossing_counts()
        
        # Semi-transparent background for text
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 30 + len(counts) * 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # Draw counts
        cv2.putText(vis_frame, f"Frame: {frame_number}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for line_name, directions in counts.items():
            for direction, count in directions.items():
                text = f"{line_name} - {direction}: {count}"
                cv2.putText(vis_frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        return vis_frame
    
    def _save_results(self, result: EnhancedAnalysisResult, output_dir: str, save_events: bool):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        
        # Save summary JSON with flow rate metrics
        summary = {
            'video_file': result.video_file.path,
            'processing_time': (result.end_time - result.start_time).total_seconds(),
            'line_crossing_counts': result.line_crossing_counts,
            'net_flows': result.net_flows,
            'total_crossings': sum(sum(dirs.values()) for dirs in result.line_crossing_counts.values()),
            'flow_rates': result.flow_rates,  # Arup/MassMotion flow rate metrics
            'peak_flow_rates': result.peak_flow_rates,  # Peak 15-minute rates
            'hourly_flow_rates': result.hourly_flow_rates  # Hourly breakdown
        }
        
        summary_file = output_path / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Save detailed flow rate report
        if result.flow_rates:
            flow_report_file = output_path / 'flow_rate_report.json'
            with open(flow_report_file, 'w') as f:
                json.dump({
                    'standard_flow_rates': result.flow_rates,
                    'peak_15min_flow_rates': result.peak_flow_rates,
                    'hourly_flow_rates': result.hourly_flow_rates,
                    'units': {
                        'flow_rate': 'pedestrians per second (peds/s)',
                        'specific_flow_rate': 'pedestrians per meter per second (peds/m/s)',
                        'total_crossings': 'pedestrians per hour (peds/hour)'
                    },
                    'standards': 'Based on Arup MassMotion and transportation engineering standards'
                }, f, indent=2, default=str)
            
            logger.info(f"Saved flow rate report to {flow_report_file}")
        
        # Save crossing events to CSV
        if save_events and result.crossing_events:
            import csv
            
            events_file = output_path / 'crossing_events.csv'
            with open(events_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id', 'line_name', 'direction', 'timestamp', 
                               'frame_number', 'crossing_x', 'crossing_y', 'confidence'])
                
                for event in result.crossing_events:
                    writer.writerow([
                        event.track_id,
                        event.line_name,
                        event.direction,
                        event.timestamp,
                        event.frame_number,
                        event.crossing_point[0],
                        event.crossing_point[1],
                        event.confidence
                    ])
            
            logger.info(f"Saved {len(result.crossing_events)} crossing events to {events_file}")
    
    def _print_summary(self, result: EnhancedAnalysisResult):
        """Print analysis summary with flow rate metrics."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nVideo: {result.video_file.name}")
        print(f"Duration: {result.video_file.duration:.1f} seconds")
        print(f"Processing time: {(result.end_time - result.start_time).total_seconds():.1f} seconds")
        
        print("\n" + "-" * 60)
        print("LINE CROSSING COUNTS")
        print("-" * 60)
        
        for line_name, directions in result.line_crossing_counts.items():
            print(f"\n{line_name}:")
            for direction, count in directions.items():
                print(f"  {direction}: {count}")
            print(f"  Net flow: {result.net_flows[line_name]}")
        
        total_crossings = sum(sum(dirs.values()) for dirs in result.line_crossing_counts.values())
        print(f"\nTotal crossings: {total_crossings}")
        print(f"Total crossing events recorded: {len(result.crossing_events)}")
        
        # Print flow rate metrics (Arup/MassMotion standards)
        if result.flow_rates:
            print("\n" + "-" * 60)
            print("FLOW RATE METRICS (Arup/MassMotion Standards)")
            print("-" * 60)
            
            for line_name, flow_data in result.flow_rates.items():
                print(f"\n{line_name}:")
                print(f"  Forward flow rate: {flow_data['forward_rate']:.2f} peds/s")
                print(f"  Backward flow rate: {flow_data['backward_rate']:.2f} peds/s")
                print(f"  Net flow rate: {flow_data['net_rate']:.2f} peds/s")
                
                if flow_data.get('forward_specific_rate') is not None:
                    print(f"  Forward specific flow: {flow_data['forward_specific_rate']:.3f} peds/m/s")
                if flow_data.get('backward_specific_rate') is not None:
                    print(f"  Backward specific flow: {flow_data['backward_specific_rate']:.3f} peds/m/s")
                if flow_data.get('width_meters'):
                    print(f"  Effective width: {flow_data['width_meters']:.2f} m")
            
            # Print peak flow rates
            if result.peak_flow_rates:
                print("\n" + "-" * 60)
                print("PEAK 15-MINUTE FLOW RATES")
                print("-" * 60)
                
                for line_name, peak_data in result.peak_flow_rates.items():
                    if peak_data.get('peak_forward_rate', 0) > 0:
                        print(f"\n{line_name}:")
                        print(f"  Peak forward flow: {peak_data['peak_forward_rate']:.2f} peds/s")
                        if peak_data.get('peak_specific_forward_rate'):
                            print(f"  Peak specific forward flow: {peak_data['peak_specific_forward_rate']:.3f} peds/m/s")
                        if peak_data.get('peak_window_start'):
                            print(f"  Peak period: {peak_data['peak_window_start']:.1f}s - {peak_data['peak_window_end']:.1f}s")
        
        print("\n" + "=" * 60)
    
    def _create_adobe_export(self, video_path: str, result: EnhancedAnalysisResult, 
                           output_dir: str, export_format: str = 'h264'):
        """Create Adobe-compatible export files."""
        try:
            logger.info("Creating Adobe-compatible export...")
            
            # Convert result to dictionary format for Adobe exporter
            analysis_data = {
                'video_file': {
                    'path': result.video_file.path,
                    'name': result.video_file.name,
                    'duration': result.video_file.duration,
                    'fps': result.video_file.fps,
                    'width': result.video_file.width,
                    'height': result.video_file.height,
                    'total_frames': result.video_file.total_frames,
                    'resolution': f"{result.video_file.width}x{result.video_file.height}"
                },
                'total_detections': result.total_detections,
                'total_ingress': result.total_ingress,
                'total_egress': result.total_egress,
                'avg_density': result.avg_density,
                'peak_density': result.peak_density,
                'avg_los': result.avg_los,
                'peak_los': result.peak_los,
                'line_crossing_counts': result.line_crossing_counts,
                'net_flows': result.net_flows,
                'frame_results': result.frame_results,
                'crossing_events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'line_name': event.line_name,
                        'direction': event.direction,
                        'pedestrian_id': event.pedestrian_id,
                        'confidence': event.confidence
                    } for event in result.crossing_events
                ]
            }
            
            # Create Adobe export
            adobe_files = create_adobe_compatible_video(
                input_video=video_path,
                analysis_data=analysis_data,
                output_dir=os.path.join(output_dir, 'adobe_export'),
                export_format=export_format
            )
            
            if adobe_files:
                logger.info("Adobe-compatible export created successfully:")
                for file_type, file_path in adobe_files.items():
                    logger.info(f"  {file_type}: {file_path}")
            else:
                logger.warning("Adobe export failed")
                
        except Exception as e:
            logger.error(f"Failed to create Adobe export: {e}")
