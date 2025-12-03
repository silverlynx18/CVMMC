"""Main processing engine for pedestrian counting and analysis."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from .sam3_detector import SAM3PedestrianDetector, Detection
from .pedestrian_tracker import PedestrianTracker
from .fruin_analysis import FruinAnalyzer, FruinLOSResult
from .camera_manager import CameraManager, CameraFrame
from .models import PedestrianDetection, PedestrianCount, ServiceLevelAnalysis, Alert
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a camera frame."""
    camera_id: int
    timestamp: datetime
    detections: List[Detection]
    tracked_pedestrians: int
    counts: Dict[str, int]
    fruin_los: Optional[FruinLOSResult]
    alerts: List[Dict]


class ProcessingEngine:
    """Main processing engine for pedestrian counting and analysis."""
    
    def __init__(self):
        """Initialize the processing engine."""
        self.sam3_detector = None
        self.camera_manager = CameraManager()
        self.trackers: Dict[int, PedestrianTracker] = {}
        self.fruin_analyzer = FruinAnalyzer()
        self.is_running = False
        self.processing_tasks = []
        
        # Database session (would be injected in real implementation)
        self.db_session = None
        
    async def initialize(self):
        """Initialize the processing engine."""
        try:
            # Initialize SAM3 detector
            self.sam3_detector = SAM3PedestrianDetector(
                model_path=settings.sam3_model_path,
                device=settings.device
            )
            logger.info("SAM3 detector initialized")
            
            # Initialize camera manager
            # In real implementation, load cameras from database
            await self._load_cameras_from_db()
            
            # Initialize trackers for each camera
            for camera_id in self.camera_manager.cameras.keys():
                tracker = PedestrianTracker(
                    max_disappeared=settings.tracking_max_disappeared,
                    max_distance=settings.tracking_max_distance
                )
                self.trackers[camera_id] = tracker
                
                # Set up zones for the tracker
                await self._setup_camera_zones(camera_id)
            
            logger.info("Processing engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processing engine: {e}")
            raise
    
    async def _load_cameras_from_db(self):
        """Load camera configurations from database."""
        # In real implementation, this would query the database
        # For now, create sample cameras
        sample_cameras = [
            {
                'id': i,
                'name': f'Camera_{i:02d}',
                'location': f'Station_Zone_{i}',
                'ip_address': f'192.168.1.{100 + i}',
                'port': 554,
                'username': 'admin',
                'password': 'password',
                'calibration_data': {
                    'area_sqm': 25.0,  # 5x5 meter area
                    'intrinsic_matrix': [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                    'distortion_coeffs': [0.1, -0.2, 0, 0, 0]
                }
            }
            for i in range(1, settings.num_cameras + 1)
        ]
        
        for camera_data in sample_cameras:
            # Create Camera model instance
            camera = Camera(**camera_data)
            self.camera_manager.add_camera(camera)
    
    async def _setup_camera_zones(self, camera_id: int):
        """Set up detection zones for a camera."""
        # Sample zone configuration
        zones = {
            'ingress_1': {
                'polygon': [(100, 100), (400, 100), (400, 300), (100, 300)],
                'type': 'ingress'
            },
            'egress_1': {
                'polygon': [(500, 100), (800, 100), (800, 300), (500, 300)],
                'type': 'egress'
            },
            'ingress_2': {
                'polygon': [(100, 400), (400, 400), (400, 600), (100, 600)],
                'type': 'ingress'
            },
            'egress_2': {
                'polygon': [(500, 400), (800, 400), (800, 600), (500, 600)],
                'type': 'egress'
            }
        }
        
        ingress_zones = ['ingress_1', 'ingress_2']
        egress_zones = ['egress_1', 'egress_2']
        
        # Set zones for camera manager
        self.camera_manager.set_camera_zones(camera_id, zones, ingress_zones, egress_zones)
        
        # Set zones for tracker
        if camera_id in self.trackers:
            self.trackers[camera_id].set_zones(zones, ingress_zones, egress_zones)
    
    async def start_processing(self):
        """Start the processing engine."""
        if self.is_running:
            logger.warning("Processing engine is already running")
            return
        
        try:
            # Connect to all cameras
            connection_results = self.camera_manager.connect_all_cameras()
            online_cameras = [cid for cid, success in connection_results.items() if success]
            
            if not online_cameras:
                logger.error("No cameras connected successfully")
                return
            
            # Start camera capture
            self.camera_manager.start_all_cameras()
            
            # Start processing tasks
            self.is_running = True
            for camera_id in online_cameras:
                task = asyncio.create_task(self._process_camera_loop(camera_id))
                self.processing_tasks.append(task)
            
            logger.info(f"Processing engine started with {len(online_cameras)} cameras")
            
        except Exception as e:
            logger.error(f"Failed to start processing engine: {e}")
            self.is_running = False
            raise
    
    async def stop_processing(self):
        """Stop the processing engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        
        # Stop cameras
        self.camera_manager.stop_all_cameras()
        self.camera_manager.disconnect_all_cameras()
        
        logger.info("Processing engine stopped")
    
    async def _process_camera_loop(self, camera_id: int):
        """Main processing loop for a camera."""
        logger.info(f"Started processing loop for camera {camera_id}")
        
        while self.is_running:
            try:
                # Get latest frame
                frame = self.camera_manager.get_latest_frames().get(camera_id)
                if frame is None:
                    await asyncio.sleep(0.1)  # Wait for next frame
                    continue
                
                # Process frame
                result = await self._process_frame(camera_id, frame)
                
                # Store results in database
                await self._store_processing_result(result)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying
        
        logger.info(f"Stopped processing loop for camera {camera_id}")
    
    async def _process_frame(self, camera_id: int, frame: CameraFrame) -> ProcessingResult:
        """Process a single camera frame."""
        # Detect pedestrians using SAM3
        detections = self.sam3_detector.detect_pedestrians(frame.frame)
        
        # Update tracker
        tracker = self.trackers.get(camera_id)
        if tracker:
            tracked_pedestrians = tracker.update(detections, frame.timestamp)
        else:
            tracked_pedestrians = []
        
        # Get counts
        counts = tracker.get_counts() if tracker else {}
        
        # Calculate Fruin's Level of Service
        fruin_los = None
        if counts and 'current_pedestrians' in counts:
            camera_area = self.camera_manager.get_camera_area(camera_id)
            if camera_area > 0:
                fruin_los = self.fruin_analyzer.analyze_zone(
                    counts['current_pedestrians'], 
                    camera_area
                )
        
        # Generate alerts
        alerts = []
        if fruin_los:
            alert = self.fruin_analyzer.generate_alert_if_needed(
                fruin_los, camera_id, "main_zone"
            )
            if alert:
                alerts.append(alert)
        
        return ProcessingResult(
            camera_id=camera_id,
            timestamp=frame.timestamp,
            detections=detections,
            tracked_pedestrians=len(tracked_pedestrians),
            counts=counts,
            fruin_los=fruin_los,
            alerts=alerts
        )
    
    async def _store_processing_result(self, result: ProcessingResult):
        """Store processing results in database."""
        # In real implementation, this would store in database
        # For now, just log the results
        logger.debug(f"Camera {result.camera_id}: {result.tracked_pedestrians} pedestrians, "
                    f"LOS: {result.fruin_los.level if result.fruin_los else 'N/A'}")
    
    def get_camera_status(self) -> Dict[int, str]:
        """Get status of all cameras."""
        return self.camera_manager.get_camera_status()
    
    def get_latest_counts(self, camera_id: Optional[int] = None) -> Dict:
        """Get latest pedestrian counts."""
        if camera_id is not None:
            if camera_id in self.trackers:
                return self.trackers[camera_id].get_counts()
            return {}
        
        # Get counts for all cameras
        all_counts = {}
        for cid, tracker in self.trackers.items():
            all_counts[cid] = tracker.get_counts()
        return all_counts
    
    def get_latest_los(self, camera_id: Optional[int] = None) -> Dict:
        """Get latest Level of Service for cameras."""
        # This would typically query the database for latest LOS data
        # For now, return sample data
        if camera_id is not None:
            return {
                'camera_id': camera_id,
                'los_level': 'C',
                'density': 0.5,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return {
            f'camera_{i}': {
                'camera_id': i,
                'los_level': 'C',
                'density': 0.5,
                'timestamp': datetime.utcnow().isoformat()
            }
            for i in range(1, settings.num_cameras + 1)
        }
    
    def get_historical_analysis(self, camera_id: int, 
                              start_time: datetime, 
                              end_time: datetime) -> Dict:
        """Get historical analysis for a camera."""
        # This would query the database for historical data
        # For now, return sample data
        return {
            'camera_id': camera_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'avg_density': 0.4,
            'peak_density': 0.8,
            'avg_los': 'B',
            'peak_los': 'D',
            'total_ingress': 150,
            'total_egress': 140,
            'peak_hour_factor': 1.3
        }