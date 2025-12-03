"""Camera management system for multiple wide-angle cameras."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
import logging
from enum import Enum

from .config import settings
from .models import Camera

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CALIBRATING = "calibrating"


@dataclass
class CameraFrame:
    """Camera frame data."""
    camera_id: int
    frame: np.ndarray
    timestamp: datetime
    frame_number: int


class WideAngleCamera:
    """Individual wide-angle camera handler."""
    
    def __init__(self, camera_config: Camera, camera_id: int):
        """
        Initialize camera handler.
        
        Args:
            camera_config: Camera configuration from database
            camera_id: Unique camera identifier
        """
        self.camera_id = camera_id
        self.config = camera_config
        self.status = CameraStatus.OFFLINE
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
        # Camera calibration data
        self.calibration = camera_config.calibration_data or {}
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.undistort_map = None
        
        # Zone definitions for this camera
        self.zones = {}
        self.ingress_zones = []
        self.egress_zones = []
        
        self._load_calibration()
    
    def _load_calibration(self):
        """Load camera calibration data."""
        if 'intrinsic_matrix' in self.calibration:
            self.intrinsic_matrix = np.array(self.calibration['intrinsic_matrix'])
        if 'distortion_coeffs' in self.calibration:
            self.distortion_coeffs = np.array(self.calibration['distortion_coeffs'])
        
        # Precompute undistortion maps if calibration is available
        if self.intrinsic_matrix is not None and self.distortion_coeffs is not None:
            h, w = settings.camera_resolution
            self.undistort_map = cv2.initUndistortRectifyMap(
                self.intrinsic_matrix, self.distortion_coeffs, None, 
                self.intrinsic_matrix, (w, h), cv2.CV_16SC2
            )
    
    def connect(self) -> bool:
        """
        Connect to the camera.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try different connection methods
            connection_strings = [
                f"rtsp://{self.config.username}:{self.config.password}@{self.config.ip_address}:{self.config.port}/stream",
                f"http://{self.config.ip_address}:{self.config.port}/video",
                f"rtsp://{self.config.ip_address}:{self.config.port}/stream"
            ]
            
            for conn_str in connection_strings:
                try:
                    self.cap = cv2.VideoCapture(conn_str)
                    if self.cap.isOpened():
                        # Set camera properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_resolution[0])
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_resolution[1])
                        self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
                        
                        # Test frame capture
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.status = CameraStatus.ONLINE
                            logger.info(f"Camera {self.camera_id} connected successfully")
                            return True
                        else:
                            self.cap.release()
                            self.cap = None
                except Exception as e:
                    logger.warning(f"Failed to connect to camera {self.camera_id} with {conn_str}: {e}")
                    continue
            
            self.status = CameraStatus.ERROR
            logger.error(f"Failed to connect to camera {self.camera_id}")
            return False
            
        except Exception as e:
            self.status = CameraStatus.ERROR
            logger.error(f"Error connecting to camera {self.camera_id}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera."""
        self.stop_capture.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status = CameraStatus.OFFLINE
        logger.info(f"Camera {self.camera_id} disconnected")
    
    def start_capture(self):
        """Start frame capture in a separate thread."""
        if self.status != CameraStatus.ONLINE:
            logger.warning(f"Cannot start capture for camera {self.camera_id} - not online")
            return
        
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Started capture for camera {self.camera_id}")
    
    def stop_capture_thread(self):
        """Stop the capture thread."""
        self.stop_capture.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        logger.info(f"Stopped capture for camera {self.camera_id}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_number = 0
        
        while not self.stop_capture.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.status = CameraStatus.ERROR
                break
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from camera {self.camera_id}")
                self.status = CameraStatus.ERROR
                break
            
            # Undistort frame if calibration is available
            if self.undistort_map is not None:
                frame = cv2.remap(frame, self.undistort_map[0], self.undistort_map[1], 
                                cv2.INTER_LINEAR)
            
            # Create frame object
            camera_frame = CameraFrame(
                camera_id=self.camera_id,
                frame=frame.copy(),
                timestamp=datetime.utcnow(),
                frame_number=frame_number
            )
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(camera_frame)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(camera_frame)
                except queue.Empty:
                    pass
            
            frame_number += 1
        
        self.status = CameraStatus.OFFLINE
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """
        Get the latest frame from the camera.
        
        Returns:
            CameraFrame object or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_zones(self, zones: Dict, ingress_zones: List[str], egress_zones: List[str]):
        """
        Set detection zones for this camera.
        
        Args:
            zones: Dictionary of zone definitions
            ingress_zones: List of ingress zone IDs
            egress_zones: List of egress zone IDs
        """
        self.zones = zones
        self.ingress_zones = ingress_zones
        self.egress_zones = egress_zones
        logger.info(f"Set {len(zones)} zones for camera {self.camera_id}")
    
    def get_calibration_area(self) -> float:
        """
        Get the calibrated area for density calculations.
        
        Returns:
            Area in square meters
        """
        if 'area_sqm' in self.calibration:
            return self.calibration['area_sqm']
        elif 'pixel_area' in self.calibration:
            return self.calibration['pixel_area'] * settings.density_calculation_area
        else:
            # Default estimation
            w, h = settings.camera_resolution
            return w * h * settings.density_calculation_area


class CameraManager:
    """Manager for multiple wide-angle cameras."""
    
    def __init__(self):
        """Initialize camera manager."""
        self.cameras: Dict[int, WideAngleCamera] = {}
        self.camera_configs: Dict[int, Camera] = {}
        self.is_running = False
        
    def add_camera(self, camera_config: Camera) -> bool:
        """
        Add a camera to the manager.
        
        Args:
            camera_config: Camera configuration
            
        Returns:
            True if camera added successfully
        """
        try:
            camera_id = camera_config.id
            self.camera_configs[camera_id] = camera_config
            
            # Create camera handler
            camera = WideAngleCamera(camera_config, camera_id)
            self.cameras[camera_id] = camera
            
            logger.info(f"Added camera {camera_id}: {camera_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add camera {camera_config.id}: {e}")
            return False
    
    def remove_camera(self, camera_id: int):
        """Remove a camera from the manager."""
        if camera_id in self.cameras:
            self.cameras[camera_id].disconnect()
            del self.cameras[camera_id]
            del self.camera_configs[camera_id]
            logger.info(f"Removed camera {camera_id}")
    
    def connect_all_cameras(self) -> Dict[int, bool]:
        """
        Connect to all cameras.
        
        Returns:
            Dictionary mapping camera IDs to connection success status
        """
        results = {}
        for camera_id, camera in self.cameras.items():
            results[camera_id] = camera.connect()
        return results
    
    def disconnect_all_cameras(self):
        """Disconnect from all cameras."""
        for camera in self.cameras.values():
            camera.disconnect()
    
    def start_all_cameras(self):
        """Start capture for all online cameras."""
        self.is_running = True
        for camera in self.cameras.values():
            if camera.status == CameraStatus.ONLINE:
                camera.start_capture()
        logger.info("Started all cameras")
    
    def stop_all_cameras(self):
        """Stop capture for all cameras."""
        self.is_running = False
        for camera in self.cameras.values():
            camera.stop_capture_thread()
        logger.info("Stopped all cameras")
    
    def get_camera_status(self) -> Dict[int, str]:
        """Get status of all cameras."""
        return {camera_id: camera.status.value for camera_id, camera in self.cameras.items()}
    
    def get_latest_frames(self) -> Dict[int, Optional[CameraFrame]]:
        """
        Get latest frames from all cameras.
        
        Returns:
            Dictionary mapping camera IDs to latest frames
        """
        frames = {}
        for camera_id, camera in self.cameras.items():
            frames[camera_id] = camera.get_latest_frame()
        return frames
    
    def set_camera_zones(self, camera_id: int, zones: Dict, 
                        ingress_zones: List[str], egress_zones: List[str]):
        """Set zones for a specific camera."""
        if camera_id in self.cameras:
            self.cameras[camera_id].set_zones(zones, ingress_zones, egress_zones)
    
    def get_camera_calibration(self, camera_id: int) -> Optional[Dict]:
        """Get calibration data for a specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].calibration
        return None
    
    def get_camera_area(self, camera_id: int) -> float:
        """Get calibrated area for a specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_calibration_area()
        return 0.0