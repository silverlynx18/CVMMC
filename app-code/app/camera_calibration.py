"""Camera calibration for pixel-to-world coordinate conversion."""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """
    Camera calibration for converting pixel coordinates to real-world coordinates.
    
    Supports two methods:
    1. Homography-based (for planar scenes)
    2. Pixels-per-meter (for simple orthogonal views)
    """
    homography_matrix: Optional[np.ndarray] = None
    pixels_per_meter: Optional[float] = None
    reference_points: List[Tuple[float, float]] = None  # (pixel_x, pixel_y, world_x, world_y)
    
    def __post_init__(self):
        """Validate calibration parameters."""
        if self.homography_matrix is None and self.pixels_per_meter is None:
            raise ValueError("Either homography_matrix or pixels_per_meter must be provided")
        
        if self.homography_matrix is not None:
            if self.homography_matrix.shape != (3, 3):
                raise ValueError("Homography matrix must be 3x3")
    
    def pixel_to_world(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates (meters).
        
        Args:
            pixel_point: (x, y) pixel coordinates
            
        Returns:
            (x, y) world coordinates in meters
        """
        if self.homography_matrix is not None:
            return self._pixel_to_world_homography(pixel_point)
        elif self.pixels_per_meter is not None:
            return self._pixel_to_world_ppm(pixel_point)
        else:
            raise RuntimeError("No calibration method available")
    
    def _pixel_to_world_homography(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert using homography matrix."""
        px, py = pixel_point
        pixel_homogeneous = np.array([px, py, 1.0])
        
        # Apply homography
        world_homogeneous = self.homography_matrix @ pixel_homogeneous
        
        # Normalize
        if world_homogeneous[2] != 0:
            world_x = world_homogeneous[0] / world_homogeneous[2]
            world_y = world_homogeneous[1] / world_homogeneous[2]
        else:
            logger.warning(f"Homography conversion failed for point {pixel_point}")
            return (0.0, 0.0)
        
        return (world_x, world_y)
    
    def _pixel_to_world_ppm(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert using pixels-per-meter ratio (assumes origin at top-left)."""
        px, py = pixel_point
        world_x = px / self.pixels_per_meter
        world_y = py / self.pixels_per_meter
        return (world_x, world_y)
    
    def calculate_real_speed(self, trajectory: List[Tuple[float, float]], 
                            fps: float) -> float:
        """
        Calculate real-world speed in m/s from a trajectory.
        
        Args:
            trajectory: List of (pixel_x, pixel_y) coordinates
            fps: Frames per second
            
        Returns:
            Speed in meters per second
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(len(trajectory) - 1):
            p1_world = self.pixel_to_world(trajectory[i])
            p2_world = self.pixel_to_world(trajectory[i + 1])
            
            dist = np.sqrt((p2_world[0] - p1_world[0])**2 + 
                          (p2_world[1] - p1_world[1])**2)
            total_distance += dist
        
        # Calculate time span
        time_span = (len(trajectory) - 1) / fps
        
        return total_distance / time_span if time_span > 0 else 0.0
    
    @classmethod
    def from_reference_points(cls, pixel_points: List[Tuple[float, float]],
                             world_points: List[Tuple[float, float]]) -> 'CameraCalibration':
        """
        Create calibration from reference point correspondences.
        
        Args:
            pixel_points: List of (x, y) pixel coordinates
            world_points: List of (x, y) world coordinates in meters
            
        Returns:
            CameraCalibration object
        """
        if len(pixel_points) != len(world_points):
            raise ValueError("Number of pixel and world points must match")
        
        if len(pixel_points) < 4:
            raise ValueError("Need at least 4 point correspondences for homography")
        
        # Convert to numpy arrays
        src_pts = np.array(pixel_points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        
        # Calculate homography
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return cls(
            homography_matrix=homography,
            reference_points=list(zip(pixel_points, world_points))
        )
    
    @classmethod
    def from_pixels_per_meter(cls, pixels_per_meter: float) -> 'CameraCalibration':
        """
        Create simple calibration using pixels-per-meter ratio.
        
        Args:
            pixels_per_meter: Conversion ratio
            
        Returns:
            CameraCalibration object
        """
        return cls(pixels_per_meter=pixels_per_meter)
    
    def save_to_file(self, filepath: str):
        """Save calibration to file."""
        import json
        
        data = {
            'pixels_per_meter': self.pixels_per_meter,
            'reference_points': self.reference_points,
        }
        
        if self.homography_matrix is not None:
            data['homography_matrix'] = self.homography_matrix.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CameraCalibration':
        """Load calibration from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        homography = None
        if 'homography_matrix' in data:
            homography = np.array(data['homography_matrix'])
        
        return cls(
            homography_matrix=homography,
            pixels_per_meter=data.get('pixels_per_meter'),
            reference_points=data.get('reference_points')
        )

