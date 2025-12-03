"""Adaptive detection tuning system for pedestrian detection optimization.

This module implements adaptive threshold adjustment based on scene conditions,
camera characteristics (especially GoPro wide-angle), and Mass Motion requirements.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import logging

from .detection_utils import Detection

logger = logging.getLogger(__name__)


@dataclass
class SceneConditions:
    """Current scene conditions affecting detection."""
    lighting_level: float = 0.5  # 0.0 (dark) to 1.0 (bright)
    pedestrian_density: float = 0.0  # Average pedestrians per frame
    average_confidence: float = 0.5  # Average detection confidence
    occlusion_level: float = 0.0  # Estimated occlusion (0.0 to 1.0)
    frame_center_pedestrian_size: float = 150.0  # Average height at frame center (pixels)
    edge_pedestrian_size: float = 80.0  # Average height at frame edges (pixels)
    size_variation: float = 0.0  # Coefficient of variation in pedestrian sizes


@dataclass
class TuningParameters:
    """Current detection tuning parameters."""
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    min_height: int = 50
    max_height: int = 300
    edge_confidence_offset: float = -0.1  # Lower confidence for edge detections
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TuningConfig:
    """Configuration for detection tuning."""
    adaptive_confidence: bool = True
    confidence_floor: float = 0.3
    confidence_ceiling: float = 0.7
    nms_adaptive: bool = True
    size_filter_adaptive: bool = True
    temporal_smoothing: float = 0.1  # Exponential moving average factor
    scene_analysis_window: int = 60  # Frames to analyze
    gopro_fov_degrees: float = 120.0  # GoPro wide-angle FOV
    enable_edge_compensation: bool = True  # Compensate for distortion at edges
    mass_motion_min_quality: float = 0.4  # Minimum confidence for Mass Motion agents


class DetectionTuner:
    """Adaptive detection parameter tuning system."""
    
    def __init__(self, config: Optional[TuningConfig] = None, 
                 image_size: Optional[Tuple[int, int]] = None,
                 stage: str = "stage2",
                 lightweight: bool = False):
        """
        Initialize detection tuner.
        
        Args:
            config: Tuning configuration (uses defaults if None)
            image_size: (width, height) of input images for GoPro FOV calculations
            stage: Workflow stage ("stage1" or "stage2") - affects update frequency
            lightweight: If True, reduces update frequency and analysis depth for speed
        """
        self.config = config or TuningConfig()
        self.image_size = image_size or (1920, 1080)  # Default GoPro resolution
        self.stage = stage
        self.lightweight = lightweight
        self.current_params = TuningParameters()
        
        # Adjust scene analysis window and update interval based on stage/lightweight mode
        if lightweight or stage == "stage1":
            # Stage 1: Faster updates, smaller history window for speed
            analysis_window = min(self.config.scene_analysis_window, 30)
            update_interval = 15
        else:
            # Stage 2: More comprehensive analysis for accuracy
            analysis_window = max(self.config.scene_analysis_window, 60)
            update_interval = 10
        
        # Scene condition history for temporal smoothing
        self.scene_history: deque = deque(maxlen=analysis_window)
        self.param_history: deque = deque(maxlen=30)  # Keep recent parameter history
        
        # Statistics tracking
        self.frame_count = 0
        self.last_update_frame = 0
        self.update_interval = update_interval  # Update parameters every N frames
        
        # GoPro wide-angle specific calculations
        self.frame_center = (self.image_size[0] / 2, self.image_size[1] / 2)
        self.frame_radius = np.sqrt(
            (self.image_size[0] / 2) ** 2 + (self.image_size[1] / 2) ** 2
        )
        
        logger.info(f"DetectionTuner initialized: image_size={self.image_size}, "
                   f"FOV={self.config.gopro_fov_degrees}°")
    
    def analyze_scene(self, detections: List[Detection], 
                     frame: Optional[np.ndarray] = None) -> SceneConditions:
        """
        Analyze current scene conditions from detections and frame.
        
        Args:
            detections: List of current detections
            frame: Optional frame for lighting analysis
            
        Returns:
            SceneConditions object
        """
        conditions = SceneConditions()
        
        if not detections:
            # No detections - likely challenging scene
            conditions.lighting_level = 0.3
            conditions.average_confidence = 0.3
            return conditions
        
        # Calculate average confidence
        confidences = [d.confidence for d in detections]
        conditions.average_confidence = np.mean(confidences) if confidences else 0.5
        
        # Pedestrian density (detections per frame)
        conditions.pedestrian_density = len(detections) / (
            self.image_size[0] * self.image_size[1] / 1000000  # Normalize by mega-pixels
        )
        
        # Analyze pedestrian sizes and positions
        heights = []
        edge_distances = []
        center_heights = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height = y2 - y1
            
            heights.append(height)
            
            # Distance from frame center (normalized)
            dist_from_center = np.sqrt(
                (center_x - self.frame_center[0]) ** 2 +
                (center_y - self.frame_center[1]) ** 2
            ) / self.frame_radius
            
            if dist_from_center < 0.3:  # Center region
                center_heights.append(height)
            elif dist_from_center > 0.7:  # Edge region
                edge_distances.append(dist_from_center)
        
        if heights:
            conditions.frame_center_pedestrian_size = (
                np.mean(center_heights) if center_heights 
                else np.mean(heights)
            )
            conditions.edge_pedestrian_size = (
                np.mean([h for i, h in enumerate(heights) 
                        if i < len(edge_distances)]) 
                if edge_distances else conditions.frame_center_pedestrian_size * 0.7
            )
            conditions.size_variation = np.std(heights) / (np.mean(heights) + 1e-6)
        
        # Estimate occlusion from confidence distribution
        if len(confidences) > 3:
            # High variance in confidences suggests occlusion/partial detections
            conf_std = np.std(confidences)
            conditions.occlusion_level = min(1.0, conf_std * 2.0)
        
        # Estimate lighting from frame if available
        if frame is not None:
            try:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                conditions.lighting_level = np.mean(gray) / 255.0
            except (ImportError, AttributeError):
                # Fallback if cv2 not available
                if len(frame.shape) == 3:
                    # Simple grayscale conversion without cv2
                    gray = np.mean(frame, axis=2)
                else:
                    gray = frame
                conditions.lighting_level = np.mean(gray) / 255.0
        
        return conditions
    
    def calculate_adaptive_confidence(self, conditions: SceneConditions) -> float:
        """
        Calculate adaptive confidence threshold based on scene conditions.
        
        Args:
            conditions: Current scene conditions
            
        Returns:
            Optimal confidence threshold
        """
        if not self.config.adaptive_confidence:
            return self.current_params.confidence_threshold
        
        base_threshold = 0.5
        adjustments = []
        
        # 1. Lighting adjustment: Lower threshold in dark scenes
        if conditions.lighting_level < 0.4:
            adjustments.append(-0.15)  # Lower threshold by 0.15
        elif conditions.lighting_level > 0.7:
            adjustments.append(0.05)  # Slightly higher in bright scenes
        
        # 2. Density adjustment: Lower threshold when sparse, higher when crowded
        if conditions.pedestrian_density < 0.5:
            adjustments.append(-0.1)  # More lenient for sparse scenes
        elif conditions.pedestrian_density > 2.0:
            adjustments.append(0.05)  # Stricter for crowded scenes
        
        # 3. Occlusion adjustment: Lower threshold when occlusion detected
        if conditions.occlusion_level > 0.3:
            adjustments.append(-0.1)
        
        # 4. Average confidence adjustment: Match threshold to observed confidence
        # If average confidence is low, lower threshold; if high, raise threshold
        confidence_diff = conditions.average_confidence - base_threshold
        adjustments.append(confidence_diff * 0.3)  # Partial adjustment
        
        # Apply adjustments
        new_threshold = base_threshold + sum(adjustments)
        
        # Clamp to floor and ceiling
        new_threshold = max(self.config.confidence_floor, 
                           min(self.config.confidence_ceiling, new_threshold))
        
        return new_threshold
    
    def calculate_adaptive_nms(self, conditions: SceneConditions) -> float:
        """
        Calculate adaptive NMS IoU threshold based on scene conditions.
        
        Args:
            conditions: Current scene conditions
            
        Returns:
            Optimal NMS IoU threshold
        """
        if not self.config.nms_adaptive:
            return self.current_params.nms_iou_threshold
        
        base_nms = 0.5
        
        # Adjust based on pedestrian density
        # Higher density -> slightly lower NMS (more aggressive suppression)
        if conditions.pedestrian_density > 1.5:
            adjustment = -0.05
        elif conditions.pedestrian_density < 0.5:
            adjustment = 0.05  # Less aggressive for sparse scenes
        else:
            adjustment = 0.0
        
        # Adjust based on size variation
        # High variation (mix of near/far) -> higher NMS threshold
        if conditions.size_variation > 0.5:
            adjustment += 0.05
        
        new_nms = base_nms + adjustment
        return max(0.3, min(0.7, new_nms))
    
    def calculate_adaptive_size_limits(self, conditions: SceneConditions) -> Tuple[int, int]:
        """
        Calculate adaptive size limits for GoPro wide-angle cameras.
        
        Args:
            conditions: Current scene conditions
            
        Returns:
            (min_height, max_height) tuple
        """
        if not self.config.size_filter_adaptive:
            return (self.current_params.min_height, self.current_params.max_height)
        
        # Base sizes for GoPro wide-angle
        # Account for wide FOV - pedestrians can be very small at edges
        min_base = 30  # Lower minimum for edge detections
        max_base = 400  # Higher maximum for close-up detections
        
        # Adjust based on observed sizes
        if conditions.frame_center_pedestrian_size > 0:
            # Use observed sizes as guide, but with safety margins
            observed_min = conditions.edge_pedestrian_size * 0.7
            observed_max = conditions.frame_center_pedestrian_size * 2.5
            
            min_height = max(min_base, int(observed_min))
            max_height = min(max_base, int(observed_max))
        else:
            min_height = min_base
            max_height = max_base
        
        return (min_height, max_height)
    
    def get_edge_confidence_offset(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate confidence offset for detections at frame edges (GoPro distortion).
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Confidence offset (negative for edges, 0 for center)
        """
        if not self.config.enable_edge_compensation:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Distance from frame center (normalized 0-1)
        dist_from_center = np.sqrt(
            (center_x - self.frame_center[0]) ** 2 +
            (center_y - self.frame_center[1]) ** 2
        ) / self.frame_radius
        
        # Apply offset: -0.15 at edge, 0 at center
        if dist_from_center > 0.8:  # Far edge
            return -0.15
        elif dist_from_center > 0.5:  # Mid edge
            return -0.08
        else:
            return 0.0
    
    def update_parameters(self, detections: List[Detection], 
                         frame: Optional[np.ndarray] = None) -> TuningParameters:
        """
        Update tuning parameters based on current scene analysis.
        
        Args:
            detections: Current frame detections
            frame: Optional frame for lighting analysis
            
        Returns:
            Updated TuningParameters
        """
        self.frame_count += 1
        
        # Analyze current scene
        conditions = self.analyze_scene(detections, frame)
        self.scene_history.append(conditions)
        
        # Update parameters periodically or if significant change
        if (self.frame_count - self.last_update_frame) >= self.update_interval:
            # Calculate temporal average of conditions for stability
            if len(self.scene_history) > 5:
                avg_conditions = self._average_scene_conditions()
            else:
                avg_conditions = conditions
            
            # Calculate new parameters
            new_params = TuningParameters()
            
            if self.config.adaptive_confidence:
                new_confidence = self.calculate_adaptive_confidence(avg_conditions)
                # Apply temporal smoothing
                if len(self.param_history) > 0:
                    old_confidence = self.current_params.confidence_threshold
                    smoothed = (1 - self.config.temporal_smoothing) * old_confidence + \
                              self.config.temporal_smoothing * new_confidence
                    new_params.confidence_threshold = smoothed
                else:
                    new_params.confidence_threshold = new_confidence
            else:
                new_params.confidence_threshold = self.current_params.confidence_threshold
            
            if self.config.nms_adaptive:
                new_params.nms_iou_threshold = self.calculate_adaptive_nms(avg_conditions)
            else:
                new_params.nms_iou_threshold = self.current_params.nms_iou_threshold
            
            if self.config.size_filter_adaptive:
                new_params.min_height, new_params.max_height = \
                    self.calculate_adaptive_size_limits(avg_conditions)
            else:
                new_params.min_height = self.current_params.min_height
                new_params.max_height = self.current_params.max_height
            
            new_params.edge_confidence_offset = self.current_params.edge_confidence_offset
            new_params.timestamp = datetime.now()
            
            # Update current parameters
            self.current_params = new_params
            self.param_history.append(new_params)
            self.last_update_frame = self.frame_count
            
            logger.debug(f"Updated tuning parameters: conf={new_params.confidence_threshold:.3f}, "
                       f"nms={new_params.nms_iou_threshold:.3f}, "
                       f"size=[{new_params.min_height}, {new_params.max_height}]")
        
        return self.current_params
    
    def _average_scene_conditions(self) -> SceneConditions:
        """Calculate average scene conditions from history."""
        if not self.scene_history:
            return SceneConditions()
        
        avg = SceneConditions()
        avg.lighting_level = np.mean([c.lighting_level for c in self.scene_history])
        avg.pedestrian_density = np.mean([c.pedestrian_density for c in self.scene_history])
        avg.average_confidence = np.mean([c.average_confidence for c in self.scene_history])
        avg.occlusion_level = np.mean([c.occlusion_level for c in self.scene_history])
        avg.frame_center_pedestrian_size = np.mean(
            [c.frame_center_pedestrian_size for c in self.scene_history 
             if c.frame_center_pedestrian_size > 0]
        )
        avg.edge_pedestrian_size = np.mean(
            [c.edge_pedestrian_size for c in self.scene_history 
             if c.edge_pedestrian_size > 0]
        )
        avg.size_variation = np.mean([c.size_variation for c in self.scene_history])
        
        return avg
    
    def get_current_parameters(self) -> TuningParameters:
        """Get current tuning parameters without updating."""
        return self.current_params
    
    def is_detection_quality_for_massmotion(self, detection: Detection) -> bool:
        """
        Check if detection quality meets Mass Motion requirements.
        
        Args:
            detection: Detection to check
            
        Returns:
            True if quality is sufficient for Mass Motion agent creation
        """
        # Apply edge compensation to confidence check
        edge_offset = self.get_edge_confidence_offset(detection.bbox)
        adjusted_confidence = detection.confidence + edge_offset
        
        return adjusted_confidence >= self.config.mass_motion_min_quality
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get statistics about tuning performance."""
        if not self.scene_history:
            return {}
        
        recent = list(self.scene_history)[-10:] if len(self.scene_history) >= 10 else list(self.scene_history)
        
        return {
            'frame_count': self.frame_count,
            'current_confidence_threshold': self.current_params.confidence_threshold,
            'current_nms_threshold': self.current_params.nms_iou_threshold,
            'current_size_limits': (self.current_params.min_height, self.current_params.max_height),
            'recent_avg_confidence': np.mean([c.average_confidence for c in recent]),
            'recent_avg_density': np.mean([c.pedestrian_density for c in recent]),
            'recent_avg_lighting': np.mean([c.lighting_level for c in recent]),
        }


# Import cv2 for lighting analysis
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available - lighting analysis will be disabled")
    cv2 = None

