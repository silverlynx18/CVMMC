"""Pedestrian tracking and counting system with Kalman filtering."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("filterpy not available. Kalman filtering disabled.")

from .detection_utils import Detection
from .models import PedestrianDetection, PedestrianCount


@dataclass
class TrackedPedestrian:
    """Tracked pedestrian object with Kalman filtering."""
    track_id: int
    detections: List[Detection]
    first_seen: datetime
    last_seen: datetime
    is_ingress: Optional[bool] = None
    zone_id: Optional[str] = None
    disappeared_frames: int = 0
    confidence_score: float = 0.5
    consecutive_misses: int = 0
    kf: Optional[Any] = None
    predicted_position: Optional[Tuple[float, float]] = None
    current_detection: Optional[Detection] = None  # Current frame's detection (if matched)


class PedestrianTracker:
    """Multi-object pedestrian tracker with Hungarian algorithm and Kalman filtering."""
    
    def __init__(self, max_disappeared: int = 90, max_distance: float = 250.0,
                 use_kalman: bool = True, min_track_confidence: float = 0.3,
                 detection_tuner=None):
        """Initialize tracker.
        
        Args:
            max_disappeared: Max frames track can be missing before removal (90 = 3 sec at 30fps)
            max_distance: Base maximum pixel distance for association (250px for wide-angle)
            use_kalman: Enable Kalman filtering for motion prediction
            min_track_confidence: Minimum confidence score to keep a track
            detection_tuner: Optional DetectionTuner for quality feedback
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_kalman = use_kalman and KALMAN_AVAILABLE
        self.min_track_confidence = min_track_confidence
        self.detection_tuner = detection_tuner
        self.next_id = 1
        self.tracked_pedestrians: Dict[int, TrackedPedestrian] = {}
        self.disappeared_count: Dict[int, int] = defaultdict(int)
        
        self.tracking_stats = {
            'total_associations': 0,
            'successful_associations': 0,
            'failed_associations': 0,
            'new_tracks_created': 0,
            'tracks_lost': 0
        }
        
        self.zones = {}
        self.ingress_zones = set()
        self.egress_zones = set()
        
        if self.use_kalman:
            logger.info("Kalman filtering enabled")
        else:
            logger.info("Kalman filtering disabled (filterpy not available)")
        
    def set_zones(self, zones: Dict[str, Dict], 
                  ingress_zones: List[str], 
                  egress_zones: List[str]):
        """Set up detection zones."""
        self.zones = zones
        self.ingress_zones = set(ingress_zones)
        self.egress_zones = set(egress_zones)
        logger.info(f"Set up {len(zones)} zones: {len(ingress_zones)} ingress, {len(egress_zones)} egress")
    
    def _get_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, detections: List[Detection], frame_timestamp: datetime) -> List[TrackedPedestrian]:
        """Update tracker with new detections.
            
        Returns:
            List of currently tracked pedestrians
        """
        # Handle empty detections
        if len(detections) == 0:
            for track_id in list(self.disappeared_count.keys()):
                self.disappeared_count[track_id] += 1
                if self.disappeared_count[track_id] > self.max_disappeared:
                    self._remove_track(track_id)
            return list(self.tracked_pedestrians.values())
        
        # Create tracks if none exist
        if len(self.tracked_pedestrians) == 0:
            for detection in detections:
                self._create_track(detection, frame_timestamp)
            return list(self.tracked_pedestrians.values())
        
        # Associate detections with existing tracks
        track_ids = list(self.tracked_pedestrians.keys())
        detection_centers = [self._get_center(det.bbox) for det in detections]
        
        # Compute distance matrix (now includes IoU)
        distances = self._compute_distance_matrix(track_ids, detection_centers, detections)
        
        # Associate using Hungarian algorithm with validation
        track_associations = self._associate_tracks(distances, track_ids, detections)
        
        # Apply associations - update existing tracks
        used_detection_indices = set()
        assigned_track_ids = set()
        
        for track_idx, det_idx in enumerate(track_associations):
            if det_idx is not None:
                if track_idx < len(track_ids) and det_idx < len(detections):
                    track_id = track_ids[track_idx]
                    detection = detections[det_idx]
                    self._update_track(track_id, detection, frame_timestamp)
                    # Mark this track as having a current detection
                    if track_id in self.tracked_pedestrians:
                        self.tracked_pedestrians[track_id].current_detection = detection
                    used_detection_indices.add(det_idx)
                    assigned_track_ids.add(track_id)
                    self.tracking_stats['successful_associations'] += 1
        
        # Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in used_detection_indices:
                self._create_track(detection, frame_timestamp)
                self.tracking_stats['new_tracks_created'] += 1
        
        # Handle tracks that weren't matched
        for track_id in track_ids:
            if track_id not in assigned_track_ids:
                track = self.tracked_pedestrians[track_id]
                track.consecutive_misses += 1
                self.disappeared_count[track_id] += 1
                # Clear current detection - track is missing
                track.current_detection = None
                
                # Predict position for missing tracks
                if track.kf is not None:
                    track.kf.predict()
                    track.predicted_position = (track.kf.x[0], track.kf.x[1])
                
                # Update confidence
                track.confidence_score = self._calculate_track_confidence(track)
                
                # Remove if too many misses
                if self.disappeared_count[track_id] > self.max_disappeared:
                    self._remove_track(track_id)
        
        self.tracking_stats['failed_associations'] = len(track_ids) - len(assigned_track_ids)
        self.tracking_stats['total_associations'] = len(track_associations)
        
        # Provide quality feedback to detection tuner
        self._provide_quality_feedback()
        
        # Filter by confidence AND only return tracks with current detections
        # This prevents empty bounding boxes from tracks that have disappeared
        valid_tracks = [
            track for track in self.tracked_pedestrians.values()
            if track.confidence_score >= self.min_track_confidence
            and track.current_detection is not None  # Only return tracks with current detection
        ]
        
        return valid_tracks
    
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_distance_matrix(self, track_ids: List[int], 
                               detection_centers: List[Tuple[float, float]],
                               detections: List[Detection]) -> np.ndarray:
        """Compute distance matrix between tracks and detections.
        
        Uses combined metric: distance + (1 - IoU) for better association.
        """
        num_tracks = len(track_ids)
        num_detections = len(detection_centers)
        distances = np.full((num_tracks, num_detections), np.inf)
        
        for i, track_id in enumerate(track_ids):
            if track_id not in self.tracked_pedestrians:
                continue
            
            track = self.tracked_pedestrians[track_id]
            
            # Get track position and bbox (use current detection if available, else last)
            if track.current_detection is not None:
                # Use current detection if available
                track_bbox = track.current_detection.bbox
                track_center = self._get_center(track_bbox)
            elif track.kf is not None:
                # Use Kalman prediction
                track.kf.predict()
                track.predicted_position = (track.kf.x[0], track.kf.x[1])
                track_center = track.predicted_position
                # Use last bbox for IoU calculation
                if track.detections:
                    track_bbox = track.detections[-1].bbox
                else:
                    continue
            elif track.detections:
                track_bbox = track.detections[-1].bbox
                track_center = self._get_center(track_bbox)
            else:
                continue  # Skip tracks with no position data
            
            # Compute distances to all detections (combine distance with IoU)
            for j, (det_center, detection) in enumerate(zip(detection_centers, detections)):
                # Euclidean distance
                dx = track_center[0] - det_center[0]
                dy = track_center[1] - det_center[1]
                euclidean_dist = np.sqrt(dx*dx + dy*dy)
                
                # IoU-based distance (1 - IoU, so higher IoU = lower distance)
                iou = self._compute_iou(track_bbox, detection.bbox)
                iou_distance = (1.0 - iou) * 200.0  # Scale IoU to pixel-equivalent distance
                
                # Combined metric: weighted combination of distance and IoU
                # IoU is more reliable for association, so weight it higher
                combined_distance = 0.4 * euclidean_dist + 0.6 * iou_distance
                distances[i, j] = combined_distance
        
        return distances
    
    def _associate_tracks(self, distances: np.ndarray, track_ids: List[int], 
                         detections: List[Detection]) -> List[Optional[int]]:
        """Associate tracks with detections using Hungarian algorithm with validation.
        
        Args:
            distances: Distance matrix between tracks and detections
            track_ids: List of track IDs corresponding to rows in distances
            detections: List of Detection objects corresponding to columns in distances
        
        Returns:
            List where result[i] = detection index for track i, or None if unassigned
        """
        from scipy.optimize import linear_sum_assignment
        
        if distances.size == 0 or distances.shape[0] == 0 or distances.shape[1] == 0:
            return [None] * distances.shape[0] if distances.shape[0] > 0 else []
        
        num_tracks, num_detections = distances.shape
        
        # Filter distances by threshold (stricter for better accuracy)
        # Use adaptive threshold based on track's disappeared count
        distance_threshold = self.max_distance * 2.0  # Reduced from 3.0 to 2.0 for stricter matching
        distances_filtered = distances.copy()
        distances_filtered[distances_filtered > distance_threshold] = np.inf
        
        # Apply stricter thresholds for tracks that have been missing
        # Tracks that disappeared recently should have tighter matching
        for i, track_id in enumerate(track_ids):
            if track_id in self.tracked_pedestrians:
                track = self.tracked_pedestrians[track_id]
                if track.consecutive_misses > 0:
                    # Stricter threshold for missing tracks
                    strict_threshold = self.max_distance * (1.0 + track.consecutive_misses * 0.1)
                    strict_threshold = min(strict_threshold, self.max_distance * 1.5)  # Cap at 1.5x
                    distances_filtered[i, distances_filtered[i] > strict_threshold] = np.inf
        
        # Replace inf with large number for Hungarian algorithm
        distances_for_hungarian = distances_filtered.copy()
        distances_for_hungarian[np.isinf(distances_for_hungarian)] = 1e10
        
        try:
            track_indices, detection_indices = linear_sum_assignment(distances_for_hungarian)
        except ValueError:
            return [None] * num_tracks
        
        # Build association list with validation
        track_associations = [None] * num_tracks
        
        for i in range(len(track_indices)):
            track_idx = int(track_indices[i])
            det_idx = int(detection_indices[i])
            
            # Basic validation: indices and distance threshold
            if (track_idx >= num_tracks or det_idx >= num_detections or
                distances_filtered[track_idx, det_idx] == np.inf or
                np.isnan(distances_filtered[track_idx, det_idx])):
                continue
            
            # Enhanced validation: check if association is valid
            # This prevents ID reuse by validating size, distance, and position consistency
            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            
            if self._is_valid_association(track_id, detection, distances[track_idx, det_idx]):
                track_associations[track_idx] = det_idx
        
        return track_associations
    
    def _is_valid_association(self, track_id: int, detection: Detection, 
                              distance: float) -> bool:
        """Validate if a detection can be associated with a track.
        
        Enhanced validation with IoU and stricter checks to prevent ID switching.
        
        Checks:
        1. IoU threshold (must have reasonable overlap)
        2. Distance moved is reasonable (not jumping across frame)
        3. Size change is reasonable (accounting for perspective and wide-angle lens)
        4. Width consistency for wheelchairs/strollers
        5. Stricter validation for tracks that have been missing
        
        Args:
            track_id: Track ID to validate
            detection: Detection candidate
            distance: Combined distance metric (includes IoU component)
        
        Returns:
            True if association is valid, False otherwise
        """
        if track_id not in self.tracked_pedestrians:
            return False
        
        track = self.tracked_pedestrians[track_id]
        
        # Need at least one previous detection to validate
        if not track.detections or len(track.detections) == 0:
            return True  # New track, accept
        
        # Use current detection if available, else last detection
        if track.current_detection is not None:
            last_detection = track.current_detection
        else:
            last_detection = track.detections[-1]
        
        last_bbox = last_detection.bbox
        new_bbox = detection.bbox
        
        # 0. IoU validation - must have minimum overlap
        iou = self._compute_iou(last_bbox, new_bbox)
        min_iou = 0.1  # Minimum IoU required (allows for some movement)
        
        # Stricter IoU for tracks that have been missing
        if track.consecutive_misses > 0:
            min_iou = 0.2  # Higher IoU required for missing tracks
        
        if iou < min_iou:
            return False
        
        # Calculate sizes
        last_x1, last_y1, last_x2, last_y2 = last_bbox
        new_x1, new_y1, new_x2, new_y2 = new_bbox
        
        last_w = last_x2 - last_x1
        last_h = last_y2 - last_y1
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        
        last_size = (last_w + last_h) / 2
        new_size = (new_w + new_h) / 2
        
        # 1. Distance validation: reject if moved too far
        # Stricter for tracks that have been missing
        if track.consecutive_misses > 0:
            max_reasonable_distance = self.max_distance * 1.5  # Stricter for missing tracks
        else:
            max_reasonable_distance = self.max_distance * 2.0  # 500px for max_distance=250
        
        if distance > max_reasonable_distance:
            return False
        
        # 2. Size consistency validation (stricter)
        size_ratio = new_size / last_size if last_size > 0 else 1.0
        
        # Stricter size validation for missing tracks
        if track.consecutive_misses > 0:
            max_size_ratio = 2.0  # Stricter for missing tracks
            min_size_ratio = 0.5
        else:
            max_size_ratio = 2.5
            min_size_ratio = 0.4
        
        if size_ratio > max_size_ratio or size_ratio < min_size_ratio:
            return False
        
        # 3. Width consistency for wheelchairs/strollers
        last_aspect = last_w / last_h if last_h > 0 else 1.0
        new_aspect = new_w / new_h if new_h > 0 else 1.0
        
        if last_aspect > 0.7:
            if new_aspect < 0.5:
                return False
            width_ratio = new_w / last_w if last_w > 0 else 1.0
            if width_ratio > 2.0 or width_ratio < 0.5:  # Stricter
                return False
        
        # 4. Position jump validation (stricter)
        last_center = self._get_center(last_bbox)
        new_center = self._get_center(new_bbox)
        
        center_distance = np.sqrt((last_center[0] - new_center[0])**2 + 
                                 (last_center[1] - new_center[1])**2)
        
        # Stricter position jump validation
        max_jump = 300 if track.consecutive_misses == 0 else 200  # Stricter for missing tracks
        if center_distance > max_jump:
            return False
        
        # 5. For very small movements, be more strict about size changes
        if center_distance < 50 and (size_ratio > 1.5 or size_ratio < 0.67):  # Stricter
            return False
        
        return True
    
    def _update_track(self, track_id: int, detection: Detection, timestamp: datetime):
        """Update existing track with new detection."""
        if track_id not in self.tracked_pedestrians:
            return
        
        track = self.tracked_pedestrians[track_id]
        track.detections.append(detection)
        track.current_detection = detection  # Set current detection
        track.last_seen = timestamp
        track.consecutive_misses = 0
        self.disappeared_count[track_id] = 0
        
        # Update Kalman filter
        if track.kf is not None:
            center = self._get_center(detection.bbox)
            track.kf.update(np.array([center[0], center[1]], dtype=float))
        
        # Update confidence
        track.confidence_score = self._calculate_track_confidence(track)
        
        # Update zone and movement direction if changed
        zone_id = self._get_zone_id(detection.bbox)
        if zone_id != track.zone_id:
            track.zone_id = zone_id
            track.is_ingress = self._determine_movement_direction(zone_id)
    
    def _create_track(self, detection: Detection, timestamp: datetime):
        """Create new track for detection."""
        track_id = self.next_id
        self.next_id += 1
        
        # Determine zone and movement direction
        zone_id = self._get_zone_id(detection.bbox)
        is_ingress = self._determine_movement_direction(zone_id)
        
        # Initialize Kalman filter if enabled
        kf = None
        if self.use_kalman:
            kf = self._init_kalman_filter(detection.bbox)
        
        track = TrackedPedestrian(
            track_id=track_id,
            detections=[detection],
            first_seen=timestamp,
            last_seen=timestamp,
            is_ingress=is_ingress,
            zone_id=zone_id,
            confidence_score=detection.confidence,
            kf=kf,
            current_detection=detection  # Set initial current detection
        )
        
        # Update Kalman filter with initial detection
        if kf is not None:
            center = self._get_center(detection.bbox)
            kf.update(np.array([center[0], center[1]], dtype=float))
        
        self.tracked_pedestrians[track_id] = track
        self.disappeared_count[track_id] = 0
    
    def _init_kalman_filter(self, bbox: Tuple[float, float, float, float]) -> Optional[Any]:
        """Initialize Kalman filter for track."""
        if not KALMAN_AVAILABLE:
            return None
        
        kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, y, vx, vy]
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        
        # Measurement matrix
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        
        # Covariance matrices
        kf.P *= 1000.0
        kf.R *= 10.0
        kf.Q *= 0.1
        
        # Initialize state with center position
        center = self._get_center(bbox)
        kf.x = np.array([center[0], center[1], 0.0, 0.0], dtype=float)
        
        return kf
    
    def _calculate_track_confidence(self, track: TrackedPedestrian) -> float:
        """Calculate confidence score for track."""
        confidence = 0.0
        
        # Factor 1: Track length
        length_factor = min(1.0, len(track.detections) / 30.0)
        confidence += 0.3 * length_factor
        
        # Factor 2: Average detection confidence
        if track.detections:
            avg_conf = np.mean([d.confidence for d in track.detections])
            confidence += 0.3 * avg_conf
        
        # Factor 3: Fewer misses
        miss_factor = 1.0 - (track.consecutive_misses / self.max_disappeared)
        confidence += 0.2 * max(0, miss_factor)
        
        return min(1.0, confidence)
    
    def _remove_track(self, track_id: int):
        """Remove track from tracking."""
        if track_id in self.tracked_pedestrians:
            del self.tracked_pedestrians[track_id]
        if track_id in self.disappeared_count:
            del self.disappeared_count[track_id]
        self.tracking_stats['tracks_lost'] += 1
    
    def _get_zone_id(self, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Determine which zone a bounding box belongs to."""
        center_x, center_y = self._get_center(bbox)
        
        for zone_id, zone_data in self.zones.items():
            if self._point_in_polygon((center_x, center_y), zone_data.get('polygon', [])):
                return zone_id
        
        return None
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _determine_movement_direction(self, zone_id: Optional[str]) -> Optional[bool]:
        """Determine if movement is ingress (True) or egress (False)."""
        if zone_id is None:
            return None
        
        if zone_id in self.ingress_zones:
            return True
        elif zone_id in self.egress_zones:
            return False
        
        return None
    
    def get_counts(self, time_window_minutes: int = 5) -> Dict[str, int]:
        """Get pedestrian counts for the specified time window."""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(minutes=time_window_minutes)
        
        ingress_count = 0
        egress_count = 0
        current_pedestrians = 0
        
        for track in self.tracked_pedestrians.values():
            # Count completed movements
            if track.is_ingress is not None:
                if track.last_seen >= cutoff_time:
                    if track.is_ingress:
                        ingress_count += 1
                    else:
                        egress_count += 1
            
            # Count current pedestrians (seen in last 30 seconds)
            if track.last_seen >= now - timedelta(seconds=30):
                current_pedestrians += 1
        
        return {
            'ingress_count': ingress_count,
            'egress_count': egress_count,
            'net_count': ingress_count - egress_count,
            'current_pedestrians': current_pedestrians
        }
    
    def get_tracked_pedestrians(self) -> List[TrackedPedestrian]:
        """Get all currently tracked pedestrians."""
        return list(self.tracked_pedestrians.values())
    
    def _provide_quality_feedback(self) -> None:
        """Provide quality feedback to detection tuner based on tracking performance."""
        if self.detection_tuner is None:
            return
        
        total = self.tracking_stats['total_associations']
        if total == 0:
            return
        
        success_rate = self.tracking_stats['successful_associations'] / total
        new_track_ratio = self.tracking_stats['new_tracks_created'] / (
            total + self.tracking_stats['new_tracks_created'] + 1e-6
        )
        
        # Log quality indicators
        if success_rate < 0.5 or new_track_ratio > 0.5:
            logger.debug(f"Tracking quality indicators: success_rate={success_rate:.2f}, "
                        f"new_track_ratio={new_track_ratio:.2f}")
        
        # Reset stats periodically for rolling average
        if total > 100:
            self.tracking_stats['total_associations'] = int(total * 0.5)
            self.tracking_stats['successful_associations'] = int(
                self.tracking_stats['successful_associations'] * 0.5
            )
            self.tracking_stats['new_tracks_created'] = int(
                self.tracking_stats['new_tracks_created'] * 0.5
            )
            self.tracking_stats['tracks_lost'] = int(
                self.tracking_stats['tracks_lost'] * 0.5
            )
    
    def get_tracking_quality_metrics(self) -> Dict[str, float]:
        """Get tracking quality metrics for analysis."""
        total = self.tracking_stats['total_associations']
        if total == 0:
            return {
                'success_rate': 0.0,
                'new_track_ratio': 0.0,
                'track_loss_rate': 0.0
            }
        
        return {
            'success_rate': self.tracking_stats['successful_associations'] / total,
            'new_track_ratio': self.tracking_stats['new_tracks_created'] / (
                total + self.tracking_stats['new_tracks_created'] + 1e-6
            ),
            'track_loss_rate': self.tracking_stats['tracks_lost'] / (
                total + self.tracking_stats['tracks_lost'] + 1e-6
            )
        }