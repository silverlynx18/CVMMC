"""
Multi-camera pedestrian tracking with re-identification (ReID).
Enables tracking the same pedestrian across multiple camera views.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PedestrianAppearance:
    """
    Appearance features for pedestrian re-identification.
    These are simplified features - in production you'd use deep learning features.
    """
    # Visual features
    color_histogram: np.ndarray  # Color distribution
    height_width_ratio: float    # Body shape
    avg_speed: float            # Movement characteristics
    
    # Metadata
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    track_id: int  # Local track ID in camera
    
    # Position features (for camera topology)
    exit_position: Optional[Tuple[float, float]] = None  # Where they left the camera view
    entry_position: Optional[Tuple[float, float]] = None  # Where they entered
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to a feature vector for similarity comparison."""
        features = []
        
        # Color histogram features (normalized)
        if self.color_histogram is not None:
            features.extend(self.color_histogram.flatten())
        
        # Geometric features
        features.append(self.height_width_ratio)
        features.append(self.avg_speed / 100.0)  # Normalize speed
        
        return np.array(features)


@dataclass  
class GlobalTrack:
    """
    A global track representing the same pedestrian across multiple cameras.
    """
    global_id: int
    camera_tracks: Dict[str, int] = field(default_factory=dict)  # camera_id -> local_track_id
    appearances: List[PedestrianAppearance] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    def add_appearance(self, appearance: PedestrianAppearance):
        """Add a new appearance (from a camera)."""
        self.appearances.append(appearance)
        self.camera_tracks[appearance.camera_id] = appearance.track_id
        
        if self.first_seen is None or appearance.first_seen < self.first_seen:
            self.first_seen = appearance.first_seen
        
        if self.last_seen is None or appearance.last_seen > self.last_seen:
            self.last_seen = appearance.last_seen
    
    def get_cameras_visited(self) -> List[str]:
        """Get list of cameras this person has been seen in."""
        return list(self.camera_tracks.keys())
    
    def get_journey_duration(self) -> Optional[timedelta]:
        """Get total journey duration across all cameras."""
        if self.first_seen and self.last_seen:
            return self.last_seen - self.first_seen
        return None


@dataclass
class CameraTopology:
    """
    Defines the spatial relationship between cameras.
    Helps with re-identification by knowing which cameras are adjacent.
    """
    camera_id: str
    adjacent_cameras: List[str]  # Cameras that pedestrians can move to
    transition_time_range: Tuple[float, float] = (2.0, 30.0)  # Min/max seconds to reach adjacent cameras
    exit_zones: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)  # zone_name -> bbox
    
    def is_adjacent_to(self, other_camera_id: str) -> bool:
        """Check if another camera is adjacent."""
        return other_camera_id in self.adjacent_cameras
    
    def get_expected_transition_time(self, target_camera: str) -> Optional[Tuple[float, float]]:
        """Get expected transition time range to target camera."""
        if self.is_adjacent_to(target_camera):
            return self.transition_time_range
        return None


class MultiCameraTracker:
    """
    Tracks pedestrians across multiple camera views using re-identification.
    """
    
    def __init__(self,
                 reid_similarity_threshold: float = 0.7,
                 max_time_gap: float = 60.0,
                 use_camera_topology: bool = True):
        """
        Initialize multi-camera tracker.
        
        Args:
            reid_similarity_threshold: Minimum similarity for matching (0-1)
            max_time_gap: Maximum time between appearances to consider a match (seconds)
            use_camera_topology: Whether to use camera topology constraints
        """
        self.reid_similarity_threshold = reid_similarity_threshold
        self.max_time_gap = max_time_gap
        self.use_camera_topology = use_camera_topology
        
        # Global tracking
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        
        # Camera topology
        self.camera_topology: Dict[str, CameraTopology] = {}
        
        # Mapping from (camera_id, local_track_id) to global_id
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        
        # Pending appearances (waiting for potential matches)
        self.pending_appearances: List[PedestrianAppearance] = []
        
    def register_camera_topology(self, topology: CameraTopology):
        """Register topology information for a camera."""
        self.camera_topology[topology.camera_id] = topology
        logger.info(f"Registered topology for camera {topology.camera_id} with {len(topology.adjacent_cameras)} adjacent cameras")
    
    def add_appearance(self, appearance: PedestrianAppearance) -> int:
        """
        Add a new pedestrian appearance from a camera.
        Attempts to match with existing global tracks.
        
        Args:
            appearance: PedestrianAppearance from a camera
            
        Returns:
            Global track ID
        """
        # Check if we already have a global ID for this camera track
        local_key = (appearance.camera_id, appearance.track_id)
        if local_key in self.local_to_global:
            global_id = self.local_to_global[local_key]
            # Update the existing global track
            if global_id in self.global_tracks:
                self.global_tracks[global_id].add_appearance(appearance)
            return global_id
        
        # Try to match with existing global tracks
        matched_global_id = self._find_matching_track(appearance)
        
        if matched_global_id is not None:
            # Found a match!
            self.global_tracks[matched_global_id].add_appearance(appearance)
            self.local_to_global[local_key] = matched_global_id
            logger.info(f"Matched camera {appearance.camera_id} track {appearance.track_id} to global ID {matched_global_id}")
            return matched_global_id
        else:
            # No match - create new global track
            global_id = self._create_global_track(appearance)
            logger.info(f"Created new global track {global_id} for camera {appearance.camera_id} track {appearance.track_id}")
            return global_id
    
    def _find_matching_track(self, appearance: PedestrianAppearance) -> Optional[int]:
        """
        Find the best matching global track for an appearance.
        
        Args:
            appearance: New appearance to match
            
        Returns:
            Global track ID or None
        """
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, global_track in self.global_tracks.items():
            # Skip if this camera is already in the track (can't be the same person twice)
            if appearance.camera_id in global_track.camera_tracks:
                continue
            
            # Check temporal constraints
            if global_track.last_seen:
                time_gap = (appearance.first_seen - global_track.last_seen).total_seconds()
                if time_gap < 0 or time_gap > self.max_time_gap:
                    continue
            
            # Check camera topology constraints
            if self.use_camera_topology and global_track.appearances:
                last_appearance = global_track.appearances[-1]
                if not self._is_valid_transition(last_appearance, appearance):
                    continue
            
            # Calculate appearance similarity
            similarity = self._calculate_similarity(global_track, appearance)
            
            if similarity > best_similarity and similarity >= self.reid_similarity_threshold:
                best_similarity = similarity
                best_match_id = global_id
        
        return best_match_id
    
    def _calculate_similarity(self, global_track: GlobalTrack, 
                             appearance: PedestrianAppearance) -> float:
        """
        Calculate similarity between a global track and a new appearance.
        
        Args:
            global_track: Existing global track
            appearance: New appearance
            
        Returns:
            Similarity score 0-1
        """
        if not global_track.appearances:
            return 0.0
        
        # Compare with all appearances in the track (use average or max)
        similarities = []
        
        for track_appearance in global_track.appearances:
            # Feature vector similarity (cosine similarity)
            vec1 = track_appearance.to_feature_vector()
            vec2 = appearance.to_feature_vector()
            
            if len(vec1) == len(vec2):
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_sim = dot_product / (norm1 * norm2)
                    similarities.append(cos_sim)
        
        if not similarities:
            return 0.0
        
        # Use maximum similarity (most similar appearance)
        return float(np.max(similarities))
    
    def _is_valid_transition(self, 
                            from_appearance: PedestrianAppearance,
                            to_appearance: PedestrianAppearance) -> bool:
        """
        Check if transition between two cameras is valid based on topology.
        
        Args:
            from_appearance: Previous appearance
            to_appearance: New appearance
            
        Returns:
            True if transition is valid
        """
        from_camera = from_appearance.camera_id
        to_camera = to_appearance.camera_id
        
        # Check if cameras are adjacent
        if from_camera in self.camera_topology:
            topology = self.camera_topology[from_camera]
            
            if not topology.is_adjacent_to(to_camera):
                return False
            
            # Check transition time
            expected_time = topology.get_expected_transition_time(to_camera)
            if expected_time:
                time_gap = (to_appearance.first_seen - from_appearance.last_seen).total_seconds()
                min_time, max_time = expected_time
                
                if time_gap < min_time or time_gap > max_time:
                    logger.debug(f"Transition time {time_gap:.1f}s outside expected range {min_time}-{max_time}s")
                    return False
        
        return True
    
    def _create_global_track(self, appearance: PedestrianAppearance) -> int:
        """Create a new global track for an appearance."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        global_track = GlobalTrack(global_id=global_id)
        global_track.add_appearance(appearance)
        
        self.global_tracks[global_id] = global_track
        self.local_to_global[(appearance.camera_id, appearance.track_id)] = global_id
        
        return global_id
    
    def get_global_track(self, camera_id: str, local_track_id: int) -> Optional[GlobalTrack]:
        """
        Get the global track for a local camera track.
        
        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID in that camera
            
        Returns:
            GlobalTrack or None
        """
        global_id = self.local_to_global.get((camera_id, local_track_id))
        if global_id:
            return self.global_tracks.get(global_id)
        return None
    
    def get_journey_statistics(self) -> Dict[str, any]:
        """
        Get statistics about pedestrian journeys across cameras.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_global_tracks': len(self.global_tracks),
            'multi_camera_tracks': 0,
            'single_camera_tracks': 0,
            'avg_cameras_per_journey': 0.0,
            'camera_transitions': defaultdict(int),  # (camera_a, camera_b) -> count
            'avg_journey_duration': 0.0
        }
        
        durations = []
        camera_counts = []
        
        for track in self.global_tracks.values():
            cameras_visited = track.get_cameras_visited()
            camera_counts.append(len(cameras_visited))
            
            if len(cameras_visited) > 1:
                stats['multi_camera_tracks'] += 1
                
                # Record transitions
                for i in range(len(track.appearances) - 1):
                    from_cam = track.appearances[i].camera_id
                    to_cam = track.appearances[i + 1].camera_id
                    if from_cam != to_cam:
                        stats['camera_transitions'][(from_cam, to_cam)] += 1
            else:
                stats['single_camera_tracks'] += 1
            
            duration = track.get_journey_duration()
            if duration:
                durations.append(duration.total_seconds())
        
        if camera_counts:
            stats['avg_cameras_per_journey'] = np.mean(camera_counts)
        
        if durations:
            stats['avg_journey_duration'] = np.mean(durations)
        
        # Convert defaultdict to regular dict for cleaner output
        stats['camera_transitions'] = dict(stats['camera_transitions'])
        
        return stats
    
    def export_journeys(self) -> List[Dict]:
        """
        Export all journeys for analysis.
        
        Returns:
            List of journey dictionaries
        """
        journeys = []
        
        for global_id, track in self.global_tracks.items():
            journey = {
                'global_id': global_id,
                'cameras_visited': track.get_cameras_visited(),
                'num_cameras': len(track.get_cameras_visited()),
                'first_seen': track.first_seen.isoformat() if track.first_seen else None,
                'last_seen': track.last_seen.isoformat() if track.last_seen else None,
                'duration_seconds': track.get_journey_duration().total_seconds() if track.get_journey_duration() else None,
                'appearances': []
            }
            
            for appearance in track.appearances:
                journey['appearances'].append({
                    'camera_id': appearance.camera_id,
                    'local_track_id': appearance.track_id,
                    'first_seen': appearance.first_seen.isoformat(),
                    'last_seen': appearance.last_seen.isoformat(),
                    'avg_speed': appearance.avg_speed
                })
            
            journeys.append(journey)
        
        return journeys


def extract_appearance_features(detections: List, 
                               frames: List[np.ndarray],
                               track_id: int,
                               camera_id: str) -> PedestrianAppearance:
    """
    Extract appearance features from detections for ReID.
    This is a simplified version - production would use deep learning features.
    
    Args:
        detections: List of Detection objects
        frames: Corresponding frames
        track_id: Track ID
        camera_id: Camera ID
        
    Returns:
        PedestrianAppearance object
    """
    # Extract color histogram from the first few good detections
    color_histograms = []
    heights = []
    widths = []
    
    for i, detection in enumerate(detections[:10]):  # Use first 10 frames
        if i < len(frames):
            x, y, w, h = detection.bbox
            roi = frames[i][y:y+h, x:x+w]
            
            if roi.size > 0:
                # Calculate color histogram (HSV is better for appearance)
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) if len(roi.shape) == 3 else roi
                hist = cv2.calcHist([roi_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                color_histograms.append(hist)
                
                heights.append(h)
                widths.append(w)
    
    # Average color histogram
    avg_histogram = np.mean(color_histograms, axis=0) if color_histograms else np.zeros(64)
    
    # Average aspect ratio
    avg_height = np.mean(heights) if heights else 0
    avg_width = np.mean(widths) if widths else 1
    height_width_ratio = avg_height / avg_width if avg_width > 0 else 1.5
    
    # Calculate average speed
    speeds = []
    for i in range(len(detections) - 1):
        x1, y1, w1, h1 = detections[i].bbox
        x2, y2, w2, h2 = detections[i + 1].bbox
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        dist = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
        speeds.append(dist)
    
    avg_speed = np.mean(speeds) if speeds else 0.0
    
    return PedestrianAppearance(
        color_histogram=avg_histogram,
        height_width_ratio=height_width_ratio,
        avg_speed=avg_speed,
        camera_id=camera_id,
        first_seen=datetime.now(),  # Should use actual timestamps
        last_seen=datetime.now(),
        track_id=track_id
    )
