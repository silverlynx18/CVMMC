"""
Trajectory-based analysis for pedestrian movement direction detection.
Supports line-crossing counting and bidirectional flow analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .detection_utils import Detection

logger = logging.getLogger(__name__)


class DirectionType(Enum):
    """Movement direction types."""
    FORWARD = "forward"      # Left to right, bottom to top
    BACKWARD = "backward"    # Right to left, top to bottom
    LATERAL = "lateral"      # Side to side movement
    STATIONARY = "stationary"  # Not moving much
    UNKNOWN = "unknown"      # Cannot determine


@dataclass
class CountingLine:
    """
    A line for counting pedestrian crossings.
    Can be used for bidirectional counting.
    
    Based on Arup's MassMotion standards and transportation engineering principles.
    """
    name: str
    start_point: Tuple[int, int]  # (x, y)
    end_point: Tuple[int, int]    # (x, y)
    direction_names: Tuple[str, str] = ("forward", "backward")  # Names for each direction
    width_meters: Optional[float] = None  # Effective width in meters (for flow rate calculations)
    
    def get_line_vector(self) -> np.ndarray:
        """Get the normalized line direction vector."""
        return np.array([
            self.end_point[0] - self.start_point[0],
            self.end_point[1] - self.start_point[1]
        ])
    
    def get_normal_vector(self) -> np.ndarray:
        """Get the normal vector to the line (perpendicular)."""
        line_vec = self.get_line_vector()
        # Rotate 90 degrees counterclockwise
        return np.array([-line_vec[1], line_vec[0]])
    
    def get_length_pixels(self) -> float:
        """Get the length of the line in pixels."""
        vec = self.get_line_vector()
        return np.linalg.norm(vec)
    
    def get_length_meters(self, pixels_per_meter: Optional[float] = None) -> Optional[float]:
        """
        Get the length of the line in meters.
        
        Args:
            pixels_per_meter: Conversion factor (if width_meters not set)
            
        Returns:
            Length in meters or None if calibration unavailable
        """
        if self.width_meters is not None:
            return self.width_meters
        
        if pixels_per_meter is not None:
            return self.get_length_pixels() / pixels_per_meter
        
        return None


@dataclass
class TrajectoryPoint:
    """A single point in a pedestrian's trajectory."""
    position: Tuple[float, float]  # (x, y) center point
    timestamp: float  # Seconds since start
    frame_number: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class LineCrossingEvent:
    """Record of a pedestrian crossing a counting line."""
    track_id: int
    line_name: str
    direction: str  # "forward" or "backward"
    timestamp: float
    frame_number: int
    crossing_point: Tuple[float, float]
    confidence: float  # 0-1, how confident we are about this crossing


class TrajectoryAnalyzer:
    """
    Analyzes pedestrian trajectories to determine movement direction.
    Supports line-crossing counting and bidirectional flow analysis.
    """
    
    def __init__(self, 
                 min_trajectory_length: int = 5,
                 smoothing_window: int = 3,
                 stationary_threshold: float = 10.0):
        """
        Initialize trajectory analyzer.
        
        Args:
            min_trajectory_length: Minimum points needed to determine direction
            smoothing_window: Window size for trajectory smoothing
            stationary_threshold: Max movement (pixels) to be considered stationary
        """
        self.min_trajectory_length = min_trajectory_length
        self.smoothing_window = smoothing_window
        self.stationary_threshold = stationary_threshold
        
        # Counting lines
        self.counting_lines: Dict[str, CountingLine] = {}
        
        # Track which pedestrians have crossed which lines (to avoid double counting)
        self.line_crossings: Dict[int, Set[str]] = {}  # track_id -> set of line names crossed
        
        # History of crossing events
        self.crossing_events: List[LineCrossingEvent] = []
        
    def add_counting_line(self, line: CountingLine):
        """Add a counting line."""
        self.counting_lines[line.name] = line
        logger.info(f"Added counting line: {line.name}")
    
    def extract_trajectory(self, detections: List[Detection], 
                          frame_numbers: Optional[List[int]] = None,
                          fps: float = 30.0) -> List[TrajectoryPoint]:
        """
        Extract trajectory points from detections.
        
        Args:
            detections: List of Detection objects
            frame_numbers: Optional frame numbers for each detection
            fps: Frames per second (for timestamp calculation)
            
        Returns:
            List of TrajectoryPoint objects
        """
        if not detections:
            return []
        
        trajectory = []
        for i, detection in enumerate(detections):
            # bbox is (x1, y1, x2, y2) format
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            frame_num = frame_numbers[i] if frame_numbers else i
            timestamp = frame_num / fps
            
            trajectory.append(TrajectoryPoint(
                position=(center_x, center_y),
                timestamp=timestamp,
                frame_number=frame_num,
                bbox=detection.bbox
            ))
        
        return trajectory
    
    def smooth_trajectory(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        Apply moving average smoothing to trajectory.
        
        Args:
            trajectory: Raw trajectory points
            
        Returns:
            Smoothed trajectory points
        """
        if len(trajectory) < self.smoothing_window:
            return trajectory
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)
            
            window_points = trajectory[start_idx:end_idx]
            avg_x = np.mean([p.position[0] for p in window_points])
            avg_y = np.mean([p.position[1] for p in window_points])
            
            smoothed.append(TrajectoryPoint(
                position=(avg_x, avg_y),
                timestamp=trajectory[i].timestamp,
                frame_number=trajectory[i].frame_number,
                bbox=trajectory[i].bbox
            ))
        
        return smoothed
    
    def calculate_direction_vector(self, trajectory: List[TrajectoryPoint]) -> Optional[np.ndarray]:
        """
        Calculate the overall direction vector of a trajectory.
        Uses first and last points for simplicity, but could use more sophisticated methods.
        
        Args:
            trajectory: List of trajectory points
            
        Returns:
            Normalized direction vector [dx, dy] or None
        """
        if len(trajectory) < self.min_trajectory_length:
            return None
        
        # Use linear regression for more robust direction estimation
        positions = np.array([p.position for p in trajectory])
        
        # Calculate displacement vector
        start_pos = positions[0]
        end_pos = positions[-1]
        displacement = end_pos - start_pos
        
        # Check if mostly stationary
        distance = np.linalg.norm(displacement)
        if distance < self.stationary_threshold:
            return None
        
        # Normalize
        direction = displacement / distance
        return direction
    
    def classify_direction(self, trajectory: List[TrajectoryPoint]) -> DirectionType:
        """
        Classify the direction of movement.
        
        Args:
            trajectory: List of trajectory points
            
        Returns:
            DirectionType enum value
        """
        direction_vec = self.calculate_direction_vector(trajectory)
        
        if direction_vec is None:
            # Check if stationary
            if len(trajectory) >= self.min_trajectory_length:
                positions = np.array([p.position for p in trajectory])
                displacement = np.linalg.norm(positions[-1] - positions[0])
                if displacement < self.stationary_threshold:
                    return DirectionType.STATIONARY
            return DirectionType.UNKNOWN
        
        dx, dy = direction_vec
        
        # Determine primary direction based on angle
        angle = np.arctan2(dy, dx)  # -π to π
        angle_deg = np.degrees(angle)
        
        # Classify based on angle quadrants
        # Could be more sophisticated based on your needs
        if abs(dx) > abs(dy) * 1.5:  # Mostly horizontal
            if dx > 0:
                return DirectionType.FORWARD  # Moving right
            else:
                return DirectionType.BACKWARD  # Moving left
        elif abs(dy) > abs(dx) * 1.5:  # Mostly vertical
            if dy > 0:
                return DirectionType.FORWARD  # Moving down
            else:
                return DirectionType.BACKWARD  # Moving up
        else:
            return DirectionType.LATERAL  # Diagonal/lateral movement
    
    def check_line_crossing(self, 
                           track_id: int,
                           trajectory: List[TrajectoryPoint]) -> List[LineCrossingEvent]:
        """
        Check if trajectory crosses any counting lines.
        
        Args:
            track_id: Track ID of the pedestrian
            trajectory: List of trajectory points
            
        Returns:
            List of LineCrossingEvent objects
        """
        if len(trajectory) < 2:
            return []
        
        new_events = []
        
        # Initialize crossing tracking for this track
        if track_id not in self.line_crossings:
            self.line_crossings[track_id] = set()
        
        # Check each counting line
        for line_name, line in self.counting_lines.items():
            # Skip if already crossed this line
            if line_name in self.line_crossings[track_id]:
                continue
            
            # Check for crossing
            crossing = self._detect_line_crossing(trajectory, line)
            
            if crossing:
                # Record the crossing
                self.line_crossings[track_id].add(line_name)
                
                event = LineCrossingEvent(
                    track_id=track_id,
                    line_name=line_name,
                    direction=crossing['direction'],
                    timestamp=crossing['timestamp'],
                    frame_number=crossing['frame_number'],
                    crossing_point=crossing['point'],
                    confidence=crossing['confidence']
                )
                
                new_events.append(event)
                self.crossing_events.append(event)
                
                logger.debug(f"Track {track_id} crossed line '{line_name}' going {crossing['direction']}")
        
        return new_events
    
    def _detect_line_crossing(self, 
                             trajectory: List[TrajectoryPoint],
                             line: CountingLine) -> Optional[Dict]:
        """
        Detect if and how a trajectory crosses a counting line.
        
        Args:
            trajectory: List of trajectory points
            line: CountingLine to check
            
        Returns:
            Dictionary with crossing info or None
        """
        line_start = np.array(line.start_point, dtype=float)
        line_end = np.array(line.end_point, dtype=float)
        
        # Get line normal vector for determining which side
        normal = line.get_normal_vector()
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Check consecutive points for crossing
        for i in range(len(trajectory) - 1):
            p1 = np.array(trajectory[i].position)
            p2 = np.array(trajectory[i + 1].position)
            
            # Calculate which side of the line each point is on
            side1 = np.dot(p1 - line_start, normal)
            side2 = np.dot(p2 - line_start, normal)
            
            # Check if crossing occurred (sign change)
            if side1 * side2 < 0:  # Different signs = crossing
                # Verify the crossing point is actually on the line segment
                crossing_point = self._line_segment_intersection(
                    p1, p2, line_start, line_end
                )
                
                if crossing_point is not None:
                    # Determine direction based on which side they came from
                    if side1 < 0:  # Crossing from negative to positive side
                        direction = line.direction_names[0]  # "forward"
                    else:  # Crossing from positive to negative side
                        direction = line.direction_names[1]  # "backward"
                    
                    # Calculate confidence based on trajectory clarity
                    confidence = self._calculate_crossing_confidence(trajectory, i)
                    
                    return {
                        'direction': direction,
                        'point': crossing_point,
                        'timestamp': trajectory[i + 1].timestamp,
                        'frame_number': trajectory[i + 1].frame_number,
                        'confidence': confidence
                    }
        
        return None
    
    def _line_segment_intersection(self,
                                   p1: np.ndarray, p2: np.ndarray,
                                   q1: np.ndarray, q2: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Find intersection point between two line segments.
        
        Args:
            p1, p2: First line segment endpoints
            q1, q2: Second line segment endpoints
            
        Returns:
            Intersection point (x, y) or None
        """
        # Direction vectors
        d1 = p2 - p1
        d2 = q2 - q1
        
        # Solve: p1 + t * d1 = q1 + s * d2
        # Using Cramer's rule
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(denominator) < 1e-10:  # Parallel lines
            return None
        
        t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / denominator
        s = ((q1[0] - p1[0]) * d1[1] - (q1[1] - p1[1]) * d1[0]) / denominator
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= s <= 1:
            intersection = p1 + t * d1
            return (float(intersection[0]), float(intersection[1]))
        
        return None
    
    def _calculate_crossing_confidence(self, 
                                      trajectory: List[TrajectoryPoint],
                                      crossing_index: int) -> float:
        """
        Calculate confidence score for a line crossing event.
        
        Args:
            trajectory: Full trajectory
            crossing_index: Index where crossing occurred
            
        Returns:
            Confidence score 0-1
        """
        confidence = 1.0
        
        # Factor 1: Trajectory length (longer = more confident)
        if len(trajectory) < self.min_trajectory_length:
            confidence *= 0.5
        
        # Factor 2: Crossing location in trajectory (middle is better)
        rel_position = crossing_index / len(trajectory)
        if rel_position < 0.2 or rel_position > 0.8:
            confidence *= 0.8  # Crossing near start/end is less reliable
        
        # Factor 3: Speed consistency (check if moving consistently)
        if crossing_index > 2 and crossing_index < len(trajectory) - 2:
            recent_positions = [p.position for p in trajectory[crossing_index-2:crossing_index+3]]
            speeds = []
            for i in range(len(recent_positions) - 1):
                dist = np.linalg.norm(
                    np.array(recent_positions[i+1]) - np.array(recent_positions[i])
                )
                speeds.append(dist)
            
            if speeds:
                speed_variance = np.std(speeds) / (np.mean(speeds) + 1e-6)
                if speed_variance > 1.0:  # High variance = erratic movement
                    confidence *= 0.7
        
        return max(0.1, min(1.0, confidence))
    
    def get_crossing_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get counts of line crossings by line and direction.
        
        Returns:
            Dictionary: {line_name: {direction: count}}
        """
        counts = {}
        
        for line_name in self.counting_lines.keys():
            line = self.counting_lines[line_name]
            counts[line_name] = {
                line.direction_names[0]: 0,
                line.direction_names[1]: 0
            }
        
        for event in self.crossing_events:
            if event.line_name in counts:
                counts[event.line_name][event.direction] += 1
        
        return counts
    
    def get_net_flow(self, line_name: str) -> int:
        """
        Get net flow (forward - backward) for a specific line.
        
        Args:
            line_name: Name of the counting line
            
        Returns:
            Net flow count (positive = more forward, negative = more backward)
        """
        if line_name not in self.counting_lines:
            return 0
        
        counts = self.get_crossing_counts()
        if line_name not in counts:
            return 0
        
        line = self.counting_lines[line_name]
        forward_count = counts[line_name].get(line.direction_names[0], 0)
        backward_count = counts[line_name].get(line.direction_names[1], 0)
        
        return forward_count - backward_count
    
    def calculate_flow_rate(self, line_name: str, 
                           time_window_seconds: float = 60.0,
                           pixels_per_meter: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive flow rate metrics across a counting line.
        
        Based on Arup's MassMotion standards and transportation engineering principles:
        - Flow rate Q (pedestrians/second)
        - Specific flow rate q = Q/W (pedestrians per meter per second)
        - Peak flow rates for standard time windows
        
        Args:
            line_name: Name of the counting line
            time_window_seconds: Time window for calculating flow rate
            pixels_per_meter: Optional calibration for pixel-to-meter conversion
            
        Returns:
            Dictionary with comprehensive flow rate metrics:
            - forward_rate: Flow rate in forward direction (peds/s)
            - backward_rate: Flow rate in backward direction (peds/s)
            - net_rate: Net flow rate (peds/s)
            - forward_specific_rate: Specific flow rate forward (peds/m/s)
            - backward_specific_rate: Specific flow rate backward (peds/m/s)
            - total_crossings: Total number of crossings
            - width_meters: Effective width of the line
        """
        from datetime import datetime
        
        if line_name not in self.counting_lines:
            return {
                'forward_rate': 0.0, 'backward_rate': 0.0, 'net_rate': 0.0,
                'forward_specific_rate': 0.0, 'backward_specific_rate': 0.0,
                'total_crossings': 0, 'width_meters': None
            }
        
        line = self.counting_lines[line_name]
        
        # Get current time or use last event time
        if self.crossing_events:
            line_events = [e for e in self.crossing_events if e.line_name == line_name]
            if line_events:
                now = max(e.timestamp for e in line_events)
            else:
                now = datetime.now().timestamp()
        else:
            now = datetime.now().timestamp()
        
        cutoff_time = now - time_window_seconds
        
        # Filter crossings in time window
        recent_crossings = [
            event for event in self.crossing_events
            if event.line_name == line_name and event.timestamp >= cutoff_time
        ]
        
        if not recent_crossings:
            width_m = line.get_length_meters(pixels_per_meter)
            return {
                'forward_rate': 0.0, 'backward_rate': 0.0, 'net_rate': 0.0,
                'forward_specific_rate': 0.0, 'backward_specific_rate': 0.0,
                'total_crossings': 0, 'width_meters': width_m
            }
        
        # Group by direction
        forward_crossings = [e for e in recent_crossings if e.direction == line.direction_names[0]]
        backward_crossings = [e for e in recent_crossings if e.direction == line.direction_names[1]]
        
        # Calculate flow rates (pedestrians per second)
        forward_rate = len(forward_crossings) / time_window_seconds
        backward_rate = len(backward_crossings) / time_window_seconds
        net_rate = forward_rate - backward_rate
        
        # Get effective width in meters
        width_m = line.width_meters or line.get_length_meters(pixels_per_meter)
        
        # Calculate specific flow rates (pedestrians per meter per second)
        # This is the standard metric used in transportation engineering
        forward_specific_rate = forward_rate / width_m if width_m and width_m > 0 else None
        backward_specific_rate = backward_rate / width_m if width_m and width_m > 0 else None
        
        result = {
            'forward_rate': forward_rate,  # peds/s
            'backward_rate': backward_rate,  # peds/s
            'net_rate': net_rate,  # peds/s
            'total_crossings': len(recent_crossings),
            'width_meters': width_m
        }
        
        if forward_specific_rate is not None:
            result['forward_specific_rate'] = forward_specific_rate  # peds/m/s
        if backward_specific_rate is not None:
            result['backward_specific_rate'] = backward_specific_rate  # peds/m/s
        
        return result
    
    def calculate_peak_flow_rates(self, line_name: str,
                                 window_minutes: int = 15,
                                 pixels_per_meter: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate peak flow rates using standard time windows (e.g., 15-minute peak).
        
        Based on Arup's standards and transportation engineering practice.
        
        Args:
            line_name: Name of the counting line
            window_minutes: Time window in minutes (default: 15 for peak hour analysis)
            pixels_per_meter: Optional calibration for pixel-to-meter conversion
            
        Returns:
            Dictionary with peak flow rate metrics:
            - peak_forward_rate: Peak forward flow rate (peds/s)
            - peak_backward_rate: Peak backward flow rate (peds/s)
            - peak_specific_forward_rate: Peak specific forward flow rate (peds/m/s)
            - peak_specific_backward_rate: Peak specific backward flow rate (peds/m/s)
            - peak_window_start: Start time of peak window
            - peak_window_end: End time of peak window
        """
        if line_name not in self.counting_lines or not self.crossing_events:
            return {
                'peak_forward_rate': 0.0, 'peak_backward_rate': 0.0,
                'peak_specific_forward_rate': None, 'peak_specific_backward_rate': None,
                'peak_window_start': None, 'peak_window_end': None
            }
        
        line = self.counting_lines[line_name]
        window_seconds = window_minutes * 60.0
        
        # Get all crossings for this line
        line_crossings = [e for e in self.crossing_events if e.line_name == line_name]
        if not line_crossings:
            return {
                'peak_forward_rate': 0.0, 'peak_backward_rate': 0.0,
                'peak_specific_forward_rate': None, 'peak_specific_backward_rate': None,
                'peak_window_start': None, 'peak_window_end': None
            }
        
        # Sort by timestamp
        line_crossings.sort(key=lambda x: x.timestamp)
        
        # Slide window across all crossings to find peak
        max_forward_rate = 0.0
        max_backward_rate = 0.0
        peak_start = None
        peak_end = None
        
        start_idx = 0
        for end_idx in range(len(line_crossings)):
            # Adjust start_idx to maintain window size
            while (line_crossings[end_idx].timestamp - line_crossings[start_idx].timestamp) > window_seconds:
                start_idx += 1
            
            # Calculate flow rate for this window
            window_crossings = line_crossings[start_idx:end_idx + 1]
            time_span = line_crossings[end_idx].timestamp - line_crossings[start_idx].timestamp
            
            if time_span > 0:
                forward_count = sum(1 for e in window_crossings if e.direction == line.direction_names[0])
                backward_count = sum(1 for e in window_crossings if e.direction == line.direction_names[1])
                
                forward_rate = forward_count / time_span
                backward_rate = backward_count / time_span
                
                if forward_rate > max_forward_rate:
                    max_forward_rate = forward_rate
                    peak_start = line_crossings[start_idx].timestamp
                    peak_end = line_crossings[end_idx].timestamp
                
                if backward_rate > max_backward_rate:
                    max_backward_rate = backward_rate
        
        # Calculate specific flow rates
        width_m = line.width_meters or line.get_length_meters(pixels_per_meter)
        peak_specific_forward = max_forward_rate / width_m if width_m and width_m > 0 else None
        peak_specific_backward = max_backward_rate / width_m if width_m and width_m > 0 else None
        
        result = {
            'peak_forward_rate': max_forward_rate,  # peds/s
            'peak_backward_rate': max_backward_rate,  # peds/s
            'peak_window_start': peak_start,
            'peak_window_end': peak_end
        }
        
        if peak_specific_forward is not None:
            result['peak_specific_forward_rate'] = peak_specific_forward  # peds/m/s
        if peak_specific_backward is not None:
            result['peak_specific_backward_rate'] = peak_specific_backward  # peds/m/s
        
        return result
    
    def calculate_hourly_flow_rates(self, line_name: str,
                                    pixels_per_meter: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate hourly flow rates for the entire observation period.
        
        Useful for peak hour analysis and design capacity planning.
        
        Args:
            line_name: Name of the counting line
            pixels_per_meter: Optional calibration for pixel-to-meter conversion
            
        Returns:
            Dictionary with hourly flow rates indexed by hour:
            {
                'hour_0': {forward_rate, backward_rate, ...},
                'hour_1': {...},
                ...
            }
        """
        if line_name not in self.counting_lines or not self.crossing_events:
            return {}
        
        from datetime import datetime
        
        line = self.counting_lines[line_name]
        line_crossings = [e for e in self.crossing_events if e.line_name == line_name]
        
        if not line_crossings:
            return {}
        
        # Group crossings by hour
        hourly_crossings = {}
        for event in line_crossings:
            dt = datetime.fromtimestamp(event.timestamp)
            hour_key = f"hour_{dt.hour}"
            
            if hour_key not in hourly_crossings:
                hourly_crossings[hour_key] = {'forward': [], 'backward': []}
            
            if event.direction == line.direction_names[0]:
                hourly_crossings[hour_key]['forward'].append(event)
            else:
                hourly_crossings[hour_key]['backward'].append(event)
        
        # Calculate flow rates for each hour
        hourly_rates = {}
        width_m = line.width_meters or line.get_length_meters(pixels_per_meter)
        
        for hour_key, crossings in hourly_crossings.items():
            forward_rate = len(crossings['forward']) / 3600.0  # per second
            backward_rate = len(crossings['backward']) / 3600.0  # per second
            
            result = {
                'forward_rate': forward_rate,  # peds/s
                'backward_rate': backward_rate,  # peds/s
                'net_rate': forward_rate - backward_rate,  # peds/s
                'forward_total': len(crossings['forward']),  # peds/hour
                'backward_total': len(crossings['backward']),  # peds/hour
                'total_crossings': len(crossings['forward']) + len(crossings['backward'])
            }
            
            if width_m and width_m > 0:
                result['forward_specific_rate'] = forward_rate / width_m  # peds/m/s
                result['backward_specific_rate'] = backward_rate / width_m  # peds/m/s
            
            hourly_rates[hour_key] = result
        
        return hourly_rates
    
    def reset_crossings(self, track_id: Optional[int] = None):
        """
        Reset crossing records.
        
        Args:
            track_id: If provided, only reset for this track. Otherwise reset all.
        """
        if track_id is not None:
            if track_id in self.line_crossings:
                del self.line_crossings[track_id]
        else:
            self.line_crossings.clear()
            self.crossing_events.clear()
