"""High-confidence dataset generation for Mass Motion integration."""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import csv
from collections import defaultdict, deque
import math

from .transit_analyzer import TransitAnalyzer, TransitMovement, TransitEntrance
from .spatial_analyzer import SpatialAnalyzer, SpatialZone, SpatialTemporalData
from .pedestrian_tracker import TrackedPedestrian

logger = logging.getLogger(__name__)


@dataclass
class JourneyPoint:
    """Single point in a pedestrian journey."""
    timestamp: datetime
    position: Tuple[float, float]
    zone_id: str
    confidence: float  # 0.0 to 1.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    activity: str = "walking"  # walking, waiting, boarding, alighting


@dataclass
class PedestrianJourney:
    """Complete journey of a pedestrian through the station."""
    journey_id: str
    pedestrian_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    journey_type: str  # boarding, alighting, transfer, circulation
    origin_zone: str
    destination_zone: str
    service_line: str
    total_distance: float
    average_speed: float
    max_speed: float
    dwell_time: float
    points: List[JourneyPoint] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class TimetableEntry:
    """Timetable entry for service line."""
    service_line: str
    platform: str
    scheduled_time: datetime
    actual_time: datetime
    passenger_count: int
    journey_type: str
    delay_minutes: float = 0.0


@dataclass
class CoordinateTimeSeries:
    """Time series of pedestrian coordinates."""
    pedestrian_id: str
    timestamps: List[datetime]
    x_coordinates: List[float]
    y_coordinates: List[float]
    velocities: List[Tuple[float, float]]
    accelerations: List[Tuple[float, float]]
    zone_ids: List[str]
    confidence_scores: List[float]


class DatasetGenerator:
    """Generates high-confidence datasets for Mass Motion integration."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize dataset generator.
        
        Args:
            confidence_threshold: Minimum confidence threshold for data inclusion
        """
        self.confidence_threshold = confidence_threshold
        self.transit_analyzer = TransitAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        
        # Data storage
        self.journeys: List[PedestrianJourney] = []
        self.timetables: Dict[str, List[TimetableEntry]] = defaultdict(list)
        self.coordinate_series: List[CoordinateTimeSeries] = []
        
        # Tracking parameters
        self.min_journey_duration = 5.0  # seconds
        self.max_journey_duration = 1800.0  # 30 minutes
        self.smoothing_window = 3  # frames for path smoothing
        self.velocity_calculation_window = 5  # frames for velocity calculation
        
    def process_video_analysis(self, analysis_results: List[Dict], 
                             zones: Dict[str, SpatialZone],
                             entrances: Dict[str, TransitEntrance]) -> Dict[str, Any]:
        """
        Process video analysis results to generate datasets.
        
        Args:
            analysis_results: List of frame-by-frame analysis results
            zones: Dictionary of spatial zones
            entrances: Dictionary of transit entrances
            
        Returns:
            Dictionary with generated datasets
        """
        logger.info(f"Processing {len(analysis_results)} analysis results")
        
        # Set up analyzers
        for zone in zones.values():
            self.spatial_analyzer.add_zone(zone)
        
        for entrance in entrances.values():
            self.transit_analyzer.add_entrance(entrance)
        
        # Process each frame
        for frame_data in analysis_results:
            timestamp = datetime.fromisoformat(frame_data['timestamp'])
            
            # Create tracked pedestrians from frame data
            tracked_pedestrians = self._create_tracked_pedestrians_from_frame(frame_data)
            
            # Update analyzers
            self.spatial_analyzer.update_pedestrian_positions(tracked_pedestrians, timestamp)
            transit_movements = self.transit_analyzer.update_pedestrian_tracking(
                tracked_pedestrians, timestamp
            )
            
            # Process movements to create journeys
            for movement in transit_movements:
                self._process_movement_to_journey(movement, tracked_pedestrians, timestamp)
        
        # Generate datasets
        datasets = {
            'journeys': self._generate_journey_dataset(),
            'timetables': self._generate_timetable_dataset(),
            'coordinate_series': self._generate_coordinate_dataset(),
            'summary_statistics': self._generate_summary_statistics()
        }
        
        logger.info(f"Generated datasets: {len(datasets['journeys'])} journeys, "
                   f"{sum(len(t) for t in datasets['timetables'].values())} timetable entries, "
                   f"{len(datasets['coordinate_series'])} coordinate series")
        
        return datasets
    
    def _create_tracked_pedestrians_from_frame(self, frame_data: Dict) -> List[TrackedPedestrian]:
        """Create tracked pedestrians from frame analysis data."""
        pedestrians = []
        
        # This would need to be implemented based on your specific frame data format
        # For now, creating a placeholder structure
        for i, detection in enumerate(frame_data.get('detections', [])):
            pedestrian = TrackedPedestrian(
                id=f"ped_{frame_data['frame_number']}_{i}",
                centroid=(detection.get('x', 0), detection.get('y', 0)),
                bbox=detection.get('bbox', [0, 0, 10, 10]),
                confidence=detection.get('confidence', 0.8),
                first_seen=datetime.fromisoformat(frame_data['timestamp']),
                last_seen=datetime.fromisoformat(frame_data['timestamp'])
            )
            pedestrians.append(pedestrian)
        
        return pedestrians
    
    def _process_movement_to_journey(self, movement: TransitMovement, 
                                   tracked_pedestrians: List[TrackedPedestrian],
                                   timestamp: datetime):
        """Process a transit movement to create a journey."""
        # Find the pedestrian
        pedestrian = None
        for ped in tracked_pedestrians:
            if ped.id == movement.pedestrian_id:
                pedestrian = ped
                break
        
        if not pedestrian:
            return
        
        # Create journey points
        journey_points = []
        for i, (x, y) in enumerate(movement.path_points):
            # Calculate confidence based on detection quality
            confidence = self._calculate_point_confidence(pedestrian, i, len(movement.path_points))
            
            # Determine zone
            zone_id = self._get_zone_for_position(x, y)
            
            # Calculate velocity and acceleration
            velocity = self._calculate_velocity(movement.path_points, i)
            acceleration = self._calculate_acceleration(movement.path_points, i)
            
            point = JourneyPoint(
                timestamp=timestamp + timedelta(seconds=i * 0.1),  # Assume 10fps
                position=(x, y),
                zone_id=zone_id or "unknown",
                confidence=confidence,
                velocity=velocity,
                acceleration=acceleration,
                activity=self._determine_activity(movement.movement_type, i, len(movement.path_points))
            )
            journey_points.append(point)
        
        # Create journey
        journey = PedestrianJourney(
            journey_id=f"journey_{movement.pedestrian_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            pedestrian_id=movement.pedestrian_id,
            start_time=movement.timestamp,
            end_time=movement.timestamp + timedelta(seconds=movement.duration),
            duration=movement.duration,
            journey_type=movement.movement_type,
            origin_zone=movement.platform_origin or movement.entrance_id,
            destination_zone=movement.platform_destination or movement.exit_id,
            service_line=movement.service_line,
            total_distance=self._calculate_total_distance(movement.path_points),
            average_speed=self._calculate_average_speed(movement.path_points, movement.duration),
            max_speed=self._calculate_max_speed(movement.path_points),
            dwell_time=self._calculate_dwell_time(movement.path_points),
            points=journey_points,
            confidence_score=self._calculate_journey_confidence(journey_points)
        )
        
        # Only include high-confidence journeys
        if journey.confidence_score >= self.confidence_threshold:
            self.journeys.append(journey)
    
    def _calculate_point_confidence(self, pedestrian: TrackedPedestrian, 
                                  point_index: int, total_points: int) -> float:
        """Calculate confidence for a journey point."""
        base_confidence = pedestrian.confidence
        
        # Reduce confidence for points at the beginning/end of journey
        position_factor = 1.0 - abs(point_index - total_points/2) / (total_points/2) * 0.2
        
        # Reduce confidence for very short journeys
        length_factor = min(1.0, total_points / 10.0)
        
        return min(1.0, base_confidence * position_factor * length_factor)
    
    def _get_zone_for_position(self, x: float, y: float) -> Optional[str]:
        """Get zone ID for a position."""
        for zone_id, zone in self.spatial_analyzer.zones.items():
            if self._point_in_polygon(x, y, zone.polygon):
                return zone_id
        return None
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon."""
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
    
    def _calculate_velocity(self, path_points: List[Tuple[float, float]], 
                          point_index: int) -> Tuple[float, float]:
        """Calculate velocity at a point."""
        if len(path_points) < 2:
            return (0.0, 0.0)
        
        # Use points around current index for velocity calculation
        start_idx = max(0, point_index - self.velocity_calculation_window // 2)
        end_idx = min(len(path_points), point_index + self.velocity_calculation_window // 2 + 1)
        
        if end_idx - start_idx < 2:
            return (0.0, 0.0)
        
        # Calculate average velocity
        dx = path_points[end_idx-1][0] - path_points[start_idx][0]
        dy = path_points[end_idx-1][1] - path_points[start_idx][1]
        dt = (end_idx - start_idx) * 0.1  # Assume 10fps
        
        if dt > 0:
            return (dx / dt, dy / dt)
        else:
            return (0.0, 0.0)
    
    def _calculate_acceleration(self, path_points: List[Tuple[float, float]], 
                              point_index: int) -> Tuple[float, float]:
        """Calculate acceleration at a point."""
        if len(path_points) < 3:
            return (0.0, 0.0)
        
        # Calculate velocity before and after current point
        v_before = self._calculate_velocity(path_points, max(0, point_index - 1))
        v_after = self._calculate_velocity(path_points, min(len(path_points) - 1, point_index + 1))
        
        # Calculate acceleration
        dt = 0.2  # 0.1s * 2 frames
        if dt > 0:
            return ((v_after[0] - v_before[0]) / dt, (v_after[1] - v_before[1]) / dt)
        else:
            return (0.0, 0.0)
    
    def _determine_activity(self, movement_type: str, point_index: int, 
                          total_points: int) -> str:
        """Determine activity at a journey point."""
        if movement_type == "boarding":
            if point_index < total_points * 0.2:
                return "walking"
            elif point_index < total_points * 0.8:
                return "boarding"
            else:
                return "walking"
        elif movement_type == "alighting":
            if point_index < total_points * 0.2:
                return "alighting"
            elif point_index < total_points * 0.8:
                return "walking"
            else:
                return "walking"
        elif movement_type == "transfer":
            if point_index < total_points * 0.1:
                return "walking"
            elif point_index < total_points * 0.9:
                return "transfer"
            else:
                return "walking"
        else:
            return "walking"
    
    def _calculate_total_distance(self, path_points: List[Tuple[float, float]]) -> float:
        """Calculate total distance of journey."""
        if len(path_points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        return total_distance
    
    def _calculate_average_speed(self, path_points: List[Tuple[float, float]], 
                               duration: float) -> float:
        """Calculate average speed of journey."""
        total_distance = self._calculate_total_distance(path_points)
        return total_distance / duration if duration > 0 else 0.0
    
    def _calculate_max_speed(self, path_points: List[Tuple[float, float]]) -> float:
        """Calculate maximum speed of journey."""
        if len(path_points) < 2:
            return 0.0
        
        max_speed = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            speed = math.sqrt(dx*dx + dy*dy) / 0.1  # Assume 10fps
            max_speed = max(max_speed, speed)
        
        return max_speed
    
    def _calculate_dwell_time(self, path_points: List[Tuple[float, float]]) -> float:
        """Calculate dwell time (time spent in one location)."""
        if len(path_points) < 2:
            return 0.0
        
        # Find periods where pedestrian is stationary
        dwell_time = 0.0
        stationary_threshold = 5.0  # pixels
        
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < stationary_threshold:
                dwell_time += 0.1  # 0.1s per frame
        
        return dwell_time
    
    def _calculate_journey_confidence(self, journey_points: List[JourneyPoint]) -> float:
        """Calculate overall confidence for a journey."""
        if not journey_points:
            return 0.0
        
        # Average confidence of all points
        avg_confidence = np.mean([point.confidence for point in journey_points])
        
        # Penalize very short journeys
        length_factor = min(1.0, len(journey_points) / 20.0)
        
        # Penalize journeys with many low-confidence points
        low_confidence_points = sum(1 for point in journey_points if point.confidence < 0.5)
        quality_factor = 1.0 - (low_confidence_points / len(journey_points)) * 0.3
        
        return min(1.0, avg_confidence * length_factor * quality_factor)
    
    def _generate_journey_dataset(self) -> List[Dict[str, Any]]:
        """Generate journey dataset for Mass Motion."""
        journey_data = []
        
        for journey in self.journeys:
            journey_entry = {
                'journey_id': journey.journey_id,
                'pedestrian_id': journey.pedestrian_id,
                'start_time': journey.start_time.isoformat(),
                'end_time': journey.end_time.isoformat(),
                'duration': journey.duration,
                'journey_type': journey.journey_type,
                'origin_zone': journey.origin_zone,
                'destination_zone': journey.destination_zone,
                'service_line': journey.service_line,
                'total_distance': journey.total_distance,
                'average_speed': journey.average_speed,
                'max_speed': journey.max_speed,
                'dwell_time': journey.dwell_time,
                'confidence_score': journey.confidence_score,
                'path_points': [
                    {
                        'timestamp': point.timestamp.isoformat(),
                        'position': point.position,
                        'zone_id': point.zone_id,
                        'confidence': point.confidence,
                        'velocity': point.velocity,
                        'acceleration': point.acceleration,
                        'activity': point.activity
                    }
                    for point in journey.points
                ]
            }
            journey_data.append(journey_entry)
        
        return journey_data
    
    def _generate_timetable_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate timetable dataset from journey patterns."""
        timetable_data = defaultdict(list)
        
        # Group journeys by service line and time
        for journey in self.journeys:
            if not journey.service_line:
                continue
            
            # Round to nearest 5 minutes for timetable
            scheduled_time = journey.start_time.replace(
                minute=(journey.start_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            
            entry = TimetableEntry(
                service_line=journey.service_line,
                platform=journey.origin_zone,
                scheduled_time=scheduled_time,
                actual_time=journey.start_time,
                passenger_count=1,  # Each journey represents one passenger
                journey_type=journey.journey_type,
                delay_minutes=(journey.start_time - scheduled_time).total_seconds() / 60
            )
            
            timetable_data[journey.service_line].append(entry)
        
        # Convert to dictionary format
        result = {}
        for service_line, entries in timetable_data.items():
            result[service_line] = [
                {
                    'platform': entry.platform,
                    'scheduled_time': entry.scheduled_time.isoformat(),
                    'actual_time': entry.actual_time.isoformat(),
                    'passenger_count': entry.passenger_count,
                    'journey_type': entry.journey_type,
                    'delay_minutes': entry.delay_minutes
                }
                for entry in entries
            ]
        
        return result
    
    def _generate_coordinate_dataset(self) -> List[Dict[str, Any]]:
        """Generate coordinate time series dataset."""
        coordinate_data = []
        
        for journey in self.journeys:
            if len(journey.points) < 2:
                continue
            
            # Extract time series data
            timestamps = [point.timestamp for point in journey.points]
            x_coords = [point.position[0] for point in journey.points]
            y_coords = [point.position[1] for point in journey.points]
            velocities = [point.velocity for point in journey.points]
            accelerations = [point.acceleration for point in journey.points]
            zone_ids = [point.zone_id for point in journey.points]
            confidences = [point.confidence for point in journey.points]
            
            coordinate_entry = {
                'pedestrian_id': journey.pedestrian_id,
                'journey_id': journey.journey_id,
                'timestamps': [ts.isoformat() for ts in timestamps],
                'x_coordinates': x_coords,
                'y_coordinates': y_coords,
                'velocities': velocities,
                'accelerations': accelerations,
                'zone_ids': zone_ids,
                'confidence_scores': confidences,
                'journey_type': journey.journey_type,
                'service_line': journey.service_line
            }
            
            coordinate_data.append(coordinate_entry)
        
        return coordinate_data
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        if not self.journeys:
            return {}
        
        # Journey statistics
        journey_types = [journey.journey_type for journey in self.journeys]
        service_lines = [journey.service_line for journey in self.journeys if journey.service_line]
        
        # Duration statistics
        durations = [journey.duration for journey in self.journeys]
        distances = [journey.total_distance for journey in self.journeys]
        speeds = [journey.average_speed for journey in self.journeys]
        confidences = [journey.confidence_score for journey in self.journeys]
        
        # Time statistics
        start_times = [journey.start_time for journey in self.journeys]
        time_range = (min(start_times), max(start_times)) if start_times else (None, None)
        
        return {
            'total_journeys': len(self.journeys),
            'journey_types': {
                journey_type: journey_types.count(journey_type) 
                for journey_type in set(journey_types)
            },
            'service_lines': {
                service_line: service_lines.count(service_line)
                for service_line in set(service_lines)
            },
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            },
            'distance_stats': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            },
            'speed_stats': {
                'mean': np.mean(speeds),
                'std': np.std(speeds),
                'min': np.min(speeds),
                'max': np.max(speeds)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'time_range': {
                'start': time_range[0].isoformat() if time_range[0] else None,
                'end': time_range[1].isoformat() if time_range[1] else None
            },
            'high_confidence_journeys': sum(1 for c in confidences if c >= self.confidence_threshold),
            'data_quality_score': np.mean(confidences)
        }
    
    def export_datasets(self, output_dir: str, format: str = "json") -> bool:
        """
        Export all datasets to files.
        
        Args:
            output_dir: Output directory
            format: Export format ("json", "csv", "both")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate datasets
            datasets = {
                'journeys': self._generate_journey_dataset(),
                'timetables': self._generate_timetable_dataset(),
                'coordinate_series': self._generate_coordinate_dataset(),
                'summary_statistics': self._generate_summary_statistics()
            }
            
            if format in ["json", "both"]:
                # Export JSON files
                with open(output_path / "journeys.json", 'w') as f:
                    json.dump(datasets['journeys'], f, indent=2, default=str)
                
                with open(output_path / "timetables.json", 'w') as f:
                    json.dump(datasets['timetables'], f, indent=2, default=str)
                
                with open(output_path / "coordinate_series.json", 'w') as f:
                    json.dump(datasets['coordinate_series'], f, indent=2, default=str)
                
                with open(output_path / "summary_statistics.json", 'w') as f:
                    json.dump(datasets['summary_statistics'], f, indent=2, default=str)
            
            if format in ["csv", "both"]:
                # Export CSV files
                self._export_journeys_csv(output_path / "journeys.csv", datasets['journeys'])
                self._export_timetables_csv(output_path / "timetables.csv", datasets['timetables'])
                self._export_coordinate_series_csv(output_path / "coordinate_series.csv", datasets['coordinate_series'])
            
            # Export Mass Motion specific format
            self._export_mass_motion_format(output_path / "mass_motion_data.json", datasets)
            
            logger.info(f"Datasets exported to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting datasets: {e}")
            return False
    
    def _export_journeys_csv(self, file_path: Path, journeys: List[Dict]):
        """Export journeys to CSV format."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'journey_id', 'pedestrian_id', 'start_time', 'end_time', 'duration',
                'journey_type', 'origin_zone', 'destination_zone', 'service_line',
                'total_distance', 'average_speed', 'max_speed', 'dwell_time', 'confidence_score'
            ])
            
            for journey in journeys:
                writer.writerow([
                    journey['journey_id'],
                    journey['pedestrian_id'],
                    journey['start_time'],
                    journey['end_time'],
                    journey['duration'],
                    journey['journey_type'],
                    journey['origin_zone'],
                    journey['destination_zone'],
                    journey['service_line'],
                    journey['total_distance'],
                    journey['average_speed'],
                    journey['max_speed'],
                    journey['dwell_time'],
                    journey['confidence_score']
                ])
    
    def _export_timetables_csv(self, file_path: Path, timetables: Dict[str, List[Dict]]):
        """Export timetables to CSV format."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'service_line', 'platform', 'scheduled_time', 'actual_time',
                'passenger_count', 'journey_type', 'delay_minutes'
            ])
            
            for service_line, entries in timetables.items():
                for entry in entries:
                    writer.writerow([
                        service_line,
                        entry['platform'],
                        entry['scheduled_time'],
                        entry['actual_time'],
                        entry['passenger_count'],
                        entry['journey_type'],
                        entry['delay_minutes']
                    ])
    
    def _export_coordinate_series_csv(self, file_path: Path, coordinate_series: List[Dict]):
        """Export coordinate series to CSV format."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pedestrian_id', 'journey_id', 'timestamp', 'x_coordinate', 'y_coordinate',
                'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
                'zone_id', 'confidence_score', 'journey_type', 'service_line'
            ])
            
            for series in coordinate_series:
                for i in range(len(series['timestamps'])):
                    writer.writerow([
                        series['pedestrian_id'],
                        series['journey_id'],
                        series['timestamps'][i],
                        series['x_coordinates'][i],
                        series['y_coordinates'][i],
                        series['velocities'][i][0],
                        series['velocities'][i][1],
                        series['accelerations'][i][0],
                        series['accelerations'][i][1],
                        series['zone_ids'][i],
                        series['confidence_scores'][i],
                        series['journey_type'],
                        series['service_line']
                    ])
    
    def _export_mass_motion_format(self, file_path: Path, datasets: Dict[str, Any]):
        """Export data in Mass Motion specific format."""
        mass_motion_data = {
            'metadata': {
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_journeys': len(datasets['journeys']),
                'total_coordinate_series': len(datasets['coordinate_series']),
                'confidence_threshold': self.confidence_threshold,
                'data_quality_score': datasets['summary_statistics'].get('data_quality_score', 0.0)
            },
            'journeys': datasets['journeys'],
            'timetables': datasets['timetables'],
            'coordinate_series': datasets['coordinate_series'],
            'summary_statistics': datasets['summary_statistics']
        }
        
        with open(file_path, 'w') as f:
            json.dump(mass_motion_data, f, indent=2, default=str)