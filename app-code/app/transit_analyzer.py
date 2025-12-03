"""Transit station specific analysis with entrance/exit tracking and Mass Motion integration."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging

from .pedestrian_tracker import PedestrianTracker, TrackedPedestrian
from .fruin_analysis import FruinAnalyzer, FruinLOSResult

logger = logging.getLogger(__name__)


@dataclass
class TransitEntrance:
    """Transit station entrance/exit definition."""
    entrance_id: str
    name: str
    entrance_type: str  # 'entrance', 'exit', 'bidirectional'
    polygon: List[List[float]]  # Zone polygon coordinates
    platform_connections: List[str] = field(default_factory=list)  # Connected platforms
    service_lines: List[str] = field(default_factory=list)  # Served transit lines
    capacity: int = 0  # Maximum capacity
    is_accessible: bool = True  # Wheelchair accessible
    notes: str = ""


@dataclass
class TransitMovement:
    """Individual pedestrian movement through transit station."""
    pedestrian_id: str
    entrance_id: str
    exit_id: str
    movement_type: str  # 'boarding', 'alighting', 'transfer', 'circulation'
    platform_origin: str = ""
    platform_destination: str = ""
    service_line: str = ""
    timestamp: datetime = None
    duration: float = 0.0  # Time spent in station
    path_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TransitMetrics:
    """Transit station specific metrics."""
    total_boardings: int = 0
    total_alightings: int = 0
    total_transfers: int = 0
    total_circulation: int = 0
    peak_boarding_hour: str = ""
    peak_alighting_hour: str = ""
    transfer_rate: float = 0.0
    average_dwell_time: float = 0.0
    platform_utilization: Dict[str, float] = field(default_factory=dict)
    entrance_utilization: Dict[str, float] = field(default_factory=dict)
    service_line_usage: Dict[str, int] = field(default_factory=dict)


class TransitAnalyzer:
    """Analyzes pedestrian flow in transit stations with entrance/exit tracking."""
    
    def __init__(self, zone_area_sqm: float = 25.0):
        """
        Initialize transit analyzer.
        
        Args:
            zone_area_sqm: Area of analysis zone in square meters
        """
        self.zone_area_sqm = zone_area_sqm
        self.fruin_analyzer = FruinAnalyzer()
        self.entrances: Dict[str, TransitEntrance] = {}
        self.movements: List[TransitMovement] = []
        self.current_pedestrians: Dict[str, TrackedPedestrian] = {}
        self.movement_history: List[TransitMovement] = []
        
        # Transit-specific parameters
        self.transfer_threshold_seconds = 300  # 5 minutes for transfer
        self.circulation_threshold_seconds = 600  # 10 minutes for circulation
        self.platform_proximity_threshold = 50  # pixels
        
    def add_entrance(self, entrance: TransitEntrance) -> bool:
        """
        Add a transit entrance/exit.
        
        Args:
            entrance: TransitEntrance object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.entrances[entrance.entrance_id] = entrance
            logger.info(f"Added entrance {entrance.entrance_id}: {entrance.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding entrance {entrance.entrance_id}: {e}")
            return False
    
    def remove_entrance(self, entrance_id: str) -> bool:
        """Remove a transit entrance/exit."""
        if entrance_id in self.entrances:
            del self.entrances[entrance_id]
            logger.info(f"Removed entrance {entrance_id}")
            return True
        return False
    
    def get_entrance_by_point(self, x: float, y: float) -> Optional[TransitEntrance]:
        """
        Find entrance containing a point.
        
        Args:
            x, y: Point coordinates
            
        Returns:
            TransitEntrance if point is inside an entrance, None otherwise
        """
        for entrance in self.entrances.values():
            if self._point_in_polygon(x, y, entrance.polygon):
                return entrance
        return None
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
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
    
    def update_pedestrian_tracking(self, tracked_pedestrians: List[TrackedPedestrian], 
                                 timestamp: datetime) -> List[TransitMovement]:
        """
        Update pedestrian tracking and detect transit movements.
        
        Args:
            tracked_pedestrians: List of currently tracked pedestrians
            timestamp: Current timestamp
            
        Returns:
            List of new movements detected
        """
        new_movements = []
        
        # Update current pedestrians
        current_pedestrian_ids = set()
        for ped in tracked_pedestrians:
            current_pedestrian_ids.add(ped.id)
            
            if ped.id in self.current_pedestrians:
                # Update existing pedestrian
                old_ped = self.current_pedestrians[ped.id]
                
                # Check for entrance/exit events
                old_entrance = self.get_entrance_by_point(old_ped.centroid[0], old_ped.centroid[1])
                new_entrance = self.get_entrance_by_point(ped.centroid[0], ped.centroid[1])
                
                if old_entrance != new_entrance:
                    # Movement detected
                    movement = self._create_movement(old_ped, ped, old_entrance, new_entrance, timestamp)
                    if movement:
                        new_movements.append(movement)
                        self.movements.append(movement)
                        self.movement_history.append(movement)
                
                # Update pedestrian
                self.current_pedestrians[ped.id] = ped
            else:
                # New pedestrian
                self.current_pedestrians[ped.id] = ped
        
        # Remove pedestrians no longer tracked
        removed_pedestrians = set(self.current_pedestrians.keys()) - current_pedestrian_ids
        for ped_id in removed_pedestrians:
            del self.current_pedestrians[ped_id]
        
        return new_movements
    
    def _create_movement(self, old_ped: TrackedPedestrian, new_ped: TrackedPedestrian,
                        old_entrance: Optional[TransitEntrance], 
                        new_entrance: Optional[TransitEntrance],
                        timestamp: datetime) -> Optional[TransitMovement]:
        """Create a movement record from pedestrian state change."""
        if not old_entrance and not new_entrance:
            return None
        
        # Determine movement type
        movement_type = self._determine_movement_type(old_entrance, new_entrance, old_ped, new_ped)
        
        # Calculate duration
        duration = (timestamp - old_ped.first_seen).total_seconds()
        
        # Create movement
        movement = TransitMovement(
            pedestrian_id=new_ped.id,
            entrance_id=old_entrance.entrance_id if old_entrance else "",
            exit_id=new_entrance.entrance_id if new_entrance else "",
            movement_type=movement_type,
            platform_origin=old_entrance.platform_connections[0] if old_entrance and old_entrance.platform_connections else "",
            platform_destination=new_entrance.platform_connections[0] if new_entrance and new_entrance.platform_connections else "",
            service_line=old_entrance.service_lines[0] if old_entrance and old_entrance.service_lines else "",
            timestamp=timestamp,
            duration=duration,
            path_points=[(old_ped.centroid[0], old_ped.centroid[1]), (new_ped.centroid[0], new_ped.centroid[1])]
        )
        
        return movement
    
    def _determine_movement_type(self, old_entrance: Optional[TransitEntrance],
                               new_entrance: Optional[TransitEntrance],
                               old_ped: TrackedPedestrian,
                               new_ped: TrackedPedestrian) -> str:
        """Determine the type of movement based on entrance/exit changes."""
        if not old_entrance and new_entrance:
            # Entering station
            if new_entrance.entrance_type in ['entrance', 'bidirectional']:
                return 'boarding'
            else:
                return 'circulation'
        
        elif old_entrance and not new_entrance:
            # Exiting station
            if old_entrance.entrance_type in ['exit', 'bidirectional']:
                return 'alighting'
            else:
                return 'circulation'
        
        elif old_entrance and new_entrance and old_entrance != new_entrance:
            # Moving between entrances
            if (old_entrance.platform_connections and new_entrance.platform_connections and
                old_entrance.platform_connections != new_entrance.platform_connections):
                return 'transfer'
            else:
                return 'circulation'
        
        else:
            return 'circulation'
    
    def calculate_transit_metrics(self, time_window: Optional[timedelta] = None) -> TransitMetrics:
        """
        Calculate transit-specific metrics.
        
        Args:
            time_window: Time window for analysis (None for all time)
            
        Returns:
            TransitMetrics object
        """
        # Filter movements by time window
        movements = self.movements
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            movements = [m for m in movements if m.timestamp >= cutoff_time]
        
        metrics = TransitMetrics()
        
        # Count movements by type
        for movement in movements:
            if movement.movement_type == 'boarding':
                metrics.total_boardings += 1
            elif movement.movement_type == 'alighting':
                metrics.total_alightings += 1
            elif movement.movement_type == 'transfer':
                metrics.total_transfers += 1
            elif movement.movement_type == 'circulation':
                metrics.total_circulation += 1
        
        # Calculate transfer rate
        total_movements = len(movements)
        if total_movements > 0:
            metrics.transfer_rate = metrics.total_transfers / total_movements
        
        # Calculate average dwell time
        dwell_times = [m.duration for m in movements if m.duration > 0]
        if dwell_times:
            metrics.average_dwell_time = np.mean(dwell_times)
        
        # Calculate peak hours
        if movements:
            hourly_counts = {}
            for movement in movements:
                hour = movement.timestamp.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            if hourly_counts:
                metrics.peak_boarding_hour = str(max(hourly_counts.keys(), key=lambda h: hourly_counts[h]))
        
        # Calculate platform utilization
        platform_counts = {}
        for movement in movements:
            if movement.platform_origin:
                platform_counts[movement.platform_origin] = platform_counts.get(movement.platform_origin, 0) + 1
            if movement.platform_destination:
                platform_counts[movement.platform_destination] = platform_counts.get(movement.platform_destination, 0) + 1
        
        total_platform_usage = sum(platform_counts.values())
        if total_platform_usage > 0:
            for platform, count in platform_counts.items():
                metrics.platform_utilization[platform] = count / total_platform_usage
        
        # Calculate entrance utilization
        entrance_counts = {}
        for movement in movements:
            if movement.entrance_id:
                entrance_counts[movement.entrance_id] = entrance_counts.get(movement.entrance_id, 0) + 1
            if movement.exit_id:
                entrance_counts[movement.exit_id] = entrance_counts.get(movement.exit_id, 0) + 1
        
        total_entrance_usage = sum(entrance_counts.values())
        if total_entrance_usage > 0:
            for entrance_id, count in entrance_counts.items():
                metrics.entrance_utilization[entrance_id] = count / total_entrance_usage
        
        # Calculate service line usage
        for movement in movements:
            if movement.service_line:
                metrics.service_line_usage[movement.service_line] = metrics.service_line_usage.get(movement.service_line, 0) + 1
        
        return metrics
    
    def export_mass_motion_data(self, output_file: str, 
                              time_window: Optional[timedelta] = None) -> bool:
        """
        Export data in Mass Motion compatible format.
        
        Args:
            output_file: Output file path
            time_window: Time window for export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter movements by time window
            movements = self.movements
            if time_window:
                cutoff_time = datetime.utcnow() - time_window
                movements = [m for m in movements if m.timestamp >= cutoff_time]
            
            # Convert to Mass Motion format
            mass_motion_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_movements': len(movements),
                    'time_window': time_window.total_seconds() if time_window else None,
                    'station_info': {
                        'entrances': len(self.entrances),
                        'zones': len(self.entrances)
                    }
                },
                'entrances': {},
                'movements': [],
                'metrics': {}
            }
            
            # Export entrance data
            for entrance_id, entrance in self.entrances.items():
                mass_motion_data['entrances'][entrance_id] = {
                    'name': entrance.name,
                    'type': entrance.entrance_type,
                    'polygon': entrance.polygon,
                    'platform_connections': entrance.platform_connections,
                    'service_lines': entrance.service_lines,
                    'capacity': entrance.capacity,
                    'is_accessible': entrance.is_accessible
                }
            
            # Export movement data
            for movement in movements:
                mass_motion_data['movements'].append({
                    'pedestrian_id': movement.pedestrian_id,
                    'entrance_id': movement.entrance_id,
                    'exit_id': movement.exit_id,
                    'movement_type': movement.movement_type,
                    'platform_origin': movement.platform_origin,
                    'platform_destination': movement.platform_destination,
                    'service_line': movement.service_line,
                    'timestamp': movement.timestamp.isoformat(),
                    'duration': movement.duration,
                    'path_points': movement.path_points
                })
            
            # Export metrics
            metrics = self.calculate_transit_metrics(time_window)
            mass_motion_data['metrics'] = {
                'total_boardings': metrics.total_boardings,
                'total_alightings': metrics.total_alightings,
                'total_transfers': metrics.total_transfers,
                'total_circulation': metrics.total_circulation,
                'transfer_rate': metrics.transfer_rate,
                'average_dwell_time': metrics.average_dwell_time,
                'platform_utilization': metrics.platform_utilization,
                'entrance_utilization': metrics.entrance_utilization,
                'service_line_usage': metrics.service_line_usage
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(mass_motion_data, f, indent=2, default=str)
            
            logger.info(f"Mass Motion data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Mass Motion data: {e}")
            return False
    
    def get_entrance_flow_summary(self, time_window: Optional[timedelta] = None) -> Dict:
        """Get summary of entrance/exit flows."""
        movements = self.movements
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            movements = [m for m in movements if m.timestamp >= cutoff_time]
        
        summary = {}
        for entrance_id, entrance in self.entrances.items():
            entrance_movements = [m for m in movements if m.entrance_id == entrance_id or m.exit_id == entrance_id]
            
            summary[entrance_id] = {
                'name': entrance.name,
                'type': entrance.entrance_type,
                'total_movements': len(entrance_movements),
                'ingress_count': len([m for m in entrance_movements if m.entrance_id == entrance_id]),
                'egress_count': len([m for m in entrance_movements if m.exit_id == entrance_id]),
                'boarding_count': len([m for m in entrance_movements if m.movement_type == 'boarding']),
                'alighting_count': len([m for m in entrance_movements if m.movement_type == 'alighting']),
                'transfer_count': len([m for m in entrance_movements if m.movement_type == 'transfer'])
            }
        
        return summary
    
    def clear_data(self):
        """Clear all tracking data."""
        self.movements.clear()
        self.movement_history.clear()
        self.current_pedestrians.clear()
        logger.info("Transit analysis data cleared")