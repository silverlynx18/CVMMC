"""Spatial-temporal analysis for density mapping and flow rate calculation."""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import math

from .pedestrian_tracker import TrackedPedestrian

logger = logging.getLogger(__name__)


@dataclass
class SpatialZone:
    """Spatial zone definition for density analysis."""
    zone_id: str
    name: str
    polygon: List[List[float]]  # Zone polygon coordinates
    area_sqm: float
    zone_type: str  # 'platform', 'concourse', 'entrance', 'corridor', 'waiting_area'
    capacity: int = 0  # Maximum capacity
    critical_density: float = 1.5  # Critical density threshold (peds/sqm)
    service_level: str = "A"  # Current service level
    connected_zones: List[str] = field(default_factory=list)


@dataclass
class DensitySnapshot:
    """Density measurement at a specific time and location."""
    timestamp: datetime
    zone_id: str
    pedestrian_count: int
    density: float  # pedestrians per square meter
    area_sqm: float
    service_level: str
    occupancy_rate: float  # percentage of capacity used


@dataclass
class FlowRateMeasurement:
    """Flow rate measurement for a specific location."""
    location_id: str
    location_type: str  # 'door', 'turnstile', 'entrance', 'exit'
    timestamp: datetime
    flow_rate: float  # pedestrians per minute
    cumulative_count: int
    direction: str  # 'in', 'out', 'bidirectional'
    width_meters: float = 1.0  # Effective width for flow calculation


@dataclass
class SpatialTemporalData:
    """Comprehensive spatial-temporal analysis data."""
    timestamp: datetime
    zone_densities: Dict[str, DensitySnapshot]
    flow_rates: Dict[str, FlowRateMeasurement]
    total_pedestrians: int
    critical_zones: List[str]  # Zones exceeding critical density
    flow_bottlenecks: List[str]  # Locations with high flow rates


class SpatialAnalyzer:
    """Analyzes spatial density and flow rates for Mass Motion integration."""
    
    def __init__(self, grid_resolution: int = 10):
        """
        Initialize spatial analyzer.
        
        Args:
            grid_resolution: Resolution for density grid (meters per cell)
        """
        self.grid_resolution = grid_resolution
        self.zones: Dict[str, SpatialZone] = {}
        self.density_history: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.flow_history: deque = deque(maxlen=1000)
        self.pedestrian_positions: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.flow_counters: Dict[str, int] = defaultdict(int)
        self.last_flow_reset: Dict[str, datetime] = {}
        
        # Analysis parameters
        self.density_calculation_interval = 1.0  # seconds
        self.flow_calculation_interval = 30.0  # seconds
        self.last_density_calculation = datetime.utcnow()
        self.last_flow_calculation = datetime.utcnow()
        
    def add_zone(self, zone: SpatialZone) -> bool:
        """Add a spatial zone for analysis."""
        try:
            self.zones[zone.zone_id] = zone
            logger.info(f"Added zone {zone.zone_id}: {zone.name} ({zone.area_sqm:.1f} sqm)")
            return True
        except Exception as e:
            logger.error(f"Error adding zone {zone.zone_id}: {e}")
            return False
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a spatial zone."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            logger.info(f"Removed zone {zone_id}")
            return True
        return False
    
    def update_pedestrian_positions(self, tracked_pedestrians: List[TrackedPedestrian], 
                                  timestamp: datetime):
        """Update pedestrian positions for density analysis."""
        # Clear previous positions
        for zone_id in self.pedestrian_positions:
            self.pedestrian_positions[zone_id].clear()
        
        # Update positions for each zone
        for pedestrian in tracked_pedestrians:
            x, y = pedestrian.centroid
            zone_id = self._get_zone_for_position(x, y)
            if zone_id:
                self.pedestrian_positions[zone_id].append((x, y))
    
    def _get_zone_for_position(self, x: float, y: float) -> Optional[str]:
        """Find which zone a position belongs to."""
        for zone_id, zone in self.zones.items():
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
    
    def calculate_density_snapshots(self, timestamp: datetime) -> Dict[str, DensitySnapshot]:
        """Calculate density for all zones."""
        snapshots = {}
        
        for zone_id, zone in self.zones.items():
            pedestrian_count = len(self.pedestrian_positions[zone_id])
            density = pedestrian_count / zone.area_sqm if zone.area_sqm > 0 else 0
            
            # Determine service level based on density
            service_level = self._determine_service_level(density, zone.critical_density)
            
            # Calculate occupancy rate
            occupancy_rate = (pedestrian_count / zone.capacity * 100) if zone.capacity > 0 else 0
            
            snapshot = DensitySnapshot(
                timestamp=timestamp,
                zone_id=zone_id,
                pedestrian_count=pedestrian_count,
                density=density,
                area_sqm=zone.area_sqm,
                service_level=service_level,
                occupancy_rate=occupancy_rate
            )
            
            snapshots[zone_id] = snapshot
        
        return snapshots
    
    def _determine_service_level(self, density: float, critical_density: float) -> str:
        """Determine service level based on density."""
        if density <= 0.2:
            return "A"
        elif density <= 0.5:
            return "B"
        elif density <= 1.0:
            return "C"
        elif density <= 1.5:
            return "D"
        elif density <= 2.0:
            return "E"
        else:
            return "F"
    
    def calculate_flow_rates(self, timestamp: datetime, 
                           flow_locations: Dict[str, Dict]) -> Dict[str, FlowRateMeasurement]:
        """Calculate flow rates for specific locations."""
        measurements = {}
        
        for location_id, location_data in flow_locations.items():
            # Get current count for this location
            current_count = self.flow_counters.get(location_id, 0)
            
            # Calculate flow rate if we have previous data
            flow_rate = 0.0
            if location_id in self.last_flow_reset:
                time_diff = (timestamp - self.last_flow_reset[location_id]).total_seconds()
                if time_diff > 0:
                    flow_rate = (current_count / time_diff) * 60  # per minute
            
            measurement = FlowRateMeasurement(
                location_id=location_id,
                location_type=location_data.get('type', 'entrance'),
                timestamp=timestamp,
                flow_rate=flow_rate,
                cumulative_count=current_count,
                direction=location_data.get('direction', 'bidirectional'),
                width_meters=location_data.get('width', 1.0)
            )
            
            measurements[location_id] = measurement
        
        return measurements
    
    def update_flow_counter(self, location_id: str, increment: int = 1):
        """Update flow counter for a specific location."""
        self.flow_counters[location_id] += increment
        self.last_flow_reset[location_id] = datetime.utcnow()
    
    def analyze_spatial_temporal_data(self, timestamp: datetime, 
                                    flow_locations: Dict[str, Dict]) -> SpatialTemporalData:
        """Perform comprehensive spatial-temporal analysis."""
        # Calculate density snapshots
        zone_densities = self.calculate_density_snapshots(timestamp)
        
        # Calculate flow rates
        flow_rates = self.calculate_flow_rates(timestamp, flow_locations)
        
        # Find critical zones (exceeding critical density)
        critical_zones = []
        for zone_id, snapshot in zone_densities.items():
            zone = self.zones[zone_id]
            if snapshot.density > zone.critical_density:
                critical_zones.append(zone_id)
        
        # Find flow bottlenecks (high flow rates)
        flow_bottlenecks = []
        for location_id, measurement in flow_rates.items():
            if measurement.flow_rate > 60:  # More than 1 person per second
                flow_bottlenecks.append(location_id)
        
        # Calculate total pedestrians
        total_pedestrians = sum(snapshot.pedestrian_count for snapshot in zone_densities.values())
        
        # Create spatial-temporal data
        data = SpatialTemporalData(
            timestamp=timestamp,
            zone_densities=zone_densities,
            flow_rates=flow_rates,
            total_pedestrians=total_pedestrians,
            critical_zones=critical_zones,
            flow_bottlenecks=flow_bottlenecks
        )
        
        # Store in history
        self.density_history.append(data)
        
        return data
    
    def get_density_heatmap(self, timestamp: datetime, 
                           image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate density heatmap for visualization."""
        height, width = image_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Create grid for density calculation
        grid_height = height // self.grid_resolution
        grid_width = width // self.grid_resolution
        
        for zone_id, zone in self.zones.items():
            # Get pedestrian positions in this zone
            positions = self.pedestrian_positions[zone_id]
            
            if not positions:
                continue
            
            # Calculate density for each grid cell in the zone
            for i in range(grid_height):
                for j in range(grid_width):
                    # Convert grid coordinates to image coordinates
                    x = j * self.grid_resolution
                    y = i * self.grid_resolution
                    
                    # Check if this grid cell is in the zone
                    if self._point_in_polygon(x, y, zone.polygon):
                        # Count pedestrians in this grid cell
                        cell_pedestrians = 0
                        for px, py in positions:
                            if (abs(px - x) < self.grid_resolution and 
                                abs(py - y) < self.grid_resolution):
                                cell_pedestrians += 1
                        
                        # Calculate density for this cell
                        cell_area = self.grid_resolution ** 2
                        density = cell_pedestrians / cell_area if cell_area > 0 else 0
                        
                        # Set heatmap value
                        heatmap[y:y+self.grid_resolution, x:x+self.grid_resolution] = density
        
        return heatmap
    
    def get_flow_rate_summary(self, time_window: timedelta = timedelta(minutes=5)) -> Dict:
        """Get flow rate summary for the specified time window."""
        cutoff_time = datetime.utcnow() - time_window
        recent_data = [d for d in self.density_history if d.timestamp >= cutoff_time]
        
        if not recent_data:
            return {}
        
        summary = {
            'time_window_seconds': time_window.total_seconds(),
            'data_points': len(recent_data),
            'zones': {},
            'flow_locations': {},
            'overall_metrics': {}
        }
        
        # Analyze zone data
        zone_metrics = defaultdict(list)
        for data in recent_data:
            for zone_id, snapshot in data.zone_densities.items():
                zone_metrics[zone_id].append({
                    'timestamp': snapshot.timestamp,
                    'density': snapshot.density,
                    'pedestrian_count': snapshot.pedestrian_count,
                    'service_level': snapshot.service_level
                })
        
        # Calculate zone summaries
        for zone_id, metrics in zone_metrics.items():
            densities = [m['density'] for m in metrics]
            pedestrian_counts = [m['pedestrian_count'] for m in metrics]
            
            summary['zones'][zone_id] = {
                'avg_density': np.mean(densities),
                'max_density': np.max(densities),
                'avg_pedestrians': np.mean(pedestrian_counts),
                'max_pedestrians': np.max(pedestrian_counts),
                'data_points': len(metrics)
            }
        
        # Analyze flow data
        flow_metrics = defaultdict(list)
        for data in recent_data:
            for location_id, measurement in data.flow_rates.items():
                flow_metrics[location_id].append({
                    'timestamp': measurement.timestamp,
                    'flow_rate': measurement.flow_rate,
                    'cumulative_count': measurement.cumulative_count
                })
        
        # Calculate flow summaries
        for location_id, metrics in flow_metrics.items():
            flow_rates = [m['flow_rate'] for m in metrics]
            
            summary['flow_locations'][location_id] = {
                'avg_flow_rate': np.mean(flow_rates),
                'max_flow_rate': np.max(flow_rates),
                'total_count': metrics[-1]['cumulative_count'] if metrics else 0,
                'data_points': len(metrics)
            }
        
        # Overall metrics
        all_densities = []
        all_flow_rates = []
        for data in recent_data:
            for snapshot in data.zone_densities.values():
                all_densities.append(snapshot.density)
            for measurement in data.flow_rates.values():
                all_flow_rates.append(measurement.flow_rate)
        
        summary['overall_metrics'] = {
            'avg_density': np.mean(all_densities) if all_densities else 0,
            'max_density': np.max(all_densities) if all_densities else 0,
            'avg_flow_rate': np.mean(all_flow_rates) if all_flow_rates else 0,
            'max_flow_rate': np.max(all_flow_rates) if all_flow_rates else 0
        }
        
        return summary
    
    def export_mass_motion_spatial_data(self, output_file: str, 
                                      time_window: Optional[timedelta] = None) -> bool:
        """Export spatial data in Mass Motion compatible format."""
        try:
            # Filter data by time window
            data_to_export = list(self.density_history)
            if time_window:
                cutoff_time = datetime.utcnow() - time_window
                data_to_export = [d for d in data_to_export if d.timestamp >= cutoff_time]
            
            # Prepare Mass Motion data structure
            mass_motion_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'time_window_seconds': time_window.total_seconds() if time_window else None,
                    'data_points': len(data_to_export),
                    'grid_resolution': self.grid_resolution,
                    'zones': len(self.zones)
                },
                'zones': {},
                'spatial_data': [],
                'flow_data': [],
                'summary_metrics': {}
            }
            
            # Export zone definitions
            for zone_id, zone in self.zones.items():
                mass_motion_data['zones'][zone_id] = {
                    'name': zone.name,
                    'polygon': zone.polygon,
                    'area_sqm': zone.area_sqm,
                    'zone_type': zone.zone_type,
                    'capacity': zone.capacity,
                    'critical_density': zone.critical_density,
                    'connected_zones': zone.connected_zones
                }
            
            # Export spatial-temporal data
            for data in data_to_export:
                spatial_entry = {
                    'timestamp': data.timestamp.isoformat(),
                    'total_pedestrians': data.total_pedestrians,
                    'critical_zones': data.critical_zones,
                    'flow_bottlenecks': data.flow_bottlenecks,
                    'zone_densities': {}
                }
                
                for zone_id, snapshot in data.zone_densities.items():
                    spatial_entry['zone_densities'][zone_id] = {
                        'pedestrian_count': snapshot.pedestrian_count,
                        'density': snapshot.density,
                        'area_sqm': snapshot.area_sqm,
                        'service_level': snapshot.service_level,
                        'occupancy_rate': snapshot.occupancy_rate
                    }
                
                mass_motion_data['spatial_data'].append(spatial_entry)
            
            # Export flow data
            for data in data_to_export:
                for location_id, measurement in data.flow_rates.items():
                    flow_entry = {
                        'timestamp': measurement.timestamp.isoformat(),
                        'location_id': measurement.location_id,
                        'location_type': measurement.location_type,
                        'flow_rate': measurement.flow_rate,
                        'cumulative_count': measurement.cumulative_count,
                        'direction': measurement.direction,
                        'width_meters': measurement.width_meters
                    }
                    mass_motion_data['flow_data'].append(flow_entry)
            
            # Export summary metrics
            summary = self.get_flow_rate_summary(time_window or timedelta(minutes=5))
            mass_motion_data['summary_metrics'] = summary
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(mass_motion_data, f, indent=2, default=str)
            
            logger.info(f"Mass Motion spatial data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Mass Motion spatial data: {e}")
            return False
    
    def clear_data(self):
        """Clear all analysis data."""
        self.density_history.clear()
        self.flow_history.clear()
        self.pedestrian_positions.clear()
        self.flow_counters.clear()
        self.last_flow_reset.clear()
        logger.info("Spatial analysis data cleared")