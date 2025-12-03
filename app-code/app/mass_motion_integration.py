"""Mass Motion integration for comprehensive transit station analysis."""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .transit_analyzer import TransitAnalyzer, TransitEntrance, TransitMetrics
from .spatial_analyzer import SpatialAnalyzer, SpatialZone, SpatialTemporalData
from .metadata_manager import CameraMetadata

logger = logging.getLogger(__name__)


@dataclass
class MassMotionAgent:
    """Mass Motion agent data structure."""
    agent_id: str
    position: Tuple[float, float]
    timestamp: datetime
    movement_type: str  # 'boarding', 'alighting', 'transfer', 'circulation'
    origin_zone: str
    destination_zone: str
    service_line: str
    dwell_time: float
    path_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MassMotionZone:
    """Mass Motion zone data structure."""
    zone_id: str
    name: str
    zone_type: str
    polygon: List[List[float]]
    area_sqm: float
    capacity: int
    current_occupancy: int
    density: float
    service_level: str
    connected_zones: List[str] = field(default_factory=list)


@dataclass
class MassMotionFlowGate:
    """Mass Motion flow gate data structure."""
    gate_id: str
    name: str
    gate_type: str  # 'door', 'turnstile', 'entrance', 'exit'
    position: Tuple[float, float]
    width_meters: float
    flow_rate: float  # pedestrians per minute
    cumulative_count: int
    direction: str  # 'in', 'out', 'bidirectional'


class MassMotionIntegration:
    """Comprehensive Mass Motion integration for transit station analysis."""
    
    def __init__(self, detection_tuner=None):
        """
        Initialize Mass Motion integration.
        
        Args:
            detection_tuner: Optional DetectionTuner for quality validation
        """
        self.transit_analyzer = TransitAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.agents: List[MassMotionAgent] = []
        self.zones: Dict[str, MassMotionZone] = {}
        self.flow_gates: Dict[str, MassMotionFlowGate] = {}
        self.detection_tuner = detection_tuner
        
        # Mass Motion specific parameters
        self.agent_lifetime_minutes = 30  # How long to keep agent data
        self.path_smoothing_factor = 0.1  # Path smoothing for visualization
        self.export_interval_seconds = 60  # Export interval for real-time data
        
        # Quality tracking
        self.quality_stats = {
            'total_detections': 0,
            'quality_filtered': 0,
            'agents_created': 0
        }
        
    def setup_transit_station(self, station_config: Dict[str, Any]) -> bool:
        """
        Set up transit station configuration for Mass Motion.
        
        Args:
            station_config: Station configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set up entrances
            for entrance_data in station_config.get('entrances', []):
                entrance = TransitEntrance(
                    entrance_id=entrance_data['id'],
                    name=entrance_data['name'],
                    entrance_type=entrance_data['type'],
                    polygon=entrance_data['polygon'],
                    platform_connections=entrance_data.get('platform_connections', []),
                    service_lines=entrance_data.get('service_lines', []),
                    capacity=entrance_data.get('capacity', 0),
                    is_accessible=entrance_data.get('is_accessible', True),
                    notes=entrance_data.get('notes', '')
                )
                self.transit_analyzer.add_entrance(entrance)
            
            # Set up spatial zones
            for zone_data in station_config.get('zones', []):
                zone = SpatialZone(
                    zone_id=zone_data['id'],
                    name=zone_data['name'],
                    polygon=zone_data['polygon'],
                    area_sqm=zone_data['area_sqm'],
                    zone_type=zone_data['type'],
                    capacity=zone_data.get('capacity', 0),
                    critical_density=zone_data.get('critical_density', 1.5),
                    connected_zones=zone_data.get('connected_zones', [])
                )
                self.spatial_analyzer.add_zone(zone)
                
                # Create Mass Motion zone
                mass_motion_zone = MassMotionZone(
                    zone_id=zone.zone_id,
                    name=zone.name,
                    zone_type=zone.zone_type,
                    polygon=zone.polygon,
                    area_sqm=zone.area_sqm,
                    capacity=zone.capacity,
                    current_occupancy=0,
                    density=0.0,
                    service_level="A",
                    connected_zones=zone.connected_zones
                )
                self.zones[zone.zone_id] = mass_motion_zone
            
            # Set up flow gates
            for gate_data in station_config.get('flow_gates', []):
                gate = MassMotionFlowGate(
                    gate_id=gate_data['id'],
                    name=gate_data['name'],
                    gate_type=gate_data['type'],
                    position=tuple(gate_data['position']),
                    width_meters=gate_data.get('width_meters', 1.0),
                    flow_rate=0.0,
                    cumulative_count=0,
                    direction=gate_data.get('direction', 'bidirectional')
                )
                self.flow_gates[gate.gate_id] = gate
            
            logger.info(f"Transit station configured: {len(self.transit_analyzer.entrances)} entrances, "
                       f"{len(self.spatial_analyzer.zones)} zones, {len(self.flow_gates)} flow gates")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up transit station: {e}")
            return False
    
    def update_analysis(self, tracked_pedestrians: List, timestamp: datetime) -> Dict[str, Any]:
        """
        Update analysis with current pedestrian data.
        
        Args:
            tracked_pedestrians: List of tracked pedestrians
            timestamp: Current timestamp
            
        Returns:
            Dictionary with current analysis results
        """
        # Update spatial analysis
        self.spatial_analyzer.update_pedestrian_positions(tracked_pedestrians, timestamp)
        
        # Update transit analysis
        transit_movements = self.transit_analyzer.update_pedestrian_tracking(
            tracked_pedestrians, timestamp
        )
        
        # Create Mass Motion agents from movements
        for movement in transit_movements:
            agent = self._create_mass_motion_agent(movement, timestamp)
            if agent:
                self.agents.append(agent)
        
        # Clean up old agents
        self._cleanup_old_agents(timestamp)
        
        # Update zone data
        self._update_zone_data(timestamp)
        
        # Update flow gate data
        self._update_flow_gate_data(timestamp)
        
        # Return current analysis
        return {
            'timestamp': timestamp,
            'total_agents': len(self.agents),
            'active_zones': len([z for z in self.zones.values() if z.current_occupancy > 0]),
            'transit_movements': len(transit_movements),
            'critical_zones': self._get_critical_zones(),
            'flow_bottlenecks': self._get_flow_bottlenecks()
        }
    
    def _create_mass_motion_agent(self, movement, timestamp: datetime) -> Optional[MassMotionAgent]:
        """Create Mass Motion agent from transit movement with quality validation."""
        try:
            # Get current position (use last known position)
            current_pedestrians = self.transit_analyzer.current_pedestrians
            if movement.pedestrian_id not in current_pedestrians:
                return None
            
            pedestrian = current_pedestrians[movement.pedestrian_id]
            position = (pedestrian.centroid[0], pedestrian.centroid[1])
            
            # Quality validation: Check if detection quality meets Mass Motion requirements
            if self.detection_tuner is not None:
                # Check if we have detection information to validate
                # Assuming pedestrian has a confidence or quality attribute
                if hasattr(pedestrian, 'detections') and pedestrian.detections:
                    # Use last detection for quality check
                    last_detection = pedestrian.detections[-1]
                    if not self.detection_tuner.is_detection_quality_for_massmotion(last_detection):
                        self.quality_stats['quality_filtered'] += 1
                        logger.debug(f"Filtered low-quality detection for agent creation: "
                                   f"confidence={last_detection.confidence:.3f}")
                        return None
            
            self.quality_stats['total_detections'] += 1
            
            # Determine origin and destination zones
            origin_zone = movement.platform_origin or movement.entrance_id
            destination_zone = movement.platform_destination or movement.exit_id
            
            # Calculate dwell time
            dwell_time = movement.duration
            
            # Validate trajectory quality
            if not movement.path_points or len(movement.path_points) < 2:
                logger.debug("Insufficient path points for agent creation")
                return None
            
            # Create agent
            agent = MassMotionAgent(
                agent_id=f"agent_{movement.pedestrian_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                position=position,
                timestamp=timestamp,
                movement_type=movement.movement_type,
                origin_zone=origin_zone,
                destination_zone=destination_zone,
                service_line=movement.service_line,
                dwell_time=dwell_time,
                path_points=movement.path_points
            )
            
            self.quality_stats['agents_created'] += 1
            return agent
            
        except Exception as e:
            logger.error(f"Error creating Mass Motion agent: {e}")
            return None
    
    def _cleanup_old_agents(self, timestamp: datetime):
        """Remove old agents to prevent memory buildup."""
        cutoff_time = timestamp - timedelta(minutes=self.agent_lifetime_minutes)
        self.agents = [agent for agent in self.agents if agent.timestamp >= cutoff_time]
    
    def _update_zone_data(self, timestamp: datetime):
        """Update Mass Motion zone data from spatial analysis."""
        # Get current density data
        density_snapshots = self.spatial_analyzer.calculate_density_snapshots(timestamp)
        
        for zone_id, snapshot in density_snapshots.items():
            if zone_id in self.zones:
                zone = self.zones[zone_id]
                zone.current_occupancy = snapshot.pedestrian_count
                zone.density = snapshot.density
                zone.service_level = snapshot.service_level
    
    def _update_flow_gate_data(self, timestamp: datetime):
        """Update flow gate data."""
        # Get flow rate data
        flow_locations = {gate_id: {'type': gate.gate_type, 'direction': gate.direction, 'width': gate.width_meters}
                         for gate_id, gate in self.flow_gates.items()}
        
        flow_rates = self.spatial_analyzer.calculate_flow_rates(timestamp, flow_locations)
        
        for gate_id, measurement in flow_rates.items():
            if gate_id in self.flow_gates:
                gate = self.flow_gates[gate_id]
                gate.flow_rate = measurement.flow_rate
                gate.cumulative_count = measurement.cumulative_count
    
    def _get_critical_zones(self) -> List[str]:
        """Get list of zones exceeding critical density."""
        critical_zones = []
        for zone_id, zone in self.zones.items():
            if zone.density > 1.5:  # Critical density threshold
                critical_zones.append(zone_id)
        return critical_zones
    
    def _get_flow_bottlenecks(self) -> List[str]:
        """Get list of flow gates with high flow rates."""
        bottlenecks = []
        for gate_id, gate in self.flow_gates.items():
            if gate.flow_rate > 60:  # More than 1 person per second
                bottlenecks.append(gate_id)
        return bottlenecks
    
    def export_mass_motion_data(self, output_file: str, 
                              time_window: Optional[timedelta] = None) -> bool:
        """
        Export comprehensive data in Mass Motion format.
        
        Args:
            output_file: Output file path
            time_window: Time window for export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter agents by time window
            agents_to_export = self.agents
            if time_window:
                cutoff_time = datetime.utcnow() - time_window
                agents_to_export = [a for a in agents_to_export if a.timestamp >= cutoff_time]
            
            # Create Mass Motion data structure
            mass_motion_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'time_window_seconds': time_window.total_seconds() if time_window else None,
                    'total_agents': len(agents_to_export),
                    'total_zones': len(self.zones),
                    'total_flow_gates': len(self.flow_gates),
                    'version': '1.0'
                },
                'zones': {},
                'flow_gates': {},
                'agents': [],
                'spatial_temporal_data': [],
                'transit_metrics': {},
                'summary_statistics': {}
            }
            
            # Export zones
            for zone_id, zone in self.zones.items():
                mass_motion_data['zones'][zone_id] = {
                    'name': zone.name,
                    'zone_type': zone.zone_type,
                    'polygon': zone.polygon,
                    'area_sqm': zone.area_sqm,
                    'capacity': zone.capacity,
                    'current_occupancy': zone.current_occupancy,
                    'density': zone.density,
                    'service_level': zone.service_level,
                    'connected_zones': zone.connected_zones
                }
            
            # Export flow gates
            for gate_id, gate in self.flow_gates.items():
                mass_motion_data['flow_gates'][gate_id] = {
                    'name': gate.name,
                    'gate_type': gate.gate_type,
                    'position': gate.position,
                    'width_meters': gate.width_meters,
                    'flow_rate': gate.flow_rate,
                    'cumulative_count': gate.cumulative_count,
                    'direction': gate.direction
                }
            
            # Export agents
            for agent in agents_to_export:
                mass_motion_data['agents'].append({
                    'agent_id': agent.agent_id,
                    'position': agent.position,
                    'timestamp': agent.timestamp.isoformat(),
                    'movement_type': agent.movement_type,
                    'origin_zone': agent.origin_zone,
                    'destination_zone': agent.destination_zone,
                    'service_line': agent.service_line,
                    'dwell_time': agent.dwell_time,
                    'path_points': agent.path_points
                })
            
            # Export spatial-temporal data
            spatial_data = self.spatial_analyzer.analyze_spatial_temporal_data(
                datetime.utcnow(), 
                {gate_id: {'type': gate.gate_type, 'direction': gate.direction, 'width': gate.width_meters}
                 for gate_id, gate in self.flow_gates.items()}
            )
            
            mass_motion_data['spatial_temporal_data'].append({
                'timestamp': spatial_data.timestamp.isoformat(),
                'total_pedestrians': spatial_data.total_pedestrians,
                'critical_zones': spatial_data.critical_zones,
                'flow_bottlenecks': spatial_data.flow_bottlenecks,
                'zone_densities': {
                    zone_id: {
                        'pedestrian_count': snapshot.pedestrian_count,
                        'density': snapshot.density,
                        'service_level': snapshot.service_level,
                        'occupancy_rate': snapshot.occupancy_rate
                    }
                    for zone_id, snapshot in spatial_data.zone_densities.items()
                },
                'flow_rates': {
                    location_id: {
                        'flow_rate': measurement.flow_rate,
                        'cumulative_count': measurement.cumulative_count,
                        'direction': measurement.direction
                    }
                    for location_id, measurement in spatial_data.flow_rates.items()
                }
            })
            
            # Export transit metrics
            transit_metrics = self.transit_analyzer.calculate_transit_metrics(time_window)
            mass_motion_data['transit_metrics'] = {
                'total_boardings': transit_metrics.total_boardings,
                'total_alightings': transit_metrics.total_alightings,
                'total_transfers': transit_metrics.total_transfers,
                'total_circulation': transit_metrics.total_circulation,
                'transfer_rate': transit_metrics.transfer_rate,
                'average_dwell_time': transit_metrics.average_dwell_time,
                'platform_utilization': transit_metrics.platform_utilization,
                'entrance_utilization': transit_metrics.entrance_utilization,
                'service_line_usage': transit_metrics.service_line_usage
            }
            
            # Export summary statistics
            summary = self.spatial_analyzer.get_flow_rate_summary(time_window or timedelta(minutes=5))
            mass_motion_data['summary_statistics'] = summary
            
            # Add detection quality metrics
            mass_motion_data['detection_quality'] = {
                'total_detections': self.quality_stats['total_detections'],
                'quality_filtered': self.quality_stats['quality_filtered'],
                'agents_created': self.quality_stats['agents_created'],
                'quality_acceptance_rate': (
                    self.quality_stats['agents_created'] / 
                    (self.quality_stats['total_detections'] + 1e-6)
                )
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(mass_motion_data, f, indent=2, default=str)
            
            logger.info(f"Mass Motion data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Mass Motion data: {e}")
            return False
    
    def export_realtime_data(self, output_file: str) -> bool:
        """Export real-time data for Mass Motion visualization."""
        try:
            current_time = datetime.utcnow()
            
            # Get current spatial data
            flow_locations = {gate_id: {'type': gate.gate_type, 'direction': gate.direction, 'width': gate.width_meters}
                             for gate_id, gate in self.flow_gates.items()}
            
            spatial_data = self.spatial_analyzer.analyze_spatial_temporal_data(
                current_time, flow_locations
            )
            
            # Create real-time data structure
            realtime_data = {
                'timestamp': current_time.isoformat(),
                'zones': {},
                'flow_gates': {},
                'agents': [],
                'summary': {
                    'total_pedestrians': spatial_data.total_pedestrians,
                    'critical_zones': spatial_data.critical_zones,
                    'flow_bottlenecks': spatial_data.flow_bottlenecks
                }
            }
            
            # Export current zone states
            for zone_id, zone in self.zones.items():
                realtime_data['zones'][zone_id] = {
                    'name': zone.name,
                    'current_occupancy': zone.current_occupancy,
                    'density': zone.density,
                    'service_level': zone.service_level,
                    'capacity_utilization': (zone.current_occupancy / zone.capacity * 100) if zone.capacity > 0 else 0
                }
            
            # Export current flow gate states
            for gate_id, gate in self.flow_gates.items():
                realtime_data['flow_gates'][gate_id] = {
                    'name': gate.name,
                    'flow_rate': gate.flow_rate,
                    'cumulative_count': gate.cumulative_count,
                    'direction': gate.direction
                }
            
            # Export recent agents (last 5 minutes)
            recent_agents = [a for a in self.agents 
                           if (current_time - a.timestamp).total_seconds() < 300]
            
            for agent in recent_agents:
                realtime_data['agents'].append({
                    'agent_id': agent.agent_id,
                    'position': agent.position,
                    'movement_type': agent.movement_type,
                    'origin_zone': agent.origin_zone,
                    'destination_zone': agent.destination_zone,
                    'service_line': agent.service_line
                })
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(realtime_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting real-time data: {e}")
            return False
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        current_time = datetime.utcnow()
        
        # Get current metrics
        transit_metrics = self.transit_analyzer.calculate_transit_metrics()
        spatial_summary = self.spatial_analyzer.get_flow_rate_summary()
        
        # Calculate additional metrics
        total_agents = len(self.agents)
        active_zones = len([z for z in self.zones.values() if z.current_occupancy > 0])
        critical_zones = self._get_critical_zones()
        flow_bottlenecks = self._get_flow_bottlenecks()
        
        # Calculate zone utilization
        zone_utilization = {}
        for zone_id, zone in self.zones.items():
            if zone.capacity > 0:
                zone_utilization[zone_id] = (zone.current_occupancy / zone.capacity) * 100
            else:
                zone_utilization[zone_id] = 0
        
        # Calculate flow gate utilization
        flow_utilization = {}
        for gate_id, gate in self.flow_gates.items():
            # Assume maximum flow rate of 120 people per minute (2 per second)
            max_flow_rate = 120
            flow_utilization[gate_id] = min((gate.flow_rate / max_flow_rate) * 100, 100)
        
        return {
            'timestamp': current_time.isoformat(),
            'total_agents': total_agents,
            'active_zones': active_zones,
            'critical_zones': critical_zones,
            'flow_bottlenecks': flow_bottlenecks,
            'zone_utilization': zone_utilization,
            'flow_utilization': flow_utilization,
            'transit_metrics': {
                'total_boardings': transit_metrics.total_boardings,
                'total_alightings': transit_metrics.total_alightings,
                'total_transfers': transit_metrics.total_transfers,
                'transfer_rate': transit_metrics.transfer_rate,
                'average_dwell_time': transit_metrics.average_dwell_time
            },
            'spatial_metrics': spatial_summary.get('overall_metrics', {}),
            'recommendations': self._generate_recommendations(critical_zones, flow_bottlenecks)
        }
    
    def _generate_recommendations(self, critical_zones: List[str], 
                                flow_bottlenecks: List[str]) -> List[str]:
        """Generate recommendations based on current analysis."""
        recommendations = []
        
        if critical_zones:
            recommendations.append(f"Critical density detected in zones: {', '.join(critical_zones)}")
            recommendations.append("Consider crowd control measures or alternative routing")
        
        if flow_bottlenecks:
            recommendations.append(f"Flow bottlenecks detected at: {', '.join(flow_bottlenecks)}")
            recommendations.append("Consider opening additional gates or improving flow management")
        
        if not critical_zones and not flow_bottlenecks:
            recommendations.append("Station operating within normal parameters")
        
        return recommendations