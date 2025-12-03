"""Comprehensive Mass Motion data export with multiple format support."""

import json
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

from .dataset_generator import DatasetGenerator, PedestrianJourney, JourneyPoint
from .transit_analyzer import TransitAnalyzer, TransitEntrance, TransitMovement
from .spatial_analyzer import SpatialAnalyzer, SpatialZone, SpatialTemporalData
from .confidence_analyzer import ConfidenceAnalyzer, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class MassMotionAgent:
    """Mass Motion agent representation."""
    agent_id: str
    journey_id: str
    start_time: datetime
    end_time: datetime
    origin_zone: str
    destination_zone: str
    journey_type: str
    service_line: str
    path_points: List[Tuple[float, float]]
    timestamps: List[datetime]
    activities: List[str]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MassMotionZone:
    """Mass Motion zone representation."""
    zone_id: str
    name: str
    zone_type: str  # platform, concourse, entrance, exit, circulation
    area_sqm: float
    capacity: int
    polygon: List[Tuple[float, float]]
    connections: List[str] = field(default_factory=list)
    flow_gates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MassMotionFlowGate:
    """Mass Motion flow gate representation."""
    gate_id: str
    name: str
    zone_from: str
    zone_to: str
    width_m: float
    capacity_per_minute: int
    current_flow_rate: float
    max_flow_rate: float
    position: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MassMotionTimetable:
    """Mass Motion timetable entry."""
    service_line: str
    platform: str
    scheduled_time: datetime
    actual_time: datetime
    passenger_count: int
    journey_type: str
    delay_minutes: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MassMotionExporter:
    """Exports data in various Mass Motion compatible formats."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize Mass Motion exporter.
        
        Args:
            confidence_threshold: Minimum confidence threshold for data inclusion
        """
        self.confidence_threshold = confidence_threshold
        self.dataset_generator = DatasetGenerator(confidence_threshold)
        self.transit_analyzer = TransitAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        
        # Data storage
        self.agents: List[MassMotionAgent] = []
        self.zones: Dict[str, MassMotionZone] = {}
        self.flow_gates: Dict[str, MassMotionFlowGate] = {}
        self.timetables: Dict[str, List[MassMotionTimetable]] = defaultdict(list)
        
    def export_comprehensive_dataset(self, analysis_results: List[Dict],
                                   zones: Dict[str, SpatialZone],
                                   entrances: Dict[str, TransitEntrance],
                                   output_dir: str,
                                   formats: List[str] = ["json", "csv", "xml"]) -> bool:
        """
        Export comprehensive dataset in multiple formats.
        
        Args:
            analysis_results: List of frame-by-frame analysis results
            zones: Dictionary of spatial zones
            entrances: Dictionary of transit entrances
            output_dir: Output directory
            formats: List of export formats
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate datasets
            datasets = self.dataset_generator.process_video_analysis(
                analysis_results, zones, entrances
            )
            
            # Create Mass Motion specific data structures
            self._create_mass_motion_structures(datasets, zones, entrances)
            
            # Export in requested formats
            success = True
            
            if "json" in formats:
                success &= self._export_json_format(output_path, datasets)
            
            if "csv" in formats:
                success &= self._export_csv_format(output_path, datasets)
            
            if "xml" in formats:
                success &= self._export_xml_format(output_path, datasets)
            
            # Export Mass Motion specific files
            success &= self._export_mass_motion_specific(output_path, datasets)
            
            # Export confidence and quality reports
            success &= self._export_quality_reports(output_path, analysis_results)
            
            logger.info(f"Comprehensive dataset exported to {output_dir}")
            return success
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive dataset: {e}")
            return False
    
    def _create_mass_motion_structures(self, datasets: Dict[str, Any],
                                     zones: Dict[str, SpatialZone],
                                     entrances: Dict[str, TransitEntrance]):
        """Create Mass Motion specific data structures."""
        # Create zones
        for zone_id, zone in zones.items():
            mass_motion_zone = MassMotionZone(
                zone_id=zone_id,
                name=zone.name,
                zone_type=zone.zone_type,
                area_sqm=zone.area_sqm,
                capacity=zone.capacity,
                polygon=zone.polygon,
                connections=zone.connections,
                flow_gates=zone.flow_gates,
                metadata=zone.metadata
            )
            self.zones[zone_id] = mass_motion_zone
        
        # Create flow gates from entrances
        for entrance_id, entrance in entrances.items():
            flow_gate = MassMotionFlowGate(
                gate_id=f"gate_{entrance_id}",
                name=entrance.name,
                zone_from=entrance.zone_from,
                zone_to=entrance.zone_to,
                width_m=entrance.width_m,
                capacity_per_minute=entrance.capacity_per_minute,
                current_flow_rate=0.0,
                max_flow_rate=entrance.capacity_per_minute,
                position=entrance.position,
                metadata=entrance.metadata
            )
            self.flow_gates[f"gate_{entrance_id}"] = flow_gate
        
        # Create agents from journeys
        for journey_data in datasets.get('journeys', []):
            if journey_data['confidence_score'] >= self.confidence_threshold:
                agent = MassMotionAgent(
                    agent_id=f"agent_{journey_data['journey_id']}",
                    journey_id=journey_data['journey_id'],
                    start_time=datetime.fromisoformat(journey_data['start_time']),
                    end_time=datetime.fromisoformat(journey_data['end_time']),
                    origin_zone=journey_data['origin_zone'],
                    destination_zone=journey_data['destination_zone'],
                    journey_type=journey_data['journey_type'],
                    service_line=journey_data['service_line'],
                    path_points=[(point['position'][0], point['position'][1]) 
                               for point in journey_data['path_points']],
                    timestamps=[datetime.fromisoformat(point['timestamp']) 
                              for point in journey_data['path_points']],
                    activities=[point['activity'] for point in journey_data['path_points']],
                    confidence_score=journey_data['confidence_score'],
                    metadata={
                        'total_distance': journey_data['total_distance'],
                        'average_speed': journey_data['average_speed'],
                        'max_speed': journey_data['max_speed'],
                        'dwell_time': journey_data['dwell_time']
                    }
                )
                self.agents.append(agent)
        
        # Create timetables
        for service_line, timetable_entries in datasets.get('timetables', {}).items():
            for entry_data in timetable_entries:
                timetable = MassMotionTimetable(
                    service_line=service_line,
                    platform=entry_data['platform'],
                    scheduled_time=datetime.fromisoformat(entry_data['scheduled_time']),
                    actual_time=datetime.fromisoformat(entry_data['actual_time']),
                    passenger_count=entry_data['passenger_count'],
                    journey_type=entry_data['journey_type'],
                    delay_minutes=entry_data['delay_minutes'],
                    metadata={}
                )
                self.timetables[service_line].append(timetable)
    
    def _export_json_format(self, output_path: Path, datasets: Dict[str, Any]) -> bool:
        """Export data in JSON format."""
        try:
            # Export main datasets
            with open(output_path / "mass_motion_data.json", 'w') as f:
                json.dump(datasets, f, indent=2, default=str)
            
            # Export agents
            agents_data = [agent.__dict__ for agent in self.agents]
            with open(output_path / "agents.json", 'w') as f:
                json.dump(agents_data, f, indent=2, default=str)
            
            # Export zones
            zones_data = {zone_id: zone.__dict__ for zone_id, zone in self.zones.items()}
            with open(output_path / "zones.json", 'w') as f:
                json.dump(zones_data, f, indent=2, default=str)
            
            # Export flow gates
            flow_gates_data = {gate_id: gate.__dict__ for gate_id, gate in self.flow_gates.items()}
            with open(output_path / "flow_gates.json", 'w') as f:
                json.dump(flow_gates_data, f, indent=2, default=str)
            
            # Export timetables
            timetables_data = {
                service_line: [timetable.__dict__ for timetable in timetables]
                for service_line, timetables in self.timetables.items()
            }
            with open(output_path / "timetables.json", 'w') as f:
                json.dump(timetables_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JSON format: {e}")
            return False
    
    def _export_csv_format(self, output_path: Path, datasets: Dict[str, Any]) -> bool:
        """Export data in CSV format."""
        try:
            # Export agents
            self._export_agents_csv(output_path / "agents.csv")
            
            # Export zones
            self._export_zones_csv(output_path / "zones.csv")
            
            # Export flow gates
            self._export_flow_gates_csv(output_path / "flow_gates.csv")
            
            # Export timetables
            self._export_timetables_csv(output_path / "timetables.csv")
            
            # Export coordinate series
            self._export_coordinate_series_csv(output_path / "coordinate_series.csv", 
                                             datasets.get('coordinate_series', []))
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting CSV format: {e}")
            return False
    
    def _export_xml_format(self, output_path: Path, datasets: Dict[str, Any]) -> bool:
        """Export data in XML format."""
        try:
            # Create root element
            root = ET.Element("MassMotionData")
            root.set("version", "1.0")
            root.set("export_timestamp", datetime.utcnow().isoformat())
            
            # Add metadata
            metadata = ET.SubElement(root, "metadata")
            metadata.set("total_agents", str(len(self.agents)))
            metadata.set("total_zones", str(len(self.zones)))
            metadata.set("total_flow_gates", str(len(self.flow_gates)))
            metadata.set("confidence_threshold", str(self.confidence_threshold))
            
            # Add zones
            zones_elem = ET.SubElement(root, "zones")
            for zone_id, zone in self.zones.items():
                zone_elem = ET.SubElement(zones_elem, "zone")
                zone_elem.set("id", zone_id)
                zone_elem.set("name", zone.name)
                zone_elem.set("type", zone.zone_type)
                zone_elem.set("area_sqm", str(zone.area_sqm))
                zone_elem.set("capacity", str(zone.capacity))
                
                # Add polygon points
                polygon_elem = ET.SubElement(zone_elem, "polygon")
                for x, y in zone.polygon:
                    point_elem = ET.SubElement(polygon_elem, "point")
                    point_elem.set("x", str(x))
                    point_elem.set("y", str(y))
            
            # Add flow gates
            flow_gates_elem = ET.SubElement(root, "flow_gates")
            for gate_id, gate in self.flow_gates.items():
                gate_elem = ET.SubElement(flow_gates_elem, "flow_gate")
                gate_elem.set("id", gate_id)
                gate_elem.set("name", gate.name)
                gate_elem.set("zone_from", gate.zone_from)
                gate_elem.set("zone_to", gate.zone_to)
                gate_elem.set("width_m", str(gate.width_m))
                gate_elem.set("capacity_per_minute", str(gate.capacity_per_minute))
                gate_elem.set("position_x", str(gate.position[0]))
                gate_elem.set("position_y", str(gate.position[1]))
            
            # Add agents
            agents_elem = ET.SubElement(root, "agents")
            for agent in self.agents:
                agent_elem = ET.SubElement(agents_elem, "agent")
                agent_elem.set("id", agent.agent_id)
                agent_elem.set("journey_id", agent.journey_id)
                agent_elem.set("start_time", agent.start_time.isoformat())
                agent_elem.set("end_time", agent.end_time.isoformat())
                agent_elem.set("origin_zone", agent.origin_zone)
                agent_elem.set("destination_zone", agent.destination_zone)
                agent_elem.set("journey_type", agent.journey_type)
                agent_elem.set("service_line", agent.service_line)
                agent_elem.set("confidence_score", str(agent.confidence_score))
                
                # Add path points
                path_elem = ET.SubElement(agent_elem, "path")
                for i, (x, y) in enumerate(agent.path_points):
                    point_elem = ET.SubElement(path_elem, "point")
                    point_elem.set("index", str(i))
                    point_elem.set("x", str(x))
                    point_elem.set("y", str(y))
                    point_elem.set("timestamp", agent.timestamps[i].isoformat())
                    point_elem.set("activity", agent.activities[i])
            
            # Write XML file
            tree = ET.ElementTree(root)
            tree.write(output_path / "mass_motion_data.xml", encoding="utf-8", xml_declaration=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting XML format: {e}")
            return False
    
    def _export_mass_motion_specific(self, output_path: Path, datasets: Dict[str, Any]) -> bool:
        """Export Mass Motion specific files."""
        try:
            # Export agent trajectories
            self._export_agent_trajectories(output_path / "agent_trajectories.json")
            
            # Export zone states
            self._export_zone_states(output_path / "zone_states.json")
            
            # Export flow gate states
            self._export_flow_gate_states(output_path / "flow_gate_states.json")
            
            # Export service line data
            self._export_service_line_data(output_path / "service_line_data.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Mass Motion specific files: {e}")
            return False
    
    def _export_quality_reports(self, output_path: Path, analysis_results: List[Dict]) -> bool:
        """Export quality and confidence reports."""
        try:
            # Generate quality analysis
            quality_reports = []
            for frame_data in analysis_results:
                timestamp = datetime.fromisoformat(frame_data['timestamp'])
                tracked_pedestrians = self.dataset_generator._create_tracked_pedestrians_from_frame(frame_data)
                
                quality_analysis = self.confidence_analyzer.analyze_tracking_quality(
                    tracked_pedestrians, timestamp
                )
                quality_reports.append(quality_analysis)
            
            # Export quality reports
            with open(output_path / "quality_reports.json", 'w') as f:
                json.dump(quality_reports, f, indent=2, default=str)
            
            # Export quality summary
            quality_summary = self.confidence_analyzer.get_confidence_summary()
            with open(output_path / "quality_summary.json", 'w') as f:
                json.dump(quality_summary, f, indent=2, default=str)
            
            # Export quality trends
            quality_trends = self.confidence_analyzer.get_quality_trends()
            with open(output_path / "quality_trends.json", 'w') as f:
                json.dump(quality_trends, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting quality reports: {e}")
            return False
    
    def _export_agents_csv(self, file_path: Path):
        """Export agents to CSV."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'agent_id', 'journey_id', 'start_time', 'end_time', 'origin_zone',
                'destination_zone', 'journey_type', 'service_line', 'confidence_score',
                'total_distance', 'average_speed', 'max_speed', 'dwell_time'
            ])
            
            for agent in self.agents:
                writer.writerow([
                    agent.agent_id,
                    agent.journey_id,
                    agent.start_time.isoformat(),
                    agent.end_time.isoformat(),
                    agent.origin_zone,
                    agent.destination_zone,
                    agent.journey_type,
                    agent.service_line,
                    agent.confidence_score,
                    agent.metadata.get('total_distance', 0),
                    agent.metadata.get('average_speed', 0),
                    agent.metadata.get('max_speed', 0),
                    agent.metadata.get('dwell_time', 0)
                ])
    
    def _export_zones_csv(self, file_path: Path):
        """Export zones to CSV."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'zone_id', 'name', 'type', 'area_sqm', 'capacity', 'connections', 'flow_gates'
            ])
            
            for zone in self.zones.values():
                writer.writerow([
                    zone.zone_id,
                    zone.name,
                    zone.zone_type,
                    zone.area_sqm,
                    zone.capacity,
                    ','.join(zone.connections),
                    ','.join(zone.flow_gates)
                ])
    
    def _export_flow_gates_csv(self, file_path: Path):
        """Export flow gates to CSV."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'gate_id', 'name', 'zone_from', 'zone_to', 'width_m',
                'capacity_per_minute', 'current_flow_rate', 'max_flow_rate',
                'position_x', 'position_y'
            ])
            
            for gate in self.flow_gates.values():
                writer.writerow([
                    gate.gate_id,
                    gate.name,
                    gate.zone_from,
                    gate.zone_to,
                    gate.width_m,
                    gate.capacity_per_minute,
                    gate.current_flow_rate,
                    gate.max_flow_rate,
                    gate.position[0],
                    gate.position[1]
                ])
    
    def _export_timetables_csv(self, file_path: Path):
        """Export timetables to CSV."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'service_line', 'platform', 'scheduled_time', 'actual_time',
                'passenger_count', 'journey_type', 'delay_minutes'
            ])
            
            for service_line, timetables in self.timetables.items():
                for timetable in timetables:
                    writer.writerow([
                        service_line,
                        timetable.platform,
                        timetable.scheduled_time.isoformat(),
                        timetable.actual_time.isoformat(),
                        timetable.passenger_count,
                        timetable.journey_type,
                        timetable.delay_minutes
                    ])
    
    def _export_coordinate_series_csv(self, file_path: Path, coordinate_series: List[Dict]):
        """Export coordinate series to CSV."""
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
    
    def _export_agent_trajectories(self, file_path: Path):
        """Export agent trajectories for Mass Motion."""
        trajectories = []
        
        for agent in self.agents:
            trajectory = {
                'agent_id': agent.agent_id,
                'journey_id': agent.journey_id,
                'start_time': agent.start_time.isoformat(),
                'end_time': agent.end_time.isoformat(),
                'journey_type': agent.journey_type,
                'service_line': agent.service_line,
                'confidence_score': agent.confidence_score,
                'trajectory': [
                    {
                        'timestamp': ts.isoformat(),
                        'position': pos,
                        'activity': activity
                    }
                    for ts, pos, activity in zip(agent.timestamps, agent.path_points, agent.activities)
                ]
            }
            trajectories.append(trajectory)
        
        with open(file_path, 'w') as f:
            json.dump(trajectories, f, indent=2, default=str)
    
    def _export_zone_states(self, file_path: Path):
        """Export zone states for Mass Motion."""
        zone_states = {}
        
        for zone_id, zone in self.zones.items():
            # Calculate current occupancy based on agents
            current_agents = [agent for agent in self.agents 
                            if agent.origin_zone == zone_id or agent.destination_zone == zone_id]
            
            zone_state = {
                'zone_id': zone_id,
                'name': zone.name,
                'type': zone.zone_type,
                'area_sqm': zone.area_sqm,
                'capacity': zone.capacity,
                'current_occupancy': len(current_agents),
                'occupancy_rate': len(current_agents) / zone.capacity if zone.capacity > 0 else 0,
                'polygon': zone.polygon,
                'connections': zone.connections,
                'flow_gates': zone.flow_gates
            }
            zone_states[zone_id] = zone_state
        
        with open(file_path, 'w') as f:
            json.dump(zone_states, f, indent=2, default=str)
    
    def _export_flow_gate_states(self, file_path: Path):
        """Export flow gate states for Mass Motion."""
        flow_gate_states = {}
        
        for gate_id, gate in self.flow_gates.items():
            # Calculate current flow rate based on agents passing through
            passing_agents = [agent for agent in self.agents 
                            if (agent.origin_zone == gate.zone_from and agent.destination_zone == gate.zone_to) or
                               (agent.origin_zone == gate.zone_to and agent.destination_zone == gate.zone_from)]
            
            flow_gate_state = {
                'gate_id': gate_id,
                'name': gate.name,
                'zone_from': gate.zone_from,
                'zone_to': gate.zone_to,
                'width_m': gate.width_m,
                'capacity_per_minute': gate.capacity_per_minute,
                'current_flow_rate': len(passing_agents),
                'max_flow_rate': gate.max_flow_rate,
                'utilization_rate': len(passing_agents) / gate.max_flow_rate if gate.max_flow_rate > 0 else 0,
                'position': gate.position
            }
            flow_gate_states[gate_id] = flow_gate_state
        
        with open(file_path, 'w') as f:
            json.dump(flow_gate_states, f, indent=2, default=str)
    
    def _export_service_line_data(self, file_path: Path):
        """Export service line data for Mass Motion."""
        service_line_data = {}
        
        for service_line, timetables in self.timetables.items():
            # Calculate service line statistics
            total_passengers = sum(timetable.passenger_count for timetable in timetables)
            average_delay = np.mean([timetable.delay_minutes for timetable in timetables])
            journey_types = [timetable.journey_type for timetable in timetables]
            
            service_line_info = {
                'service_line': service_line,
                'total_passengers': total_passengers,
                'total_services': len(timetables),
                'average_delay_minutes': average_delay,
                'journey_type_distribution': {
                    journey_type: journey_types.count(journey_type)
                    for journey_type in set(journey_types)
                },
                'timetables': [timetable.__dict__ for timetable in timetables]
            }
            service_line_data[service_line] = service_line_info
        
        with open(file_path, 'w') as f:
            json.dump(service_line_data, f, indent=2, default=str)