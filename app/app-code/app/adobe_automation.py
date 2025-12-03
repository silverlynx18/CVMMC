"""
MassMotion-Enhanced Adobe Integration

Focuses on enhancing MassMotion exports for professional presentation,
leveraging MassMotion's built-in visualization capabilities while adding
Adobe-specific enhancements for client presentations.
"""

import os
import json
import csv
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class MassMotionAdobeEnhancer:
    """
    Enhances MassMotion exports for Adobe Creative Cloud presentation.
    
    Key Features:
    - MassMotion data export enhancement
    - Automated timeline markers from MassMotion analysis
    - Data-driven motion graphics using MassMotion metrics
    - Professional presentation templates
    - Non-intrusive, completely optional
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.template_dir = Path(__file__).parent / "massmotion_adobe_templates"
        self.template_dir.mkdir(exist_ok=True)
    
    def enhance_massmotion_export(self, 
                                 massmotion_data: Dict,
                                 analysis_results: Dict,
                                 output_dir: str,
                                 presentation_type: str = "client") -> Dict[str, str]:
        """
        Enhance MassMotion export for Adobe presentation.
        
        Args:
            massmotion_data: MassMotion analysis data
            analysis_results: Your analysis results
            output_dir: Output directory
            presentation_type: 'client', 'technical', or 'executive'
        
        Returns:
            Dict with paths to enhanced files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(analysis_results.get('video_file', {}).get('path', 'analysis')).stem
            created_files = {}
            
            # 1. Create enhanced MassMotion CSV for Adobe
            enhanced_csv = self._create_enhanced_massmotion_csv(
                massmotion_data, analysis_results, output_path, base_name
            )
            created_files['enhanced_massmotion_csv'] = enhanced_csv
            
            # 2. Create MassMotion timeline markers
            markers_file = self._create_massmotion_markers(
                massmotion_data, analysis_results, output_path, base_name
            )
            created_files['massmotion_markers'] = markers_file
            
            # 3. Create presentation templates based on MassMotion data
            template_file = self._create_presentation_template(
                massmotion_data, analysis_results, output_path, base_name, presentation_type
            )
            created_files['presentation_template'] = template_file
            
            # 4. Create MassMotion visualization overlays
            overlay_file = self._create_massmotion_overlays(
                massmotion_data, analysis_results, output_path, base_name
            )
            created_files['massmotion_overlays'] = overlay_file
            
            # 5. Create Adobe automation script for MassMotion data
            automation_script = self._create_massmotion_automation(
                massmotion_data, analysis_results, output_path, base_name
            )
            created_files['automation_script'] = automation_script
            
            logger.info(f"Enhanced MassMotion export for Adobe: {output_dir}")
            return created_files
            
        except Exception as e:
            logger.error(f"Failed to enhance MassMotion export: {e}")
            return {}
    
    def _create_enhanced_massmotion_csv(self, massmotion_data: Dict, analysis_results: Dict,
                                      output_path: Path, base_name: str) -> str:
        """Create enhanced CSV combining MassMotion data with analysis results."""
        
        csv_file = output_path / f"{base_name}_massmotion_enhanced.csv"
        
        # Combine MassMotion metrics with your analysis
        enhanced_data = []
        
        # MassMotion core metrics
        if 'agents' in massmotion_data:
            for agent in massmotion_data['agents']:
                enhanced_data.append({
                    'metric_type': 'agent_journey',
                    'metric_name': f'Agent {agent.get("agent_id", "unknown")}',
                    'value': agent.get('journey_duration', 0),
                    'unit': 'seconds',
                    'category': 'MassMotion Journey',
                    'origin_zone': agent.get('origin_zone', ''),
                    'destination_zone': agent.get('destination_zone', ''),
                    'service_line': agent.get('service_line', ''),
                    'confidence': agent.get('confidence_score', 0)
                })
        
        # MassMotion zone data
        if 'zones' in massmotion_data:
            for zone_id, zone_data in massmotion_data['zones'].items():
                enhanced_data.append({
                    'metric_type': 'zone_occupancy',
                    'metric_name': f'Zone {zone_id} Occupancy',
                    'value': zone_data.get('current_occupancy', 0),
                    'unit': 'people',
                    'category': 'MassMotion Zone',
                    'zone_capacity': zone_data.get('capacity', 0),
                    'occupancy_rate': zone_data.get('occupancy_rate', 0),
                    'peak_occupancy': zone_data.get('peak_occupancy', 0)
                })
        
        # MassMotion flow gates
        if 'flow_gates' in massmotion_data:
            for gate_id, gate_data in massmotion_data['flow_gates'].items():
                enhanced_data.append({
                    'metric_type': 'flow_gate',
                    'metric_name': f'Gate {gate_id} Flow',
                    'value': gate_data.get('flow_rate', 0),
                    'unit': 'people/hour',
                    'category': 'MassMotion Flow',
                    'direction': gate_data.get('direction', ''),
                    'peak_flow': gate_data.get('peak_flow', 0),
                    'average_flow': gate_data.get('average_flow', 0)
                })
        
        # Your analysis results integrated with MassMotion
        enhanced_data.append({
            'metric_type': 'analysis_summary',
            'metric_name': 'Total Pedestrians Detected',
            'value': analysis_results.get('total_detections', 0),
            'unit': 'people',
            'category': 'Analysis Results',
            'massmotion_agents': len(massmotion_data.get('agents', [])),
            'detection_accuracy': 'High'
        })
        
        enhanced_data.append({
            'metric_type': 'analysis_summary',
            'metric_name': 'Average Density',
            'value': analysis_results.get('avg_density', 0),
            'unit': 'peds/m²',
            'category': 'Analysis Results',
            'massmotion_zones': len(massmotion_data.get('zones', {})),
            'fruin_los': analysis_results.get('avg_los', 'N/A')
        })
        
        # Write enhanced CSV
        fieldnames = [
            'metric_type', 'metric_name', 'value', 'unit', 'category',
            'origin_zone', 'destination_zone', 'service_line', 'confidence',
            'zone_capacity', 'occupancy_rate', 'peak_occupancy',
            'direction', 'peak_flow', 'average_flow',
            'massmotion_agents', 'detection_accuracy', 'massmotion_zones', 'fruin_los'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_data)
        
        logger.info(f"Created enhanced MassMotion CSV: {csv_file}")
        return str(csv_file)
    
    def _create_massmotion_markers(self, massmotion_data: Dict, analysis_results: Dict,
                                 output_path: Path, base_name: str) -> str:
        """Create timeline markers based on MassMotion analysis."""
        
        markers_file = output_path / f"{base_name}_massmotion_markers.csv"
        
        markers = []
        
        # MassMotion agent journey markers
        if 'agents' in massmotion_data:
            for agent in massmotion_data['agents']:
                start_time = agent.get('start_time', 0)
                markers.append({
                    'time': start_time,
                    'name': f"Journey Start - Agent {agent.get('agent_id', 'unknown')}",
                    'type': 'massmotion_journey',
                    'color': 'blue',
                    'description': f"From {agent.get('origin_zone', 'unknown')} to {agent.get('destination_zone', 'unknown')}"
                })
        
        # MassMotion zone occupancy peaks
        if 'zones' in massmotion_data:
            for zone_id, zone_data in massmotion_data['zones'].items():
                peak_time = zone_data.get('peak_occupancy_time', 0)
                if peak_time > 0:
                    markers.append({
                        'time': peak_time,
                        'name': f"Peak Occupancy - Zone {zone_id}",
                        'type': 'massmotion_peak',
                        'color': 'red',
                        'description': f"Peak occupancy: {zone_data.get('peak_occupancy', 0)} people"
                    })
        
        # MassMotion flow gate peaks
        if 'flow_gates' in massmotion_data:
            for gate_id, gate_data in massmotion_data['flow_gates'].items():
                peak_time = gate_data.get('peak_flow_time', 0)
                if peak_time > 0:
                    markers.append({
                        'time': peak_time,
                        'name': f"Peak Flow - Gate {gate_id}",
                        'type': 'massmotion_flow',
                        'color': 'green',
                        'description': f"Peak flow: {gate_data.get('peak_flow', 0)} people/hour"
                    })
        
        # Analysis results markers
        frame_results = analysis_results.get('frame_results', [])
        avg_density = analysis_results.get('avg_density', 0)
        
        for i, frame_data in enumerate(frame_results):
            density = frame_data.get('density', 0)
            
            # Mark high density periods
            if density > avg_density * 1.5:
                markers.append({
                    'time': i * 5,  # 5-second intervals
                    'name': f"High Density: {density:.2f} peds/m²",
                    'type': 'analysis_peak',
                    'color': 'orange',
                    'description': f'Density {density:.2f} peds/m² (avg: {avg_density:.2f})'
                })
        
        # Summary markers
        markers.append({
            'time': 0,
            'name': 'MassMotion Analysis Start',
            'type': 'summary',
            'color': 'purple',
            'description': f"Total agents: {len(massmotion_data.get('agents', []))}, Zones: {len(massmotion_data.get('zones', {}))}"
        })
        
        # Write markers CSV
        with open(markers_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'name', 'type', 'color', 'description'])
            writer.writeheader()
            writer.writerows(markers)
        
        logger.info(f"Created MassMotion markers: {markers_file}")
        return str(markers_file)
    
    def _create_presentation_template(self, massmotion_data: Dict, analysis_results: Dict,
                                   output_path: Path, base_name: str, presentation_type: str) -> str:
        """Create presentation template based on MassMotion data."""
        
        template_file = output_path / f"{base_name}_presentation_template.json"
        
        # Template configuration based on presentation type
        if presentation_type == "client":
            template_config = {
                'template_name': f'{base_name}_client_presentation',
                'presentation_type': 'client',
                'composition_settings': {
                    'width': 1920,
                    'height': 1080,
                    'frame_rate': 30,
                    'duration': analysis_results.get('video_file', {}).get('duration', 0)
                },
                'overlay_elements': [
                    {
                        'name': 'massmotion_dashboard',
                        'type': 'dashboard',
                        'position': {'x': 50, 'y': 50},
                        'style': {
                            'width': 500,
                            'height': 400,
                            'background': 'rgba(0,0,0,0.9)',
                            'border_radius': 15,
                            'border': '2px solid #00FFFF'
                        },
                        'children': [
                            {
                                'name': 'title',
                                'type': 'text',
                                'text': f'{base_name.replace("_", " ").title()} Analysis',
                                'style': {'font_size': 32, 'color': '#00FFFF', 'font_weight': 'bold'}
                            },
                            {
                                'name': 'massmotion_summary',
                                'type': 'text',
                                'text': f'MassMotion Agents: {len(massmotion_data.get("agents", []))}',
                                'style': {'font_size': 24, 'color': '#FFFFFF'}
                            },
                            {
                                'name': 'zone_summary',
                                'type': 'text',
                                'text': f'Active Zones: {len(massmotion_data.get("zones", {}))}',
                                'style': {'font_size': 24, 'color': '#FFFFFF'}
                            }
                        ]
                    }
                ]
            }
        elif presentation_type == "technical":
            template_config = {
                'template_name': f'{base_name}_technical_presentation',
                'presentation_type': 'technical',
                'composition_settings': {
                    'width': 1920,
                    'height': 1080,
                    'frame_rate': 30,
                    'duration': analysis_results.get('video_file', {}).get('duration', 0)
                },
                'overlay_elements': [
                    {
                        'name': 'technical_panel',
                        'type': 'panel',
                        'position': {'x': 50, 'y': 50},
                        'style': {
                            'width': 600,
                            'height': 500,
                            'background': 'rgba(0,0,0,0.8)',
                            'border_radius': 10
                        },
                        'children': [
                            {
                                'name': 'massmotion_metrics',
                                'type': 'metrics_grid',
                                'columns': 2,
                                'children': [
                                    {'name': 'total_agents', 'type': 'metric_card', 'data_source': 'MassMotion Agents'},
                                    {'name': 'active_zones', 'type': 'metric_card', 'data_source': 'Active Zones'},
                                    {'name': 'flow_gates', 'type': 'metric_card', 'data_source': 'Flow Gates'},
                                    {'name': 'peak_occupancy', 'type': 'metric_card', 'data_source': 'Peak Occupancy'}
                                ]
                            }
                        ]
                    }
                ]
            }
        else:  # executive
            template_config = {
                'template_name': f'{base_name}_executive_presentation',
                'presentation_type': 'executive',
                'composition_settings': {
                    'width': 1920,
                    'height': 1080,
                    'frame_rate': 30,
                    'duration': analysis_results.get('video_file', {}).get('duration', 0)
                },
                'overlay_elements': [
                    {
                        'name': 'executive_summary',
                        'type': 'summary_panel',
                        'position': {'x': 100, 'y': 100},
                        'style': {
                            'width': 800,
                            'height': 300,
                            'background': 'rgba(0,0,0,0.9)',
                            'border_radius': 20
                        },
                        'children': [
                            {
                                'name': 'key_findings',
                                'type': 'text',
                                'text': 'Key Findings',
                                'style': {'font_size': 36, 'color': '#FFFFFF', 'font_weight': 'bold'}
                            },
                            {
                                'name': 'massmotion_insights',
                                'type': 'text',
                                'text': f'MassMotion Analysis: {len(massmotion_data.get("agents", []))} pedestrian journeys tracked',
                                'style': {'font_size': 24, 'color': '#00FFFF'}
                            }
                        ]
                    }
                ]
            }
        
        # Add MassMotion-specific automation instructions
        template_config['massmotion_integration'] = {
            'instructions': [
                '1. Import MassMotion enhanced CSV into Premiere Pro',
                '2. Use Data-Driven Motion Graphics to link MassMotion data',
                '3. Apply MassMotion timeline markers',
                '4. Create visualizations based on MassMotion zones and flows',
                '5. Export professional presentation'
            ],
            'csv_file': f'{base_name}_massmotion_enhanced.csv',
            'markers_file': f'{base_name}_massmotion_markers.csv',
            'massmotion_version': 'Latest',
            'required_extensions': ['Data-Driven Motion Graphics']
        }
        
        with open(template_file, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        logger.info(f"Created presentation template: {template_file}")
        return str(template_file)
    
    def _create_massmotion_overlays(self, massmotion_data: Dict, analysis_results: Dict,
                                  output_path: Path, base_name: str) -> str:
        """Create MassMotion-specific overlay configurations."""
        
        overlay_file = output_path / f"{base_name}_massmotion_overlays.json"
        
        overlays = {
            'overlay_config': {
                'name': f'{base_name}_massmotion_overlays',
                'version': '1.0',
                'massmotion_integration': True
            },
            'overlay_elements': [
                {
                    'name': 'massmotion_agent_counter',
                    'type': 'counter',
                    'data_source': 'MassMotion Agents',
                    'position': {'x': 100, 'y': 100},
                    'style': {
                        'font_size': 48,
                        'color': '#00FFFF',
                        'background': 'rgba(0,0,0,0.7)',
                        'animation': 'count_up'
                    }
                },
                {
                    'name': 'zone_occupancy_gauge',
                    'type': 'gauge',
                    'data_source': 'Zone Occupancy',
                    'position': {'x': 100, 'y': 200},
                    'style': {
                        'width': 200,
                        'height': 200,
                        'color': '#00FF00',
                        'background': '#333333',
                        'animation': 'gauge_fill'
                    }
                },
                {
                    'name': 'flow_rate_chart',
                    'type': 'line_chart',
                    'data_source': 'Flow Gates',
                    'position': {'x': 400, 'y': 100},
                    'style': {
                        'width': 400,
                        'height': 200,
                        'color': '#FF00FF',
                        'background': 'rgba(0,0,0,0.8)',
                        'animation': 'line_draw'
                    }
                }
            ],
            'massmotion_specific': {
                'agent_visualization': {
                    'show_journey_paths': True,
                    'path_color': '#00FFFF',
                    'path_width': 3,
                    'path_opacity': 0.7
                },
                'zone_visualization': {
                    'show_occupancy_heatmap': True,
                    'heatmap_colors': ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
                    'heatmap_opacity': 0.5
                },
                'flow_visualization': {
                    'show_flow_arrows': True,
                    'arrow_color': '#FF00FF',
                    'arrow_size': 'dynamic',
                    'arrow_opacity': 0.8
                }
            }
        }
        
        with open(overlay_file, 'w') as f:
            json.dump(overlays, f, indent=2)
        
        logger.info(f"Created MassMotion overlays: {overlay_file}")
        return str(overlay_file)
    
    def _create_massmotion_automation(self, massmotion_data: Dict, analysis_results: Dict,
                                    output_path: Path, base_name: str) -> str:
        """Create automation script for MassMotion-enhanced Adobe workflow."""
        
        script_file = output_path / f"{base_name}_massmotion_automation.py"
        
        script_content = f"""
#!/usr/bin/env python3
\"\"\"
MassMotion-Enhanced Adobe Automation Script for {base_name}
Automates Adobe presentation creation using MassMotion data.
\"\"\"

import os
import json
import csv
from pathlib import Path

def automate_massmotion_adobe(project_path, video_path, massmotion_csv, markers_csv):
    \"\"\"
    Automate Adobe workflow using MassMotion data.
    
    This script provides automation instructions for:
    1. Creating MassMotion-enhanced presentations
    2. Using MassMotion data for motion graphics
    3. Applying MassMotion timeline markers
    4. Creating professional visualizations
    \"\"\"
    
    print(f"Automating MassMotion-Enhanced Adobe project: {{project_path}}")
    print(f"Video file: {{video_path}}")
    print(f"MassMotion CSV: {{massmotion_csv}}")
    print(f"MassMotion Markers: {{markers_csv}}")
    
    # MassMotion-specific automation instructions
    instructions = [
        "1. Open Premiere Pro and create new project",
        "2. Import video file and MassMotion enhanced CSV",
        "3. Create new sequence matching video properties",
        "4. Add video to timeline",
        "5. Use Data-Driven Motion Graphics with MassMotion data",
        "6. Link MassMotion metrics to overlay elements",
        "7. Import MassMotion timeline markers",
        "8. Create MassMotion zone visualizations",
        "9. Add MassMotion flow animations",
        "10. Export professional MassMotion presentation"
    ]
    
    for instruction in instructions:
        print(f"  {{instruction}}")
    
    # Create MassMotion automation log
    log_file = Path(project_path).parent / "massmotion_automation_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"MassMotion-Enhanced Adobe Automation Log\\n")
        f.write(f"Project: {{project_path}}\\n")
        f.write(f"Video: {{video_path}}\\n")
        f.write(f"MassMotion Data: {{massmotion_csv}}\\n")
        f.write(f"MassMotion Markers: {{markers_csv}}\\n\\n")
        
        f.write("MassMotion Integration Features:\\n")
        f.write("- Agent journey visualization\\n")
        f.write("- Zone occupancy heatmaps\\n")
        f.write("- Flow gate animations\\n")
        f.write("- Professional presentation templates\\n\\n")
        
        for instruction in instructions:
            f.write(f"{{instruction}}\\n")
    
    print(f"\\nMassMotion automation log saved to: {{log_file}}")

if __name__ == "__main__":
    # Configuration
    project_path = "{base_name}_massmotion_premiere_project.prproj"
    video_path = "{base_name}_analyzed.mp4"
    massmotion_csv = "{base_name}_massmotion_enhanced.csv"
    markers_csv = "{base_name}_massmotion_markers.csv"
    
    automate_massmotion_adobe(project_path, video_path, massmotion_csv, markers_csv)
"""
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        logger.info(f"Created MassMotion automation script: {script_file}")
        return str(script_file)


def enhance_massmotion_for_adobe(massmotion_data: Dict, analysis_results: Dict,
                                output_dir: str, presentation_type: str = "client") -> Dict[str, str]:
    """
    Convenience function to enhance MassMotion exports for Adobe presentation.
    
    Args:
        massmotion_data: MassMotion analysis data
        analysis_results: Your analysis results
        output_dir: Output directory
        presentation_type: 'client', 'technical', or 'executive'
    
    Returns:
        Dict with paths to enhanced files
    """
    enhancer = MassMotionAdobeEnhancer()
    
    # Enhance MassMotion export for Adobe
    enhanced_files = enhancer.enhance_massmotion_export(
        massmotion_data=massmotion_data,
        analysis_results=analysis_results,
        output_dir=output_dir,
        presentation_type=presentation_type
    )
    
    return enhanced_files
