"""Fruin's Level of Service analysis for pedestrian facilities."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .models import PedestrianCount, ServiceLevelAnalysis
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class FruinLOSResult:
    """Fruin's Level of Service analysis result."""
    level: str  # A, B, C, D, E, F
    density: float  # pedestrians per square meter
    description: str
    comfort_level: str
    flow_characteristics: str


class FruinAnalyzer:
    """Fruin's Level of Service analyzer for pedestrian facilities."""
    
    def __init__(self):
        """Initialize the Fruin analyzer."""
        self.thresholds = settings.fruin_thresholds
        self.density_calculation_area = settings.density_calculation_area
        
        # Fruin Level of Service descriptions
        self.los_descriptions = {
            "A": {
                "description": "Free flow - pedestrians can move freely without conflicts",
                "comfort": "Excellent",
                "flow": "Unimpeded movement, no queuing"
            },
            "B": {
                "description": "Reasonably free flow - minor conflicts possible",
                "comfort": "Good",
                "flow": "Slight reduction in walking speed"
            },
            "C": {
                "description": "Stable flow - some conflicts and queuing",
                "comfort": "Acceptable",
                "flow": "Moderate reduction in walking speed"
            },
            "D": {
                "description": "Approaching unstable flow - frequent conflicts",
                "comfort": "Poor",
                "flow": "Significant reduction in walking speed"
            },
            "E": {
                "description": "Unstable flow - frequent conflicts and queuing",
                "comfort": "Very Poor",
                "flow": "Severe reduction in walking speed"
            },
            "F": {
                "description": "Forced flow - breakdown of flow, queuing",
                "comfort": "Unacceptable",
                "flow": "Movement severely restricted"
            }
        }
    
    def calculate_density(self, pedestrian_count: int, area_sqm: float) -> float:
        """
        Calculate pedestrian density in pedestrians per square meter.
        
        Args:
            pedestrian_count: Number of pedestrians in the area
            area_sqm: Area in square meters
            
        Returns:
            Density in pedestrians per square meter
        """
        if area_sqm <= 0:
            return 0.0
        
        return pedestrian_count / area_sqm
    
    def calculate_density_from_pixels(self, pedestrian_count: int, 
                                    pixel_area: float) -> float:
        """
        Calculate density from pixel-based area measurements.
        
        Args:
            pedestrian_count: Number of pedestrians
            pixel_area: Area in pixels
            
        Returns:
            Density in pedestrians per square meter
        """
        area_sqm = pixel_area * self.density_calculation_area
        return self.calculate_density(pedestrian_count, area_sqm)
    
    def determine_los(self, density: float) -> FruinLOSResult:
        """
        Determine Fruin's Level of Service based on density.
        
        Args:
            density: Pedestrian density in pedestrians per square meter
            
        Returns:
            FruinLOSResult object
        """
        # Find the appropriate LOS level
        level = "F"  # Default to worst case
        for los_level in ["A", "B", "C", "D", "E", "F"]:
            if density <= self.thresholds[los_level]:
                level = los_level
                break
        
        # Get description for the level
        description_data = self.los_descriptions[level]
        
        return FruinLOSResult(
            level=level,
            density=density,
            description=description_data["description"],
            comfort_level=description_data["comfort"],
            flow_characteristics=description_data["flow"]
        )
    
    def analyze_zone(self, pedestrian_count: int, 
                    zone_area_sqm: float) -> FruinLOSResult:
        """
        Analyze a specific zone's Level of Service.
        
        Args:
            pedestrian_count: Number of pedestrians in the zone
            zone_area_sqm: Zone area in square meters
            
        Returns:
            FruinLOSResult for the zone
        """
        density = self.calculate_density(pedestrian_count, zone_area_sqm)
        return self.determine_los(density)
    
    def analyze_camera_view(self, pedestrian_count: int, 
                          camera_calibration: Dict) -> FruinLOSResult:
        """
        Analyze Level of Service for a camera view.
        
        Args:
            pedestrian_count: Number of pedestrians in the camera view
            camera_calibration: Camera calibration data with area information
            
        Returns:
            FruinLOSResult for the camera view
        """
        # Extract area information from calibration
        if 'area_sqm' in camera_calibration:
            area_sqm = camera_calibration['area_sqm']
        elif 'pixel_area' in camera_calibration:
            area_sqm = camera_calibration['pixel_area'] * self.density_calculation_area
        else:
            # Default area estimation based on camera resolution
            width_px = camera_calibration.get('width', 1920)
            height_px = camera_calibration.get('height', 1080)
            pixel_area = width_px * height_px
            area_sqm = pixel_area * self.density_calculation_area
        
        return self.analyze_zone(pedestrian_count, area_sqm)
    
    def analyze_time_period(self, counts: List[PedestrianCount], 
                          zone_area_sqm: float) -> Dict[str, any]:
        """
        Analyze Level of Service over a time period.
        
        Args:
            counts: List of pedestrian counts over time
            zone_area_sqm: Zone area in square meters
            
        Returns:
            Analysis results including average and peak LOS
        """
        if not counts:
            return {
                'avg_density': 0.0,
                'peak_density': 0.0,
                'avg_los': 'A',
                'peak_los': 'A',
                'total_ingress': 0,
                'total_egress': 0,
                'peak_hour_factor': 0.0
            }
        
        # Calculate densities and LOS for each count
        densities = []
        los_levels = []
        
        for count in counts:
            density = self.calculate_density(count.total_pedestrians, zone_area_sqm)
            los_result = self.determine_los(density)
            densities.append(density)
            los_levels.append(los_result.level)
        
        # Calculate statistics
        avg_density = np.mean(densities)
        peak_density = np.max(densities)
        avg_los = self.determine_los(avg_density).level
        peak_los = self.determine_los(peak_density).level
        
        # Calculate total movements
        total_ingress = sum(count.ingress_count for count in counts)
        total_egress = sum(count.egress_count for count in counts)
        
        # Calculate peak hour factor (ratio of peak 15-min rate to average rate)
        if len(counts) >= 4:  # Need at least 4 counts for 15-min analysis
            # Group counts into 15-minute periods
            counts_by_15min = {}
            for count in counts:
                # Round to nearest 15 minutes
                time_key = count.timestamp.replace(minute=(count.timestamp.minute // 15) * 15, 
                                                 second=0, microsecond=0)
                if time_key not in counts_by_15min:
                    counts_by_15min[time_key] = []
                counts_by_15min[time_key].append(count)
            
            # Calculate 15-minute rates
            rates_15min = []
            for time_key, period_counts in counts_by_15min.items():
                period_ingress = sum(c.ingress_count for c in period_counts)
                period_egress = sum(c.egress_count for c in period_counts)
                rates_15min.append(period_ingress + period_egress)
            
            if rates_15min:
                peak_15min_rate = max(rates_15min)
                avg_15min_rate = np.mean(rates_15min)
                peak_hour_factor = peak_15min_rate / avg_15min_rate if avg_15min_rate > 0 else 0.0
            else:
                peak_hour_factor = 0.0
        else:
            peak_hour_factor = 0.0
        
        return {
            'avg_density': float(avg_density),
            'peak_density': float(peak_density),
            'avg_los': avg_los,
            'peak_los': peak_los,
            'total_ingress': total_ingress,
            'total_egress': total_egress,
            'peak_hour_factor': float(peak_hour_factor)
        }
    
    def get_los_summary(self) -> Dict[str, Dict[str, str]]:
        """
        Get a summary of all Fruin Level of Service levels.
        
        Returns:
            Dictionary with LOS levels and their characteristics
        """
        summary = {}
        for level in ["A", "B", "C", "D", "E", "F"]:
            density_range = f"≤ {self.thresholds[level]:.1f}"
            if level != "F":
                next_level = chr(ord(level) + 1)
                density_range = f"{density_range} peds/m²"
            else:
                density_range = f"> {self.thresholds[level]:.1f} peds/m²"
            
            summary[level] = {
                'density_range': density_range,
                'description': self.los_descriptions[level]['description'],
                'comfort': self.los_descriptions[level]['comfort'],
                'flow': self.los_descriptions[level]['flow']
            }
        
        return summary
    
    def generate_alert_if_needed(self, los_result: FruinLOSResult, 
                               camera_id: int, zone_id: str) -> Optional[Dict]:
        """
        Generate an alert if the Level of Service is poor.
        
        Args:
            los_result: Fruin LOS analysis result
            camera_id: Camera ID
            zone_id: Zone ID
            
        Returns:
            Alert dictionary or None
        """
        # Alert thresholds
        alert_levels = {
            "D": "medium",
            "E": "high", 
            "F": "critical"
        }
        
        if los_result.level in alert_levels:
            return {
                'camera_id': camera_id,
                'zone_id': zone_id,
                'alert_type': 'density_high',
                'severity': alert_levels[los_result.level],
                'message': f"High pedestrian density detected: {los_result.level} "
                          f"({los_result.density:.2f} peds/m²) - {los_result.description}",
                'metadata': {
                    'density': los_result.density,
                    'los_level': los_result.level,
                    'comfort_level': los_result.comfort_level
                }
            }
        
        return None