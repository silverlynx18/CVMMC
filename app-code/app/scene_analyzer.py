"""Scene analysis module for characterizing video scene conditions.

Provides detailed analysis of lighting, density, occlusion, and pedestrian
characteristics for detection tuning optimization.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import logging

from .detection_utils import Detection

logger = logging.getLogger(__name__)


@dataclass
class LightingAnalysis:
    """Lighting condition analysis."""
    overall_brightness: float  # 0.0 (dark) to 1.0 (bright)
    contrast_level: float  # 0.0 (low) to 1.0 (high)
    shadow_areas: float  # Proportion of frame in shadow
    overexposed_areas: float  # Proportion of frame overexposed
    is_daytime: bool  # Estimated day/night


@dataclass
class DensityAnalysis:
    """Pedestrian density analysis."""
    detections_per_frame: float
    detections_per_megapixel: float
    spatial_distribution: Dict[str, int]  # Distribution across frame regions
    crowding_level: str  # 'sparse', 'normal', 'crowded', 'very_crowded'


@dataclass
class OcclusionAnalysis:
    """Occlusion level analysis."""
    occlusion_level: float  # 0.0 (none) to 1.0 (high)
    partial_detection_ratio: float  # Ratio of partial detections
    overlap_ratio: float  # Average IoU between detections
    occlusion_hotspots: List[Tuple[int, int]]  # Regions with high occlusion


@dataclass
class SizeDistribution:
    """Pedestrian size distribution analysis."""
    mean_height: float
    std_height: float
    min_height: float
    max_height: float
    center_region_mean: float
    edge_region_mean: float
    size_variation_coefficient: float
    size_percentiles: Dict[str, float]  # 10th, 50th, 90th percentiles


@dataclass
class SceneAnalysis:
    """Comprehensive scene analysis."""
    timestamp: datetime
    lighting: LightingAnalysis
    density: DensityAnalysis
    occlusion: OcclusionAnalysis
    size_distribution: SizeDistribution
    recommendations: List[str]  # Recommendations for detection tuning


class SceneAnalyzer:
    """Analyzes scene characteristics for detection optimization."""
    
    def __init__(self, image_size: Tuple[int, int] = (1920, 1080),
                 gopro_fov_degrees: float = 120.0):
        """
        Initialize scene analyzer.
        
        Args:
            image_size: (width, height) of input frames
            gopro_fov_degrees: Field of view for GoPro wide-angle camera
        """
        self.image_size = image_size
        self.gopro_fov_degrees = gopro_fov_degrees
        self.frame_center = (image_size[0] / 2, image_size[1] / 2)
        self.frame_radius = np.sqrt(
            (image_size[0] / 2) ** 2 + (image_size[1] / 2) ** 2
        )
        
        # Define frame regions for spatial analysis
        w, h = image_size
        self.center_region = (w * 0.25, h * 0.25, w * 0.75, h * 0.75)
        self.edge_region_threshold = 0.7  # Normalized distance from center
        
        logger.info(f"SceneAnalyzer initialized: size={image_size}, FOV={gopro_fov_degrees}°")
    
    def analyze_lighting(self, frame: np.ndarray) -> LightingAnalysis:
        """
        Analyze lighting conditions in frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            LightingAnalysis object
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Overall brightness
        overall_brightness = np.mean(gray) / 255.0
        
        # Contrast level (standard deviation of intensities)
        contrast_level = np.std(gray) / 255.0
        
        # Shadow areas (pixels below threshold)
        shadow_threshold = 50
        shadow_areas = np.sum(gray < shadow_threshold) / gray.size
        
        # Overexposed areas (pixels above threshold)
        overexposed_threshold = 240
        overexposed_areas = np.sum(gray > overexposed_threshold) / gray.size
        
        # Estimate day/night (heuristic: brightness > 0.4 = daytime)
        is_daytime = overall_brightness > 0.4
        
        return LightingAnalysis(
            overall_brightness=float(overall_brightness),
            contrast_level=float(contrast_level),
            shadow_areas=float(shadow_areas),
            overexposed_areas=float(overexposed_areas),
            is_daytime=is_daytime
        )
    
    def analyze_density(self, detections: List[Detection]) -> DensityAnalysis:
        """
        Analyze pedestrian density in current frame.
        
        Args:
            detections: List of detections
            
        Returns:
            DensityAnalysis object
        """
        detections_per_frame = len(detections)
        
        # Normalize by frame area (megapixels)
        megapixels = (self.image_size[0] * self.image_size[1]) / 1000000
        detections_per_megapixel = detections_per_frame / megapixels
        
        # Spatial distribution across frame regions
        spatial_distribution = {
            'center': 0,
            'mid': 0,
            'edge': 0
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            dist_from_center = np.sqrt(
                (center_x - self.frame_center[0]) ** 2 +
                (center_y - self.frame_center[1]) ** 2
            ) / self.frame_radius
            
            if dist_from_center < 0.3:
                spatial_distribution['center'] += 1
            elif dist_from_center < 0.6:
                spatial_distribution['mid'] += 1
            else:
                spatial_distribution['edge'] += 1
        
        # Determine crowding level
        if detections_per_frame < 2:
            crowding_level = 'sparse'
        elif detections_per_frame < 5:
            crowding_level = 'normal'
        elif detections_per_frame < 10:
            crowding_level = 'crowded'
        else:
            crowding_level = 'very_crowded'
        
        return DensityAnalysis(
            detections_per_frame=float(detections_per_frame),
            detections_per_megapixel=float(detections_per_megapixel),
            spatial_distribution=spatial_distribution,
            crowding_level=crowding_level
        )
    
    def analyze_occlusion(self, detections: List[Detection]) -> OcclusionAnalysis:
        """
        Analyze occlusion levels in detections.
        
        Args:
            detections: List of detections
            
        Returns:
            OcclusionAnalysis object
        """
        if len(detections) < 2:
            return OcclusionAnalysis(
                occlusion_level=0.0,
                partial_detection_ratio=0.0,
                overlap_ratio=0.0,
                occlusion_hotspots=[]
            )
        
        # Calculate IoU between all detection pairs
        ious = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                iou = self._calculate_iou(det1.bbox, det2.bbox)
                ious.append(iou)
        
        overlap_ratio = np.mean(ious) if ious else 0.0
        
        # Estimate partial detections from confidence distribution
        # Low confidence with normal size suggests partial occlusion
        confidences = [d.confidence for d in detections]
        heights = [(d.bbox[3] - d.bbox[1]) for d in detections]
        
        if heights:
            mean_height = np.mean(heights)
            # Detections significantly smaller than average or low confidence
            partial_count = sum(
                1 for i, (conf, h) in enumerate(zip(confidences, heights))
                if conf < 0.4 or h < mean_height * 0.6
            )
            partial_detection_ratio = partial_count / len(detections)
        else:
            partial_detection_ratio = 0.0
        
        # Overall occlusion level
        occlusion_level = min(1.0, overlap_ratio * 2.0 + partial_detection_ratio)
        
        # Find occlusion hotspots (regions with many overlapping detections)
        occlusion_hotspots = []
        if overlap_ratio > 0.3:
            # Find regions with high detection density
            grid_size = 10
            grid_counts = np.zeros((grid_size, grid_size))
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                center_x = int((x1 + x2) / 2 / self.image_size[0] * grid_size)
                center_y = int((y1 + y2) / 2 / self.image_size[1] * grid_size)
                center_x = max(0, min(grid_size - 1, center_x))
                center_y = max(0, min(grid_size - 1, center_y))
                grid_counts[center_y, center_x] += 1
            
            # Find hotspots (cells with >2 detections)
            hotspot_coords = np.argwhere(grid_counts > 2)
            for y, x in hotspot_coords:
                occlusion_hotspots.append((
                    int(x * self.image_size[0] / grid_size),
                    int(y * self.image_size[1] / grid_size)
                ))
        
        return OcclusionAnalysis(
            occlusion_level=float(occlusion_level),
            partial_detection_ratio=float(partial_detection_ratio),
            overlap_ratio=float(overlap_ratio),
            occlusion_hotspots=occlusion_hotspots
        )
    
    def analyze_size_distribution(self, detections: List[Detection]) -> SizeDistribution:
        """
        Analyze pedestrian size distribution, accounting for GoPro wide-angle distortion.
        
        Args:
            detections: List of detections
            
        Returns:
            SizeDistribution object
        """
        if not detections:
            return SizeDistribution(
                mean_height=0.0,
                std_height=0.0,
                min_height=0.0,
                max_height=0.0,
                center_region_mean=0.0,
                edge_region_mean=0.0,
                size_variation_coefficient=0.0,
                size_percentiles={}
            )
        
        heights = []
        center_heights = []
        edge_heights = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height = y2 - y1
            
            heights.append(height)
            
            dist_from_center = np.sqrt(
                (center_x - self.frame_center[0]) ** 2 +
                (center_y - self.frame_center[1]) ** 2
            ) / self.frame_radius
            
            if dist_from_center < 0.3:
                center_heights.append(height)
            elif dist_from_center > 0.7:
                edge_heights.append(height)
        
        heights_array = np.array(heights)
        
        mean_height = float(np.mean(heights_array))
        std_height = float(np.std(heights_array))
        min_height = float(np.min(heights_array))
        max_height = float(np.max(heights_array))
        
        center_region_mean = float(np.mean(center_heights)) if center_heights else mean_height
        edge_region_mean = float(np.mean(edge_heights)) if edge_heights else mean_height * 0.7
        
        size_variation_coefficient = std_height / (mean_height + 1e-6)
        
        size_percentiles = {
            '10th': float(np.percentile(heights_array, 10)),
            '50th': float(np.percentile(heights_array, 50)),
            '90th': float(np.percentile(heights_array, 90))
        }
        
        return SizeDistribution(
            mean_height=mean_height,
            std_height=std_height,
            min_height=min_height,
            max_height=max_height,
            center_region_mean=center_region_mean,
            edge_region_mean=edge_region_mean,
            size_variation_coefficient=size_variation_coefficient,
            size_percentiles=size_percentiles
        )
    
    def analyze_comprehensive(self, detections: List[Detection],
                              frame: Optional[np.ndarray] = None) -> SceneAnalysis:
        """
        Perform comprehensive scene analysis.
        
        Args:
            detections: List of detections
            frame: Optional frame for lighting analysis
            
        Returns:
            SceneAnalysis object with recommendations
        """
        timestamp = datetime.now()
        
        # Perform all analyses
        lighting = self.analyze_lighting(frame) if frame is not None else LightingAnalysis(
            overall_brightness=0.5, contrast_level=0.5, shadow_areas=0.0,
            overexposed_areas=0.0, is_daytime=True
        )
        density = self.analyze_density(detections)
        occlusion = self.analyze_occlusion(detections)
        size_dist = self.analyze_size_distribution(detections)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            lighting, density, occlusion, size_dist
        )
        
        return SceneAnalysis(
            timestamp=timestamp,
            lighting=lighting,
            density=density,
            occlusion=occlusion,
            size_distribution=size_dist,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, lighting: LightingAnalysis,
                                  density: DensityAnalysis,
                                  occlusion: OcclusionAnalysis,
                                  size_dist: SizeDistribution) -> List[str]:
        """Generate tuning recommendations based on analysis."""
        recommendations = []
        
        # Lighting recommendations
        if lighting.overall_brightness < 0.3:
            recommendations.append("Low light detected - consider lowering confidence threshold")
        elif lighting.overall_brightness > 0.8:
            recommendations.append("Bright scene - can use higher confidence threshold")
        
        if lighting.shadow_areas > 0.3:
            recommendations.append("High shadow areas - may need shadow-robust detection")
        
        # Density recommendations
        if density.crowding_level == 'very_crowded':
            recommendations.append("Very crowded scene - consider stricter NMS to reduce overlaps")
        elif density.crowding_level == 'sparse':
            recommendations.append("Sparse scene - can use lower confidence threshold")
        
        # Occlusion recommendations
        if occlusion.occlusion_level > 0.5:
            recommendations.append("High occlusion detected - lower confidence threshold may help")
        
        # Size distribution recommendations (GoPro-specific)
        if size_dist.size_variation_coefficient > 0.6:
            recommendations.append("High size variation (wide-angle effect) - use adaptive size filtering")
        
        if size_dist.edge_region_mean < size_dist.center_region_mean * 0.6:
            recommendations.append("Significant size difference between center and edges - enable edge compensation")
        
        return recommendations
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

