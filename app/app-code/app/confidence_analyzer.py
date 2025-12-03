"""Confidence analysis and quality assessment for pedestrian tracking data."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for tracking data."""
    detection_confidence: float
    tracking_consistency: float
    path_smoothness: float
    temporal_continuity: float
    spatial_accuracy: float
    overall_quality: float


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different quality levels."""
    high_confidence: float = 0.8
    medium_confidence: float = 0.6
    low_confidence: float = 0.4
    minimum_acceptable: float = 0.3


class ConfidenceAnalyzer:
    """Analyzes confidence and quality of pedestrian tracking data."""
    
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """
        Initialize confidence analyzer.
        
        Args:
            thresholds: Confidence thresholds for quality assessment
        """
        self.thresholds = thresholds or ConfidenceThresholds()
        self.quality_history = []
        
    def analyze_tracking_quality(self, tracked_pedestrians: List[Any], 
                               frame_timestamp: datetime) -> Dict[str, Any]:
        """
        Analyze quality of current tracking frame.
        
        Args:
            tracked_pedestrians: List of tracked pedestrians
            frame_timestamp: Current frame timestamp
            
        Returns:
            Dictionary with quality analysis results
        """
        if not tracked_pedestrians:
            return {
                'frame_timestamp': frame_timestamp.isoformat(),
                'total_pedestrians': 0,
                'quality_metrics': None,
                'confidence_distribution': {},
                'recommendations': ['No pedestrians detected in frame']
            }
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(tracked_pedestrians)
        
        # Analyze confidence distribution
        confidence_dist = self._analyze_confidence_distribution(tracked_pedestrians)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics, confidence_dist)
        
        # Store quality history
        self.quality_history.append({
            'timestamp': frame_timestamp,
            'quality_metrics': quality_metrics,
            'confidence_distribution': confidence_dist
        })
        
        return {
            'frame_timestamp': frame_timestamp.isoformat(),
            'total_pedestrians': len(tracked_pedestrians),
            'quality_metrics': quality_metrics.__dict__,
            'confidence_distribution': confidence_dist,
            'recommendations': recommendations
        }
    
    def _calculate_quality_metrics(self, tracked_pedestrians: List[Any]) -> QualityMetrics:
        """Calculate quality metrics for tracked pedestrians."""
        if not tracked_pedestrians:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Detection confidence
        detection_confidences = [ped.confidence for ped in tracked_pedestrians]
        detection_confidence = np.mean(detection_confidences)
        
        # Tracking consistency (based on bbox stability)
        tracking_consistency = self._calculate_tracking_consistency(tracked_pedestrians)
        
        # Path smoothness (based on movement patterns)
        path_smoothness = self._calculate_path_smoothness(tracked_pedestrians)
        
        # Temporal continuity (based on tracking duration)
        temporal_continuity = self._calculate_temporal_continuity(tracked_pedestrians)
        
        # Spatial accuracy (based on bbox quality)
        spatial_accuracy = self._calculate_spatial_accuracy(tracked_pedestrians)
        
        # Overall quality (weighted average)
        overall_quality = (
            detection_confidence * 0.3 +
            tracking_consistency * 0.25 +
            path_smoothness * 0.2 +
            temporal_continuity * 0.15 +
            spatial_accuracy * 0.1
        )
        
        return QualityMetrics(
            detection_confidence=detection_confidence,
            tracking_consistency=tracking_consistency,
            path_smoothness=path_smoothness,
            temporal_continuity=temporal_continuity,
            spatial_accuracy=spatial_accuracy,
            overall_quality=overall_quality
        )
    
    def _calculate_tracking_consistency(self, tracked_pedestrians: List[Any]) -> float:
        """Calculate tracking consistency based on bbox stability."""
        if not tracked_pedestrians:
            return 0.0
        
        consistency_scores = []
        
        for ped in tracked_pedestrians:
            if hasattr(ped, 'bbox_history') and len(ped.bbox_history) > 1:
                # Calculate bbox stability over time
                bbox_stability = self._calculate_bbox_stability(ped.bbox_history)
                consistency_scores.append(bbox_stability)
            else:
                # If no history, use current confidence as proxy
                consistency_scores.append(ped.confidence)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_bbox_stability(self, bbox_history: List[List[float]]) -> float:
        """Calculate stability of bounding box over time."""
        if len(bbox_history) < 2:
            return 0.0
        
        # Calculate center point stability
        centers = []
        for bbox in bbox_history:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        # Calculate movement variance
        movements = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        
        if not movements:
            return 0.0
        
        # Stability is inverse of movement variance
        movement_variance = np.var(movements)
        stability = max(0.0, 1.0 - min(1.0, movement_variance / 100.0))  # Normalize
        
        return stability
    
    def _calculate_path_smoothness(self, tracked_pedestrians: List[Any]) -> float:
        """Calculate path smoothness based on movement patterns."""
        if not tracked_pedestrians:
            return 0.0
        
        smoothness_scores = []
        
        for ped in tracked_pedestrians:
            if hasattr(ped, 'centroid_history') and len(ped.centroid_history) > 2:
                # Calculate path smoothness
                path_smoothness = self._calculate_path_smoothness_for_pedestrian(ped.centroid_history)
                smoothness_scores.append(path_smoothness)
            else:
                # If no history, assume moderate smoothness
                smoothness_scores.append(0.5)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def _calculate_path_smoothness_for_pedestrian(self, centroid_history: List[Tuple[float, float]]) -> float:
        """Calculate path smoothness for a single pedestrian."""
        if len(centroid_history) < 3:
            return 0.0
        
        # Calculate acceleration changes (jerk)
        accelerations = []
        for i in range(2, len(centroid_history)):
            # Calculate velocity
            v1 = (
                centroid_history[i][0] - centroid_history[i-1][0],
                centroid_history[i][1] - centroid_history[i-1][1]
            )
            v2 = (
                centroid_history[i-1][0] - centroid_history[i-2][0],
                centroid_history[i-1][1] - centroid_history[i-2][1]
            )
            
            # Calculate acceleration
            acc = (v1[0] - v2[0], v1[1] - v2[1])
            acc_magnitude = math.sqrt(acc[0]*acc[0] + acc[1]*acc[1])
            accelerations.append(acc_magnitude)
        
        if not accelerations:
            return 0.0
        
        # Smoothness is inverse of acceleration variance
        acc_variance = np.var(accelerations)
        smoothness = max(0.0, 1.0 - min(1.0, acc_variance / 10.0))  # Normalize
        
        return smoothness
    
    def _calculate_temporal_continuity(self, tracked_pedestrians: List[Any]) -> float:
        """Calculate temporal continuity based on tracking duration."""
        if not tracked_pedestrians:
            return 0.0
        
        continuity_scores = []
        
        for ped in tracked_pedestrians:
            if hasattr(ped, 'first_seen') and hasattr(ped, 'last_seen'):
                duration = (ped.last_seen - ped.first_seen).total_seconds()
                # Longer tracking duration indicates better continuity
                continuity = min(1.0, duration / 30.0)  # Normalize to 30 seconds
                continuity_scores.append(continuity)
            else:
                # If no duration info, assume low continuity
                continuity_scores.append(0.2)
        
        return np.mean(continuity_scores) if continuity_scores else 0.0
    
    def _calculate_spatial_accuracy(self, tracked_pedestrians: List[Any]) -> float:
        """Calculate spatial accuracy based on bbox quality."""
        if not tracked_pedestrians:
            return 0.0
        
        accuracy_scores = []
        
        for ped in tracked_pedestrians:
            if hasattr(ped, 'bbox'):
                # Calculate bbox quality based on size and aspect ratio
                bbox_quality = self._calculate_bbox_quality(ped.bbox)
                accuracy_scores.append(bbox_quality)
            else:
                accuracy_scores.append(0.0)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _calculate_bbox_quality(self, bbox: List[float]) -> float:
        """Calculate quality of bounding box."""
        if len(bbox) != 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return 0.0
        
        # Check for reasonable size (not too small, not too large)
        area = width * height
        if area < 100:  # Too small
            return 0.2
        elif area > 10000:  # Too large
            return 0.3
        
        # Check aspect ratio (should be roughly human-like)
        aspect_ratio = width / height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Unrealistic aspect ratio
            return 0.4
        
        # Good bbox quality
        return 0.8
    
    def _analyze_confidence_distribution(self, tracked_pedestrians: List[Any]) -> Dict[str, int]:
        """Analyze distribution of confidence scores."""
        if not tracked_pedestrians:
            return {}
        
        confidences = [ped.confidence for ped in tracked_pedestrians]
        
        distribution = {
            'high_confidence': sum(1 for c in confidences if c >= self.thresholds.high_confidence),
            'medium_confidence': sum(1 for c in confidences if self.thresholds.medium_confidence <= c < self.thresholds.high_confidence),
            'low_confidence': sum(1 for c in confidences if self.thresholds.low_confidence <= c < self.thresholds.medium_confidence),
            'very_low_confidence': sum(1 for c in confidences if c < self.thresholds.low_confidence),
            'total': len(confidences)
        }
        
        return distribution
    
    def _generate_recommendations(self, quality_metrics: QualityMetrics, 
                                confidence_dist: Dict[str, int]) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        # Overall quality recommendations
        if quality_metrics.overall_quality < self.thresholds.low_confidence:
            recommendations.append("Overall tracking quality is low. Consider adjusting detection parameters.")
        elif quality_metrics.overall_quality < self.thresholds.medium_confidence:
            recommendations.append("Tracking quality is moderate. Some improvements may be needed.")
        else:
            recommendations.append("Tracking quality is good.")
        
        # Specific metric recommendations
        if quality_metrics.detection_confidence < self.thresholds.medium_confidence:
            recommendations.append("Detection confidence is low. Consider improving lighting or camera angle.")
        
        if quality_metrics.tracking_consistency < self.thresholds.medium_confidence:
            recommendations.append("Tracking consistency is low. Consider adjusting tracking parameters.")
        
        if quality_metrics.path_smoothness < self.thresholds.medium_confidence:
            recommendations.append("Path smoothness is low. Consider applying smoothing filters.")
        
        if quality_metrics.temporal_continuity < self.thresholds.medium_confidence:
            recommendations.append("Temporal continuity is low. Consider improving tracking algorithm.")
        
        if quality_metrics.spatial_accuracy < self.thresholds.medium_confidence:
            recommendations.append("Spatial accuracy is low. Consider improving detection resolution.")
        
        # Confidence distribution recommendations
        if confidence_dist.get('very_low_confidence', 0) > confidence_dist.get('total', 0) * 0.3:
            recommendations.append("High number of very low confidence detections. Consider filtering or improving detection.")
        
        if confidence_dist.get('high_confidence', 0) < confidence_dist.get('total', 0) * 0.5:
            recommendations.append("Low number of high confidence detections. Consider improving detection quality.")
        
        return recommendations
    
    def get_quality_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Get quality trends over time."""
        if len(self.quality_history) < window_size:
            return {'trend': 'insufficient_data', 'recommendations': []}
        
        # Get recent quality metrics
        recent_metrics = [entry['quality_metrics'] for entry in self.quality_history[-window_size:]]
        
        # Calculate trends
        overall_qualities = [m.overall_quality for m in recent_metrics]
        detection_confidences = [m.detection_confidence for m in recent_metrics]
        tracking_consistencies = [m.tracking_consistency for m in recent_metrics]
        
        # Calculate trend direction
        overall_trend = self._calculate_trend_direction(overall_qualities)
        detection_trend = self._calculate_trend_direction(detection_confidences)
        tracking_trend = self._calculate_trend_direction(tracking_consistencies)
        
        # Generate trend recommendations
        trend_recommendations = []
        
        if overall_trend == 'declining':
            trend_recommendations.append("Overall quality is declining. Check system performance.")
        elif overall_trend == 'improving':
            trend_recommendations.append("Overall quality is improving. System is performing well.")
        
        if detection_trend == 'declining':
            trend_recommendations.append("Detection confidence is declining. Check lighting conditions.")
        
        if tracking_trend == 'declining':
            trend_recommendations.append("Tracking consistency is declining. Check tracking parameters.")
        
        return {
            'trend': overall_trend,
            'overall_quality_trend': overall_trend,
            'detection_confidence_trend': detection_trend,
            'tracking_consistency_trend': tracking_trend,
            'recent_average_quality': np.mean(overall_qualities),
            'recommendations': trend_recommendations
        }
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'stable'
        
        # Calculate slope using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 'stable'
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of confidence analysis."""
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No quality data available'}
        
        # Calculate overall statistics
        all_qualities = [entry['quality_metrics'].overall_quality for entry in self.quality_history]
        all_confidences = []
        for entry in self.quality_history:
            for ped in entry.get('tracked_pedestrians', []):
                if hasattr(ped, 'confidence'):
                    all_confidences.append(ped.confidence)
        
        if not all_confidences:
            all_confidences = [entry['quality_metrics'].detection_confidence for entry in self.quality_history]
        
        return {
            'status': 'data_available',
            'total_frames_analyzed': len(self.quality_history),
            'average_overall_quality': np.mean(all_qualities),
            'average_detection_confidence': np.mean(all_confidences),
            'quality_std': np.std(all_qualities),
            'confidence_std': np.std(all_confidences),
            'high_quality_frames': sum(1 for q in all_qualities if q >= self.thresholds.high_confidence),
            'low_quality_frames': sum(1 for q in all_qualities if q < self.thresholds.low_confidence),
            'data_quality_score': np.mean(all_qualities)
        }