"""Utility functions for pedestrian detection improvements."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Detection:
    """Detection object for pedestrians."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: np.ndarray
    center: Tuple[int, int]
    confidence: float
    track_id: Optional[int] = None


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                 bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: (x1, y1, x2, y2) first bounding box
        bbox2: (x1, y1, x2, y2) second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    # Bboxes are already in (x_min, y_min, x_max, y_max) format
    box1 = bbox1
    box2 = bbox2
    
    # Calculate intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def apply_nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for suppression (default: 0.5)
        
    Returns:
        Filtered list of Detection objects
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    
    while sorted_detections:
        # Keep highest confidence detection
        current = sorted_detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        sorted_detections = [
            det for det in sorted_detections
            if calculate_iou(current.bbox, det.bbox) < iou_threshold
        ]
    
    return keep


def apply_context_aware_nms(detections: List[Detection], 
                            base_iou_threshold: float = 0.5,
                            adaptive_iou: Optional[float] = None,
                            confidence_weighted: bool = False,
                            scale_aware: bool = False,
                            image_size: Optional[Tuple[int, int]] = None) -> List[Detection]:
    """
    Apply context-aware Non-Maximum Suppression with adaptive thresholds.
    
    Args:
        detections: List of Detection objects
        base_iou_threshold: Base IoU threshold (default: 0.5)
        adaptive_iou: Optional adaptive IoU threshold from scene analysis
        confidence_weighted: If True, adjust IoU threshold based on confidence difference
        scale_aware: If True, use different thresholds for different detection scales
        image_size: (width, height) for scale-aware processing
        
    Returns:
        Filtered list of Detection objects
    """
    if not detections:
        return []
    
    # Use adaptive threshold if provided, otherwise use base
    iou_threshold = adaptive_iou if adaptive_iou is not None else base_iou_threshold
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    
    while sorted_detections:
        current = sorted_detections.pop(0)
        keep.append(current)
        
        remaining = []
        for det in sorted_detections:
            iou = calculate_iou(current.bbox, det.bbox)
            
            # Calculate adaptive threshold for this pair
            threshold = iou_threshold
            
            if confidence_weighted:
                # Adjust threshold based on confidence difference
                # Larger confidence gap -> more aggressive suppression
                conf_diff = abs(current.confidence - det.confidence)
                if conf_diff > 0.2:  # Significant confidence difference
                    threshold *= 0.8  # More aggressive suppression
                elif conf_diff < 0.05:  # Similar confidence
                    threshold *= 1.1  # Less aggressive (allow both)
            
            if scale_aware and image_size:
                # Adjust threshold based on detection sizes
                # Smaller detections (further away) need more lenient NMS
                current_h = current.bbox[3] - current.bbox[1]
                det_h = det.bbox[3] - det.bbox[1]
                avg_size = (current_h + det_h) / 2
                
                # Normalize by image height
                normalized_size = avg_size / image_size[1]
                if normalized_size < 0.1:  # Small detections (far away)
                    threshold *= 1.2  # More lenient
                elif normalized_size > 0.2:  # Large detections (close)
                    threshold *= 0.9  # More aggressive
            
            # Keep detection if IoU is below threshold
            if iou < threshold:
                remaining.append(det)
        
        sorted_detections = remaining
    
    return keep


def find_peaks(arr: np.ndarray, min_height: Optional[float] = None) -> List[int]:
    """
    Find peaks in a 1D array (for body proportion validation).
    
    Args:
        arr: 1D numpy array
        min_height: Minimum peak height (if None, uses 10% of max)
        
    Returns:
        List of peak indices
    """
    if len(arr) < 3:
        return []
    
    peaks = []
    if min_height is None:
        min_height = np.max(arr) * 0.1
    
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > min_height:
            peaks.append(i)
    
    return peaks

