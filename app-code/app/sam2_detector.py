"""Segment Anything Model 2 integration for pedestrian detection."""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from ultralytics import YOLO
import logging
import hashlib

from .detection_utils import apply_nms, apply_context_aware_nms, find_peaks
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .detection_tuner import DetectionTuner

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Pedestrian detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    mask: np.ndarray
    confidence: float
    track_id: Optional[int] = None


class SAM2PedestrianDetector:
    """Pedestrian detector using Segment Anything Model 2 with YOLO-first pipeline."""
    
    def __init__(self, model_path: str, device: str = "cuda", 
                 yolo_model_path: Optional[str] = None,
                 use_yolo_first: bool = True,
                 cache_embeddings: bool = True,
                 detection_tuner: Optional['DetectionTuner'] = None):
        """
        Initialize the SAM2 detector.
        
        Args:
            model_path: Path to SAM2 model file
            device: Device to run inference on ("cuda" or "cpu")
            yolo_model_path: Path to YOLOv8 model (default: "yolov8n.pt")
            use_yolo_first: Use YOLO as first-pass detector (much faster)
            cache_embeddings: Cache SAM embeddings for similar frames
            detection_tuner: Optional DetectionTuner for adaptive parameters
        """
        self.device = device
        self.model_path = model_path
        self.use_yolo_first = use_yolo_first
        self.cache_embeddings = cache_embeddings
        self.detection_tuner = detection_tuner
        self.sam = None
        self.predictor = None
        self.yolo_model = None
        self.transform = ResizeLongestSide(1024)
        
        # Embedding cache for similar frames
        self.last_image_hash = None
        self.last_embedding_set = False
        
        # Load models
        self._load_model()
        
        # Load YOLO if using YOLO-first pipeline
        if self.use_yolo_first:
            yolo_path = yolo_model_path or "yolov8n.pt"
            try:
                self.yolo_model = YOLO(yolo_path)
                logger.info(f"YOLO model loaded from {yolo_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Falling back to grid-based detection.")
                self.use_yolo_first = False
        
        # Pedestrian detection parameters
        self.base_confidence_threshold = 0.5
        self.base_nms_iou_threshold = 0.5
        self.min_height = 50
        self.max_height = 300
        self.confidence_threshold = 0.5
        self.nms_iou_threshold = 0.5
        self.use_adaptive_nms = detection_tuner is not None
        
    def _load_model(self):
        """Load the SAM2 model."""
        try:
            # Load SAM2 model (using vit_h architecture)
            sam = sam_model_registry["vit_h"](checkpoint=self.model_path)
            sam.to(device=self.device)
            self.sam = sam
            self.predictor = SamPredictor(sam)
            logger.info(f"SAM2 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise
    
    def detect_pedestrians(self, image: np.ndarray, 
                          prompt_points: Optional[List[Tuple[int, int]]] = None,
                          prompt_labels: Optional[List[int]] = None) -> List[Detection]:
        """
        Detect pedestrians in the given image using optimized pipeline.
        
        Args:
            image: Input image (BGR format)
            prompt_points: Optional prompt points for guided detection
            prompt_labels: Optional labels for prompt points (1 for foreground, 0 for background)
            
        Returns:
            List of Detection objects (with NMS applied)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        # Update adaptive parameters if tuner is available
        if self.detection_tuner is not None:
            params = self.detection_tuner.get_current_parameters()
            self.confidence_threshold = params.confidence_threshold
            self.nms_iou_threshold = params.nms_iou_threshold
            self.min_height = params.min_height
            self.max_height = params.max_height
        
        # Generate prompt points if not provided
        if prompt_points is None:
            if self.use_yolo_first and self.yolo_model is not None:
                # Use YOLO-first approach (much faster)
                prompt_points, prompt_labels = self._generate_yolo_prompts(image)
            else:
                # Fallback to grid-based (slower)
                prompt_points, prompt_labels = self._generate_grid_prompts(image)
        
        # Check if we can reuse embedding from previous frame
        should_set_image = True
        if self.cache_embeddings:
            image_hash = self._hash_image(image)
            if image_hash == self.last_image_hash and self.last_embedding_set:
                should_set_image = False
            else:
                self.last_image_hash = image_hash
        
        # Set image for prediction (only if needed)
        if should_set_image:
            self.predictor.set_image(image)
            self.last_embedding_set = True
        
        # Get masks and scores
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(prompt_points),
            point_labels=np.array(prompt_labels),
            multimask_output=True
        )
        
        # Filter and process detections
        detections = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score < self.confidence_threshold:
                continue
                
            # Get bounding box from mask
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue
                
            # Filter by pedestrian size
            x, y, w, h = bbox
            if not (self.min_height <= h <= self.max_height):
                continue
            
            # Enhanced pedestrian validation
            if self._is_pedestrian_like(mask, bbox):
                detection = Detection(
                    bbox=bbox,
                    mask=mask,
                    confidence=float(score)
                )
                detections.append(detection)
        
        # Update tuner with detections for next frame analysis
        if self.detection_tuner is not None:
            self.detection_tuner.update_parameters(detections, image)
        
        # Apply Non-Maximum Suppression (context-aware if tuner available)
        if detections:
            if self.use_adaptive_nms and self.detection_tuner is not None:
                params = self.detection_tuner.get_current_parameters()
                detections = apply_context_aware_nms(
                    detections,
                    base_iou_threshold=self.base_nms_iou_threshold,
                    adaptive_iou=params.nms_iou_threshold,
                    confidence_weighted=True,
                    scale_aware=True,
                    image_size=(image.shape[1], image.shape[0])
                )
            else:
                detections = apply_nms(detections, self.nms_iou_threshold)
        
        return detections
    
    def _generate_yolo_prompts(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Generate prompt points using YOLO detection (much faster than grid).
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (prompt_points, prompt_labels)
        """
        if self.yolo_model is None:
            return self._generate_grid_prompts(image)
        
        # Run YOLO detection
        results = self.yolo_model(image, classes=[0], verbose=False, conf=0.25)  # class 0 = person
        
        prompt_points = []
        prompt_labels = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    # Use center of bounding box as prompt point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    prompt_points.append((center_x, center_y))
                    prompt_labels.append(1)  # Foreground point
        
        # If no YOLO detections, fallback to grid
        if not prompt_points:
            logger.debug("No YOLO detections, falling back to grid prompts")
            return self._generate_grid_prompts(image)
        
        logger.debug(f"Generated {len(prompt_points)} YOLO-based prompts")
        return prompt_points, prompt_labels
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Generate hash of image for caching."""
        # Use a simple hash of image dimensions and mean values
        h, w = image.shape[:2]
        mean_val = np.mean(image)
        hash_input = f"{h}_{w}_{mean_val:.2f}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _generate_grid_prompts(self, image: np.ndarray, 
                              grid_size: int = 20) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Generate grid-based prompt points for detection."""
        h, w = image.shape[:2]
        points = []
        labels = []
        
        # Create a grid of points
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                points.append((j, i))
                labels.append(1)  # All points as foreground prompts
        
        return points, labels
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask to bounding box."""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _is_pedestrian_like(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Enhanced validation if detection looks like a pedestrian.
        
        Args:
            mask: Segmentation mask
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            True if detection appears to be a pedestrian
        """
        x, y, w, h = bbox
        
        # 1. Check aspect ratio (more strict: typical human 1.8-2.5)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 1.5 or aspect_ratio > 3.5:
            return False
        
        # 2. Check size reasonableness
        area = w * h
        if area < 500 or area > 50000:  # Too small or too large
            return False
        
        # 3. Check mask density (higher threshold)
        # Ensure bbox is within mask bounds
        mask_h, mask_w = mask.shape[:2]
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(mask_w, x + w)
        y_max = min(mask_h, y + h)
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        mask_roi = mask[y_min:y_max, x_min:x_max]
        mask_area = np.sum(mask_roi)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        density = mask_area / bbox_area if bbox_area > 0 else 0
        
        if density < 0.5:  # Increased from 0.3
            return False
        
        # 4. Body proportion check (head/torso/legs)
        # Check if mask has reasonable vertical distribution
        if mask_roi.size > 0 and mask_roi.shape[0] > 3:
            vertical_profile = np.sum(mask_roi, axis=1)
            peaks = find_peaks(vertical_profile, min_height=np.max(vertical_profile) * 0.2)
            
            # Should have at least 2 peaks (head and body, or body and legs)
            if len(peaks) < 2:
                # Allow single peak if it's well-distributed
                if np.std(vertical_profile) < np.mean(vertical_profile) * 0.3:
                    return False
        
        return True
    
    def segment_pedestrian(self, image: np.ndarray, 
                          point: Tuple[int, int]) -> Optional[Detection]:
        """
        Segment a specific pedestrian at the given point.
        
        Args:
            image: Input image
            point: (x, y) coordinates of the pedestrian
            
        Returns:
            Detection object or None
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        if len(masks) == 0:
            return None
        
        # Get the best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]
        
        if score < self.confidence_threshold:
            return None
        
        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            return None
        
        return Detection(
            bbox=bbox,
            mask=mask,
            confidence=float(score)
        )
    
    def update_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for detection."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def update_size_limits(self, min_height: int, max_height: int):
        """Update the size limits for pedestrian detection."""
        self.min_height = min_height
        self.max_height = max_height
        logger.info(f"Updated size limits: {min_height}-{max_height} pixels")