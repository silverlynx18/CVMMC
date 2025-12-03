"""ONNX-based pedestrian detector using DirectML for AMD GPU acceleration."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import detection utils
from .detection_utils import apply_nms, Detection


class ONNXPedestrianDetector:
    """ONNX-based pedestrian detector with DirectML support for AMD GPU."""
    
    def __init__(self, onnx_model_path: str = "yolov8n.onnx",
                 confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.5,
                 use_directml: bool = True):
        """
        Initialize the ONNX detector.
        
        Args:
            onnx_model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            use_directml: Use DirectML provider for AMD GPU acceleration
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.onnx_model_path = onnx_model_path
        
        # Pedestrian detection parameters
        self.min_height = 50
        self.max_height = 300
        
        # Load ONNX model
        try:
            import onnxruntime as ort
            
            # Set up providers - DirectML only if requested
            if use_directml:
                providers = ['DmlExecutionProvider']
                logger.info("✅ Using DirectML provider for AMD GPU acceleration")
            else:
                providers = ['CPUExecutionProvider']
                logger.info("Using CPU provider")
            
            # Create inference session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            try:
                self.session = ort.InferenceSession(
                    onnx_model_path,
                    sess_options=sess_options,
                    providers=providers
                )
            except Exception as e:
                if use_directml and 'DmlExecutionProvider' in str(e):
                    raise RuntimeError(
                        f"DirectML provider not available. Error: {e}\n"
                        "Please check:\n"
                        "1. Is onnxruntime-directml installed? (pip install onnxruntime-directml)\n"
                        "2. Are your AMD GPU drivers up to date?\n"
                        "3. If you want to use CPU instead, set use_directml=False"
                    )
                raise
            
            # Verify provider is active
            active_providers = self.session.get_providers()
            if use_directml and 'DmlExecutionProvider' not in active_providers:
                raise RuntimeError(
                    f"DirectML provider requested but not active. Active providers: {active_providers}\n"
                    "GPU acceleration is not available. If you want to use CPU instead, set use_directml=False"
                )
            
            if 'DmlExecutionProvider' in active_providers:
                logger.info(f"✅ DirectML provider active - AMD GPU acceleration enabled!")
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (int(input_shape[2]), int(input_shape[3]))  # (width, height)
            logger.info(f"✅ ONNX model loaded: {onnx_model_path}")
            logger.info(f"   Input size: {self.input_size}")
            
        except ImportError:
            raise ImportError("onnxruntime-directml not installed. Install with: pip install onnxruntime-directml")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def detect_pedestrians(self, image: np.ndarray) -> List[Detection]:
        """
        Detect pedestrians in the given image using ONNX model.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of Detection objects (with NMS applied)
        """
        # Preprocess image
        input_image = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_image})
        
        # Parse outputs (YOLOv8 format: [batch, num_detections, 4+num_classes])
        detections = self._parse_outputs(outputs[0], image.shape[:2])
        
        # Apply NMS
        if detections:
            detections = apply_nms(detections, iou_threshold=self.nms_iou_threshold)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model input."""
        # Resize to model input size
        h, w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # Resize maintaining aspect ratio
        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)  # Gray padding
        padded[:new_h, :new_w] = resized
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        
        return input_tensor
    
    def _parse_outputs(self, outputs: np.ndarray, image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Parse ONNX model outputs to Detection objects.
        
        YOLOv8 output format: [batch, num_detections, 84]
        Where 84 = 4 (bbox coords) + 80 (classes) or similar structure
        """
        detections = []
        h, w = image_shape
        input_w, input_h = self.input_size
        
        # YOLOv8 output is typically [1, num_detections, 84]
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Calculate scale factors
        scale_x = w / input_w
        scale_y = h / input_h
        
        for detection in outputs:
            # YOLOv8 format: [x_center, y_center, width, height, class0_score, class1_score, ...]
            # Or sometimes: [x1, y1, x2, y2, conf, class0, class1, ...]
            
            # Try to parse - YOLOv8 usually has bbox in first 4 values
            if len(detection) >= 5:
                # Common format: [x_center, y_center, width, height, ...]
                x_center = float(detection[0]) * scale_x
                y_center = float(detection[1]) * scale_y
                bbox_w = float(detection[2]) * scale_x
                bbox_h = float(detection[3]) * scale_y
                
                # Convert center+size to x1,y1,x2,y2
                x1 = int(x_center - bbox_w / 2)
                y1 = int(y_center - bbox_h / 2)
                x2 = int(x_center + bbox_w / 2)
                y2 = int(y_center + bbox_h / 2)
                
                # Get confidence (class 0 = person)
                if len(detection) > 4:
                    # Check if 5th element is confidence or if we need to look at class scores
                    confidence = float(detection[4]) if detection[4] > 0.1 else 0.0
                    
                    # If it's class scores format, find person class (usually index 0)
                    if len(detection) > 84:
                        # Might be class logits, use max or person class score
                        confidence = max(float(detection[4]), float(detection[5])) if len(detection) > 5 else float(detection[4])
                
                # Filter by confidence and size
                if confidence >= self.confidence_threshold:
                    height = y2 - y1
                    if self.min_height <= height <= self.max_height:
                        # Create mask (bbox as mask)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
                        
                        detection_obj = Detection(
                            bbox=(x1, y1, x2, y2),
                            mask=mask,
                            center=(int(x_center), int(y_center)),
                            confidence=confidence
                        )
                        detections.append(detection_obj)
        
        return detections

