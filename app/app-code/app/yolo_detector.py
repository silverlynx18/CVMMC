"""YOLO pedestrian detector with DirectML GPU support via ONNX Runtime."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
from .detection_utils import apply_nms, apply_context_aware_nms, Detection
from .path_resolver import resolve_model_path, get_models_dir
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .detection_tuner import DetectionTuner
    from .frame_preprocessor import FramePreprocessor

logger = logging.getLogger(__name__)


class YOLOPedestrianDetector:
    """YOLO pedestrian detector with DirectML GPU acceleration via ONNX."""
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt", 
                 device: str = "cpu",
                 confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.5,
                 detection_tuner: Optional['DetectionTuner'] = None,
                 frame_preprocessor: Optional['FramePreprocessor'] = None):
        """
        Initialize the YOLO detector.
        
        Args:
            yolo_model_path: Path to YOLOv8 model (default: "yolov8n.pt")
            device: Device to run inference on ("mps", "cuda", "directml", or "cpu")
            confidence_threshold: Minimum confidence for detections (base value)
            nms_iou_threshold: IoU threshold for NMS (base value)
            detection_tuner: Optional DetectionTuner for adaptive parameters
        """
        self.base_confidence_threshold = confidence_threshold
        self.base_nms_iou_threshold = nms_iou_threshold
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.detection_tuner = detection_tuner
        self.frame_preprocessor = frame_preprocessor
        
        # Pedestrian detection parameters
        self.min_height = 50
        self.max_height = 300
        self.use_adaptive_nms = detection_tuner is not None
        
        # Ensure model exists (auto-download if needed)
        # First, try to resolve using path_resolver
        model_name = Path(yolo_model_path).name
        resolved_path = resolve_model_path(model_name, check_exists=False)
        
        # If provided path is absolute and exists, use it; otherwise use resolved path
        provided_path = Path(yolo_model_path)
        if provided_path.is_absolute() and provided_path.exists():
            model_path = provided_path
            yolo_model_path = str(model_path)
        elif resolved_path and resolved_path.exists():
            model_path = resolved_path
            yolo_model_path = str(model_path)
            logger.info(f"Using model from: {model_path}")
        else:
            # Model not found - try to download
            logger.info(f"Model not found at {yolo_model_path} - checking for auto-download...")
            from app.setup_manager import SetupManager
            setup_manager = SetupManager()
            models_status = setup_manager.ensure_models()
            
            # Re-check resolved path after download attempt
            resolved_path = resolve_model_path(model_name, check_exists=False)
            if resolved_path and resolved_path.exists():
                model_path = resolved_path
                yolo_model_path = str(model_path)
                logger.info(f"Using model from: {model_path}")
            else:
                # Use original path - YOLO will try to download
                model_path = Path(yolo_model_path)
        
        # Handle DirectML (AMD GPU on Windows) - Try multiple methods
        if device == "directml":
            # Try method 1: ONNX Runtime with DirectML (preferred)
            try:
                self._use_onnx_directml = True
                self.yolo_model = None  # Won't use Ultralytics YOLO
                logger.info("✅ DirectML detected - attempting ONNX Runtime with DirectML for AMD GPU acceleration")
                self._load_onnx_directml(yolo_model_path)
                # Success - we're done
                return
            except Exception as e:
                logger.warning(f"Method 1 (ONNX DirectML) failed: {e}")
                logger.info("Trying alternative GPU method...")
                
                # Try method 2: torch-directml with Ultralytics YOLO
                try:
                    import torch_directml
                    if torch_directml.is_available():
                        self._use_onnx_directml = False
                        from ultralytics import YOLO
                        dml_device = torch_directml.device()
                        self.yolo_model = YOLO(yolo_model_path)
                        # Try to use DirectML device (may not work but worth trying)
                        self.device = dml_device
                        logger.info(f"✅ Using torch-directml with Ultralytics YOLO: {dml_device}")
                        logger.info("   Note: May fallback to CPU in inference if YOLO doesn't support DirectML device")
                        return
                except ImportError:
                    logger.warning("torch-directml not available for method 2")
                except Exception as e2:
                    logger.warning(f"Method 2 (torch-directml) failed: {e2}")
                
                # If both methods failed, ask user if they want CPU
                logger.error("="*80)
                logger.error("❌ ALL DIRECTML GPU METHODS FAILED")
                logger.error("="*80)
                logger.error(f"Method 1 (ONNX DirectML): {e}")
                logger.error(f"Method 2 (torch-directml): Failed or not available")
                logger.error("\nCPU processing will be 4-10x SLOWER and take HOURS!")
                logger.error("\nPlease try:")
                logger.error("1. Install onnxruntime-directml: pip install onnxruntime-directml")
                logger.error("2. Install torch-directml: pip install torch-directml")
                logger.error("3. Update AMD GPU drivers")
                logger.error("4. Restart the application")
                logger.error("\n" + "="*80)
                
                try:
                    response = input("\n⚠️  Proceed with CPU anyway? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        logger.warning("⚠️  Proceeding with CPU as requested (MUCH SLOWER)")
                        # Fall through to CPU initialization
                        self._use_onnx_directml = False
                        from ultralytics import YOLO
                        self.yolo_model = YOLO(yolo_model_path)
                        self.device = "cpu"
                        logger.warning("⚠️  CPU MODE - Processing will be 4-10x slower than GPU")
                        return
                    else:
                        raise RuntimeError("User cancelled - GPU acceleration required")
                except (EOFError, OSError):
                    # Non-interactive - raise error
                    raise RuntimeError(
                        f"All DirectML GPU methods failed and non-interactive mode.\n"
                        f"To proceed with CPU, explicitly specify --device cpu"
                    )
        elif device == "mps":
            # Use Ultralytics YOLO for MPS (Apple Silicon)
            self._use_onnx_directml = False
            try:
                from ultralytics import YOLO
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    self.yolo_model = YOLO(yolo_model_path)
                    logger.info("✅ Using M4 GPU (MPS) for YOLO inference")
                else:
                    # MPS requested but not available - ask user
                    logger.error("="*80)
                    logger.error("❌ MPS (Apple Silicon GPU) NOT AVAILABLE")
                    logger.error("="*80)
                    logger.error("CPU processing will be 4-10x SLOWER and take HOURS!")
                    logger.error("\nPlease verify:")
                    logger.error("1. Running on macOS with Apple Silicon (M1/M2/M3/M4)")
                    logger.error("2. PyTorch with MPS support is installed")
                    logger.error("3. macOS version supports Metal")
                    logger.error("\n" + "="*80)
                    
                    try:
                        response = input("\n⚠️  Proceed with CPU anyway? (yes/no): ").strip().lower()
                        if response in ['yes', 'y']:
                            logger.warning("⚠️  Proceeding with CPU as requested (MUCH SLOWER)")
                            self.device = "cpu"
                            self.yolo_model = YOLO(yolo_model_path)
                            logger.warning("⚠️  CPU MODE - Processing will be 4-10x slower than GPU")
                            return
                        else:
                            raise RuntimeError("User cancelled - GPU acceleration required")
                    except (EOFError, OSError):
                        raise RuntimeError(
                            f"MPS not available and non-interactive mode.\n"
                            f"To proceed with CPU, explicitly specify --device cpu"
                        )
            except RuntimeError:
                raise  # Re-raise our RuntimeErrors
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize MPS (Apple Silicon GPU): {e}\n"
                    f"This will cause hours of delay with CPU processing.\n\n"
                    f"To use CPU (NOT RECOMMENDED), explicitly specify --device cpu"
                )
        elif device == "cuda":
            # Use Ultralytics YOLO for CUDA (NVIDIA GPU)
            self._use_onnx_directml = False
            try:
                from ultralytics import YOLO
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.yolo_model = YOLO(yolo_model_path)
                    device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
                    logger.info(f"✅ Using CUDA (NVIDIA GPU) for YOLO inference: {device_name}")
                else:
                    # CUDA requested but not available - ask user
                    logger.error("="*80)
                    logger.error("❌ CUDA (NVIDIA GPU) NOT AVAILABLE")
                    logger.error("="*80)
                    logger.error("CPU processing will be 4-10x SLOWER and take HOURS!")
                    logger.error("\nPlease verify:")
                    logger.error("1. NVIDIA GPU is installed and recognized")
                    logger.error("2. PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                    logger.error("3. CUDA drivers are up to date")
                    logger.error("4. GPU is available: python -c 'import torch; print(torch.cuda.is_available())'")
                    logger.error("\n" + "="*80)
                    
                    try:
                        response = input("\n⚠️  Proceed with CPU anyway? (yes/no): ").strip().lower()
                        if response in ['yes', 'y']:
                            logger.warning("⚠️  Proceeding with CPU as requested (MUCH SLOWER)")
                            self.device = "cpu"
                            self.yolo_model = YOLO(yolo_model_path)
                            logger.warning("⚠️  CPU MODE - Processing will be 4-10x slower than GPU")
                            return
                        else:
                            raise RuntimeError("User cancelled - GPU acceleration required")
                    except (EOFError, OSError):
                        raise RuntimeError(
                            f"CUDA not available and non-interactive mode.\n"
                            f"To proceed with CPU, explicitly specify --device cpu"
                        )
            except RuntimeError:
                raise  # Re-raise our RuntimeErrors
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize CUDA (NVIDIA GPU): {e}\n"
                    f"This will cause hours of delay with CPU processing.\n\n"
                    f"To use CPU (NOT RECOMMENDED), explicitly specify --device cpu"
                )
        elif device == "cpu":
            # Use Ultralytics YOLO on CPU (explicitly requested)
            self._use_onnx_directml = False
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
            self.device = "cpu"
            logger.warning("⚠️  CPU MODE SELECTED - Processing will be MUCH SLOWER")
            logger.warning(f"   Expected speed: ~6-10 FPS (vs ~40-75 FPS on GPU)")
            logger.warning(f"   Processing time will be 4-10x longer than GPU mode")
            logger.info(f"YOLO model loaded from {yolo_model_path} on CPU")
        else:
            # Unknown device - default to CPU with warning
            logger.warning(f"⚠️  Unknown device '{device}', defaulting to CPU")
            self._use_onnx_directml = False
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
            self.device = "cpu"
            logger.warning("⚠️  CPU MODE - Processing will be MUCH SLOWER")
    
    def _load_onnx_directml(self, yolo_model_path: str):
        """Load ONNX model with DirectML provider for AMD GPU."""
        try:
            import onnxruntime as ort
            
            # Convert .pt path to .onnx path
            if yolo_model_path.endswith('.pt'):
                onnx_path = yolo_model_path.replace('.pt', '.onnx')
            else:
                onnx_path = "yolov8n.onnx"
            
            # Check if ONNX model exists, create if not
            if not Path(onnx_path).exists():
                logger.info(f"ONNX model not found at {onnx_path}, exporting from PyTorch...")
                from ultralytics import YOLO
                model = YOLO(yolo_model_path)
                model.export(format='onnx', opset=12)
                logger.info(f"✅ Exported ONNX model to {onnx_path}")
            
            # Set up DirectML provider (will fallback to CPU if not available)
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            
            # Create inference session with DirectML
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=providers
                )
            except Exception as e:
                # If DirectML fails, ask user before falling back to CPU
                logger.error(f"❌ DirectML provider initialization failed: {e}")
                raise RuntimeError(
                    f"DirectML GPU acceleration failed. This will cause hours of delay with CPU processing.\n"
                    f"Error: {e}\n\n"
                    f"Please try:\n"
                    f"1. Ensure onnxruntime-directml is installed: pip install onnxruntime-directml\n"
                    f"2. Update AMD GPU drivers\n"
                    f"3. Restart the application\n"
                    f"4. If problem persists, try alternative GPU method: --device cuda (if available)\n\n"
                    f"If you want to proceed with CPU anyway (NOT RECOMMENDED - will be very slow), "
                    f"use --device cpu explicitly."
                )
            
            # Verify DirectML is active
            active_providers = self.session.get_providers()
            if 'DmlExecutionProvider' in active_providers:
                logger.info(f"✅ DirectML provider active - AMD GPU acceleration enabled!")
                logger.info(f"   Using GPU via DirectML")
            else:
                # DirectML provider not active - raise error instead of silent fallback
                logger.error(f"❌ DirectML provider requested but not active!")
                logger.error(f"   Active providers: {active_providers}")
                raise RuntimeError(
                    f"DirectML GPU acceleration is not available.\n"
                    f"Active providers: {active_providers}\n\n"
                    f"Please:\n"
                    f"1. Install onnxruntime-directml: pip install onnxruntime-directml\n"
                    f"2. Update AMD GPU drivers\n"
                    f"3. Verify GPU is recognized in Windows Device Manager\n\n"
                    f"To use CPU (NOT RECOMMENDED - very slow), explicitly specify --device cpu"
                )
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (int(input_shape[2]), int(input_shape[3]))  # (width, height)
            logger.info(f"✅ ONNX model loaded: {onnx_path}")
            logger.info(f"   Input size: {self.input_size}")
            
        except ImportError as e:
            raise ImportError(f"onnxruntime not installed. Error: {e}\nInstall with: pip install onnxruntime-directml")
        except Exception as e:
            logger.error(f"Failed to load ONNX model with DirectML: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def detect_pedestrians(self, image: np.ndarray) -> List[Detection]:
        """
        Detect pedestrians in the given image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of Detection objects (with NMS applied)
        """
        # Apply preprocessing if enabled
        processed_image = image
        if self.frame_preprocessor is not None:
            processed_image = self.frame_preprocessor.preprocess(image)
        
        # Update adaptive parameters if tuner is available
        if self.detection_tuner is not None:
            # Get current parameters from tuner (will be updated internally)
            params = self.detection_tuner.get_current_parameters()
            self.confidence_threshold = params.confidence_threshold
            self.nms_iou_threshold = params.nms_iou_threshold
            self.min_height = params.min_height
            self.max_height = params.max_height
        
        if self._use_onnx_directml:
            # Use ONNX Runtime with DirectML (AMD GPU)
            detections = self._detect_onnx(processed_image)
        else:
            # Use Ultralytics YOLO (CPU, CUDA, or MPS)
            detections = self._detect_yolo(processed_image)
        
        # Update tuner with detections for next frame analysis
        if self.detection_tuner is not None:
            self.detection_tuner.update_parameters(detections, processed_image)
        
        return detections
    
    def _detect_yolo(self, image: np.ndarray) -> List[Detection]:
        """Detect using Ultralytics YOLO."""
        results = self.yolo_model(image, conf=self.confidence_threshold, device=self.device, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # YOLO class 0 is 'person'
                if int(box.cls) == 0:  # person class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate center and dimensions
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Filter by size
                    if self.min_height <= height <= self.max_height:
                        # Create a simple mask (bbox as mask for now)
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        mask[int(y1):int(y2), int(x1):int(x2)] = 255
                        
                        detection = Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            mask=mask,
                            center=(center_x, center_y),
                            confidence=confidence
                        )
                        detections.append(detection)
        
        # Apply NMS (context-aware if tuner available)
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
                detections = apply_nms(detections, iou_threshold=self.nms_iou_threshold)
        
        return detections
    
    def _detect_onnx(self, image: np.ndarray) -> List[Detection]:
        """Detect using ONNX Runtime with DirectML (AMD GPU)."""
        # Preprocess image
        input_image = self._preprocess_onnx(image)
        
        # Run inference on GPU via DirectML
        outputs = self.session.run(None, {self.input_name: input_image})
        
        # Parse outputs (YOLOv8 format)
        detections = self._parse_onnx_outputs(outputs[0], image.shape[:2])
        
        # Apply NMS (context-aware if tuner available)
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
                detections = apply_nms(detections, iou_threshold=self.nms_iou_threshold)
        
        return detections
    
    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model input."""
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
    
    def _parse_onnx_outputs(self, outputs: np.ndarray, image_shape: Tuple[int, int]) -> List[Detection]:
        """Parse ONNX model outputs to Detection objects using Ultralytics post-processing."""
        detections = []
        h, w = image_shape
        
        # YOLOv8 ONNX output format: [1, 84, 8400] or [1, 84, num_detections]
        # Transpose to [1, 8400, 84] for easier processing
        if len(outputs.shape) == 3:
            outputs = np.transpose(outputs[0], (1, 0))  # [8400, 84]
        
        # Calculate scale factors (output is in input image coordinates, need to scale to original)
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]
        
        # Process each detection candidate
        for detection in outputs:
            if len(detection) < 84:
                continue
            
            # YOLOv8 format: first 4 are bbox (x_center, y_center, width, height) in input image coords
            # Remaining 80 are class scores (COCO classes)
            x_center = float(detection[0]) * scale_x
            y_center = float(detection[1]) * scale_y
            bbox_w = float(detection[2]) * scale_x
            bbox_h = float(detection[3]) * scale_y
            
            # Get person class confidence (class 0 in COCO, index 4 in detection array)
            person_conf = float(detection[4])  # Class 0 score
            
            # Filter by confidence
            if person_conf >= self.confidence_threshold:
                # Convert center+size to x1,y1,x2,y2
                x1 = int(x_center - bbox_w / 2)
                y1 = int(y_center - bbox_h / 2)
                x2 = int(x_center + bbox_w / 2)
                y2 = int(y_center + bbox_h / 2)
                
                # Clip to image boundaries
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                
                height = y2 - y1
                
                # Filter by size
                if height > 0 and self.min_height <= height <= self.max_height:
                    # Create minimal mask (just bbox region) to save memory
                    # Use sparse representation: only create mask for bbox area
                    mask_h = max(1, y2 - y1)
                    mask_w = max(1, x2 - x1)
                    mask = np.ones((mask_h, mask_w), dtype=np.uint8) * 255
                    
                    detection_obj = Detection(
                        bbox=(x1, y1, x2, y2),
                        mask=mask,
                        center=(int(x_center), int(y_center)),
                        confidence=person_conf
                    )
                    detections.append(detection_obj)
        
        return detections

