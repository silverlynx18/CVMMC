"""
GPU-accelerated frame preprocessing using PyTorch DirectML for AMD GPUs.

This is an alternative implementation that attempts to use GPU for preprocessing
operations where possible. Note: Some operations may still fall back to CPU
due to DirectML limitations.
"""

import cv2
import numpy as np
import torch
import torch_directml
from typing import Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)

# Try to initialize DirectML device
try:
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        logger.info(f"✅ DirectML device available for preprocessing: {DEVICE}")
    else:
        DEVICE = None
        logger.warning("DirectML not available for preprocessing - will use CPU")
except Exception as e:
    DEVICE = None
    logger.warning(f"Could not initialize DirectML for preprocessing: {e}")


def numpy_to_torch(frame: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor on GPU."""
    if DEVICE is None:
        return torch.from_numpy(frame).float()
    # Convert BGR to RGB and normalize to [0, 1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    # Change from HWC to CHW format
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor back to numpy array."""
    # Move to CPU if on GPU
    if tensor.is_cuda or tensor.device.type == 'privateuseone':
        tensor = tensor.cpu()
    # Remove batch dimension and convert CHW to HWC
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    # Denormalize and convert to uint8
    frame = (tensor * 255.0).clamp(0, 255).numpy().astype(np.uint8)
    # Convert RGB back to BGR
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def gamma_correction_gpu(frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply gamma correction using GPU (PyTorch DirectML).
    
    Note: This is faster for large batches but may have overhead for single frames.
    """
    if DEVICE is None:
        # Fallback to CPU
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return cv2.LUT(frame, lookup_table)
    
    try:
        # Convert to tensor
        tensor = numpy_to_torch(frame)
        
        # Apply gamma correction on GPU
        inv_gamma = 1.0 / gamma
        tensor = tensor ** inv_gamma
        
        # Convert back
        return torch_to_numpy(tensor)
    except Exception as e:
        logger.warning(f"GPU gamma correction failed, falling back to CPU: {e}")
        # Fallback to CPU
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return cv2.LUT(frame, lookup_table)


def lab_enhancement_gpu(frame: np.ndarray, luminance_boost: float = 1.1) -> np.ndarray:
    """
    Apply LAB enhancement using GPU where possible.
    
    Note: Color space conversion still uses OpenCV (CPU), but luminance
    multiplication can be done on GPU.
    """
    if DEVICE is None:
        # Fallback to CPU version
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = np.clip(l * luminance_boost, 0, 255).astype(np.uint8)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    try:
        # Color space conversion on CPU (OpenCV doesn't have DirectML support)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Convert L channel to tensor and apply boost on GPU
        l_tensor = torch.from_numpy(l.astype(np.float32)).to(DEVICE)
        l_enhanced = (l_tensor * luminance_boost).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # Merge back (CPU)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"GPU LAB enhancement failed, falling back to CPU: {e}")
        # Fallback to CPU
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = np.clip(l * luminance_boost, 0, 255).astype(np.uint8)
        enhanced = cv2.merge([l_2, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

