"""Helper functions for stage-aware detection tuning integration."""

from typing import Optional, Tuple, Dict, Any
import logging

from .detection_tuner import DetectionTuner, TuningConfig, TuningConfig as TuningConfigClass
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def create_stage_tuner(stage: str, image_size: Tuple[int, int],
                      config_manager: Optional[ConfigManager] = None,
                      custom_config: Optional[Dict[str, Any]] = None) -> Optional[DetectionTuner]:
    """
    Create detection tuner appropriate for the workflow stage.
    
    Args:
        stage: Workflow stage ("stage1" or "stage2")
        image_size: (width, height) of input images
        config_manager: Optional ConfigManager instance
        custom_config: Optional custom tuning configuration dictionary
        
    Returns:
        DetectionTuner instance or None if tuning disabled
    """
    if config_manager is None:
        from config.config_manager import get_config
        config_manager = get_config()
    
    tuning_settings = config_manager.get_detection_tuning_settings()
    
    # Check if tuning is enabled for this stage
    stage_config = tuning_settings.get(stage, {})
    if not stage_config.get('enabled', True):
        logger.info(f"Detection tuning disabled for {stage}")
        return None
    
    # Build tuning config with stage-specific overrides
    base_config = {
        'adaptive_confidence': tuning_settings.get('adaptive_confidence', True),
        'confidence_floor': tuning_settings.get('confidence_floor', 0.3),
        'confidence_ceiling': tuning_settings.get('confidence_ceiling', 0.7),
        'nms_adaptive': tuning_settings.get('nms_adaptive', True),
        'size_filter_adaptive': tuning_settings.get('size_filter_adaptive', True),
        'temporal_smoothing': tuning_settings.get('temporal_smoothing', 0.1),
        'scene_analysis_window': tuning_settings.get('scene_analysis_window', 60),
        'gopro_fov_degrees': tuning_settings.get('gopro_fov_degrees', 120.0),
        'enable_edge_compensation': tuning_settings.get('enable_edge_compensation', True),
        'mass_motion_min_quality': tuning_settings.get('mass_motion_min_quality', 0.4),
    }
    
    # Apply stage-specific overrides
    if stage == "stage1":
        # Stage 1: Optimized for speed
        base_config.update({
            'adaptive_confidence': stage_config.get('adaptive_confidence', True),
            'temporal_smoothing': stage_config.get('temporal_smoothing', 0.15),
            'scene_analysis_window': stage_config.get('scene_analysis_window', 30),
        })
        lightweight = stage_config.get('lightweight_mode', True)
    elif stage == "stage2":
        # Stage 2: Optimized for accuracy
        base_config.update({
            'adaptive_confidence': stage_config.get('adaptive_confidence', True),
            'temporal_smoothing': stage_config.get('temporal_smoothing', 0.08),
            'scene_analysis_window': stage_config.get('scene_analysis_window', 90),
            'mass_motion_min_quality': max(
                base_config['mass_motion_min_quality'],
                stage_config.get('mass_motion_min_quality', 0.4)
            ),
        })
        lightweight = stage_config.get('lightweight_mode', False)
    else:
        logger.warning(f"Unknown stage '{stage}', using defaults")
        lightweight = False
    
    # Apply custom config overrides if provided
    if custom_config:
        base_config.update(custom_config)
    
    # Create TuningConfig object
    tuning_config = TuningConfig(
        adaptive_confidence=base_config['adaptive_confidence'],
        confidence_floor=base_config['confidence_floor'],
        confidence_ceiling=base_config['confidence_ceiling'],
        nms_adaptive=base_config['nms_adaptive'],
        size_filter_adaptive=base_config['size_filter_adaptive'],
        temporal_smoothing=base_config['temporal_smoothing'],
        scene_analysis_window=base_config['scene_analysis_window'],
        gopro_fov_degrees=base_config['gopro_fov_degrees'],
        enable_edge_compensation=base_config['enable_edge_compensation'],
        mass_motion_min_quality=base_config['mass_motion_min_quality'],
    )
    
    # Create tuner
    tuner = DetectionTuner(
        config=tuning_config,
        image_size=image_size,
        stage=stage,
        lightweight=lightweight
    )
    
    logger.info(f"Created {stage} detection tuner: "
               f"lightweight={lightweight}, "
               f"analysis_window={base_config['scene_analysis_window']}")
    
    return tuner


def get_stage_from_fps(fps: float) -> str:
    """
    Infer workflow stage from frame rate.
    
    Args:
        fps: Frame rate of video processing
        
    Returns:
        "stage1" or "stage2"
    """
    # Stage 1 typically uses 1fps or 15fps
    # Stage 2 typically uses 30fps
    if fps >= 25:
        return "stage2"
    else:
        return "stage1"

