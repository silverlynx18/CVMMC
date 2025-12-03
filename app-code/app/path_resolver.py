"""
Path Resolver - Handles file paths correctly in both source and executable modes.

This module provides utilities to resolve file paths that work correctly whether
the application is running from source code or from a PyInstaller executable.
"""

import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    """
    Check if the application is running from a PyInstaller executable.
    
    Returns:
        True if running from executable, False if running from source
    """
    return getattr(sys, 'frozen', False)


def get_base_path() -> Path:
    """
    Get the base path of the application.
    
    When running from executable: Returns the directory containing the .exe
    When running from source: Returns the project root directory
    
    Returns:
        Path object pointing to the base directory
    """
    if is_frozen():
        # Running from PyInstaller executable
        # sys.executable points to the .exe file
        return Path(sys.executable).parent
    else:
        # Running from source
        # Go up from app/path_resolver.py to project root
        return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """
    Get the models directory path.
    
    Models are stored in a 'models' folder relative to the base path.
    
    Returns:
        Path object pointing to the models directory
    """
    base_path = get_base_path()
    models_dir = base_path / 'models'
    return models_dir


def get_config_dir() -> Path:
    """
    Get the configuration directory path.
    
    Config files are stored in a 'config' folder relative to the base path.
    
    Returns:
        Path object pointing to the config directory
    """
    base_path = get_base_path()
    config_dir = base_path / 'config'
    return config_dir


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path object pointing to the data directory
    """
    base_path = get_base_path()
    data_dir = base_path / 'data'
    return data_dir


def get_outputs_dir() -> Path:
    """
    Get the outputs directory path.
    
    Returns:
        Path object pointing to the outputs directory
    """
    base_path = get_base_path()
    outputs_dir = base_path / 'outputs'
    return outputs_dir


def resolve_model_path(model_name: str, check_exists: bool = True) -> Optional[Path]:
    """
    Resolve the path to a model file.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        check_exists: If True, return None if file doesn't exist
        
    Returns:
        Path to the model file, or None if not found and check_exists=True
    """
    models_dir = get_models_dir()
    model_path = models_dir / model_name
    
    if check_exists and not model_path.exists():
        return None
    
    return model_path


def resolve_config_path(config_name: str, check_exists: bool = True) -> Optional[Path]:
    """
    Resolve the path to a configuration file.
    
    Args:
        config_name: Name of the config file (e.g., 'default_config.json')
        check_exists: If True, return None if file doesn't exist
        
    Returns:
        Path to the config file, or None if not found and check_exists=True
    """
    config_dir = get_config_dir()
    config_path = config_dir / config_name
    
    if check_exists and not config_path.exists():
        return None
    
    return config_path


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        The path object (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_resource_path(relative_path: str) -> Path:
    """
    Get the path to a resource file.
    
    This is a general-purpose function for resolving paths to any resource
    file relative to the base directory.
    
    Args:
        relative_path: Relative path from base directory (e.g., 'data/videos')
        
    Returns:
        Path object pointing to the resource
    """
    base_path = get_base_path()
    return base_path / relative_path


# Convenience functions for common paths
def get_logs_dir() -> Path:
    """Get the logs directory path."""
    base_path = get_base_path()
    logs_dir = base_path / 'logs'
    return logs_dir


def get_temp_dir() -> Path:
    """Get the temporary files directory path."""
    base_path = get_base_path()
    temp_dir = base_path / 'temp'
    return temp_dir

