"""Configuration settings for the pedestrian counting application."""

import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database settings
    database_url: str = "postgresql://user:password@localhost/pedestrian_db"
    redis_url: str = "redis://localhost:6379"
    
    # Camera settings
    num_cameras: int = 10
    camera_resolution: tuple = (1920, 1080)
    fps: int = 30
    
    # SAM3 settings
    sam3_model_path: str = "sam3_hiera_large.pt"
    # Detector selection: choose 'YOLO' or 'SAM3'
    detector_type: str = "SAM3"
    # HuggingFace token (optional). Better to set via environment variable `HF_TOKEN`.
    huggingface_token: str | None = None
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"
    confidence_threshold: float = 0.5
    
    # Pedestrian detection settings
    min_pedestrian_height: int = 50  # pixels
    max_pedestrian_height: int = 300  # pixels
    tracking_max_disappeared: int = 30  # frames
    tracking_max_distance: int = 100  # pixels
    
    # Fruin's Level of Service thresholds (pedestrians per square meter)
    fruin_thresholds: Dict[str, float] = {
        "A": 0.0,    # Free flow
        "B": 0.3,    # Reasonably free flow
        "C": 0.7,    # Stable flow
        "D": 1.1,    # Approaching unstable flow
        "E": 1.5,    # Unstable flow
        "F": 2.0     # Forced flow
    }
    
    # Analysis settings
    analysis_window_minutes: int = 5
    density_calculation_area: float = 1.0  # square meters per pixel
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"


settings = Settings()