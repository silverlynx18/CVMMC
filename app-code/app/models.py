"""Database models for pedestrian counting and analysis."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Camera(Base):
    """Camera configuration and status."""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    location = Column(String(200))
    ip_address = Column(String(45))
    port = Column(Integer, default=554)
    username = Column(String(100))
    password = Column(String(100))
    is_active = Column(Boolean, default=True)
    calibration_data = Column(JSON)  # Camera calibration parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PedestrianDetection(Base):
    """Individual pedestrian detection record."""
    __tablename__ = "pedestrian_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)
    confidence = Column(Float)
    track_id = Column(Integer, index=True)
    is_ingress = Column(Boolean)  # True for entering, False for exiting
    zone_id = Column(String(50))  # Specific zone/area within the camera view


class PedestrianCount(Base):
    """Aggregated pedestrian counts by time period."""
    __tablename__ = "pedestrian_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    time_window_minutes = Column(Integer, default=5)
    ingress_count = Column(Integer, default=0)
    egress_count = Column(Integer, default=0)
    net_count = Column(Integer, default=0)  # ingress - egress
    total_pedestrians = Column(Integer, default=0)  # current count in area
    density_per_sqm = Column(Float)  # pedestrians per square meter
    fruin_los = Column(String(1))  # A, B, C, D, E, F
    zone_id = Column(String(50))


class ServiceLevelAnalysis(Base):
    """Fruin's Level of Service analysis results."""
    __tablename__ = "service_level_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    analysis_period_minutes = Column(Integer, default=60)
    avg_density = Column(Float)
    peak_density = Column(Float)
    avg_fruin_los = Column(String(1))
    peak_fruin_los = Column(String(1))
    total_ingress = Column(Integer)
    total_egress = Column(Integer)
    peak_hour_factor = Column(Float)
    zone_id = Column(String(50))


class Alert(Base):
    """System alerts and notifications."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50))  # density_high, camera_offline, etc.
    severity = Column(String(20))  # low, medium, high, critical
    message = Column(Text)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    alert_metadata = Column(JSON)