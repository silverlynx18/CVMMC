"""Metadata management for organizing video clips by camera."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """Video clip metadata."""
    file_path: str
    camera_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    file_size: int
    resolution: Tuple[int, int]
    fps: float
    codec: str
    tags: List[str]
    notes: str = ""


@dataclass
class CameraMetadata:
    """Camera metadata and configuration."""
    camera_id: str
    name: str
    location: str
    description: str
    zone_area_sqm: float
    calibration_data: Dict
    clips: List[VideoClip]
    created_at: datetime
    updated_at: datetime


class MetadataManager:
    """Manages video metadata and camera organization."""
    
    def __init__(self, metadata_file: str = "camera_metadata.json"):
        """
        Initialize metadata manager.
        
        Args:
            metadata_file: Path to metadata JSON file
        """
        self.metadata_file = metadata_file
        self.cameras: Dict[str, CameraMetadata] = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from file."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Convert data back to dataclasses
                for camera_id, camera_data in data.items():
                    clips = []
                    for clip_data in camera_data.get('clips', []):
                        clip = VideoClip(
                            file_path=clip_data['file_path'],
                            camera_id=clip_data['camera_id'],
                            start_time=datetime.fromisoformat(clip_data['start_time']),
                            end_time=datetime.fromisoformat(clip_data['end_time']),
                            duration=clip_data['duration'],
                            file_size=clip_data['file_size'],
                            resolution=tuple(clip_data['resolution']),
                            fps=clip_data['fps'],
                            codec=clip_data['codec'],
                            tags=clip_data.get('tags', []),
                            notes=clip_data.get('notes', '')
                        )
                        clips.append(clip)
                    
                    camera = CameraMetadata(
                        camera_id=camera_id,
                        name=camera_data['name'],
                        location=camera_data['location'],
                        description=camera_data.get('description', ''),
                        zone_area_sqm=camera_data.get('zone_area_sqm', 25.0),
                        calibration_data=camera_data.get('calibration_data', {}),
                        clips=clips,
                        created_at=datetime.fromisoformat(camera_data['created_at']),
                        updated_at=datetime.fromisoformat(camera_data['updated_at'])
                    )
                    self.cameras[camera_id] = camera
                
                logger.info(f"Loaded metadata for {len(self.cameras)} cameras")
            else:
                logger.info("No existing metadata file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.cameras = {}
    
    def save_metadata(self):
        """Save metadata to file."""
        try:
            data = {}
            for camera_id, camera in self.cameras.items():
                camera_data = {
                    'name': camera.name,
                    'location': camera.location,
                    'description': camera.description,
                    'zone_area_sqm': camera.zone_area_sqm,
                    'calibration_data': camera.calibration_data,
                    'created_at': camera.created_at.isoformat(),
                    'updated_at': camera.updated_at.isoformat(),
                    'clips': []
                }
                
                for clip in camera.clips:
                    clip_data = {
                        'file_path': clip.file_path,
                        'camera_id': clip.camera_id,
                        'start_time': clip.start_time.isoformat(),
                        'end_time': clip.end_time.isoformat(),
                        'duration': clip.duration,
                        'file_size': clip.file_size,
                        'resolution': list(clip.resolution),
                        'fps': clip.fps,
                        'codec': clip.codec,
                        'tags': clip.tags,
                        'notes': clip.notes
                    }
                    camera_data['clips'].append(clip_data)
                
                data[camera_id] = camera_data
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metadata saved to {self.metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_camera(self, camera_id: str, name: str, location: str, 
                   description: str = "", zone_area_sqm: float = 25.0,
                   calibration_data: Optional[Dict] = None) -> bool:
        """
        Add a new camera to metadata.
        
        Args:
            camera_id: Unique camera identifier
            name: Camera name
            location: Camera location
            description: Camera description
            zone_area_sqm: Zone area in square meters
            calibration_data: Camera calibration data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if camera_id in self.cameras:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            
            camera = CameraMetadata(
                camera_id=camera_id,
                name=name,
                location=location,
                description=description,
                zone_area_sqm=zone_area_sqm,
                calibration_data=calibration_data or {},
                clips=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.cameras[camera_id] = camera
            self.save_metadata()
            logger.info(f"Added camera {camera_id}: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {e}")
            return False
    
    def add_video_clip(self, camera_id: str, file_path: str, 
                      start_time: datetime, end_time: datetime,
                      duration: float, file_size: int,
                      resolution: Tuple[int, int], fps: float,
                      codec: str, tags: List[str] = None,
                      notes: str = "") -> bool:
        """
        Add a video clip to a camera's metadata.
        
        Args:
            camera_id: Camera identifier
            file_path: Path to video file
            start_time: Clip start time
            end_time: Clip end time
            duration: Clip duration in seconds
            file_size: File size in bytes
            resolution: Video resolution (width, height)
            fps: Frames per second
            codec: Video codec
            tags: Optional tags
            notes: Optional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            clip = VideoClip(
                file_path=file_path,
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                file_size=file_size,
                resolution=resolution,
                fps=fps,
                codec=codec,
                tags=tags or [],
                notes=notes
            )
            
            self.cameras[camera_id].clips.append(clip)
            self.cameras[camera_id].updated_at = datetime.utcnow()
            self.save_metadata()
            
            logger.info(f"Added clip to camera {camera_id}: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding clip to camera {camera_id}: {e}")
            return False
    
    def get_camera_clips(self, camera_id: str) -> List[VideoClip]:
        """Get all clips for a specific camera."""
        if camera_id not in self.cameras:
            return []
        return self.cameras[camera_id].clips
    
    def get_camera(self, camera_id: str) -> Optional[CameraMetadata]:
        """Get camera metadata by ID."""
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[CameraMetadata]:
        """Get all cameras."""
        return list(self.cameras.values())
    
    def get_clips_by_time_range(self, camera_id: str, 
                               start_time: datetime, 
                               end_time: datetime) -> List[VideoClip]:
        """Get clips within a specific time range."""
        if camera_id not in self.cameras:
            return []
        
        clips = []
        for clip in self.cameras[camera_id].clips:
            if (clip.start_time <= end_time and clip.end_time >= start_time):
                clips.append(clip)
        
        return clips
    
    def get_clips_by_duration(self, camera_id: str, 
                             min_duration: float = 0,
                             max_duration: float = float('inf')) -> List[VideoClip]:
        """Get clips within a specific duration range."""
        if camera_id not in self.cameras:
            return []
        
        clips = []
        for clip in self.cameras[camera_id].clips:
            if min_duration <= clip.duration <= max_duration:
                clips.append(clip)
        
        return clips
    
    def get_clips_by_tags(self, camera_id: str, tags: List[str]) -> List[VideoClip]:
        """Get clips that have any of the specified tags."""
        if camera_id not in self.cameras:
            return []
        
        clips = []
        for clip in self.cameras[camera_id].clips:
            if any(tag in clip.tags for tag in tags):
                clips.append(clip)
        
        return clips
    
    def update_camera_calibration(self, camera_id: str, calibration_data: Dict) -> bool:
        """Update camera calibration data."""
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            self.cameras[camera_id].calibration_data = calibration_data
            self.cameras[camera_id].updated_at = datetime.utcnow()
            self.save_metadata()
            
            logger.info(f"Updated calibration for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating calibration for camera {camera_id}: {e}")
            return False
    
    def update_zone_area(self, camera_id: str, zone_area_sqm: float) -> bool:
        """Update camera zone area."""
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            self.cameras[camera_id].zone_area_sqm = zone_area_sqm
            self.cameras[camera_id].updated_at = datetime.utcnow()
            self.save_metadata()
            
            logger.info(f"Updated zone area for camera {camera_id}: {zone_area_sqm} sqm")
            return True
            
        except Exception as e:
            logger.error(f"Error updating zone area for camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera and all its clips."""
        try:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False
            
            del self.cameras[camera_id]
            self.save_metadata()
            
            logger.info(f"Removed camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing camera {camera_id}: {e}")
            return False
    
    def remove_clip(self, camera_id: str, file_path: str) -> bool:
        """Remove a specific clip from a camera."""
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            # Find and remove clip
            for i, clip in enumerate(self.cameras[camera_id].clips):
                if clip.file_path == file_path:
                    del self.cameras[camera_id].clips[i]
                    self.cameras[camera_id].updated_at = datetime.utcnow()
                    self.save_metadata()
                    
                    logger.info(f"Removed clip from camera {camera_id}: {os.path.basename(file_path)}")
                    return True
            
            logger.warning(f"Clip not found: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error removing clip from camera {camera_id}: {e}")
            return False
    
    def get_metadata_summary(self) -> Dict:
        """Get a summary of all metadata."""
        summary = {
            'total_cameras': len(self.cameras),
            'total_clips': sum(len(camera.clips) for camera in self.cameras.values()),
            'cameras': []
        }
        
        for camera in self.cameras.values():
            camera_summary = {
                'camera_id': camera.camera_id,
                'name': camera.name,
                'location': camera.location,
                'clips_count': len(camera.clips),
                'total_duration': sum(clip.duration for clip in camera.clips),
                'zone_area_sqm': camera.zone_area_sqm,
                'created_at': camera.created_at.isoformat(),
                'updated_at': camera.updated_at.isoformat()
            }
            summary['cameras'].append(camera_summary)
        
        return summary