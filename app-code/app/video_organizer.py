"""Video file organization and metadata creation utility."""

import os
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re

from .metadata_manager import MetadataManager, VideoClip, CameraMetadata

logger = logging.getLogger(__name__)


class VideoOrganizer:
    """Organizes video files and creates metadata for cameras."""
    
    def __init__(self, metadata_file: str = "camera_metadata.json"):
        """
        Initialize video organizer.
        
        Args:
            metadata_file: Path to metadata JSON file
        """
        self.metadata_manager = MetadataManager(metadata_file)
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def scan_directory(self, directory: str, camera_id: str = None) -> Dict[str, List[str]]:
        """
        Scan directory for video files and organize by camera.
        
        Args:
            directory: Directory to scan
            camera_id: Optional camera ID to filter files
            
        Returns:
            Dictionary mapping camera IDs to lists of video files
        """
        logger.info(f"Scanning directory: {directory}")
        
        video_files = {}
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return video_files
        
        # Find all video files
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                # Try to extract camera ID from filename or path
                detected_camera_id = self.extract_camera_id(file_path, camera_id)
                
                if detected_camera_id not in video_files:
                    video_files[detected_camera_id] = []
                
                video_files[detected_camera_id].append(str(file_path))
        
        logger.info(f"Found {sum(len(files) for files in video_files.values())} video files")
        for cam_id, files in video_files.items():
            logger.info(f"Camera {cam_id}: {len(files)} files")
        
        return video_files
    
    def extract_camera_id(self, file_path: Path, default_camera_id: str = None) -> str:
        """
        Extract camera ID from file path or use default.
        
        Args:
            file_path: Path to video file
            default_camera_id: Default camera ID if extraction fails
            
        Returns:
            Camera ID string
        """
        if default_camera_id:
            return default_camera_id
        
        # Try to extract from filename
        filename = file_path.stem.lower()
        
        # Common patterns for camera IDs
        patterns = [
            r'cam(\d+)',           # cam1, cam2, etc.
            r'camera(\d+)',        # camera1, camera2, etc.
            r'cam_(\d+)',          # cam_1, cam_2, etc.
            r'camera_(\d+)',       # camera_1, camera_2, etc.
            r'(\d+)_cam',          # 1_cam, 2_cam, etc.
            r'(\d+)_camera',       # 1_camera, 2_camera, etc.
            r'cam(\w+)',           # camA, camB, etc.
            r'camera(\w+)',        # cameraA, cameraB, etc.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return f"camera_{match.group(1)}"
        
        # Try to extract from parent directory
        parent_dir = file_path.parent.name.lower()
        for pattern in patterns:
            match = re.search(pattern, parent_dir)
            if match:
                return f"camera_{match.group(1)}"
        
        # Use filename as camera ID
        return f"camera_{filename}"
    
    def get_video_info(self, file_path: str) -> Optional[Dict]:
        """
        Get video file information.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {file_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size': file_size,
                'codec': codec
            }
            
        except Exception as e:
            logger.error(f"Error getting video info for {file_path}: {e}")
            return None
    
    def create_camera_metadata(self, camera_id: str, name: str, location: str,
                              description: str = "", zone_area_sqm: float = 25.0) -> bool:
        """
        Create camera metadata.
        
        Args:
            camera_id: Camera identifier
            name: Camera name
            location: Camera location
            description: Camera description
            zone_area_sqm: Zone area in square meters
            
        Returns:
            True if successful, False otherwise
        """
        return self.metadata_manager.add_camera(
            camera_id, name, location, description, zone_area_sqm
        )
    
    def add_video_clips(self, camera_id: str, video_files: List[str],
                       start_time: datetime = None, tags: List[str] = None) -> int:
        """
        Add video clips to camera metadata.
        
        Args:
            camera_id: Camera identifier
            video_files: List of video file paths
            start_time: Start time for first clip (defaults to current time)
            tags: Optional tags for clips
            
        Returns:
            Number of clips successfully added
        """
        if not start_time:
            start_time = datetime.utcnow()
        
        if not tags:
            tags = []
        
        added_count = 0
        current_time = start_time
        
        for video_file in video_files:
            # Get video info
            video_info = self.get_video_info(video_file)
            if not video_info:
                logger.warning(f"Skipping file due to error: {video_file}")
                continue
            
            # Calculate end time
            end_time = current_time + timedelta(seconds=video_info['duration'])
            
            # Add clip to metadata
            success = self.metadata_manager.add_video_clip(
                camera_id=camera_id,
                file_path=video_file,
                start_time=current_time,
                end_time=end_time,
                duration=video_info['duration'],
                file_size=video_info['file_size'],
                resolution=(video_info['width'], video_info['height']),
                fps=video_info['fps'],
                codec=video_info['codec'],
                tags=tags,
                notes=f"Auto-imported from {os.path.basename(video_file)}"
            )
            
            if success:
                added_count += 1
                logger.info(f"Added clip: {os.path.basename(video_file)}")
            else:
                logger.error(f"Failed to add clip: {video_file}")
            
            # Update current time for next clip
            current_time = end_time
        
        logger.info(f"Added {added_count} clips to camera {camera_id}")
        return added_count
    
    def organize_videos(self, source_directory: str, 
                       camera_mappings: Dict[str, Dict] = None) -> Dict[str, int]:
        """
        Organize videos and create metadata.
        
        Args:
            source_directory: Directory containing video files
            camera_mappings: Dictionary mapping camera IDs to metadata
            
        Returns:
            Dictionary with counts of organized files per camera
        """
        logger.info(f"Organizing videos from: {source_directory}")
        
        # Scan directory for video files
        video_files = self.scan_directory(source_directory)
        
        if not video_files:
            logger.warning("No video files found")
            return {}
        
        # Organize files by camera
        organized_counts = {}
        
        for camera_id, files in video_files.items():
            logger.info(f"Processing camera {camera_id} with {len(files)} files")
            
            # Create camera metadata if it doesn't exist
            if not self.metadata_manager.get_camera(camera_id):
                camera_info = camera_mappings.get(camera_id, {}) if camera_mappings else {}
                
                success = self.create_camera_metadata(
                    camera_id=camera_id,
                    name=camera_info.get('name', f"Camera {camera_id}"),
                    location=camera_info.get('location', "Unknown"),
                    description=camera_info.get('description', ""),
                    zone_area_sqm=camera_info.get('zone_area_sqm', 25.0)
                )
                
                if not success:
                    logger.error(f"Failed to create camera metadata for {camera_id}")
                    continue
            
            # Add video clips
            added_count = self.add_video_clips(camera_id, files)
            organized_counts[camera_id] = added_count
        
        logger.info(f"Organization complete: {organized_counts}")
        return organized_counts
    
    def create_test_clips(self, camera_id: str, duration_seconds: int = 60) -> List[str]:
        """
        Create test clips from existing videos.
        
        Args:
            camera_id: Camera identifier
            duration_seconds: Duration of test clips in seconds
            
        Returns:
            List of created test clip file paths
        """
        camera = self.metadata_manager.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return []
        
        if not camera.clips:
            logger.warning(f"No clips found for camera {camera_id}")
            return []
        
        test_clips = []
        
        for clip in camera.clips:
            if clip.duration < duration_seconds:
                logger.warning(f"Clip {clip.file_path} too short for test duration")
                continue
            
            # Create test clip filename
            clip_path = Path(clip.file_path)
            test_filename = f"{clip_path.stem}_test_{duration_seconds}s{clip_path.suffix}"
            test_path = clip_path.parent / test_filename
            
            # Extract test clip using OpenCV
            try:
                cap = cv2.VideoCapture(clip.file_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video: {clip.file_path}")
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(test_path), fourcc, fps, (width, height))
                
                # Extract frames for test duration
                frame_count = int(fps * duration_seconds)
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                cap.release()
                out.release()
                
                # Add test clip to metadata
                test_start_time = clip.start_time
                test_end_time = test_start_time + timedelta(seconds=duration_seconds)
                
                success = self.metadata_manager.add_video_clip(
                    camera_id=camera_id,
                    file_path=str(test_path),
                    start_time=test_start_time,
                    end_time=test_end_time,
                    duration=duration_seconds,
                    file_size=os.path.getsize(test_path),
                    resolution=(width, height),
                    fps=fps,
                    codec=clip.codec,
                    tags=clip.tags + ['test'],
                    notes=f"Test clip extracted from {os.path.basename(clip.file_path)}"
                )
                
                if success:
                    test_clips.append(str(test_path))
                    logger.info(f"Created test clip: {test_filename}")
                else:
                    logger.error(f"Failed to add test clip to metadata: {test_filename}")
                
            except Exception as e:
                logger.error(f"Error creating test clip from {clip.file_path}: {e}")
        
        logger.info(f"Created {len(test_clips)} test clips for camera {camera_id}")
        return test_clips
    
    def export_metadata_summary(self, output_file: str = "metadata_summary.json"):
        """
        Export metadata summary to JSON file.
        
        Args:
            output_file: Output file path
        """
        summary = self.metadata_manager.get_metadata_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Metadata summary exported to {output_file}")
    
    def cleanup_metadata(self, remove_missing_files: bool = True):
        """
        Clean up metadata by removing references to missing files.
        
        Args:
            remove_missing_files: Whether to remove clips with missing files
        """
        if not remove_missing_files:
            return
        
        logger.info("Cleaning up metadata...")
        
        for camera in self.metadata_manager.get_all_cameras():
            clips_to_remove = []
            
            for clip in camera.clips:
                if not os.path.exists(clip.file_path):
                    clips_to_remove.append(clip.file_path)
                    logger.warning(f"Missing file: {clip.file_path}")
            
            # Remove missing clips
            for file_path in clips_to_remove:
                self.metadata_manager.remove_clip(camera.camera_id, file_path)
        
        logger.info("Metadata cleanup complete")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize video files and create metadata")
    parser.add_argument("source_dir", help="Source directory containing video files")
    parser.add_argument("--camera-id", help="Specific camera ID to process")
    parser.add_argument("--metadata-file", default="camera_metadata.json", 
                       help="Metadata file path")
    parser.add_argument("--create-tests", action="store_true", 
                       help="Create test clips")
    parser.add_argument("--test-duration", type=int, default=60, 
                       help="Test clip duration in seconds")
    parser.add_argument("--export-summary", help="Export summary to file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create organizer
    organizer = VideoOrganizer(args.metadata_file)
    
    # Organize videos
    organized_counts = organizer.organize_videos(args.source_dir)
    
    if organized_counts:
        print(f"Organized {sum(organized_counts.values())} video files")
        for camera_id, count in organized_counts.items():
            print(f"  {camera_id}: {count} files")
    
    # Create test clips if requested
    if args.create_tests:
        for camera_id in organized_counts.keys():
            test_clips = organizer.create_test_clips(camera_id, args.test_duration)
            print(f"Created {len(test_clips)} test clips for {camera_id}")
    
    # Export summary if requested
    if args.export_summary:
        organizer.export_metadata_summary(args.export_summary)
    
    # Cleanup
    organizer.cleanup_metadata()


if __name__ == "__main__":
    main()