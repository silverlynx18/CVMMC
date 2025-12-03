"""Command-line interface for batch analysis."""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .video_processor import VideoProcessor
from .fruin_analysis import FruinAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Command-line runner for pedestrian counting analysis."""
    
    def __init__(self):
        """Initialize the analysis runner."""
        self.video_processor = None
        self.fruin_analyzer = FruinAnalyzer()
        
    def setup_processor(self, model_path: str, device: str = "cuda"):
        """Setup the video processor with SAM2 model."""
        try:
            self.video_processor = VideoProcessor(model_path, device)
            logger.info(f"Video processor initialized with model: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video processor: {e}")
            return False
    
    def load_zones(self, zones_file: str) -> Optional[Dict]:
        """Load zone configuration from file."""
        try:
            with open(zones_file, 'r') as f:
                zones = json.load(f)
            logger.info(f"Loaded zones from {zones_file}")
            return zones
        except Exception as e:
            logger.error(f"Failed to load zones from {zones_file}: {e}")
            return None
    
    def process_single_video(self, video_path: str, output_dir: str, 
                           zones: Optional[Dict] = None) -> bool:
        """Process a single video file."""
        if not self.video_processor:
            logger.error("Video processor not initialized")
            return False
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process video
            result = self.video_processor.process_video(video_path, zones)
            
            if result:
                # Save result
                output_file = os.path.join(output_dir, f"{Path(video_path).stem}_analysis.json")
                self.video_processor.save_result(result, output_file)
                
                # Print summary
                self.print_analysis_summary(result)
                return True
            else:
                logger.error(f"Failed to process video: {video_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return False
    
    def process_batch(self, video_dir: str, output_dir: str, 
                     zones: Optional[Dict] = None) -> bool:
        """Process multiple video files in a directory."""
        if not self.video_processor:
            logger.error("Video processor not initialized")
            return False
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process batch
            results = self.video_processor.process_batch(video_dir, output_dir, zones)
            
            if results:
                logger.info(f"Batch processing completed: {len(results)} videos processed")
                
                # Print batch summary
                self.print_batch_summary(results)
                return True
            else:
                logger.error("Batch processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return False
    
    def print_analysis_summary(self, result):
        """Print analysis summary for a single video."""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Video File: {result.video_file.name}")
        print(f"Duration: {result.video_file.duration:.2f} seconds")
        print(f"Total Frames: {result.video_file.total_frames}")
        print(f"Resolution: {result.video_file.width}x{result.video_file.height}")
        print(f"FPS: {result.video_file.fps:.2f}")
        print("-"*60)
        print("DETECTION RESULTS")
        print("-"*60)
        print(f"Total Detections: {result.total_detections}")
        print(f"Total Ingress: {result.total_ingress}")
        print(f"Total Egress: {result.total_egress}")
        print(f"Net Count: {result.total_ingress - result.total_egress}")
        print("-"*60)
        print("LEVEL OF SERVICE ANALYSIS")
        print("-"*60)
        print(f"Average Density: {result.avg_density:.3f} peds/m²")
        print(f"Peak Density: {result.peak_density:.3f} peds/m²")
        print(f"Average LOS: {result.avg_los}")
        print(f"Peak LOS: {result.peak_los}")
        
        # LOS description
        los_descriptions = {
            "A": "Free flow - pedestrians can move freely without conflicts",
            "B": "Reasonably free flow - minor conflicts possible",
            "C": "Stable flow - some conflicts and queuing",
            "D": "Approaching unstable flow - frequent conflicts",
            "E": "Unstable flow - frequent conflicts and queuing",
            "F": "Forced flow - breakdown of flow, queuing"
        }
        
        print(f"Average LOS Description: {los_descriptions.get(result.avg_los, 'Unknown')}")
        print(f"Peak LOS Description: {los_descriptions.get(result.peak_los, 'Unknown')}")
        print("="*60)
    
    def print_batch_summary(self, results: List):
        """Print batch analysis summary."""
        print("\n" + "="*80)
        print("BATCH ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total Videos Processed: {len(results)}")
        print(f"Total Detections: {sum(r.total_detections for r in results)}")
        print(f"Total Ingress: {sum(r.total_ingress for r in results)}")
        print(f"Total Egress: {sum(r.total_egress for r in results)}")
        print(f"Net Count: {sum(r.total_ingress - r.total_egress for r in results)}")
        print("-"*80)
        
        # LOS distribution
        los_counts = {}
        for result in results:
            los_counts[result.avg_los] = los_counts.get(result.avg_los, 0) + 1
        
        print("LEVEL OF SERVICE DISTRIBUTION (Average)")
        print("-"*80)
        for los, count in sorted(los_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"LOS {los}: {count} videos ({percentage:.1f}%)")
        
        # Peak LOS distribution
        peak_los_counts = {}
        for result in results:
            peak_los_counts[result.peak_los] = peak_los_counts.get(result.peak_los, 0) + 1
        
        print("\nLEVEL OF SERVICE DISTRIBUTION (Peak)")
        print("-"*80)
        for los, count in sorted(peak_los_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"LOS {los}: {count} videos ({percentage:.1f}%)")
        
        # Density statistics
        avg_densities = [r.avg_density for r in results]
        peak_densities = [r.peak_density for r in results]
        
        print(f"\nDENSITY STATISTICS")
        print("-"*80)
        print(f"Average Density - Mean: {sum(avg_densities)/len(avg_densities):.3f} peds/m²")
        print(f"Average Density - Min: {min(avg_densities):.3f} peds/m²")
        print(f"Average Density - Max: {max(avg_densities):.3f} peds/m²")
        print(f"Peak Density - Mean: {sum(peak_densities)/len(peak_densities):.3f} peds/m²")
        print(f"Peak Density - Min: {min(peak_densities):.3f} peds/m²")
        print(f"Peak Density - Max: {max(peak_densities):.3f} peds/m²")
        print("="*80)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Pedestrian Counting and Service Level Analysis Tool"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Path to SAM2 model file (.pt)"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to video file or directory containing videos"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--zones",
        help="Path to zone configuration JSON file"
    )
    
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)"
    )
    
    parser.add_argument(
        "--zone-area",
        type=float,
        default=25.0,
        help="Zone area in square meters (default: 25.0)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AnalysisRunner()
    
    # Setup processor
    if not runner.setup_processor(args.model, args.device):
        return 1
    
    # Load zones if provided
    zones = None
    if args.zones:
        zones = runner.load_zones(args.zones)
        if zones is None:
            return 1
    
    # Update processor parameters
    runner.video_processor.set_analysis_parameters(
        frame_skip=args.frame_skip,
        analysis_window_minutes=5,
        zone_area_sqm=args.zone_area
    )
    runner.video_processor.sam2_detector.update_confidence_threshold(args.confidence)
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        # Single video file
        success = runner.process_single_video(args.input, args.output, zones)
    elif os.path.isdir(args.input):
        # Directory of videos
        success = runner.process_batch(args.input, args.output, zones)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())