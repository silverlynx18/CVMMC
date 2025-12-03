"""
Setup Manager - Handles first-time app setup automatically.

Downloads models, checks dependencies, and sets up required components
without sending users to external sites or requiring terminal commands.
"""

import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .path_resolver import (
    get_base_path,
    get_models_dir,
    get_config_dir,
    get_data_dir,
    get_outputs_dir,
    ensure_directory,
)

logger = logging.getLogger(__name__)


class SetupManager:
    """Manages automatic app setup including model downloads and dependency checks."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize setup manager.
        
        Args:
            base_dir: Base directory for app (uses path_resolver if None)
        """
        if base_dir is None:
            base_dir = get_base_path()
        self.base_dir = Path(base_dir)
        self.models_dir = get_models_dir()
        self.setup_complete_flag = self.base_dir / ".setup_complete"
        
    def check_setup_complete(self) -> bool:
        """Check if initial setup has been completed."""
        return self.setup_complete_flag.exists()
    
    def mark_setup_complete(self):
        """Mark setup as complete."""
        try:
            self.setup_complete_flag.touch()
        except Exception as e:
            logger.warning(f"Could not create setup flag: {e}")
    
    def ensure_models(self) -> Dict[str, bool]:
        """
        Ensure required models are downloaded.
        
        Returns:
            Dict mapping model name to success status
        """
        models_status = {}
        
        # YOLO model (PyTorch)
        yolo_pt = self.models_dir / "yolov8n.pt"
        if not yolo_pt.exists():
            logger.info("YOLO PyTorch model not found - downloading...")
            if self._download_yolo_model():
                models_status["yolov8n.pt"] = True
            else:
                models_status["yolov8n.pt"] = False
        else:
            models_status["yolov8n.pt"] = True
            logger.debug(f"YOLO PyTorch model found: {yolo_pt}")
        
        # ONNX model will be generated from .pt if needed (handled by detector)
        yolo_onnx = self.models_dir / "yolov8n.onnx"
        if not yolo_onnx.exists():
            logger.debug("ONNX model not found - will be generated when needed")
            models_status["yolov8n.onnx"] = False  # Will be auto-generated
        else:
            models_status["yolov8n.onnx"] = True
        
        return models_status
    
    def _download_yolo_model(self) -> bool:
        """
        Download YOLO model using Ultralytics.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            logger.info("Downloading YOLOv8n model (this may take a moment)...")
            model = YOLO("yolov8n.pt")  # This downloads if not present
            model_path = self.models_dir / "yolov8n.pt"
            
            # Check if model was downloaded
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"Model downloaded successfully: {size_mb:.1f} MB")
                return True
            else:
                logger.error("Model download completed but file not found")
                return False
                
        except ImportError:
            logger.error("Ultralytics not installed - cannot download model")
            logger.info("Please install: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to download YOLO model: {e}")
            return False
    
    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """
        Check for required and optional dependencies.
        
        Returns:
            Tuple of (missing_required, missing_optional)
        """
        required = [
            ("cv2", "opencv-python", "OpenCV"),
            ("numpy", "numpy", "NumPy"),
            ("ultralytics", "ultralytics", "Ultralytics YOLO"),
            ("onnxruntime", "onnxruntime", "ONNX Runtime"),
        ]
        
        optional = [
            ("customtkinter", "customtkinter", "GUI framework"),
            ("onnxruntime-directml", "onnxruntime-directml", "AMD GPU support"),
            ("torch", "torch", "PyTorch"),
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required
        for module_name, package_name, description in required:
            try:
                __import__(module_name)
            except ImportError:
                missing_required.append((module_name, package_name, description))
        
        # Check optional
        for module_name, package_name, description in optional:
            try:
                __import__(module_name)
            except ImportError:
                missing_optional.append((module_name, package_name, description))
        
        return missing_required, missing_optional
    
    def install_dependency(self, package_name: str, description: str = None) -> bool:
        """
        Install a Python package using pip.
        
        Args:
            package_name: Package to install
            description: Human-readable description
            
        Returns:
            True if successful, False otherwise
        """
        if description:
            logger.info(f"Installing {description} ({package_name})...")
        else:
            logger.info(f"Installing {package_name}...")
        
        try:
            # Use subprocess to install package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name}")
                return True
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Installation of {package_name} timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing {package_name}: {e}")
            return False
    
    def install_missing_dependencies(self, auto_install: bool = False) -> Dict[str, bool]:
        """
        Install missing dependencies.
        
        Args:
            auto_install: If True, installs automatically. If False, prompts user.
            
        Returns:
            Dict mapping package name to installation success status
        """
        missing_required, missing_optional = self.check_dependencies()
        results = {}
        
        # Handle required dependencies
        if missing_required:
            logger.warning(f"Missing {len(missing_required)} required dependency(ies)")
            
            if not auto_install:
                print("\n" + "="*80)
                print("REQUIRED DEPENDENCIES MISSING")
                print("="*80)
                for module, package, desc in missing_required:
                    print(f"  - {desc} ({package})")
                print("\nThe app can install these automatically.")
                
                try:
                    response = input("\nInstall missing dependencies now? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        logger.error("User declined dependency installation")
                        return results
                except (EOFError, OSError):
                    # Non-interactive - try auto-install
                    logger.info("Non-interactive mode - attempting auto-install")
            
            # Install required
            for module, package, desc in missing_required:
                results[package] = self.install_dependency(package, desc)
        
        # Handle optional dependencies (inform user but don't auto-install)
        if missing_optional:
            logger.info(f"Missing {len(missing_optional)} optional dependency(ies)")
            if auto_install:
                for module, package, desc in missing_optional:
                    logger.info(f"Installing optional: {desc}")
                    results[package] = self.install_dependency(package, desc)
            else:
                print("\nOptional dependencies (for enhanced features):")
                for module, package, desc in missing_optional:
                    print(f"  - {desc} ({package})")
                print("  (These can be installed later if needed)")
        
        return results
    
    def run_first_time_setup(self, auto_install_deps: bool = False) -> bool:
        """
        Run complete first-time setup.
        
        Args:
            auto_install_deps: If True, automatically installs missing dependencies
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("="*80)
        logger.info("FIRST-TIME SETUP")
        logger.info("="*80)
        logger.info("Setting up app for first use...\n")
        
        # Check dependencies
        missing_required, missing_optional = self.check_dependencies()
        
        if missing_required:
            logger.info("Checking and installing dependencies...")
            dep_results = self.install_missing_dependencies(auto_install=auto_install_deps)
            
            # Check if all required were installed
            failed = [pkg for pkg, success in dep_results.items() if not success]
            if failed:
                logger.error(f"Failed to install: {', '.join(failed)}")
                logger.error("Please install manually: pip install " + " ".join(failed))
                return False
            
            # Re-check after installation
            missing_required, _ = self.check_dependencies()
            if missing_required:
                logger.error("Some dependencies still missing after installation")
                return False
        
        # Download models
        logger.info("\nChecking models...")
        models_status = self.ensure_models()
        
        failed_models = [name for name, success in models_status.items() if not success and name != "yolov8n.onnx"]
        if failed_models:
            logger.warning(f"Some models not available: {', '.join(failed_models)}")
            logger.info("Models will be downloaded when first needed")
        
        # Create necessary directories
        logger.info("\nCreating directories...")
        dirs_to_create = [
            get_outputs_dir(),
            get_data_dir(),
            get_config_dir(),
            get_models_dir(),
        ]
        
        for dir_path in dirs_to_create:
            ensure_directory(dir_path)
            logger.debug(f"Created/verified directory: {dir_path}")
        
        # Mark setup complete
        self.mark_setup_complete()
        
        logger.info("\n" + "="*80)
        logger.info("SETUP COMPLETE")
        logger.info("="*80)
        logger.info("App is ready to use!")
        
        return True
    
    def verify_setup(self) -> Tuple[bool, List[str]]:
        """
        Verify that setup is complete and all components are available.
        
        Returns:
            Tuple of (is_ready, list_of_warnings)
        """
        warnings = []
        is_ready = True
        
        # Check dependencies
        missing_required, missing_optional = self.check_dependencies()
        if missing_required:
            is_ready = False
            warnings.append(f"Missing required dependencies: {[d[2] for d in missing_required]}")
        
        # Check models (ONNX can be auto-generated)
        models_status = self.ensure_models()
        if not models_status.get("yolov8n.pt", False):
            is_ready = False
            warnings.append("YOLO PyTorch model not found")
        
        return is_ready, warnings

