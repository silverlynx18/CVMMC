#!/usr/bin/env python3
"""Launch the Workflow GUI application."""

import sys
import logging
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for GUI launcher."""
    # Run first-time setup if needed
    from app.setup_manager import SetupManager
    
    setup_manager = SetupManager()
    
    if not setup_manager.check_setup_complete():
        logger.info("Running first-time setup...")
        if not setup_manager.run_first_time_setup(auto_install_deps=True):
            logger.error("Setup failed. Please check errors above.")
            sys.exit(1)
    
    # Verify setup
    is_ready, warnings = setup_manager.verify_setup()
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    if not is_ready:
        logger.error("Setup incomplete. Please run setup again.")
        sys.exit(1)
    
    # Launch GUI
    try:
        from app.workflow_gui import WorkflowGUI
        
        logger.info("Launching Pedestrian Detection Workflow GUI...")
        app = WorkflowGUI()
        app.run()
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.info("\nAttempting to install missing dependencies...")
        
        # Try to install missing deps
        setup_manager.install_missing_dependencies(auto_install=True)
        
        # Try again
        try:
            from app.workflow_gui import WorkflowGUI
            logger.info("Retrying GUI launch...")
            app = WorkflowGUI()
            app.run()
        except Exception as e2:
            logger.error(f"Failed after dependency installation: {e2}")
            logger.info("\nPlease install manually:")
            logger.info("  pip install customtkinter")
            logger.info("  pip install -r requirements.txt")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


