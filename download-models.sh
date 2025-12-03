#!/bin/bash
# Script to download required models for the application
# This can be run before building the Docker image to include models in the image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/app-code/models"

echo "Downloading models to: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+"
    exit 1
fi

# Download YOLOv8n model using Ultralytics
echo ""
echo "Downloading YOLOv8n model..."
python3 << EOF
import sys
try:
    from ultralytics import YOLO
    import os
    
    models_dir = "$MODELS_DIR"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "yolov8n.pt")
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        sys.exit(0)
    
    print("Downloading YOLOv8n model (this may take a moment)...")
    model = YOLO("yolov8n.pt")
    
    # Save to models directory
    import shutil
    default_path = os.path.expanduser("~/.ultralytics/models/yolov8n.pt")
    if os.path.exists(default_path):
        shutil.copy2(default_path, model_path)
        print(f"✅ Model downloaded successfully: {model_path}")
    else:
        # Try to find where ultralytics saved it
        import ultralytics
        ultralytics_dir = os.path.dirname(ultralytics.__file__)
        possible_paths = [
            os.path.join(ultralytics_dir, "models", "yolov8n.pt"),
            os.path.expanduser("~/yolov8n.pt"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                shutil.copy2(path, model_path)
                print(f"✅ Model downloaded successfully: {model_path}")
                sys.exit(0)
        print("⚠️  Model downloaded but could not locate file. It will be auto-downloaded during runtime.")
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    print("   Model will be auto-downloaded during Docker build or runtime.")
except Exception as e:
    print(f"⚠️  Error downloading model: {e}")
    print("   Model will be auto-downloaded during Docker build or runtime.")
EOF

echo ""
echo "✅ Model download script completed."
echo ""
echo "Note: SAM3 model (sam3_hiera_large.pt) can be downloaded from HuggingFace"
echo "      during runtime or you can add it manually to the models directory."

