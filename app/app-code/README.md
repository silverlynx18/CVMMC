# Application Code for Docker Build

This directory contains all the application code and files needed to build the Docker image.

## Structure

```
app-code/
├── app/                    # Main application package
├── config/                 # Configuration files
│   └── default_config.json
├── models/                 # Model files (will be downloaded during build)
├── data/                   # Data directory (mounted as volume)
├── outputs/                # Output directory (mounted as volume)
├── processed_videos/      # Processed videos directory (mounted as volume)
├── launch_gui.py          # Application entry point
├── setup.py               # Python package setup
├── counting_zone.json     # Zone configuration
├── requirements-cuda-docker.txt  # Python dependencies
└── .dockerignore          # Files to exclude from Docker build
```

## Models

Models will be automatically downloaded during the Docker build process:
- `yolov8n.pt` - YOLOv8 nano model (auto-downloaded via Ultralytics)
- `sam3_hiera_large.pt` - SAM3 model (optional, can download from HuggingFace)

If you want to pre-download models, place them in the `models/` directory before building.

## Usage

This directory is used by the Dockerfile during the build process. The Dockerfile copies everything from this directory into the container at `/app/`.
