# Application Code Structure

This document explains how the application code is organized for Docker builds.

## Directory Structure

```
gcp-build/
├── app-code/                    # Self-contained application code
│   ├── app/                      # Main application package
│   │   ├── __init__.py
│   │   ├── workflow_gui.py      # GUI application
│   │   ├── yolo_detector.py     # YOLO detector
│   │   ├── sam3_detector.py     # SAM3 detector
│   │   └── ... (all app modules)
│   ├── config/                   # Configuration files
│   │   └── default_config.json
│   ├── models/                   # Model files (auto-downloaded)
│   ├── data/                     # Data directory (volume mount)
│   ├── outputs/                  # Output directory (volume mount)
│   ├── processed_videos/        # Processed videos (volume mount)
│   ├── launch_gui.py            # Application entry point
│   ├── setup.py                 # Python package setup
│   ├── counting_zone.json      # Zone configuration
│   ├── requirements-cuda-docker.txt  # Python dependencies
│   └── README.md                # This file
├── Dockerfile.cuda              # CUDA Dockerfile
├── cloudbuild.yaml              # Cloud Build configuration
├── prepare-app-code.sh          # Script to prepare app-code
└── download-models.sh           # Script to download models
```

## How It Works

### 1. Preparing Application Code

Run the preparation script to copy all necessary files:

```bash
cd gcp-build
./prepare-app-code.sh
```

This script:
- Copies all application code from `../app/` to `app-code/app/`
- Copies entry point (`launch_gui.py`) and setup files
- Copies configuration files
- Creates directory structure for models, data, outputs
- Creates `.dockerignore` for the app-code directory

### 2. Including Models

Models can be included in two ways:

**Option A: Auto-download during build (Recommended)**
- Models are automatically downloaded when the application runs
- YOLOv8n model downloads via Ultralytics
- No action needed

**Option B: Pre-download models**
```bash
# Download models before building
./download-models.sh

# Or manually download and place in app-code/models/
# - yolov8n.pt (YOLOv8 nano model)
# - sam3_hiera_large.pt (SAM3 model, optional)
```

### 3. Docker Build Process

When Cloud Build runs:

1. **Repository is cloned** - Cloud Build clones your GitHub repository
2. **Build context is repository root** - The `.` in `cloudbuild.yaml` means repository root
3. **Dockerfile copies from `gcp-build/app-code/`** - The Dockerfile copies everything from this directory to `/app/` in the container
4. **Application runs** - The container runs `launch_gui.py` as the entry point

## Files Included

### Application Code
- **`app/`** - Complete application package with all modules
- **`launch_gui.py`** - Main entry point for the GUI application
- **`setup.py`** - Python package configuration

### Configuration
- **`config/default_config.json`** - Default application configuration
- **`counting_zone.json`** - Zone definitions for counting

### Dependencies
- **`requirements-cuda-docker.txt`** - Python dependencies (PyTorch installed separately)

### Directories (Created but Empty)
- **`models/`** - Model files (auto-downloaded or pre-downloaded)
- **`data/`** - Input data (mounted as volume in production)
- **`outputs/`** - Output files (mounted as volume in production)
- **`processed_videos/`** - Processed videos (mounted as volume in production)

## Updating Application Code

When you make changes to your application:

1. **Update source code** in the main repository (`../app/`, etc.)
2. **Run preparation script** to copy changes:
   ```bash
   cd gcp-build
   ./prepare-app-code.sh
   ```
3. **Commit and push** to GitHub:
   ```bash
   git add gcp-build/app-code/
   git commit -m "Update application code"
   git push
   ```
4. **Cloud Build** will automatically build with the new code

## Models

### Required Models

1. **YOLOv8n (`yolov8n.pt`)**
   - Auto-downloaded via Ultralytics when first needed
   - Size: ~6 MB
   - Used for initial pedestrian detection

2. **SAM3 (`sam3_hiera_large.pt`)** - Optional
   - Can be downloaded from HuggingFace
   - Size: ~1.5 GB
   - Used for advanced segmentation (if SAM3 detector is enabled)

### Model Download

Models are automatically downloaded when the application runs. If you want to include them in the Docker image:

```bash
# Download models
cd gcp-build
./download-models.sh

# Verify models are in place
ls -lh app-code/models/
```

## Docker Build Context

The Docker build context is the **repository root** (`.`), but the Dockerfile only copies from `gcp-build/app-code/`. This means:

- ✅ Only application code is included (not entire repository)
- ✅ Smaller build context = faster builds
- ✅ Clean separation of build files and application code
- ✅ `.dockerignore` in app-code excludes unnecessary files

## Troubleshooting

### Models Not Found
- Models will auto-download on first run
- Check `app-code/models/` directory exists
- Verify write permissions

### Missing Files
- Run `./prepare-app-code.sh` again
- Check that source files exist in repository root
- Verify file paths in the script

### Build Fails
- Check Dockerfile paths are correct
- Verify `gcp-build/app-code/` exists
- Check Cloud Build logs for specific errors

