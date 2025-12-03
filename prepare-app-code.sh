#!/bin/bash
# Script to copy all application code and necessary files to gcp-build/app-code

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_CODE_DIR="$SCRIPT_DIR/app-code"

echo -e "${BLUE}=========================================="
echo "Preparing Application Code for Docker Build"
echo "==========================================${NC}"
echo ""

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "$APP_CODE_DIR/app"
mkdir -p "$APP_CODE_DIR/config"
mkdir -p "$APP_CODE_DIR/models"
mkdir -p "$APP_CODE_DIR/data"
mkdir -p "$APP_CODE_DIR/outputs"
mkdir -p "$APP_CODE_DIR/processed_videos"

# Copy application code (exclude cache and macOS files)
echo -e "${GREEN}Copying application code...${NC}"
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='._*' --exclude='.DS_Store' \
    "$REPO_ROOT/app/" "$APP_CODE_DIR/app/" 2>/dev/null || \
    cp -r "$REPO_ROOT/app/"* "$APP_CODE_DIR/app/" 2>/dev/null || true

# Clean up any cache files that were copied
find "$APP_CODE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$APP_CODE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$APP_CODE_DIR" -type f -name "._*" -delete 2>/dev/null || true
find "$APP_CODE_DIR" -type f -name ".DS_Store" -delete 2>/dev/null || true

# Copy entry point
echo -e "${GREEN}Copying entry point...${NC}"
cp "$REPO_ROOT/launch_gui.py" "$APP_CODE_DIR/" 2>/dev/null || true

# Copy setup.py
echo -e "${GREEN}Copying setup.py...${NC}"
cp "$REPO_ROOT/setup.py" "$APP_CODE_DIR/" 2>/dev/null || true

# Copy config files
echo -e "${GREEN}Copying config files...${NC}"
if [ -f "$REPO_ROOT/config/default_config.json" ]; then
    cp "$REPO_ROOT/config/default_config.json" "$APP_CODE_DIR/config/" 2>/dev/null || true
fi
if [ -f "$REPO_ROOT/counting_zone.json" ]; then
    cp "$REPO_ROOT/counting_zone.json" "$APP_CODE_DIR/" 2>/dev/null || true
fi

# Copy requirements file
echo -e "${GREEN}Copying requirements file...${NC}"
if [ -f "$SCRIPT_DIR/requirements-cuda-docker.txt" ]; then
    cp "$SCRIPT_DIR/requirements-cuda-docker.txt" "$APP_CODE_DIR/" 2>/dev/null || true
fi

# Create .dockerignore for app-code
echo -e "${GREEN}Creating .dockerignore...${NC}"
cat > "$APP_CODE_DIR/.dockerignore" << 'EOF'
# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/
.venv

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Build artifacts
build/
dist/
*.egg-info/
*.egg

# Test files
.pytest_cache/
.coverage
htmlcov/
*.log
test_*.py
tests/

# Documentation
docs/
*.md
!README.md

# Data directories (will be mounted as volumes)
data/*
!data/.gitkeep
outputs/*
!outputs/.gitkeep
processed_videos/*
!processed_videos/.gitkeep

# Large files
*.mp4
*.avi
*.mov
*.mkv

# Jupyter notebooks
*.ipynb
.ipynb_checkpoints
EOF

# Create placeholder files for directories
echo -e "${GREEN}Creating placeholder files...${NC}"
touch "$APP_CODE_DIR/models/.gitkeep"
touch "$APP_CODE_DIR/data/.gitkeep"
touch "$APP_CODE_DIR/outputs/.gitkeep"
touch "$APP_CODE_DIR/processed_videos/.gitkeep"

# Create README for app-code
echo -e "${GREEN}Creating README...${NC}"
cat > "$APP_CODE_DIR/README.md" << 'EOF'
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
EOF

echo ""
echo -e "${GREEN}✅ Application code prepared successfully!${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  - Application code: $APP_CODE_DIR/app/"
echo "  - Entry point: $APP_CODE_DIR/launch_gui.py"
echo "  - Config files: $APP_CODE_DIR/config/"
echo "  - Models directory: $APP_CODE_DIR/models/"
echo ""
echo -e "${YELLOW}Note: Models will be downloaded automatically during Docker build.${NC}"
echo -e "${YELLOW}If you want to include pre-downloaded models, copy them to: $APP_CODE_DIR/models/${NC}"

