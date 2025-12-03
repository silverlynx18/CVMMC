# Quick Setup Guide - Application Code in Docker

## Overview

The application code is now self-contained in `gcp-build/app-code/`. This folder contains everything needed to build and run the Docker image.

## Initial Setup

1. **Prepare the application code:**
   ```bash
   cd gcp-build
   ./prepare-app-code.sh
   ```

2. **Optional: Download models (or let them auto-download):**
   ```bash
   ./download-models.sh
   ```

3. **Verify structure:**
   ```bash
   ls -la app-code/
   # Should show: app/, config/, models/, launch_gui.py, setup.py, etc.
   ```

## What's Included

The `app-code/` folder contains:
- ✅ All application code (`app/` directory)
- ✅ Entry point (`launch_gui.py`)
- ✅ Configuration files (`config/`, `counting_zone.json`)
- ✅ Python dependencies (`requirements-cuda-docker.txt`)
- ✅ Directory structure for models, data, outputs

## How Docker Build Works

1. **Cloud Build clones your repository** (entire repo)
2. **Build context is repository root** (`.`)
3. **Dockerfile copies from `gcp-build/app-code/`** to `/app/` in container
4. **Only app-code contents** are included in the image

## Updating Code

When you update your application:

```bash
# 1. Make changes to your source code (in repository root)
# 2. Update app-code folder:
cd gcp-build
./prepare-app-code.sh

# 3. Commit and push:
git add gcp-build/app-code/
git commit -m "Update application code"
git push

# 4. Cloud Build will automatically build with new code
```

## Models

Models are **automatically downloaded** when the application runs. You don't need to include them unless you want to:

- **Pre-download:** Run `./download-models.sh` before building
- **Auto-download:** Let the app download them on first run (recommended)

## File Structure

```
gcp-build/
├── app-code/              ← Self-contained application
│   ├── app/               ← All application modules
│   ├── config/            ← Configuration files
│   ├── models/            ← Model files (auto-downloaded)
│   ├── launch_gui.py     ← Entry point
│   └── ...
├── Dockerfile.cuda        ← Builds from app-code/
├── cloudbuild.yaml        ← Cloud Build config
└── prepare-app-code.sh    ← Updates app-code/
```

## Next Steps

1. ✅ Application code is ready in `gcp-build/app-code/`
2. ✅ Dockerfile is configured to use it
3. ✅ Cloud Build is configured correctly
4. 🚀 Ready to build!

See `APP_CODE_README.md` for detailed documentation.

