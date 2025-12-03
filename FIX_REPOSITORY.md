# Fix: Repository Not Found Error

## Problem
The build is failing with:
```
name unknown: Repository "cv-mm-images" not found
```

## Solution: Create Artifact Registry Repository

The Artifact Registry repository needs to be created before images can be pushed.

### Quick Fix (Command Line)

```bash
# Set your project (if not already set)
gcloud config set project alert-snowfall-472121-b5

# Create the Artifact Registry repository
gcloud artifacts repositories create cv-mm-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV_MM Docker images for CUDA and AMD GPU"
```

### Via Cloud Console

1. Go to: https://console.cloud.google.com/artifacts
2. Click "Create Repository"
3. Fill in:
   - **Name**: `cv-mm-images`
   - **Format**: Docker
   - **Mode**: Standard
   - **Region**: `us-central1` (or your preferred region)
4. Click "Create"

### Verify Repository Exists

```bash
gcloud artifacts repositories list --location=us-central1
```

You should see `cv-mm-images` in the list.

## After Creating Repository

Once the repository is created, you can:

1. **Re-run the build** (if using a trigger, it will automatically retry)
2. **Or manually trigger**:
   ```bash
   gcloud builds triggers run YOUR_TRIGGER_NAME --branch=main
   ```

## Note

The CUDA image built successfully! Once the repository exists, both images will be pushed successfully.

