# How to Update Your Repository

The build is still failing because your GitHub repository still has the old Dockerfiles with `libdc1394-22-dev`. You need to update the files in your repository.

## Quick Fix Steps

### Option 1: Update Files Directly in GitHub (Easiest)

1. Go to your repository: https://github.com/silverlynx18/CVMMC
2. Navigate to `Dockerfile.cuda` and `Dockerfile.amd` in the root
3. Edit each file and remove the line:
   ```
   libdc1394-22-dev \
   ```
4. Save/commit the changes
5. Re-run the Cloud Build

### Option 2: Update via Git (Recommended)

```bash
# 1. Copy the fixed files from gcp-build to your repo root
cd /Volumes/Extreme\ SSD/CV_MM
cp gcp-build/Dockerfile.cuda .
cp gcp-build/Dockerfile.amd .

# 2. Verify the files are correct (should NOT contain libdc1394-22-dev)
grep -n "libdc1394" Dockerfile.cuda Dockerfile.amd
# Should return nothing or show it's been removed

# 3. Commit and push
git add Dockerfile.cuda Dockerfile.amd
git commit -m "Fix: Remove libdc1394-22-dev package (not available in Ubuntu 22.04)"
git push origin main

# 4. Re-run Cloud Build
gcloud builds submit --config=cloudbuild.yaml
```

### Option 3: Manual Edit in GitHub Web Interface

1. Go to: https://github.com/silverlynx18/CVMMC/blob/main/Dockerfile.cuda
2. Click the pencil icon (Edit)
3. Find line with `libdc1394-22-dev \` and delete it
4. Commit changes
5. Repeat for `Dockerfile.amd`
6. Re-run Cloud Build

## Verify the Fix

After updating, verify the files don't contain the problematic package:

```bash
# Check locally
grep "libdc1394" Dockerfile.cuda Dockerfile.amd
# Should return nothing

# Or check in GitHub web interface
# Search for "libdc1394" - should find nothing
```

## What Changed

**Before (causing error):**
```dockerfile
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \    ← This line causes the error
    wget \
```

**After (fixed):**
```dockerfile
    libtbb2 \
    libtbb-dev \
    wget \                 ← Line removed
```

## After Updating

Once you've updated the files in your repository and pushed the changes:

1. **Re-trigger the build** (if using triggers, just push again)
2. **Or manually submit**: `gcloud builds submit --config=cloudbuild.yaml`

The build should now proceed past the package installation step.

