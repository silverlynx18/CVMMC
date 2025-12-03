#!/bin/bash
# Script to push the fixed Dockerfiles to GitHub

set -e

echo "=========================================="
echo "Pushing Fixed Dockerfiles to Repository"
echo "=========================================="
echo

cd "/Volumes/Extreme SSD/CV_MM"

# Check if files are fixed
echo "Checking if files are fixed..."
if grep -q "libdc1394-22-dev" Dockerfile.cuda Dockerfile.amd 2>/dev/null; then
    echo "ERROR: Files still contain libdc1394-22-dev!"
    exit 1
fi
echo "✓ Files are fixed locally"

# Add files to git
echo
echo "Adding files to git..."
git add Dockerfile.cuda Dockerfile.amd cloudbuild.yaml requirements-cuda-docker.txt .dockerignore 2>/dev/null || true

# Check status
echo
echo "Files ready to commit:"
git status --short | grep -E "Dockerfile|cloudbuild|requirements-cuda|dockerignore" || echo "No changes detected"

echo
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo
echo "1. Review changes:"
echo "   git diff --cached Dockerfile.cuda Dockerfile.amd"
echo
echo "2. Commit the changes:"
echo "   git commit -m 'Fix: Remove libdc1394-22-dev package (not available in Ubuntu 22.04)'"
echo
echo "3. Push to repository:"
echo "   git push origin main"
echo
echo "4. Re-run Cloud Build:"
echo "   gcloud builds submit --config=cloudbuild.yaml"
echo

