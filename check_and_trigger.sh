#!/bin/bash
# Script to check GitHub status and trigger Cloud Build

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "Cloud Build Trigger Helper"
echo "==========================================${NC}"
echo

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get latest local commit
LOCAL_COMMIT=$(git rev-parse HEAD)
LOCAL_SHORT=$(git rev-parse --short HEAD)

echo -e "${GREEN}Local Repository:${NC}"
echo "  Latest commit: $LOCAL_SHORT"
echo "  Full SHA: $LOCAL_COMMIT"
echo

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    echo "  Consider committing and pushing first"
    echo
fi

# Try to get remote commit (if available)
REMOTE_COMMIT=$(git ls-remote origin HEAD 2>/dev/null | cut -f1 || echo "")
if [ -n "$REMOTE_COMMIT" ]; then
    REMOTE_SHORT=$(echo $REMOTE_COMMIT | cut -c1-7)
    echo -e "${GREEN}Remote Repository (GitHub):${NC}"
    echo "  Latest commit: $REMOTE_SHORT"
    echo "  Full SHA: $REMOTE_COMMIT"
    echo
    
    if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
        echo -e "${YELLOW}⚠ Local and remote commits differ!${NC}"
        echo "  Make sure you've pushed your changes:"
        echo "    git push origin main"
        echo
    else
        echo -e "${GREEN}✓ Local and remote are in sync${NC}"
        echo
    fi
else
    echo -e "${YELLOW}Could not fetch remote commit info${NC}"
    echo "  Make sure remote is configured: git remote -v"
    echo
fi

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}Google Cloud SDK not found${NC}"
    echo "  Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get current project
PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -z "$PROJECT" ]; then
    echo -e "${YELLOW}No GCP project set${NC}"
    echo "  Set with: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}Google Cloud:${NC}"
echo "  Project: $PROJECT"
echo

# Check for triggers
echo -e "${BLUE}Checking Cloud Build triggers...${NC}"
TRIGGERS=$(gcloud builds triggers list --format="value(name)" 2>/dev/null || echo "")

if [ -z "$TRIGGERS" ]; then
    echo -e "${YELLOW}No triggers found${NC}"
    echo
    echo "Options:"
    echo "  1. Create a trigger (recommended):"
    echo "     See TRIGGER_BUILD.md for instructions"
    echo
    echo "  2. Manual build submission:"
    echo "     gcloud builds submit --config=cloudbuild.yaml"
    echo
else
    echo -e "${GREEN}Found triggers:${NC}"
    echo "$TRIGGERS" | while read trigger; do
        echo "  - $trigger"
    done
    echo
    echo "To trigger a build:"
    echo "  gcloud builds triggers run TRIGGER_NAME --branch=main"
    echo
fi

# Ask if user wants to submit build
echo -e "${BLUE}=========================================="
echo "Next Steps"
echo "==========================================${NC}"
echo
echo "1. If you have a trigger set up:"
echo "   - Go to: https://console.cloud.google.com/cloud-build/triggers"
echo "   - Click 'Run' on your trigger"
echo "   - Or: gcloud builds triggers run TRIGGER_NAME --branch=main"
echo
echo "2. Manual build (uses local files or GitHub if configured):"
echo "   gcloud builds submit --config=cloudbuild.yaml"
echo
echo "3. View recent builds:"
echo "   gcloud builds list --limit=5"
echo

