#!/bin/bash
# Script to commit and push all gcp-build files to GitHub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Committing gcp-build Directory to Git"
echo "=========================================="
echo ""

cd "$REPO_ROOT"

# Check if gcp-build exists
if [ ! -d "gcp-build" ]; then
    echo "Error: gcp-build directory not found!"
    exit 1
fi

# Check if app-code exists
if [ ! -d "gcp-build/app-code" ]; then
    echo "Warning: gcp-build/app-code directory not found!"
    echo "Run ./prepare-app-code.sh first to create it."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check git status
echo "Checking git status..."
if [ -n "$(git status --porcelain gcp-build/)" ]; then
    echo "Found changes in gcp-build/"
    
    # Add entire gcp-build directory
    echo "Adding gcp-build/ to git..."
    git add gcp-build/
    
    # Show what will be committed
    echo ""
    echo "Files to be committed:"
    git status --short gcp-build/ | head -30
    
    FILE_COUNT=$(git status --short gcp-build/ | wc -l)
    if [ $FILE_COUNT -gt 30 ]; then
        echo "... and $((FILE_COUNT - 30)) more files"
    fi
    
    echo ""
    echo "Total files: $FILE_COUNT"
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -m "Add Google Cloud Build configuration and application code

- Dockerfiles for CUDA and AMD GPU support
- Cloud Build configuration (cloudbuild.yaml)
- Application code in app-code/ directory
- Setup and utility scripts
- Documentation and guides

Ready for Cloud Build Docker image creation"
        
        echo ""
        echo "✅ Committed successfully!"
        echo ""
        echo "Next step: Push to GitHub"
        echo "  git push origin main"
        echo ""
        read -p "Push to GitHub now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push origin main
            echo ""
            echo "✅ Pushed to GitHub!"
            echo ""
            echo "Cloud Build will now be able to find all required files."
            echo ""
            echo "You can now trigger Cloud Build:"
            echo "  - Via Cloud Console: https://console.cloud.google.com/cloud-build/triggers"
            echo "  - Or it will trigger automatically if you have a trigger set up"
        else
            echo ""
            echo "⚠️  Remember to push: git push origin main"
        fi
    else
        echo "Cancelled. Changes are staged but not committed."
        echo "To commit manually: git commit -m 'Add Google Cloud Build configuration'"
    fi
else
    echo "No changes detected in gcp-build/"
    echo "Directory may already be committed."
    echo ""
    echo "Current git status:"
    git status --short gcp-build/ || echo "No changes"
fi

