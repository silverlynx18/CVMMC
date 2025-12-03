#!/bin/bash
# Script to commit and push the app-code directory to GitHub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Committing app-code to Git"
echo "=========================================="
echo ""

cd "$REPO_ROOT"

# Check if app-code exists
if [ ! -d "gcp-build/app-code" ]; then
    echo "Error: gcp-build/app-code directory not found!"
    echo "Run ./prepare-app-code.sh first to create it."
    exit 1
fi

# Check git status
echo "Checking git status..."
if [ -n "$(git status --porcelain gcp-build/app-code/)" ]; then
    echo "Found changes in gcp-build/app-code/"
    
    # Add app-code directory
    echo "Adding gcp-build/app-code/ to git..."
    git add gcp-build/app-code/
    
    # Show what will be committed
    echo ""
    echo "Files to be committed:"
    git status --short gcp-build/app-code/ | head -20
    
    if [ $(git status --short gcp-build/app-code/ | wc -l) -gt 20 ]; then
        echo "... and more files"
    fi
    
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -m "Add application code for Docker build
        
- Includes all application modules from app/
- Includes entry point (launch_gui.py)
- Includes configuration files
- Includes requirements file
- Ready for Cloud Build Docker image creation"
        
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
            echo "Cloud Build will now be able to find gcp-build/app-code/"
        else
            echo ""
            echo "⚠️  Remember to push: git push origin main"
        fi
    else
        echo "Cancelled. Changes are staged but not committed."
        echo "To commit manually: git commit -m 'Add application code for Docker build'"
    fi
else
    echo "No changes detected in gcp-build/app-code/"
    echo "Directory may already be committed, or is empty."
fi

