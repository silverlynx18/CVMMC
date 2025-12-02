#!/bin/bash
# Setup script for Google Cloud Build
# This script helps set up the necessary GCP resources

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud SDK is not installed."
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_info "Google Cloud SDK found: $(gcloud --version | head -n 1)"

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")

if [ -z "$CURRENT_PROJECT" ]; then
    print_error "No project set. Please set a project:"
    echo "  gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_info "Current project: $CURRENT_PROJECT"

# Prompt for confirmation
read -p "Continue with project '$CURRENT_PROJECT'? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warn "Setup cancelled."
    exit 0
fi

# Prompt for region
echo
print_step "Setting up Artifact Registry..."
read -p "Enter Artifact Registry region (default: us-central1): " REGION
REGION=${REGION:-us-central1}
print_info "Using region: $REGION"

# Prompt for repository name
read -p "Enter repository name (default: cv-mm-images): " REPO_NAME
REPO_NAME=${REPO_NAME:-cv-mm-images}
print_info "Using repository: $REPO_NAME"

# Check if repository exists
print_step "Checking if repository exists..."
if gcloud artifacts repositories describe $REPO_NAME --location=$REGION &>/dev/null; then
    print_info "Repository '$REPO_NAME' already exists in $REGION"
else
    print_step "Creating Artifact Registry repository..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="CV_MM Docker images for CUDA and AMD GPU"
    
    if [ $? -eq 0 ]; then
        print_info "Repository created successfully!"
    else
        print_error "Failed to create repository"
        exit 1
    fi
fi

# Enable necessary APIs
print_step "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable containerregistry.googleapis.com
print_info "APIs enabled"

# Set up permissions
print_step "Setting up Cloud Build permissions..."
PROJECT_NUMBER=$(gcloud projects describe $CURRENT_PROJECT --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

print_info "Granting Artifact Registry Writer role to Cloud Build service account..."
gcloud projects add-iam-policy-binding $CURRENT_PROJECT \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/artifactregistry.writer" \
    --condition=None 2>/dev/null || print_warn "Role may already be granted"

# Configure Docker authentication
print_step "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
print_info "Docker authentication configured"

# Update cloudbuild.yaml with correct values
print_step "Updating cloudbuild.yaml with your settings..."
if [ -f "cloudbuild.yaml" ]; then
    # Create backup
    cp cloudbuild.yaml cloudbuild.yaml.backup
    
    # Update substitutions (simple sed replacement)
    sed -i.bak "s/_GCR_REGION: '.*'/_GCR_REGION: '${REGION}'/" cloudbuild.yaml
    sed -i.bak "s/_GCR_REPOSITORY: '.*'/_GCR_REPOSITORY: '${REPO_NAME}'/" cloudbuild.yaml
    
    # Remove backup files
    rm -f cloudbuild.yaml.bak
    
    print_info "cloudbuild.yaml updated"
else
    print_warn "cloudbuild.yaml not found in current directory"
fi

# Summary
echo
print_info "=========================================="
print_info "Setup Complete!"
print_info "=========================================="
echo
print_info "Project ID: $CURRENT_PROJECT"
print_info "Region: $REGION"
print_info "Repository: $REPO_NAME"
echo
print_step "Next steps:"
echo "  1. Copy files from gcp-build/ to your repository root"
echo "  2. Submit build: gcloud builds submit --config=cloudbuild.yaml"
echo "  3. Or create a trigger in Cloud Console for automatic builds"
echo
print_info "Repository URL:"
echo "  ${REGION}-docker.pkg.dev/${CURRENT_PROJECT}/${REPO_NAME}"
echo

