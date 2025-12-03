#!/bin/bash
# Script to fix Cloud Build permissions for Artifact Registry

set -e

PROJECT_ID="alert-snowfall-472121-b5"
REGION="us-central1"
REPOSITORY="cv-mm-images"

echo "🔧 Fixing Cloud Build permissions for Artifact Registry..."
echo ""

# Get the Cloud Build service account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "Project ID: $PROJECT_ID"
echo "Project Number: $PROJECT_NUMBER"
echo "Cloud Build Service Account: $CLOUD_BUILD_SA"
echo ""

# Check if repository exists
echo "📦 Checking if Artifact Registry repository exists..."
if gcloud artifacts repositories describe $REPOSITORY \
    --location=$REGION \
    --project=$PROJECT_ID &>/dev/null; then
    echo "✅ Repository '$REPOSITORY' exists"
else
    echo "❌ Repository '$REPOSITORY' not found!"
    echo "Creating repository..."
    gcloud artifacts repositories create $REPOSITORY \
        --repository-format=docker \
        --location=$REGION \
        --description="CV_MM Docker images for CUDA GPU" \
        --project=$PROJECT_ID
    echo "✅ Repository created"
fi

echo ""
echo "🔐 Granting Artifact Registry Writer role to Cloud Build service account..."

# Grant Artifact Registry Writer role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/artifactregistry.writer" \
    --condition=None

echo "✅ Permissions granted!"
echo ""
echo "🎉 Setup complete! You can now run Cloud Build and it should be able to push images."
echo ""
echo "To test, run:"
echo "  gcloud builds submit --config=cloudbuild.yaml"

