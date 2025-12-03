# Fix Artifact Registry Permissions

## Problem
The error `name unknown: Repository "cv-mm-images" not found` occurs because Cloud Build doesn't have permission to push to Artifact Registry.

## Solution

Run the fix script:

```bash
cd gcp-build
./fix-permissions.sh
```

Or manually run these commands:

```bash
PROJECT_ID="alert-snowfall-472121-b5"
REGION="us-central1"
REPOSITORY="cv-mm-images"

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Grant Artifact Registry Writer role to Cloud Build service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

## Verify Repository Exists

```bash
gcloud artifacts repositories list --location=us-central1 --project=alert-snowfall-472121-b5
```

If it doesn't exist, create it:

```bash
gcloud artifacts repositories create cv-mm-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV_MM Docker images for CUDA GPU" \
    --project=alert-snowfall-472121-b5
```

## After Fixing Permissions

1. Commit and push your changes to GitHub
2. Re-run your Cloud Build trigger
3. The build should now successfully push the CUDA image

