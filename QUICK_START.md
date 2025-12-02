# Quick Start - 5 Minute Setup

## Step 1: Run Setup Script

```bash
cd gcp-build
./setup.sh
```

This will:
- ✅ Check Google Cloud SDK installation
- ✅ Create Artifact Registry repository
- ✅ Enable necessary APIs
- ✅ Set up permissions
- ✅ Configure Docker authentication
- ✅ Update cloudbuild.yaml with your settings

## Step 2: Copy Files to Repository Root

**IMPORTANT:** Cloud Build needs these files in your repository root.

```bash
# If using git repository
cd /path/to/your/repo
cp gcp-build/Dockerfile.cuda .
cp gcp-build/Dockerfile.amd .
cp gcp-build/cloudbuild.yaml .
cp gcp-build/requirements-cuda-docker.txt .
cp gcp-build/.dockerignore .

# Commit and push
git add Dockerfile.* cloudbuild.yaml requirements-cuda-docker.txt .dockerignore
git commit -m "Add Docker build configuration"
git push
```

## Step 3: Submit Build

```bash
# From repository root
gcloud builds submit --config=cloudbuild.yaml
```

Or create a trigger in Cloud Console for automatic builds on every push.

## That's It! 🎉

Your images will be built and pushed to:
```
{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/cv-mm:{TAG}
```

## Pull Images

```bash
docker pull {REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/cv-mm:cuda-latest
docker pull {REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/cv-mm:amd-latest
```

## Need Help?

See `README.md` for detailed documentation and troubleshooting.

