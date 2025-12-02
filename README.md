# Google Cloud Build Setup Guide

This folder contains all the necessary files to build Docker images for CV_MM using Google Cloud Build.

## 📁 Files Included

- `Dockerfile.cuda` - Docker image for NVIDIA GPU (CUDA) support
- `Dockerfile.amd` - Docker image for AMD GPU (ROCm) support
- `cloudbuild.yaml` - Google Cloud Build configuration
- `requirements-cuda-docker.txt` - Python dependencies (without PyTorch)
- `.dockerignore` - Files to exclude from Docker build context

## 🚀 Quick Start

### Step 1: Prerequisites

1. **Install Google Cloud SDK** (if not already installed):
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set your project**:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

### Step 2: Create Artifact Registry Repository

Create a Docker repository in Artifact Registry to store your images:

```bash
gcloud artifacts repositories create cv-mm-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV_MM Docker images for CUDA and AMD GPU"
```

**Note:** Change `us-central1` to your preferred region if needed.

### Step 3: Copy Files to Repository Root

**IMPORTANT:** These files need to be in your repository root (not in a subfolder) for Cloud Build to work correctly.

If you're uploading to Cloud Source Repositories or GitHub:

1. Copy all files from `gcp-build/` to your repository root:
   ```bash
   cp gcp-build/* /path/to/your/repo/
   ```

2. Or if using git, commit them:
   ```bash
   git add Dockerfile.cuda Dockerfile.amd cloudbuild.yaml requirements-cuda-docker.txt .dockerignore
   git commit -m "Add Docker build configuration for Google Cloud Build"
   git push
   ```

### Step 4: Configure Cloud Build

Edit `cloudbuild.yaml` and update the substitution variables if needed:

```yaml
substitutions:
  _GCR_REGION: 'us-central1'  # Your Artifact Registry region
  _GCR_REPOSITORY: 'cv-mm-images'  # Your repository name
```

### Step 5: Submit Build

#### Option A: Manual Build (from repository root)

```bash
# From your repository root directory
gcloud builds submit --config=cloudbuild.yaml
```

#### Option B: Manual Build with Custom Substitutions

```bash
gcloud builds submit --config=cloudbuild.yaml \
    --substitutions=_GCR_REGION=us-central1,_GCR_REPOSITORY=cv-mm-images
```

#### Option C: Create Cloud Build Trigger (Recommended)

1. Go to [Google Cloud Console](https://console.cloud.google.com/cloud-build/triggers)
2. Click "Create Trigger"
3. Connect your repository (GitHub, Cloud Source Repositories, etc.)
4. Set configuration:
   - **Type:** Cloud Build configuration file (yaml or json)
   - **Location:** `cloudbuild.yaml`
5. Set substitution variables:
   - `_GCR_REGION`: `us-central1`
   - `_GCR_REPOSITORY`: `cv-mm-images`
6. Click "Create"

Now every push to your repository will automatically trigger a build!

## 📋 Build Process

The build process will:

1. **Build CUDA Image** (`Dockerfile.cuda`)
   - Uses NVIDIA CUDA 12.1 base image
   - Installs PyTorch with CUDA support
   - Tags: `cv-mm:cuda-{SHA}` and `cv-mm:cuda-latest`

2. **Build AMD Image** (`Dockerfile.amd`)
   - Uses ROCm PyTorch base image
   - Installs ONNX Runtime with DirectML
   - Tags: `cv-mm:amd-{SHA}` and `cv-mm:amd-latest`

3. **Push Images** to Artifact Registry

## 🔍 Verify Build

After the build completes:

1. **Check build status**:
   ```bash
   gcloud builds list --limit=5
   ```

2. **View build logs**:
   ```bash
   gcloud builds log BUILD_ID
   ```

3. **List images** in Artifact Registry:
   ```bash
   gcloud artifacts docker images list \
       us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cv-mm-images
   ```

## 📥 Pulling Images

After successful build, pull images:

```bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull CUDA image
docker pull us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cv-mm-images/cv-mm:cuda-latest

# Pull AMD image
docker pull us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cv-mm-images/cv-mm:amd-latest
```

## 🏃 Running Containers

### CUDA Container

```bash
docker run --gpus all \
    -v /path/to/videos:/app/data/videos \
    -v /path/to/models:/app/models \
    -v /path/to/outputs:/app/outputs \
    -p 8000:8000 \
    -p 8501:8501 \
    us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cv-mm-images/cv-mm:cuda-latest
```

### AMD Container

```bash
docker run --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v /path/to/videos:/app/data/videos \
    -v /path/to/models:/app/models \
    -v /path/to/outputs:/app/outputs \
    -p 8000:8000 \
    -p 8501:8501 \
    us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cv-mm-images/cv-mm:amd-latest
```

## ⚙️ Configuration Options

### Machine Type

Edit `cloudbuild.yaml` to change build machine:

```yaml
options:
  machineType: 'E2_HIGHCPU_8'  # Options: E2_HIGHCPU_8, E2_HIGHCPU_16, E2_HIGHMEM_8, etc.
```

### Timeout

Adjust build timeout if needed:

```yaml
timeout: '7200s'  # 2 hours (default)
```

### Build Only One Image

To build only CUDA or AMD image, comment out the other in `cloudbuild.yaml`:

```yaml
steps:
  # Comment out AMD build steps if only building CUDA
  # - name: 'gcr.io/cloud-builders/docker'
  #   id: 'build-amd'
  #   ...
```

## 🐛 Troubleshooting

### Build Fails with "Permission Denied"

Ensure Cloud Build has necessary permissions:

```bash
# Grant Cloud Build service account permissions
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

### Build Times Out

- Increase timeout in `cloudbuild.yaml`
- Use larger machine type (E2_HIGHCPU_16)
- Check for large files being copied (use `.dockerignore`)

### Images Not Found After Build

- Verify Artifact Registry repository exists
- Check image tags match repository name
- Ensure you're looking in the correct region

### Docker Build Context Too Large

- Review `.dockerignore` file
- Exclude unnecessary files (videos, large data files)
- Use Cloud Storage for large files instead

## 📊 Build Costs

- **Cloud Build**: Free tier includes 120 build-minutes per day
- **Artifact Registry**: ~$0.10 per GB/month for storage
- **Network Egress**: Varies by region

## 🔗 Additional Resources

- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Docker Documentation](https://docs.docker.com/)

## 📝 Notes

- Builds typically take 15-30 minutes depending on machine type
- Images are tagged with commit SHA for versioning
- `latest` tags are updated on each successful build
- Both CUDA and AMD images are built in parallel

## ✅ Checklist

Before submitting your first build:

- [ ] Google Cloud SDK installed and authenticated
- [ ] Project ID set correctly
- [ ] Artifact Registry repository created
- [ ] Files copied to repository root
- [ ] `cloudbuild.yaml` substitution variables updated
- [ ] Cloud Build service account has necessary permissions
- [ ] `.dockerignore` excludes unnecessary files

---

**Need Help?** Check the build logs in Cloud Console or run:
```bash
gcloud builds log --stream
```

