# Files Checklist

Use this checklist to ensure you have all necessary files before uploading to Google Cloud.

## ✅ Required Files

- [x] `Dockerfile.cuda` - CUDA (NVIDIA GPU) Docker image definition
- [x] `Dockerfile.amd` - AMD GPU Docker image definition  
- [x] `cloudbuild.yaml` - Google Cloud Build configuration
- [x] `requirements-cuda-docker.txt` - Python dependencies (without PyTorch)
- [x] `.dockerignore` - Files to exclude from Docker build

## 📚 Documentation Files (Optional but Recommended)

- [x] `README.md` - Complete setup and usage guide
- [x] `QUICK_START.md` - Quick 5-minute setup guide
- [x] `setup.sh` - Automated setup script
- [x] `FILES_CHECKLIST.md` - This file

## 📋 Before Uploading

### 1. Verify All Files Are Present

```bash
cd gcp-build
ls -la
```

You should see:
- Dockerfile.cuda
- Dockerfile.amd
- cloudbuild.yaml
- requirements-cuda-docker.txt
- .dockerignore
- README.md
- QUICK_START.md
- setup.sh

### 2. Review Configuration

- [ ] Check `cloudbuild.yaml` substitution variables match your setup
- [ ] Verify `_GCR_REGION` is set to your preferred region
- [ ] Verify `_GCR_REPOSITORY` matches your Artifact Registry repository name

### 3. Test Setup Script (Optional)

```bash
cd gcp-build
./setup.sh
```

This will:
- Create Artifact Registry repository if needed
- Set up permissions
- Update cloudbuild.yaml with your settings

## 🚀 Upload Instructions

### Option 1: Upload to Cloud Source Repositories

1. Create a new repository in Cloud Source Repositories
2. Copy all files from `gcp-build/` to repository root
3. Commit and push

### Option 2: Upload to GitHub/GitLab

1. Copy files from `gcp-build/` to your repository root
2. Commit and push
3. Connect repository to Cloud Build

### Option 3: Manual Upload via Cloud Console

1. Zip the `gcp-build` folder contents
2. Upload to Cloud Storage
3. Extract and use for Cloud Build

## ⚠️ Important Notes

1. **File Location**: Cloud Build expects these files in the repository **root**, not in a subfolder
2. **Application Code**: The Dockerfiles use `COPY . /app/` which copies all files from the build context (repository root)
3. **Build Context**: Cloud Build uses the repository root as the build context by default

## 📝 File Descriptions

### Dockerfile.cuda
- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Installs PyTorch with CUDA 12.1 support
- Includes all system dependencies for OpenCV and video processing

### Dockerfile.amd
- Base: `rocm/pytorch:rocm5.7_ubuntu22.04_py3.11_pytorch_2.1.1`
- Includes ROCm support for AMD GPUs
- Adds ONNX Runtime with DirectML

### cloudbuild.yaml
- Defines build steps for both CUDA and AMD images
- Configures parallel builds
- Sets up image tagging and pushing

### requirements-cuda-docker.txt
- Python dependencies without PyTorch/torchvision
- Used by both Dockerfiles (PyTorch installed separately)

### .dockerignore
- Excludes unnecessary files from Docker build context
- Reduces build time and image size
- Excludes test files, documentation, videos, etc.

## ✅ Verification

After uploading, verify files are in the correct location:

```bash
# In your repository root
ls -la | grep -E "(Dockerfile|cloudbuild|requirements-cuda|\.dockerignore)"
```

You should see all four files listed.

## 🆘 Troubleshooting

**Build fails with "file not found"**
- Ensure files are in repository root, not in a subfolder
- Check file names match exactly (case-sensitive)

**Build context too large**
- Review `.dockerignore` file
- Ensure large files (videos, data) are excluded

**Permission errors**
- Run `setup.sh` to configure permissions
- Or manually grant Cloud Build service account necessary roles

