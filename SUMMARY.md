# Google Cloud Build Package Summary

This folder contains everything you need to build and deploy Docker images for CV_MM using Google Cloud Build.

## 📦 Package Contents

### Core Build Files (Required)

1. **Dockerfile.cuda** (1,976 bytes)
   - NVIDIA GPU (CUDA) Docker image
   - Uses CUDA 12.1 with cuDNN 8
   - Includes PyTorch with CUDA support

2. **Dockerfile.amd** (1,985 bytes)
   - AMD GPU Docker image
   - Uses ROCm 5.7 with PyTorch
   - Includes ONNX Runtime with DirectML

3. **cloudbuild.yaml** (2,773 bytes)
   - Google Cloud Build configuration
   - Builds both CUDA and AMD images in parallel
   - Configures image tagging and pushing

4. **requirements-cuda-docker.txt** (443 bytes)
   - Python dependencies (without PyTorch)
   - Used by both Dockerfiles

5. **.dockerignore** (894 bytes)
   - Excludes unnecessary files from build context
   - Reduces build time and image size

### Documentation Files

6. **README.md** (7,596 bytes)
   - Complete setup and deployment guide
   - Step-by-step instructions
   - Troubleshooting section

7. **QUICK_START.md** (1,405 bytes)
   - 5-minute quick setup guide
   - Essential steps only

8. **FILES_CHECKLIST.md** (3,500+ bytes)
   - Pre-upload checklist
   - File verification guide

9. **SUMMARY.md** (This file)
   - Package overview

### Helper Scripts

10. **setup.sh** (4,448 bytes)
    - Automated setup script
    - Creates Artifact Registry repository
    - Configures permissions
    - Updates cloudbuild.yaml

## 🚀 Quick Start

1. **Run setup script**:
   ```bash
   cd gcp-build
   ./setup.sh
   ```

2. **Copy files to repository root**:
   ```bash
   cp gcp-build/Dockerfile.* .
   cp gcp-build/cloudbuild.yaml .
   cp gcp-build/requirements-cuda-docker.txt .
   cp gcp-build/.dockerignore .
   ```

3. **Submit build**:
   ```bash
   gcloud builds submit --config=cloudbuild.yaml
   ```

## 📋 What Gets Built

### CUDA Image
- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Tags**: `cv-mm:cuda-{SHA}` and `cv-mm:cuda-latest`
- **Size**: ~3-4 GB (estimated)
- **Use Case**: NVIDIA GPU servers/workstations

### AMD Image
- **Base**: `rocm/pytorch:rocm5.7_ubuntu22.04_py3.11_pytorch_2.1.1`
- **Tags**: `cv-mm:amd-{SHA}` and `cv-mm:amd-latest`
- **Size**: ~4-5 GB (estimated)
- **Use Case**: AMD GPU servers/workstations

## ⏱️ Build Time

- **CUDA Image**: ~15-20 minutes
- **AMD Image**: ~20-25 minutes
- **Total**: ~25-30 minutes (parallel builds)

## 💰 Estimated Costs

- **Cloud Build**: Free tier includes 120 build-minutes/day
- **Artifact Registry Storage**: ~$0.10/GB/month
- **Network Egress**: Varies by region

## 📍 File Locations After Upload

When uploading to your repository, place files in the **root directory**:

```
your-repo/
├── Dockerfile.cuda          ← From gcp-build/
├── Dockerfile.amd           ← From gcp-build/
├── cloudbuild.yaml          ← From gcp-build/
├── requirements-cuda-docker.txt  ← From gcp-build/
├── .dockerignore            ← From gcp-build/
├── app/                     ← Your application code
├── config/                  ← Your config files
└── ...                      ← Other application files
```

## ✅ Pre-Upload Checklist

- [ ] All 5 core build files present
- [ ] `cloudbuild.yaml` substitution variables reviewed
- [ ] `.dockerignore` excludes large files
- [ ] Google Cloud SDK installed
- [ ] Authenticated with `gcloud auth login`
- [ ] Project ID set with `gcloud config set project`
- [ ] Artifact Registry repository created (or run `setup.sh`)

## 🔗 Next Steps

1. **Upload files** to your repository (GitHub, Cloud Source Repositories, etc.)
2. **Run setup script** or manually create Artifact Registry repository
3. **Submit build** or create Cloud Build trigger
4. **Pull images** after successful build
5. **Deploy** to your infrastructure

## 📚 Documentation

- **Quick Start**: See `QUICK_START.md`
- **Full Guide**: See `README.md`
- **Checklist**: See `FILES_CHECKLIST.md`

## 🆘 Support

If you encounter issues:

1. Check build logs: `gcloud builds log BUILD_ID`
2. Review `README.md` troubleshooting section
3. Verify all files are in repository root
4. Ensure Cloud Build service account has permissions

---

**Package Version**: 1.0  
**Created**: December 2024  
**Compatible with**: Google Cloud Build, Artifact Registry

