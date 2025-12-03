# Commit Instructions for gcp-build

## Current Status

The `gcp-build` directory contains all the files needed for Cloud Build, but it's not yet committed to GitHub.

## What Needs to be Committed

The entire `gcp-build/` directory needs to be committed, including:
- `Dockerfile.cuda` - Updated to copy from `app-code/`
- `cloudbuild.yaml` - References `Dockerfile.cuda` at root
- `app-code/` - Contains all application code
- All scripts and documentation

## Quick Commit

Run this from the repository root:

```bash
# Add gcp-build directory (excluding macOS metadata)
find gcp-build -name "._*" -delete
git add gcp-build/

# Commit
git commit -m "Add gcp-build directory with Docker configuration and application code

- Dockerfile.cuda copies all code from app-code/
- cloudbuild.yaml configured for Cloud Build
- Complete application code in app-code/
- Setup scripts and documentation"

# Push
git push origin main
```

## After Committing

Once `gcp-build/` is in GitHub, Cloud Build will:
1. Find `gcp-build/cloudbuild.yaml`
2. Use `Dockerfile.cuda` at repository root
3. Copy everything from `app-code/` into the container
4. Build a complete, self-contained Docker image

## Verify

After pushing, verify in GitHub that:
- `gcp-build/` directory exists
- `gcp-build/cloudbuild.yaml` exists
- `gcp-build/Dockerfile.cuda` exists
- `gcp-build/app-code/` contains all application files

