# Build Error Fix: Missing app-code Directory

## Problem

The Cloud Build failed with error: `step exited with non-zero status: 2`

This happened because the `gcp-build/app-code/` directory doesn't exist in your GitHub repository. When Cloud Build clones your repo, it can't find the files that the Dockerfile is trying to copy.

## Solution

You need to commit and push the `app-code` directory to GitHub.

### Quick Fix

Run the commit script:

```bash
cd gcp-build
./commit-app-code.sh
```

This will:
1. Check if `app-code` exists locally
2. Stage all files in `gcp-build/app-code/`
3. Commit them with an appropriate message
4. Optionally push to GitHub

### Manual Fix

If you prefer to do it manually:

```bash
# From repository root
cd /Volumes/Extreme\ SSD/CV_MM

# Add app-code directory
git add gcp-build/app-code/

# Commit
git commit -m "Add application code for Docker build"

# Push to GitHub
git push origin main
```

### Verify

After pushing, verify the files are in GitHub:

```bash
# Check git status
git status

# Verify files are tracked
git ls-files gcp-build/app-code/ | head -10
```

## Why This Happened

The `app-code` directory was created locally using `./prepare-app-code.sh`, but it wasn't committed to git. Cloud Build only has access to files that are in your GitHub repository.

## Next Steps

1. ✅ Commit and push `gcp-build/app-code/`
2. ✅ Trigger Cloud Build again
3. ✅ Build should now succeed

## Note

The `app-code` directory contains copies of your application code. When you update your source code, you'll need to:

1. Run `./prepare-app-code.sh` to update the copies
2. Commit and push the changes
3. Cloud Build will use the updated code

Alternatively, you could set up a GitHub Action or pre-commit hook to automatically update `app-code` when source files change.

