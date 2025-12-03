# How to Trigger Cloud Build with Latest Commit

Since you've pushed the changes to GitHub, here's how to get Cloud Build to use the updated commit.

## Option 1: Manual Build Submission (Easiest)

This will automatically use the latest commit from your default branch:

```bash
# Make sure you're authenticated
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Submit build (uses latest commit from default branch)
gcloud builds submit --config=cloudbuild.yaml
```

**Note:** If you're running this from your local machine, it will use your local files. To use the GitHub repository instead, use Option 2 or 3.

## Option 2: Trigger Build from GitHub Repository

If you want Cloud Build to pull from GitHub (not local files):

```bash
# Get your latest commit SHA from GitHub
# Or use: git rev-parse HEAD

# Submit build from GitHub source
gcloud builds submit --config=cloudbuild.yaml \
    --source=https://source.developers.google.com/p/YOUR_PROJECT_ID/r/YOUR_REPO_NAME \
    --substitutions=COMMIT_SHA=YOUR_LATEST_COMMIT_SHA
```

## Option 3: Use Cloud Build Trigger (Recommended)

If you have a trigger set up, it should automatically build on push. To manually trigger it:

1. **Go to Cloud Console:**
   - Navigate to: https://console.cloud.google.com/cloud-build/triggers
   - Find your trigger
   - Click "Run" button

2. **Or use gcloud command:**
   ```bash
   # List your triggers
   gcloud builds triggers list
   
   # Run a specific trigger
   gcloud builds triggers run TRIGGER_NAME \
       --branch=main
   ```

## Option 4: Create/Update Cloud Build Trigger

If you don't have a trigger set up, create one:

### Via Console:
1. Go to: https://console.cloud.google.com/cloud-build/triggers
2. Click "Create Trigger"
3. Connect your GitHub repository
4. Configuration:
   - **Type:** Cloud Build configuration file
   - **Location:** `cloudbuild.yaml`
   - **Branch:** `^main$` (or your default branch)
5. Substitution variables:
   - `_GCR_REGION`: `us-central1` (or your region)
   - `_GCR_REPOSITORY`: `cv-mm-images` (or your repo name)
6. Click "Create"

### Via Command Line:
```bash
gcloud builds triggers create github \
    --name="cv-mm-build" \
    --repo-name="CVMMC" \
    --repo-owner="silverlynx18" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --substitutions=_GCR_REGION=us-central1,_GCR_REPOSITORY=cv-mm-images
```

## Option 5: Build from Specific Commit SHA

If you want to build a specific commit:

```bash
# Get commit SHA from GitHub
COMMIT_SHA="3f6e0c0ef71d918f39fb6b82ccd728591a3d896c"  # Replace with your latest

# Build from that commit
gcloud builds submit --config=cloudbuild.yaml \
    --substitutions=SHORT_SHA=${COMMIT_SHA:0:7}
```

## Verify Latest Commit is Being Used

Check the build logs to see which commit is being used:

```bash
# List recent builds
gcloud builds list --limit=5

# View specific build details
gcloud builds describe BUILD_ID

# Check the GitCommit field in the output
```

## Troubleshooting

### Build Still Uses Old Commit

1. **Check which branch Cloud Build is using:**
   - Verify trigger branch pattern matches your branch name
   - Default branch might be `main` or `master`

2. **Force rebuild:**
   ```bash
   # Delete old build cache (if any)
   # Then trigger new build
   ```

3. **Check trigger configuration:**
   ```bash
   gcloud builds triggers describe TRIGGER_NAME
   ```

### Build Uses Local Files Instead of GitHub

If `gcloud builds submit` uses local files:
- Use a trigger instead (Option 3)
- Or use `--source` flag to specify GitHub source

## Quick Check: Is Your Commit in GitHub?

Verify your commit is in GitHub:
```bash
# Check latest commit
git log -1 --oneline

# Verify it's pushed
git log origin/main -1 --oneline
```

If the commits match, your code is in GitHub and Cloud Build should use it.

