# AIRR-ML 2025 Phase 2 - Submission Checklist

## Required Materials (Due: January 15, 2026)

### 1. Open-Source Code Repository ✓

- [ ] Create public GitHub repository
- [ ] Upload complete `winningApproach/` code
- [ ] Include comprehensive README with:
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Dataset-specific notes
  - [ ] System requirements
- [ ] Include methodology documentation
- [ ] Add LICENSE file
- [ ] Test that others can clone and run

**Suggested repository name**: `airrml-2025-winning-solution`

**Repository URL**: `_________________________________` (fill in after creation)

---

### 2. Kaggle Submission File ✓

- [x] Located at: `/home/dsi/solefroni/AIRRML/submission/BEST_SUBMISSION_0.84003.csv`
- [ ] Verify scores match:
  - Public Score: 0.84003
  - Private Score: 0.69353
- [ ] Check file format (6 columns: ID, dataset, label_positive_probability, junction_aa, v_call, j_call)
- [ ] Will be included in submission package automatically

---

### 3. Docker Container ✓

- [ ] Build Docker image: `./build_docker.sh`
- [ ] Test Docker image: `./test_docker.sh`
- [ ] Verify all 8 datasets work
- [ ] Test on sample Phase 2 data (when available)
- [ ] Save Docker image: Created by `./prepare_submission.sh`

**Docker image name**: `airrml2025-winning-solution:v1.0`

---

## Pre-Submission Testing

### Local Testing (Before Building Docker)

```bash
# Test individual dataset predictors
cd /home/dsi/solefroni/AIRRML/winningApproach/ds1/code
python predict.py  # Should run without errors

# Repeat for ds2-ds8
```

### Docker Testing

```bash
cd /home/dsi/solefroni/AIRRML/dockers

# 1. Build
./build_docker.sh

# 2. Run tests
./test_docker.sh

# 3. Test on actual data (if available)
docker run --rm \
  -v /path/to/test/data:/data \
  -v /tmp/output:/output \
  airrml2025-winning-solution:latest \
  --train_dir /data/train_dataset_1 \
  --test_dir /data/test_dataset_1_1 \
  --out_dir /output \
  --n_jobs 4 --device cpu

# 4. Verify output
ls -la /tmp/output/
```

---

## Creating GitHub Repository

### Step 1: Initialize Repository

```bash
cd /home/dsi/solefroni/AIRRML/
git init airrml-2025-submission
cd airrml-2025-submission

# Copy winning approach
cp -r ../winningApproach/* .

# Copy Docker files
mkdir docker
cp -r ../dockers/* docker/

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/
*.pkl.bak
*.model.bak
.DS_Store
.vscode/
.idea/
EOF

# Initial commit
git add .
git commit -m "Initial commit: AIRR-ML 2025 winning solution"
```

### Step 2: Push to GitHub

```bash
# Create repository on GitHub (web interface)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/airrml-2025-winning-solution.git
git branch -M main
git push -u origin main
```

### Step 3: Add Documentation

Ensure repository includes:
- [ ] README.md (main documentation)
- [ ] METHODOLOGY.md (algorithm details)
- [ ] LICENSE (MIT or similar)
- [ ] requirements.txt
- [ ] Docker instructions (in docker/ folder)

---

## Preparing Final Submission Package

### Step 1: Run Preparation Script

```bash
cd /home/dsi/solefroni/AIRRML/dockers
./prepare_submission.sh
```

This creates `submission_package/` with:
- Docker image (compressed .tar.gz)
- Kaggle submission.csv
- Repository information
- Usage instructions
- Checksums

### Step 2: Update Repository Information

```bash
cd submission_package
nano REPOSITORY_INFO.txt

# Update:
# - GitHub URL
# - Contact email
# - Any additional notes
```

### Step 3: Compress Package

```bash
cd /home/dsi/solefroni/AIRRML/dockers
tar -czf submission_package.tar.gz submission_package/

# Check size
du -h submission_package.tar.gz

# If too large (>5GB), upload to cloud storage and provide link
```

---

## Submission Email Template

```
Subject: AIRR-ML 2025 Phase 2 Submission - Team [YOUR_TEAM_NAME]

Dear AIRR-ML Organizers,

We are excited to participate in Phase 2 of the AIRR-ML Challenge.

Participation Confirmation:
- We confirm participation in Phase 2
- We choose Option B (provide Docker container, you run analysis)

Required Materials:

1. Open-Source Code Repository:
   https://github.com/[YOUR_USERNAME]/airrml-2025-winning-solution

2. Kaggle Submission File:
   Included in submission package (BEST_SUBMISSION_0.84003.csv)
   - Public Score: 0.84003
   - Private Score: 0.69353

3. Docker Container:
   Attached as: submission_package.tar.gz
   [OR if too large: Download link: https://...]
   
   To load:
   docker load < airrml2025-winning-solution_v1.0.tar.gz
   
   To test:
   docker run --rm airrml2025-winning-solution:v1.0 --help
   
   To run:
   docker run --rm \
     -v /path/to/train:/data/train \
     -v /path/to/test:/data/test \
     -v /path/to/output:/output \
     airrml2025-winning-solution:v1.0 \
     --train_dir /data/train/train_dataset_X \
     --test_dir /data/test/test_dataset_X_Y \
     --out_dir /output \
     --n_jobs 4 --device cpu

System Requirements:
- CPU: 4+ cores (8+ recommended for DS8)
- RAM: 16GB minimum, 32GB recommended for DS8
- Disk: ~20GB for container + data
- Docker 20.10+

Additional Notes:
- All 8 datasets have pre-trained models included
- DS8 requires automatic downsampling (handled by container)
- Container follows uniform interface from github.com/uio-bmi/predict-airr
- Detailed usage instructions included in package

Contact Information:
- Name: [YOUR NAME]
- Email: [YOUR EMAIL]
- Affiliation: [YOUR INSTITUTION]

We are available for any questions or clarifications.

Best regards,
[YOUR NAME]
```

---

## Timeline

- [ ] **By January 10, 2026**: Create GitHub repository
- [ ] **By January 12, 2026**: Test Docker thoroughly
- [ ] **By January 14, 2026**: Prepare submission package
- [ ] **By January 15, 2026**: Submit everything to organizers

---

## Pre-Submission Verification

Before submitting, verify:

### Code Repository
- [ ] Repository is public
- [ ] README is comprehensive
- [ ] Code is well-documented
- [ ] Can be cloned and run by others
- [ ] LICENSE file included

### Docker Container
- [ ] Builds without errors
- [ ] All tests pass
- [ ] Works on sample data
- [ ] Follows required interface
- [ ] Size is reasonable (<10GB compressed)

### Submission File
- [ ] Correct format (6 columns)
- [ ] Scores match (Public: 0.84003, Private: 0.69353)
- [ ] All 8 datasets included
- [ ] No missing values in critical columns

### Documentation
- [ ] REPOSITORY_INFO.txt updated
- [ ] Usage instructions clear
- [ ] Contact information provided
- [ ] System requirements documented

---

## After Submission

When you receive Phase 2 datasets (expected before end of January 2026):

### Option B Processing (Docker provided, organizers run)
- [ ] Wait for organizers to run your container
- [ ] Be available for questions/debugging
- [ ] Review results when provided

### If Issues Arise
- [ ] Respond promptly to organizer questions
- [ ] Provide fixes/patches if needed
- [ ] Test on similar data if possible

### Expected Timeline for Option B
- [ ] Late January: Receive Phase 2 data notification
- [ ] February-March: Organizers run analysis
- [ ] March 30, 2026: Results due

---

## Troubleshooting Guide

### Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
cd /home/dsi/solefroni/AIRRML/dockers
rm -rf winningApproach
./build_docker.sh
```

### Tests fail
```bash
# Check individual components
cd /home/dsi/solefroni/AIRRML/winningApproach/ds1/code
python -c "import sys; print(sys.path)"
python -c "import numpy, pandas, sklearn, xgboost, torch; print('OK')"

# Check predict script
python predict.py
```

### Docker image too large
```bash
# Check image size
docker images | grep airrml2025

# Try to reduce (remove unnecessary files via .dockerignore)
# Or upload to cloud storage (Google Drive, Dropbox, etc.)
```

### GitHub push fails
```bash
# Check file sizes
find . -type f -size +100M

# Use Git LFS for large files if needed
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

---

## Contact Information

**Competition Organizers**: [From their email]

**Team Lead**: [YOUR NAME]

**Emergency Contact**: [YOUR PHONE/EMAIL]

---

## Notes

- Keep all communications with organizers
- Document any issues/solutions
- Maintain version control
- Back up everything

---

**Status**: 🟡 In Progress

Update this checklist as you complete each item!
