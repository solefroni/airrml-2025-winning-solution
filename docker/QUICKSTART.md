# AIRR-ML 2025 Docker Submission - Quick Start Guide

## For Competition Organizers

This Docker container provides a uniform interface for running our winning solution (Private Score: 0.69353) on Phase 2 datasets.

### Step 1: Build the Container

```bash
cd /dsi/efroni-lab/sol/AIRRML_from_home/dockers
./build_docker.sh
```

Expected output:
```
[1/4] Copying winningApproach to build context...
[2/4] Building Docker image...
[3/4] Tagging as latest...
[4/4] Cleaning up...
✓ Docker image built successfully!
```

### Step 2: Test the Container

```bash
./test_docker.sh
```

All tests should pass:
```
[Test 1] Checking if Docker image exists... ✓
[Test 2] Checking image size... ✓
[Test 3] Testing help command... ✓
[Test 4] Checking Python packages... ✓
[Test 5] Checking winning approach files... ✓
[Test 6] Checking submission module... ✓
✓ ALL TESTS PASSED
```

### Step 3: Run on Data

Single dataset:
```bash
docker run --rm \
  -v /path/to/train_datasets:/data/train \
  -v /path/to/test_datasets:/data/test \
  -v /path/to/output:/output \
  airrml2025-winning-solution:latest \
  --train_dir /data/train/train_dataset_1 \
  --test_dir /data/test/test_dataset_1_1 /data/test/test_dataset_1_2 \
  --out_dir /output \
  --n_jobs 4 \
  --device cpu
```

All datasets (batch processing):
```bash
./run_all_datasets.sh /path/to/train /path/to/test /path/to/output 4 cpu
```

### Step 4: Prepare Submission Package

```bash
./prepare_submission.sh
```

This creates a `submission_package/` directory containing:
- Docker image (compressed tar.gz)
- Kaggle submission file
- Repository information
- Usage instructions
- Checksums

## For Team Members

### Quick Build & Test

```bash
# Build
cd /dsi/efroni-lab/sol/AIRRML_from_home/dockers
./build_docker.sh

# Test
./test_docker.sh

# Prepare submission
./prepare_submission.sh

# Update repository info in submission_package/REPOSITORY_INFO.txt
# Then compress:
tar -czf submission_package.tar.gz submission_package/
```

### Local Testing (without Docker)

```bash
cd /dsi/efroni-lab/sol/AIRRML_from_home/dockers
python3 -m submission.main \
  --train_dir /path/to/train_dataset_1 \
  --test_dir /path/to/test_dataset_1_1 \
  --out_dir /tmp/output \
  --n_jobs 4 \
  --device cpu
```

## Troubleshooting

### Build fails: "winningApproach directory not found"
```bash
# Ensure you're in the dockers directory
cd /dsi/efroni-lab/sol/AIRRML_from_home/dockers
# And that ../winningApproach exists
ls -la ../winningApproach/
```

### Permission denied executing scripts
```bash
chmod +x *.sh
```

### Docker daemon not running
```bash
sudo systemctl start docker
```

### Out of disk space
```bash
# Clean up old Docker images
docker system prune -a

# Check disk usage
df -h
```

## System Requirements

### Minimum
- 4 CPU cores
- 16 GB RAM
- 50 GB disk space
- Docker 20.10+

### Recommended (for DS8)
- 8+ CPU cores
- 32+ GB RAM
- 100 GB disk space
- SSD storage

## Expected Performance

| Dataset | Training Time | Prediction Time | Memory |
|---------|--------------|-----------------|---------|
| DS1     | ~5 min       | ~2 min          | 4 GB    |
| DS2     | ~10 min      | ~5 min          | 8 GB    |
| DS3     | ~5 min       | ~2 min          | 4 GB    |
| DS4     | ~8 min       | ~3 min          | 6 GB    |
| DS5     | ~8 min       | ~3 min          | 6 GB    |
| DS6     | ~5 min       | ~2 min          | 4 GB    |
| DS7     | ~15 min      | ~5 min          | 8 GB    |
| DS8     | ~60-120 min  | ~30-60 min      | 16-32 GB|

**Note**: Times are approximate and depend on dataset size and hardware.

## File Structure

```
dockers/
├── build_docker.sh           # Build the Docker image
├── test_docker.sh            # Test the container
├── prepare_submission.sh     # Package everything for submission
├── run_all_datasets.sh       # Batch process all datasets
├── Dockerfile                # Docker image definition
├── requirements.txt          # Python dependencies
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── submission/              # Uniform interface
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── predictor.py         # ImmuneStatePredictor class
│   └── utils.py             # Helper functions
└── [winningApproach/]       # Copied during build (not in git)
```

## Support

Questions? Check:
1. `README.md` - Detailed documentation
2. `../winningApproach/README.md` - Solution methodology
3. Test logs: `./test_docker.sh 2>&1 | tee test.log`

## Submission Checklist

- [ ] Docker image builds successfully
- [ ] All tests pass
- [ ] Tested on sample dataset
- [ ] GitHub repository is public
- [ ] Kaggle submission.csv included
- [ ] REPOSITORY_INFO.txt updated with GitHub URL and contact
- [ ] Submission package created and compressed
- [ ] Package size is reasonable (<10 GB compressed)

## Contact Organizers

Submit to: [organizer email from instructions]

Include:
1. `submission_package.tar.gz` (or provide download link if too large)
2. GitHub repository URL
3. Confirmation of Option B (Docker submission)

---

**Ready to submit!** 🚀
