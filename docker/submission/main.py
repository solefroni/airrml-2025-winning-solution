"""
Main entry point for AIRR-ML 2025 submission.

This script provides the uniform interface expected by the competition organizers:
    python3 -m submission.main --train_dir /path --test_dir /path --out_dir /path --n_jobs 4 --device cpu
"""

import argparse
import sys
from pathlib import Path
from typing import List
from submission.predictor import ImmuneStatePredictor
from submission.utils import get_dataset_pairs, concatenate_output_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AIRR-ML 2025 - Winning Solution (Private Score: 0.69353)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing training data for a single dataset"
    )
    
    parser.add_argument(
        "--test_dir",
        type=str,
        nargs='+',
        required=True,
        help="Directory (or directories) containing test data"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save predictions and trained models"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and use pre-trained models"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Convert paths
    train_dir = Path(args.train_dir)
    test_dirs = [Path(td) for td in args.test_dir]
    out_dir = Path(args.out_dir)
    
    # Validate paths
    if not train_dir.exists():
        print(f"Error: Training directory does not exist: {train_dir}")
        sys.exit(1)
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            print(f"Error: Test directory does not exist: {test_dir}")
            sys.exit(1)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AIRR-ML 2025 - Winning Solution")
    print("Private Leaderboard Score: 0.69353")
    print("=" * 80)
    print(f"Train dir: {train_dir}")
    print(f"Test dirs: {test_dirs}")
    print(f"Output dir: {out_dir}")
    print(f"n_jobs: {args.n_jobs}")
    print(f"device: {args.device}")
    print("=" * 80)
    print()
    
    # Initialize predictor
    predictor = ImmuneStatePredictor(n_jobs=args.n_jobs, device=args.device)
    
    # Train (or skip if using pre-trained)
    if not args.skip_training:
        print("TRAINING PHASE")
        print("-" * 80)
        predictor.train(train_dir, out_dir)
        print()
    else:
        print("Skipping training (using pre-trained models)")
        predictor.dataset_num = predictor._detect_dataset_number(train_dir)
        print(f"Detected dataset: DS{predictor.dataset_num}")
        print()
    
    # Predict
    print("PREDICTION PHASE")
    print("-" * 80)
    predictions = predictor.predict(test_dirs, out_dir)
    print()
    
    # Summary
    print("=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print(f"Generated {len(predictions)} predictions")
    print(f"Outputs saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
