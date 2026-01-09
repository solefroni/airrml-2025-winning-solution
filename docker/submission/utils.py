"""
Utility functions for handling datasets and output files.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


def get_dataset_pairs(train_datasets_dir: str, test_datasets_dir: str) -> List[Tuple[Path, List[Path]]]:
    """
    Get pairs of training and test dataset directories.
    
    Args:
        train_datasets_dir: Root directory containing training datasets
        test_datasets_dir: Root directory containing test datasets
        
    Returns:
        List of tuples (train_dir, [test_dir1, test_dir2, ...])
    """
    train_dir = Path(train_datasets_dir)
    test_dir = Path(test_datasets_dir)
    
    if not train_dir.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not test_dir.exists():
        raise ValueError(f"Test directory does not exist: {test_dir}")
    
    pairs = []
    
    # Find all dataset directories (assuming naming pattern dataset_1, dataset_2, etc.)
    for dataset_num in range(1, 9):
        # Look for training directory
        possible_train_names = [
            f"train_dataset_{dataset_num}",
            f"dataset_{dataset_num}",
            f"train_ds{dataset_num}",
            f"ds{dataset_num}"
        ]
        
        train_dataset_dir = None
        for name in possible_train_names:
            candidate = train_dir / name
            if candidate.exists() and candidate.is_dir():
                train_dataset_dir = candidate
                break
        
        if train_dataset_dir is None:
            print(f"Warning: Could not find training directory for dataset {dataset_num}")
            continue
        
        # Look for test directories (may have multiple test sets per dataset)
        test_dataset_dirs = []
        for test_idx in range(1, 10):  # Support up to 9 test sets per dataset
            possible_test_names = [
                f"test_dataset_{dataset_num}_{test_idx}",
                f"test_dataset_{dataset_num}",
                f"test_ds{dataset_num}_{test_idx}",
                f"test_ds{dataset_num}"
            ]
            
            for name in possible_test_names:
                candidate = test_dir / name
                if candidate.exists() and candidate.is_dir():
                    test_dataset_dirs.append(candidate)
                    break
        
        if not test_dataset_dirs:
            # Try without index if numbered versions not found
            for name in [f"test_dataset_{dataset_num}", f"test_ds{dataset_num}"]:
                candidate = test_dir / name
                if candidate.exists() and candidate.is_dir():
                    test_dataset_dirs.append(candidate)
                    break
        
        if test_dataset_dirs:
            pairs.append((train_dataset_dir, test_dataset_dirs))
        else:
            print(f"Warning: Could not find test directories for dataset {dataset_num}")
    
    return pairs


def concatenate_output_files(out_dir: str, output_filename: str = "submission.csv") -> None:
    """
    Concatenate all prediction files into a single submission file.
    
    Args:
        out_dir: Directory containing individual prediction files
        output_filename: Name of the final combined output file
    """
    out_path = Path(out_dir)
    
    if not out_path.exists():
        raise ValueError(f"Output directory does not exist: {out_path}")
    
    # Required columns in submission format
    required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    
    # Find all prediction files
    prediction_files = []
    for dataset_num in range(1, 9):
        # Look for predictions for this dataset
        pattern_files = list(out_path.glob(f"ds{dataset_num}_predictions.csv"))
        prediction_files.extend(pattern_files)
    
    if not prediction_files:
        print(f"Warning: No prediction files found in {out_path}")
        return
    
    # Load and concatenate all predictions
    all_predictions = []
    for pred_file in sorted(prediction_files):
        try:
            df = pd.read_csv(pred_file)
            all_predictions.append(df)
            print(f"Loaded {len(df)} predictions from {pred_file.name}")
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue
    
    if not all_predictions:
        print("Error: No valid prediction files could be loaded")
        return
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    # Ensure all required columns exist
    for col in required_cols:
        if col not in combined_df.columns:
            combined_df[col] = -999.0
    
    # Save combined file
    output_file = out_path / output_filename
    combined_df[required_cols].to_csv(output_file, index=False)
    
    print()
    print("=" * 80)
    print(f"SUBMISSION FILE CREATED: {output_file}")
    print(f"Total predictions: {len(combined_df)}")
    print(f"Datasets included: {sorted(combined_df['dataset'].unique())}")
    print("=" * 80)


def validate_submission_file(submission_file: str) -> Dict[str, any]:
    """
    Validate a submission file and return statistics.
    
    Args:
        submission_file: Path to submission CSV file
        
    Returns:
        Dictionary with validation results and statistics
    """
    submission_path = Path(submission_file)
    
    if not submission_path.exists():
        return {"valid": False, "error": f"File not found: {submission_file}"}
    
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        return {"valid": False, "error": f"Could not read file: {e}"}
    
    # Check required columns
    required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return {"valid": False, "error": f"Missing required columns: {missing_cols}"}
    
    # Check for missing values in critical columns
    critical_cols = ['ID', 'dataset']
    missing_values = {col: df[col].isna().sum() for col in critical_cols}
    
    # Statistics
    stats = {
        "valid": True,
        "total_rows": len(df),
        "datasets": sorted(df['dataset'].unique().tolist()),
        "rows_per_dataset": df.groupby('dataset').size().to_dict(),
        "missing_values": missing_values,
        "probability_range": (df['label_positive_probability'].min(), df['label_positive_probability'].max())
    }
    
    return stats
