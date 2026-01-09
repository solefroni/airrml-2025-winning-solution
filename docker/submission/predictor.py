"""
ImmuneStatePredictor - Uniform Interface for AIRR-ML Competition

This class wraps the winning solution's 8 different approaches (one per dataset)
and provides a unified interface for training and prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import subprocess
import shutil


class ImmuneStatePredictor:
    """
    Unified predictor class that wraps all 8 dataset-specific approaches.
    
    This class automatically detects which dataset is being processed and
    routes to the appropriate model implementation.
    """
    
    def __init__(self, n_jobs: int = 4, device: str = "cpu"):
        """
        Initialize the predictor.
        
        Args:
            n_jobs: Number of parallel jobs for training
            device: Device for computation ('cpu' or 'cuda')
        """
        self.n_jobs = n_jobs
        self.device = device
        self.dataset_num = None
        self.model_trained = False
        
        # Path to winning approach code
        self.winning_approach_dir = Path(__file__).parent.parent / "winningApproach"
        
    def _detect_dataset_number(self, train_dir: Path) -> int:
        """
        Detect which dataset (1-8) based on directory name or metadata.
        
        Args:
            train_dir: Training data directory
            
        Returns:
            Dataset number (1-8)
        """
        # Try to extract from directory name
        dir_name = train_dir.name.lower()
        for i in range(1, 9):
            if f"dataset_{i}" in dir_name or f"dataset{i}" in dir_name or f"ds{i}" in dir_name:
                return i
        
        # Try to extract from parent directory
        parent_name = train_dir.parent.name.lower()
        for i in range(1, 9):
            if f"dataset_{i}" in parent_name or f"dataset{i}" in parent_name or f"ds{i}" in parent_name:
                return i
                
        # If can't detect, raise error
        raise ValueError(
            f"Could not detect dataset number from directory: {train_dir}. "
            f"Directory name should contain 'dataset_X' or 'datasetX' where X is 1-8."
        )
    
    def _get_dataset_code_dir(self, dataset_num: int) -> Path:
        """Get the code directory for a specific dataset."""
        return self.winning_approach_dir / f"ds{dataset_num}" / "code"
    
    def _get_dataset_model_dir(self, dataset_num: int) -> Path:
        """Get the model directory for a specific dataset."""
        return self.winning_approach_dir / f"ds{dataset_num}" / "model"
    
    def _get_dataset_output_dir(self, dataset_num: int) -> Path:
        """Get the output directory for a specific dataset."""
        output_dir = self.winning_approach_dir / f"ds{dataset_num}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def train(self, train_dir: Path, out_dir: Path) -> None:
        """
        Train the model on the provided training data.
        
        Args:
            train_dir: Directory containing training data
            out_dir: Directory to save trained model
        """
        print(f"Training on data from: {train_dir}")
        
        # Detect dataset number
        self.dataset_num = self._detect_dataset_number(train_dir)
        print(f"Detected dataset: DS{self.dataset_num}")
        
        # Get paths
        code_dir = self._get_dataset_code_dir(self.dataset_num)
        train_script = code_dir / "train.py"
        
        # Check if training script exists
        if not train_script.exists():
            print(f"Warning: No training script found for DS{self.dataset_num} at {train_script}")
            print("Using pre-trained model.")
            self.model_trained = True
            return
        
        # Set environment variables for the training script
        env = os.environ.copy()
        env[f"DS{self.dataset_num}_TRAIN_DATA"] = str(train_dir)
        
        # For DS8, check if downsampling is needed
        if self.dataset_num == 8:
            downsample_script = code_dir / "downsample_samples.py"
            if downsample_script.exists():
                print("DS8: Running downsampling preprocessing...")
                result = subprocess.run(
                    [sys.executable, str(downsample_script)],
                    cwd=str(code_dir),
                    env=env,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Warning: Downsampling had non-zero exit code: {result.returncode}")
                    print(f"stderr: {result.stderr}")
        
        # Run training script
        print(f"Running training script: {train_script}")
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(code_dir),
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Training stderr:\n{result.stderr}")
            raise RuntimeError(f"Training failed with exit code {result.returncode}")
        
        print(f"Training stdout:\n{result.stdout}")
        self.model_trained = True
        
        # Copy trained model to output directory if different
        model_dir = self._get_dataset_model_dir(self.dataset_num)
        if out_dir != model_dir and model_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            for model_file in model_dir.glob("*"):
                if model_file.is_file():
                    shutil.copy2(model_file, out_dir / model_file.name)
            print(f"Copied trained model to: {out_dir}")
    
    def predict(self, test_dirs: List[Path], out_dir: Path) -> pd.DataFrame:
        """
        Generate predictions for test data.
        
        Args:
            test_dirs: List of directories containing test data
            out_dir: Directory to save predictions
            
        Returns:
            DataFrame with predictions in competition format
        """
        if self.dataset_num is None:
            raise ValueError("Must train or specify dataset number before predicting")
        
        print(f"Generating predictions for DS{self.dataset_num}")
        
        # Get paths
        code_dir = self._get_dataset_code_dir(self.dataset_num)
        predict_script = code_dir / "predict.py"
        
        if not predict_script.exists():
            raise FileNotFoundError(f"Prediction script not found: {predict_script}")
        
        # Set environment variables for each test directory
        all_predictions = []
        
        for test_idx, test_dir in enumerate(test_dirs, 1):
            print(f"Processing test set {test_idx}/{len(test_dirs)}: {test_dir}")
            
            env = os.environ.copy()
            env[f"DS{self.dataset_num}_TEST_DATA"] = str(test_dir)
            
            # For DS8, run downsampling for test data
            if self.dataset_num == 8:
                downsample_test_script = code_dir / "downsample_test.py"
                if downsample_test_script.exists():
                    print(f"DS8: Running test downsampling for set {test_idx}...")
                    env[f"DS8_TEST_CACHE"] = str(self._get_dataset_output_dir(self.dataset_num) / f"cache/test{test_idx}")
                    result = subprocess.run(
                        [sys.executable, str(downsample_test_script)],
                        cwd=str(code_dir),
                        env=env,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        print(f"Warning: Test downsampling had non-zero exit code: {result.returncode}")
            
            # Run prediction script
            result = subprocess.run(
                [sys.executable, str(predict_script)],
                cwd=str(code_dir),
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Prediction stderr:\n{result.stderr}")
                raise RuntimeError(f"Prediction failed with exit code {result.returncode}")
            
            print(f"Prediction stdout:\n{result.stdout}")
        
        # Load and combine predictions from output directory
        dataset_output_dir = self._get_dataset_output_dir(self.dataset_num)
        
        # Load test predictions
        test_pred_files = list(dataset_output_dir.glob("*test_predictions.csv"))
        ranked_seq_files = list(dataset_output_dir.glob("*ranked_sequences.csv"))
        
        for pred_file in test_pred_files:
            df = pd.read_csv(pred_file)
            all_predictions.append(df)
            print(f"Loaded {len(df)} test predictions from {pred_file.name}")
        
        for ranked_file in ranked_seq_files:
            df = pd.read_csv(ranked_file)
            all_predictions.append(df)
            print(f"Loaded {len(df)} ranked sequences from {ranked_file.name}")
        
        if not all_predictions:
            raise RuntimeError(f"No predictions found in {dataset_output_dir}")
        
        # Combine all predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        # Ensure correct format
        required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
        for col in required_cols:
            if col not in combined_df.columns:
                combined_df[col] = -999.0
        
        # Save to output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"ds{self.dataset_num}_predictions.csv"
        combined_df[required_cols].to_csv(output_file, index=False)
        print(f"Saved predictions to: {output_file}")
        
        return combined_df[required_cols]
    
    def train_and_predict(self, train_dir: Path, test_dirs: List[Path], out_dir: Path) -> pd.DataFrame:
        """
        Convenience method to train and predict in one call.
        
        Args:
            train_dir: Directory containing training data
            test_dirs: List of directories containing test data
            out_dir: Directory to save outputs
            
        Returns:
            DataFrame with predictions
        """
        self.train(train_dir, out_dir)
        return self.predict(test_dirs, out_dir)
