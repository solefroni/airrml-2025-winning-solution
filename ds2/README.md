# DS2: Synthetic Dataset

## Approach
- **Model**: XGBoost with gapped k-mer features
- **Tokenizer**: Dynamic exhaustive gapped (lengths 4, 5, 6; max 3 gaps)
- **Feature selection**: Chi-squared, top 5,000 features
- **Classifier**: XGBoost (n_estimators=300, max_depth=5, learning_rate=0.05)

## Files
- `code/train.py` — Train the pipeline (requires `DS2_TRAIN_DATA` pointing to training directory with metadata.csv and .tsv repertoires)
- `code/predict.py` — Load trained model and generate predictions
- `code/witness_ranking.py` — Ranking of sequences for submission
- `model/` — Pre-trained pipeline and artifacts (saved by train.py or provided)

## Quick start
```bash
# Training (optional; pre-trained model is included)
export DS2_TRAIN_DATA=/path/to/train_dataset_2
python code/train.py

# Prediction
export DS2_TEST_DATA=/path/to/test
python code/predict.py
```

## Note
The pre-trained model in `model/` was produced by this same pipeline. Adding `train.py` ensures full reproducibility for Phase 2 evaluation.
