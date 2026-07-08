# AIRR-ML 2025 — Method Descriptions (Team gordianknot, Rank-1)

**Author:** Sol Efroni, Faculty of Life Sciences, Bar-Ilan University, Ramat-Gan, Israel  
**Repository:** https://github.com/solefroni/airrml-2025-winning-solution  

This document describes each dataset-specific method submitted for Phase 2, following the organizers' requested format (~200 words per method): training data and preprocessing, model training and hyperparameters, prediction probabilities, and top-50k sequence ranking. Descriptions match the current `main` branch of the submission repository (including the DS8 inference-time XGBoost-only fallback for samples whose graph build fails).

---

## R1-M6: DS1 (Synthetic T1D)

For DS1, we used antigen-specific TCR matching against external reference sequences from VDJdb (Proinsulin and GAD65). For each repertoire, we counted the number of unique CDR3 amino acid sequences matching each reference set (two base features: Proinsulin diversity and GAD65 diversity). Features were log-transformed (`log1p`), expanded with degree-2 polynomial terms (including an interaction), and standardized. We trained scikit-learn Logistic Regression with isotonic calibration (`CalibratedClassifierCV`) for binary repertoire classification; hyperparameters were fixed (no extensive search) given the low feature dimensionality. Test probabilities are the calibrated positive-class outputs. For the top 50,000 training sequences, we ranked unique CDR3s by witness enrichment: log2 fold-change of per-sample presence frequency in positive versus negative repertoires, combined with model-based importance for sequences matching Proinsulin or GAD65 references in VDJdb.

---

## R1-M1: DS2 (Synthetic)

For DS2, each repertoire was encoded as gapped k-mer token frequencies. We used a dynamic gapped tokenizer over CDR3 lengths 4–6 (anchor positions fixed, up to three internal gaps), producing approximately 2.2 million candidate features. No external antigen database was used. Features were selected with Chi-squared univariate selection (5,000 features retained). We trained an XGBoost binary classifier (`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, with L1/L2 regularization) on the Phase-1 training split; hyperparameters were set from cross-validation on training data. Test repertoire probabilities are `predict_proba` positive-class scores from the fitted pipeline (tokenizer → selector → XGBoost). For interpretability, the top 50,000 training sequences were ranked by witness enrichment (log2 fold-change of sample-level frequency in positive vs. negative repertoires), combined with XGBoost feature importance mapped back to gapped k-mer tokens associated with each sequence.

---

## R1-M7: DS3 (Synthetic SARS-CoV-2)

For DS3, repertoires were represented as abundance vectors over a fixed list of SARS-CoV-2-reactive CDR3 sequences from the Parse Bioscience reference database (validated positive reactivity only). Each dimension is the count of a reference CDR3 in the repertoire (sparse high-dimensional features). Features were standardized with `StandardScaler`. We trained Logistic Regression with balanced class weights and L2 regularization; the regularization strength was chosen by cross-validation on training data. Test probabilities are the positive-class outputs of the logistic model. For the top 50,000 sequences, training CDR3s were ranked by association with the positive class using model coefficients (absolute logistic weights for matching reference dimensions) and enrichment frequency in positive versus negative training repertoires.

---

## R1-M5: DS4 (Synthetic)

For DS4, we first held out a stratified train/validation split; all pattern discovery used training data only to avoid leakage. CDR3 sequences were tokenized into standard and gapped k-mers (k = 3–6). Disease-associated patterns were identified with Fisher's exact test (p < 0.0001, fold-change > 4.0, max 50 patterns per k). Per-repertoire features were aggregated pattern statistics (counts, abundances, diversity) plus V/J gene usage summaries, then scaled with `RobustScaler`. We trained XGBoost (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`). Test probabilities are XGBoost `predict_proba` scores. Top 50,000 training sequences were ranked by pattern-level statistical significance (Fisher p-values and fold-change among patterns present in each sequence) and repertoire-level enrichment in positive samples.

---

## R1-M3: DS5 (Synthetic)

For DS5, k-mer patterns (k = 3, 4, 5) were discovered on training data using Fisher's exact test with FDR correction (Benjamini–Hochberg). Per-repertoire inputs were aggregated pattern features: pattern counts, total abundance, Shannon diversity, and normalized ratios across k sizes. Features were standardized before classification. The deployed model is Logistic Regression (L2-regularized, hyperparameters chosen by cross-validation on training folds). Test probabilities are logistic positive-class scores. For interpretability, the top 50,000 training sequences were ranked by their contribution to significant disease-associated patterns (pattern enrichment scores and association with high-weight logistic features derived from aggregated pattern presence).

---

## R1-M8: DS6 (Synthetic HER2/neu)

For DS6, repertoires were encoded as counts over a fixed HER2/neu-reactive CDR3 list from Parse Bioscience (positive reactivity only), analogous to DS3. Features were standardized and classified with Logistic Regression (balanced class weights, L2 regularization, C chosen by cross-validation). Test probabilities are logistic positive-class outputs. For the top 50,000 sequences, training CDR3s were ranked using logistic regression coefficients on the HER2/neu reference dimensions (sequences matching high-weight reference CDR3s) combined with per-sequence enrichment counts in positive versus negative training repertoires.

---

## R1-M2: DS7 (HSV, real-world)

For DS7, training used repertoire-level features only: 600 binary indicators for differentially abundant CDR3s (selected by positive-minus-negative sample fraction on training data), Shannon/Simpson/clonality diversity metrics, V/J gene diversity statistics, and CDR3 length distribution summaries. Negative training samples were undersampled (`RandomUnderSampler`, 2:1 negative:positive) before fitting. XGBoost was trained with a small hyperparameter grid (`n_estimators` 100–200, `max_depth` 3–4) selected by cross-validation; decision threshold was tuned on validation data (Youden's J). Test probabilities are XGBoost `predict_proba` scores. Top 50,000 training sequences were ranked by witness enrichment (log2 fold-change of sample presence in positive vs. negative repertoires), with additional weight for sequences contributing to the 600 differential features.

---

## R1-M4: DS8 (T1D, real-world)

For DS8, repertoires were preprocessed by downsampling to 10,000 templates per repertoire (multinomial sampling proportional to template counts, seed 42). Two models were combined: (1) a Graph Convolutional Network on CVC-embedded KNN graphs (k = 30; node features: frequency, length, amino acid composition, centrality, CVC embedding slice; GCN: 3 layers, hidden dim 160, dropout 0.47; hyperparameters from Optuna on validation AUC); (2) XGBoost on approximately 200 repertoire-level features (k-mers, V/J usage, diversity, length, optional antigen counts). A logistic regression meta-learner stacked GCN and XGBoost validation predictions (out-of-fold for training the meta-model). At inference, repertoire probabilities are meta-learner positive-class scores when both graph and feature extraction succeed; if graph construction fails for a sample, the pipeline falls back to XGBoost-only probabilities for that sample (no model retraining). Top 50,000 training sequences were ranked by log2 fold-change of template-weighted abundance in positive vs. negative training repertoires. Only the CVC embedder was used in the final winning pipeline (not TCRformer or other embedders).

---

## Summary

| Method ID | Dataset | Classifier | Key signal |
|-----------|---------|------------|------------|
| R1-M6 | DS1 | Calibrated logistic regression | VDJdb Proinsulin/GAD65 TCR diversity |
| R1-M1 | DS2 | XGBoost on gapped k-mers | Chi-squared selected gapped k-mers (k=4–6) |
| R1-M7 | DS3 | Logistic regression | Parse Bioscience SARS-CoV-2 CDR3 counts |
| R1-M5 | DS4 | XGBoost | Fisher-selected gapped k-mer patterns (train-only) |
| R1-M3 | DS5 | Logistic regression | FDR-corrected multi-k pattern aggregates |
| R1-M8 | DS6 | Logistic regression | Parse Bioscience HER2/neu CDR3 counts |
| R1-M2 | DS7 | XGBoost | Differential sequences + diversity + V/J features |
| R1-M4 | DS8 | GCN + XGBoost stacking | CVC graph embeddings + repertoire features |
