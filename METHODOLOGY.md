# AIRRML Challenge: Winning Methodology

## Complete Technical Documentation

This document provides a detailed explanation of the methodology used to solve each of the eight datasets in the AIRRML challenge. Each section describes the approach, feature engineering, model architecture, and key insights for one dataset.

---

# Page 1: Dataset 1 (DS1) — Synthetic

## Overview

Dataset 1 is a synthetic dataset simulating Type 1 Diabetes (T1D). The solution leverages **domain knowledge about T1D autoantigens** to create highly predictive features using only **2 key features**: counts of TCRs matching known Proinsulin and GAD65 reactive sequences.

## Key Insight: T1D Autoantigen-Specific TCRs

Type 1 Diabetes is an autoimmune disease where the immune system attacks pancreatic beta cells. Two major autoantigens are:
- **Proinsulin** — the precursor to insulin, a primary T1D autoantigen
- **GAD65** (Glutamic Acid Decarboxylase 65) — an enzyme in beta cells, another major T1D autoantigen

The hypothesis: **Positive samples (T1D) should have more TCRs that recognize these autoantigens.**

## Methodology

### 1. External Reference Data: VDJdb

VDJdb is a curated database of TCR sequences with known antigen specificity. We extracted:

```python
# From VDJdb, filter for T1D-related antigens
proinsulin_cdr3s = vdjdb[vdjdb['antigen'] == 'Proinsulin']['cdr3_aa'].unique()
gad65_cdr3s = vdjdb[vdjdb['antigen'] == 'GAD65']['cdr3_aa'].unique()
```

These reference sets contain experimentally validated TCR CDR3 sequences known to bind Proinsulin or GAD65 peptides.

### 2. Feature Engineering: Diversity Counts

For each repertoire, we compute only **2 features**:

```python
def extract_features(repertoire_path, proinsulin_set, gad65_set):
    rep_df = pd.read_csv(repertoire_path, sep='\t')
    rep_unique_sequences = set(rep_df['junction_aa'].dropna())
    
    # Count unique matches to reference sequences
    proinsulin_diversity = len(rep_unique_sequences & proinsulin_set)
    gad65_diversity = len(rep_unique_sequences & gad65_set)
    
    return [proinsulin_diversity, gad65_diversity]
```

**Why "Diversity" (unique counts)?**
- Counting unique matches is more robust than total abundance
- Avoids bias from clonal expansion (one sequence appearing many times)
- Captures breadth of autoantigen recognition

### 3. Feature Transformation

The raw counts are transformed for better model performance:

**Log Transformation**:
```python
X_log = np.log1p(X)  # log(1 + x) handles zeros gracefully
```

**Polynomial Expansion (degree 2)**:
```python
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_log)
# Creates: [proinsulin, gad65, proinsulin², gad65², proinsulin×gad65]
```

This captures:
- Individual effects of each antigen
- Squared terms (diminishing/accelerating returns)
- Interaction term (combined effect)

### 4. Classification Model

**Logistic Regression** with isotonic calibration:
```python
model = LogisticRegression()
calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

Why logistic regression works well:
- Only 5 features after polynomial expansion
- Clear linear relationship in log space
- Calibration ensures reliable probability estimates

### 5. The Complete Pipeline

```
Repertoire → Extract CDR3s → Match to VDJdb → Count matches → Log → Polynomial → Logistic Regression → Probability
```

### 6. Why This Works

1. **Domain Knowledge**: T1D is characterized by autoimmune responses to specific antigens
2. **External Validation**: VDJdb provides experimentally confirmed TCR-antigen pairs
3. **Simple Signal**: The synthetic dataset was designed with this signal embedded
4. **Minimal Overfitting**: Only 2 base features means no risk of overfitting to noise

### 7. Results

This simple 2-feature model achieves excellent performance because:
- The signal is **biologically grounded** (T1D autoantigens)
- The features are **externally validated** (VDJdb)
- The model is **appropriately simple** for the signal strength

---

# Page 2: Dataset 2 (DS2) — Synthetic

## Overview

Dataset 2 is a synthetic dataset with a diffuse, subtle signal. Unlike DS1, there is **no known external database** of disease-associated sequences. The solution uses **shared pattern aggregation** and **forward feature selection** to find predictive patterns without overfitting.

## Methodology

### 1. Challenge

DS2 presented significant challenges:
- No external reference database (unlike DS1, DS3, DS6)
- Signal is spread across many sequences (no dominant pattern)
- Pattern-based approaches showed severe overfitting (CV AUC ~1.0, test AUC ~0.6)
- Multiple sophisticated approaches failed to improve beyond baseline

### 2. Feature Engineering

The approach uses **aggregated pattern features** rather than individual sequence matching:

**K-mer Frequencies (~8,000 features)**
```python
def extract_kmer_features(repertoire_df, k=3):
    all_kmers = Counter()
    for seq in repertoire_df['junction_aa']:
        for i in range(len(seq) - k + 1):
            all_kmers[seq[i:i+k]] += 1
    total = sum(all_kmers.values())
    return {kmer: count/total for kmer, count in all_kmers.items()}
```

**V/J Gene Usage (~200 features)**
- Frequency of each V gene and J gene
- Captures germline gene usage differences

**CDR3 Length Distribution (~26 features)**
- Mean, std, min, max, median length
- Binned length distribution (5-26 amino acids)

**Diversity Metrics (6 features)**
- Unique sequence count and ratio
- Shannon entropy, Simpson's index
- Gini coefficient, top clone frequency

### 3. Forward Feature Selection

To prevent overfitting with thousands of features:

```python
def forward_selection(X, y, max_features=200):
    selected = []
    remaining = list(range(X.shape[1]))
    
    for _ in range(max_features):
        best_score, best_feature = -1, None
        for f in remaining:
            X_subset = X[:, selected + [f]]
            score = cross_val_score(XGBClassifier(), X_subset, y, cv=5).mean()
            if score > best_score:
                best_score, best_feature = score, f
        selected.append(best_feature)
        remaining.remove(best_feature)
    
    return selected
```

This greedy approach:
1. Starts with 0 features
2. Evaluates each candidate feature using 5-fold CV
3. Selects the feature that maximizes ROC-AUC
4. Repeats until reaching 200 features

### 4. Classification Model

**XGBoost Classifier** with regularization:
```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0  # L2 regularization
)
```

### 5. Key Insights

DS2 demonstrated important lessons:
- **No shortcut**: Without an external reference database, we must learn patterns from data
- **Overfitting risk**: Thousands of features on ~400 samples leads to severe overfitting
- **Conservative approach**: Forward selection with regularization outperformed complex methods
- **Baseline value**: Sometimes the simple baseline approach is hard to beat

### 6. Results

Final performance: ~0.82 ROC-AUC

The model relies on aggregated repertoire statistics rather than specific sequence matching, reflecting the diffuse nature of the underlying signal.

---

# Page 3: Dataset 3 (DS3) — Synthetic

## Overview

Dataset 3 is a synthetic dataset where positive samples contain **SARS-CoV-2 reactive TCRs**. The solution uses **antigen-specific sequence matching** with the Parse Bioscience SARS-CoV-2 TCR database.

## Methodology

### 1. External Reference Database

The Parse Bioscience dataset contains experimentally validated TCRs that react to SARS-CoV-2 antigens. We extract:
- CDR3 sequences labeled as "SARSCoV2"
- Only sequences marked as `is_positive = True` (validated reactivity)
- Result: ~60,000+ unique SARS-CoV-2 specific CDR3 sequences

### 2. Feature Engineering

For each repertoire, create a feature vector where each dimension represents one reference CDR3:

```python
feature_vector = [count(cdr3_i) for cdr3_i in sarscov2_cdr3s]
```

This creates a sparse, high-dimensional feature space where:
- Positive repertoires should have more matches to SARS-CoV-2 TCRs
- The specific CDR3s that match are biologically informative

### 3. Classification Models

Three classifiers are trained and compared:

**Logistic Regression**
- StandardScaler normalization
- Balanced class weights
- L2 regularization

**Random Forest**
- 100 trees
- Balanced class weights
- Feature importance analysis

**XGBoost**
- Scale_pos_weight for imbalance
- max_depth = 5

### 4. Feature Importance Analysis

The Random Forest model provides feature importance scores, identifying which SARS-CoV-2 CDR3 sequences are most predictive. The top features represent:
- CDR3s that are common in positive repertoires
- CDR3s that rarely appear in negative repertoires

### 5. Results

This approach works well because:
- The signal is explicitly defined (SARS-CoV-2 reactivity)
- External database provides biological ground truth
- Simple models can capture the linear relationship

---

# Page 4: Dataset 4 (DS4) — Synthetic

## Overview

Dataset 4 is a synthetic dataset solved using **hybrid gapped k-mers with XGBoost**. A critical insight was ensuring **no data leakage** during pattern selection.

## Methodology

### 1. Preventing Data Leakage

The key to DS4 was proper train/test splitting BEFORE pattern discovery:

```python
# STEP 1: Split data FIRST
train_indices, test_indices = train_test_split(
    metadata.index,
    test_size=0.2,
    stratify=metadata['label_positive']
)

# STEP 2: Identify patterns using ONLY training data
patterns = identify_disease_patterns(train_kmers, y_train, ...)
```

If patterns are selected using all data (including test), the model sees test data during training, leading to inflated performance estimates.

### 2. Gapped K-mer Tokenization

Standard k-mers (contiguous sequences) may miss structural patterns. Gapped k-mers allow flexibility:

- **Standard 4-mer**: `ABCD`
- **Gapped 4-mers**: `A.CD`, `AB.D`, `ABC.` (where `.` is any amino acid)

This captures:
- Conserved anchor positions
- Variable middle regions
- Structural motifs that span insertions

### 3. Statistical Pattern Selection

For each k-mer pattern, compute disease association using Fisher's exact test:

```
Contingency table:
                    Positive samples    Negative samples
Pattern present:    pos_count           neg_count
Pattern absent:     n_pos - pos_count   n_neg - neg_count
```

Selection criteria (strict to prevent overfitting):
- p-value < 0.0001
- fold_change > 4.0
- Maximum 50 patterns per k-size

### 4. Aggregated Pattern Features

Instead of using individual patterns as features (which can overfit), aggregate statistics:

- **Count**: Number of significant patterns present
- **Abundance**: Total occurrences of significant patterns
- **Diversity**: Shannon entropy of pattern distribution

This reduces feature dimensionality from potentially thousands to ~12-16 features.

### 5. V/J Gene Features

Additional features from germline gene usage:
- V gene unique count and diversity
- J gene unique count and diversity

### 6. Classification Model

**XGBoost Classifier**:
- 200 estimators
- max_depth = 4
- learning_rate = 0.05
- RobustScaler preprocessing

### 7. Results

The leakage-free approach achieved:
- Realistic CV estimates
- Generalization to held-out test set
- Biologically interpretable patterns

---

# Page 5: Dataset 5 (DS5) — Synthetic

## Overview

Dataset 5 is a synthetic dataset solved using **shared sequence patterns with statistical selection** and an **ensemble of multiple classifiers**.

## Methodology

### 1. Statistical Pattern Discovery

Enhanced pattern identification with multiple criteria:

**Fisher's Exact Test**: Statistical significance of pattern enrichment

**FDR Correction**: Benjamini-Hochberg procedure for multiple testing:
```python
from statsmodels.stats.multitest import multipletests
_, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
```

**Combined Scoring**:
```python
combined_score = -log10(p_value) * fold_change * log1p(abundance)
```

### 2. Multi-K-mer Approach

Patterns are extracted for multiple k-mer sizes (k=3, 4, 5):
- Shorter k-mers: More general patterns, higher statistical power
- Longer k-mers: More specific patterns, biological precision

### 3. Advanced Feature Engineering

**Basic Pattern Features**:
- Pattern count (number of significant patterns present)
- Pattern abundance (total occurrences)
- Pattern diversity (Shannon index)
- Max/mean abundance

**Normalized Features**:
- Pattern ratio (patterns / total sequences)
- Abundance ratio (pattern abundance / total k-mers)
- Count per 1000 sequences

**Interaction Features**:
- Ratio between different k-mer sizes (e.g., k3_count / k4_count)
- Combined features across all k-sizes

### 4. Ensemble Model Training

Multiple algorithms are trained with grid search:

**Logistic Regression**:
- C values: [0.001, 0.01, 0.1, 1.0, 10.0]
- Penalties: L1, L2

**Random Forest**:
- n_estimators: [100, 200, 300]
- max_depth: [5, 10, 15, None]

**Gradient Boosting**:
- n_estimators: [100, 200]
- learning_rate: [0.01, 0.1, 0.2]

**XGBoost**:
- Similar hyperparameter grid

### 5. Voting Ensemble

Top 3 models (by CV AUC) are combined:
```python
ensemble = VotingClassifier(
    estimators=top_models,
    voting='soft'  # Average probabilities
)
```

### 6. Nested Cross-Validation

To avoid leakage during pattern selection:
1. Outer loop: 5-fold CV for final evaluation
2. Inner loop: Pattern selection + model training on fold's training portion
3. Report outer loop performance as unbiased estimate

---

# Page 6: Dataset 6 (DS6) — Synthetic

## Overview

Dataset 6 is a synthetic dataset where positive samples contain **HER2/neu-reactive TCRs**. The solution uses **antigen-specific sequence matching**, similar to DS3 but targeting a cancer antigen.

## Methodology

### 1. External Reference Database

The Parse Bioscience dataset contains TCRs reactive to various antigens, including HER2/neu (human epidermal growth factor receptor 2). We extract:

```python
her2neu_chunk = chunk[
    (chunk['sample'] == 'Her2Neu') & 
    (chunk['is_positive'] == True) & 
    (chunk['cdr3_aa'].notna())
]
```

### 2. Feature Matrix Construction

For each repertoire, count occurrences of each HER2/neu CDR3:

```python
rep_cdr3_counts = Counter(rep_df['junction_aa'])
feature_vector = [rep_cdr3_counts.get(cdr3, 0) for cdr3 in her2neu_cdr3s]
```

The resulting feature matrix is:
- Dimensions: (n_samples × n_her2neu_sequences)
- Sparse (most entries are 0)
- Direct biological interpretation

### 3. Classification Models

Same three-model approach as DS3:

**Logistic Regression**:
- StandardScaler normalization
- class_weight = 'balanced'

**Random Forest**:
- n_estimators = 100
- class_weight = 'balanced'

**XGBoost**:
- scale_pos_weight = n_neg / n_pos

### 4. Feature Importance

Random Forest feature importance identifies:
- Which HER2/neu CDR3s are most predictive
- These likely represent true positive-sample-specific sequences

### 5. Model Selection

The best model is selected based on:
- 5-fold cross-validation AUC
- Held-out test set AUC
- Typically Logistic Regression or Random Forest perform best

### 6. Key Insights

DS6 demonstrates:
- Cancer antigen-specific TCRs can be detected in repertoire data
- External databases provide powerful prior knowledge
- Simple linear models work when signal is clear

---

# Page 7: Dataset 7 (DS7) — HSV (Real-World)

## Overview

Dataset 7 contains real-world repertoire data from HSV (Herpes Simplex Virus) studies. The solution uses a **multi-modal approach** combining differential sequences, diversity metrics, and gene usage features.

## Methodology

### 1. Challenge: Class Imbalance

DS7 has severe class imbalance (many more negative than positive samples). Key strategy: **undersampling** negative samples.

```python
undersampler = RandomUnderSampler(
    sampling_strategy=0.5,  # 2:1 ratio
    random_state=42
)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train, y_train)
```

### 2. Differential Sequence Discovery

Identify sequences that are differentially shared between classes:

```python
# For each sequence, compute differential score
score = (fraction in positive) - (fraction in negative)

# Select top 600 sequences by score
differential_sequences = sorted_by_score[:600]
```

These 600 sequences become binary features (present/absent in each repertoire).

### 3. Diversity Features

Repertoire-level statistics capture overall structure:

**Shannon Entropy**:
```python
diversity_shannon = -sum(p * log(p) for p in frequencies)
```

**Simpson's Index**:
```python
diversity_simpson = 1 - sum(p^2 for p in frequencies)
```

**Clonality**:
```python
clonality = 1 / (1 + shannon_entropy)
```

**Clone Distribution**:
- Unique ratio (unique / total)
- Top clone frequency
- Top 10 clones cumulative frequency

### 4. V/J Gene Features

Germline gene usage patterns:
- V gene diversity (Shannon entropy)
- V gene unique count
- Top V gene frequency
- Same metrics for J genes

### 5. CDR3 Length Features

Length distribution statistics:
- Mean, std, min, max, median length
- Binned distribution (8-12, 12-16, 16-20, 20-24)

### 6. Classification Model

**XGBoost Classifier** with hyperparameter search:
```python
configs = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
]
```

### 7. Optimal Threshold

For imbalanced data, the default 0.5 threshold is suboptimal. Use Youden's J statistic:
```python
j_scores = tpr - fpr
optimal_threshold = thresholds[argmax(j_scores)]
```

---

# Page 8: Dataset 8 (DS8) — Type 1 Diabetes (Real-World)

## Overview

Dataset 8 contains real-world repertoire data from Type 1 Diabetes studies with very large repertoires (millions of sequences). The solution uses a **GCN + XGBoost ensemble** with careful preprocessing.

## Methodology

### 1. Preprocessing: Downsampling

Large repertoires must be downsampled for computational tractability:

**Target**: 10,000 templates per repertoire

**Method**: Immunarch-style multinomial sampling
```python
# Sample proportional to template counts
probabilities = template_counts / total
sampled_counts = np.random.multinomial(10000, probabilities)
```

This preserves relative abundance while reducing size.

### 2. Component 1: Graph Convolutional Network (GCN)

**Graph Construction**:
1. Each CDR3 sequence is a node
2. Edges connect k-nearest neighbors (k=30) based on sequence similarity
3. Similarity computed using CVC (Clustered Viral Copy) embeddings

**Node Features (59 dimensions)**:
- Frequency (1)
- CDR3 length (1)
- Amino acid composition (20)
- Graph centrality measures (4)
- Network density (1)
- CVC embedding (32 dimensions)

**GCN Architecture**:
- 3 layers
- Hidden dimension: 160
- Dropout: 0.47
- Global mean pooling for graph-level prediction

### 3. Component 2: XGBoost on Repertoire Features

Extract ~200 repertoire-level features:
- K-mer frequencies
- V/J gene usage
- Diversity metrics
- CDR3 length statistics
- Antigen-specific sequence counts (Proinsulin, GAD65)

Standard XGBoost classifier with feature selection.

### 4. Rank Scaling for Calibration

GCN and XGBoost predictions may have different scales. Apply rank scaling:
```python
def rank_scale(preds):
    ranks = np.argsort(np.argsort(preds))
    return ranks / (len(preds) - 1)
```

This normalizes predictions to [0, 1] while preserving ordering.

### 5. Meta-Learner Ensemble

Combine GCN and XGBoost using a meta-learner:

```python
# Stack predictions as features
meta_features = np.column_stack([gcn_probs, xgb_probs])

# Train logistic regression as meta-learner
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)
```

### 6. Out-of-Fold Predictions

To avoid leakage in the meta-learner:
1. Generate OOF (out-of-fold) predictions for each component model
2. Train meta-learner on OOF predictions only
3. This ensures the meta-learner never sees predictions on training samples

### 7. Final Prediction

For test samples:
1. Run GCN to get probability
2. Run XGBoost to get probability
3. Apply rank scaling
4. Combine with meta-learner weights

### 8. Results

The ensemble achieves better performance than either component alone by:
- GCN captures inter-sequence relationships (graph structure)
- XGBoost captures repertoire-level statistics
- Meta-learner optimally weights their contributions

---

# Summary

| Dataset | Type | Key Approach | Model |
|---------|------|--------------|-------|
| DS1 | Synthetic (T1D) | VDJdb antigen-specific TCR counts (Proinsulin, GAD65 diversity) | Calibrated Logistic Regression |
| DS2 | Synthetic | Aggregated k-mer/gene features + forward feature selection | XGBoost |
| DS3 | Synthetic | Parse Bioscience SARS-CoV-2 TCR database matching | Logistic Regression |
| DS4 | Synthetic | Gapped k-mers with strict train/test separation | XGBoost |
| DS5 | Synthetic | Statistical pattern selection + FDR correction | Voting Ensemble |
| DS6 | Synthetic | Parse Bioscience HER2/neu TCR database matching | Logistic Regression |
| DS7 | Real (HSV) | Differential sequences + diversity + undersampling | XGBoost |
| DS8 | Real (T1D) | GCN (graph structure) + XGBoost (repertoire stats) | Stacking Ensemble |

## Key Principles

1. **Prevent data leakage**: Always split data before pattern/feature discovery
2. **Use domain knowledge**: External databases (VDJdb, Parse) provide biological signal
3. **Match complexity to signal**: Simple models for clear signals, ensembles for noisy data
4. **Handle imbalance carefully**: Undersampling, class weights, and threshold tuning
5. **Combine orthogonal information**: Different feature types capture different aspects of the signal

