# File manifest: DS2, DS4, DS5, DS7, DS8

All code files present as of the latest submission update.

| Dataset | Code files |
|---------|------------|
| **DS2** | `train.py`, `predict.py`, `witness_ranking.py` |
| **DS4** | `train.py`, `predict.py`, `rank_sequences.py` |
| **DS5** | `train.py`, `predict.py` |
| **DS7** | `train.py`, `predict.py` |
| **DS8** | `cache_utils.py`, `cvc_embedder.py`, `downsample_samples.py`, `downsample_test.py`, `ensemble_predictor.py`, `extract_repertoire_features.py`, `generate_ranked_sequences_only.py`, `graph_builder.py`, `graph_classification.py`, `predict.py`, `tcrdist3_embedder.py`, `tcremp_embedder.py`, `tcrformer_embedder.py`, `train_gcn_best_params.py` |

DS8 embedder modules (`tcrformer_embedder.py`, `tcremp_embedder.py`, `tcrdist3_embedder.py`) are included so that `graph_classification.py` can be run with different embedder types. The submitted model uses CVC only (`embedder_type: "cvc"` in `ds8/model/ensemble_config.json`).
