# Semantic Core Translator Experiment

Local RU-EN experiment for testing whether a compact semantic bottleneck can make representations more coherent while preserving translation quality.

This MVP intentionally uses a small synthetic domain and `numpy`, not an external LLM API. It compares:

- `baseline`: direct phrase-table translation from observed training pairs.
- `semantic_translator`: text -> compact semantic vector -> nearest meaning -> target text.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 scripts/build_dataset.py --config configs/default.yaml
python3 scripts/train_semantic_core.py --config configs/default.yaml
python3 scripts/train_baseline.py --config configs/default.yaml
python3 scripts/train_semantic_translator.py --config configs/default.yaml
python3 scripts/evaluate.py --config configs/default.yaml
python3 scripts/demo.py --text "кошка сидит на столе" --target en
```

## Outputs

- `data/processed/*.jsonl` synthetic train/valid/test splits.
- `checkpoints/semantic_encoder.npz`
- `checkpoints/baseline_translator.json`
- `checkpoints/semantic_translator.json`
- `reports/metrics.json`
- `reports/examples.md`
- `reports/semantic_clusters.csv`
- `reports/semantic_clusters.png` if `matplotlib` is installed.

## Metrics

- `semantic_gap = intra_cluster_similarity - inter_cluster_similarity`
- `cycle_consistency`: cosine between original and after-translation semantic vectors.
- `exact_match`, `token_accuracy`, `char_similarity`
- `ablation_exact_match`: quality after shuffling semantic vectors. It should drop sharply.
- latency and checkpoint sizes.

The experiment is promising only if the semantic model has a positive semantic gap, high cycle consistency, and the ablation test damages translation quality.

