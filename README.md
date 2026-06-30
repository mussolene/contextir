# SIR Translator

Local research project for building a Semantic Intermediate Representation
(SIR) translator: language text is compiled into a compact semantic state,
processed by a small core, and decompiled back into language.

This repository is now the single home for the experiment. Nearby SIR/LLM lab
projects should be treated as historical benchmark work, not as the active
architecture.

## Architecture Direction

```text
language text
  -> precompiler
  -> SIR vector / SIR graph
  -> small ML core
  -> decompiler
  -> language text / action / relation
```

The project currently has two layers:

- synthetic RU/EN translator MVP for measuring a compact semantic bottleneck;
- lexical SIR + tiny ML core for concept grounding and future pretraining.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run Synthetic RU/EN MVP

```bash
python3 scripts/build_dataset.py --config configs/default.yaml
python3 scripts/train_semantic_core.py --config configs/default.yaml
python3 scripts/train_baseline.py --config configs/default.yaml
python3 scripts/train_semantic_translator.py --config configs/default.yaml
python3 scripts/evaluate.py --config configs/default.yaml
python3 scripts/demo.py --text "кошка сидит на столе" --target en
```

## Run SIR Concept Core

Use the local source cache if present, or add `--download` to fetch Open
Multilingual Wordnet and WordNet sources:

```bash
python3 scripts/build_concept_sources.py --limit 0
python3 scripts/train_sir_ml_core.py train --limit 500 --epochs 4 --out checkpoints/sir_ml_core_smoke.npz
python3 scripts/train_sir_ml_core.py eval --limit 500 --checkpoint checkpoints/sir_ml_core_smoke.npz
python3 scripts/train_sir_ml_core.py demo --text "кошка" --target en --limit 0 --checkpoint checkpoints/sir_ml_core_smoke.npz
```

Expected demo:

```text
Input: кошка
SIR nearest concept: 02121620-n
Output: cat
```

## Outputs

- `data/processed/*.jsonl` synthetic train/valid/test splits.
- `data/concepts/concept_records.jsonl` normalized lexical concept records.
- `checkpoints/semantic_encoder.npz`
- `checkpoints/baseline_translator.json`
- `checkpoints/semantic_translator.json`
- `checkpoints/sir_ml_core_smoke.npz`
- `reports/metrics.json`
- `reports/examples.md`
- `reports/sir_ml_core_smoke_eval.json`

## Metrics

- `semantic_gap = intra_cluster_similarity - inter_cluster_similarity`
- `cycle_consistency`: cosine between original and after-translation semantic vectors.
- `exact_match`, `token_accuracy`, `char_similarity`
- `ablation_exact_match`: quality after shuffling semantic vectors. It should drop sharply.
- latency and checkpoint sizes.

The experiment is promising only if semantic coherence is positive, ablations
damage quality, and the precompiler/core/decompiler boundary remains inspectable.

## Git Milestones

Work should advance through small commits after each validated milestone:

1. baseline synthetic translator;
2. concept source ingestion;
3. SIR ML core smoke;
4. graph relations and source adapters;
5. RU phraseology and Japanese WordNet.
