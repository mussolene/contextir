# SIR ML Core Architecture

This project is the single home for the SIR translator experiment.

The intended runtime shape is:

```text
language text
  -> precompiler
  -> SIR vector / SIR graph
  -> small ML core
  -> decompiler
  -> language text / action / relation
```

## Source Policy

Russian lexical grounding must be pluggable. RuWordNet can be used for research
if its license constraints are acceptable, but the architecture must not depend
on it directly.

Source order:

1. Open Multilingual Wordnet plus Wiktionary/DBnary-derived Russian lemmas.
2. RuWordNet as a research/non-commercial source if needed.
3. Project-local Russian idioms, particles, interjections, and phraseology.
4. Japanese WordNet once the RU/EN concept graph path is stable.

All sources should compile into normalized records:

```json
{
  "concept_id": "02121620-n",
  "en": ["cat"],
  "ru": ["кошка"],
  "definition_en": "feline mammal usually having thick soft fur",
  "pos": "n",
  "source": "omw+wordnet30"
}
```

## Current v0

The current v0 is hybrid:

```text
exact lexical precompiler
  -> learned fallback projection
  -> SIR vector
  -> nearest concept
  -> decompiler target lemma
```

A pure linear projection from text features to concepts failed as a translator,
which is useful evidence. The precompiler is not optional; the ML core should
sit above compilation rather than replace all lexical grounding.

## Commands

```bash
python3 scripts/build_concept_sources.py --download --limit 0
python3 scripts/train_sir_ml_core.py train --limit 500 --epochs 4 --out checkpoints/sir_ml_core_smoke.npz
python3 scripts/train_sir_ml_core.py eval --limit 500 --checkpoint checkpoints/sir_ml_core_smoke.npz
python3 scripts/train_sir_ml_core.py demo --text "кошка" --target en --limit 0 --checkpoint checkpoints/sir_ml_core_smoke.npz
```

## Next Milestones

1. Add a `concept_graph.jsonl` builder with explicit relations.
2. Add source adapters for DBnary/Wiktionary, RuWordNet, Japanese WordNet, and
   a curated Russian idiom pack.
3. Replace flat concept retrieval with graph tasks:
   relation prediction, contradiction detection, missing-edge recovery, and
   idiom-to-literal-SIR compilation.
4. Keep every milestone behind a benchmark and a git commit.

## Current Graph Milestone

The first graph layer builds WordNet pointer relations among the current
concept records:

```bash
python3 scripts/build_concept_graph.py --min-relation-count 3
python3 scripts/train_sir_graph_core.py train --out checkpoints/sir_graph_core_smoke.npz
python3 scripts/train_sir_graph_core.py eval --checkpoint checkpoints/sir_graph_core_smoke.npz --out reports/sir_graph_core_eval.json
```

Model shape:

```text
source concept vector + relation-specific linear map -> target concept vector
```

The held-out validation metric is the main metric. Full-graph eval is useful as
a fit/debug check, not as proof of generalization.

Current held-out result:

```text
relations: 7,126
hit@1: 0.0772
hit@5: 0.1364
hit@10: 0.1662
MRR@10: 0.1033
```

The relation core is therefore a benchmarkable first graph substrate, not a
finished reasoner. Broad relations such as `hypernym` and `hyponym` remain weak;
more constrained relations such as `instance_hypernym`, `domain_topic`,
`domain_usage`, and several part/domain relations already show useful signal.

Next architectural step: replace relation-specific linear maps with a small
graph encoder trained with negative sampling and relation-aware neighborhoods.

## Trainable Graph Embedding Milestone

`semantic_core/sir_graph_embedding_core.py` adds a trainable TransE-style graph
core:

```text
source concept embedding + relation embedding ~= target concept embedding
```

Training uses margin ranking with negative target sampling. This is still small
and inspectable, but it is a real learned graph substrate rather than a closed
form relation map.

Full graph training command:

```bash
python3 scripts/train_sir_graph_embedding_core.py train --dim 96 --epochs 4 --out checkpoints/sir_graph_embedding_core_smoke.npz
```

Held-out result:

```text
relations: 7,126
hit@1: 0.0946
hit@5: 0.1953
hit@10: 0.2397
MRR@10: 0.1369
```

This improves over the relation-matrix held-out baseline:

```text
matrix hit@1: 0.0772
matrix hit@10: 0.1662
embedding hit@1: 0.0946
embedding hit@10: 0.2397
```

The `cat + hypernym` probe now puts `animal` in the top 2. That is not yet good
enough for product inference, but it confirms the training direction.

## Long Training + Hard Negatives

The next run used 12 epochs, 128 dimensions, and hard negatives every second
epoch:

```bash
python3 scripts/train_sir_graph_embedding_core.py train \
  --dim 128 \
  --epochs 12 \
  --lr 0.02 \
  --hard-negative-every 2 \
  --hard-negative-k 32 \
  --out checkpoints/sir_graph_embedding_core_long.npz \
  --report reports/sir_graph_embedding_long_valid_eval.json
```

Held-out result:

```text
relations: 7,126
hit@1: 0.2548
hit@5: 0.5267
hit@10: 0.6214
MRR@10: 0.3689
```

This is a large improvement over the first embedding run:

```text
short hit@10: 0.2397
long  hit@10: 0.6214
```

Caveat: full-graph eval is much higher because it includes training edges. The
held-out report remains the decision metric.

The `cat + hypernym` demo is still not perfect: it retrieves `domestic cat`
before broader concepts. That exposes the next data/model issue: relation tasks
need typed negatives and ancestor-aware evaluation, not only random and nearest
hard negatives.
