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

