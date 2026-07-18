---
language:
- en
- ru
task_categories:
- feature-extraction
- sentence-similarity
- text-classification
pretty_name: ContextIR Research Concept Graph
---

# ContextIR Research Concept Graph

Normalized concept and relation triples for the ContextIR research layer.

## Files

- `concepts.jsonl`: concept records with English/Russian lemmas and English definitions.
- `relations.jsonl`: all positive concept relation triples.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: relation prediction splits.

## Counts

- Concepts: 19983
- Relations: 35628
- Train triples: 28504
- Validation triples: 3562
- Test triples: 3562

## Top Relations

| relation | count |
|---|---:|
| hyponym | 8299 |
| hypernym | 8299 |
| derivationally_related | 7051 |
| part_meronym | 2016 |
| part_holonym | 2016 |
| similar_to | 966 |
| instance_hyponym | 929 |
| instance_hypernym | 929 |
| member_of_domain_topic | 799 |
| domain_topic | 799 |
| antonym | 686 |
| pertainym | 509 |
| also_see | 417 |
| member_holonym | 296 |
| member_meronym | 296 |
| attribute | 194 |
| domain_region | 168 |
| member_of_domain_region | 168 |
| domain_usage | 152 |
| member_of_domain_usage | 152 |

## Intended Uses

- multilingual concept grounding;
- relation prediction;
- negative-sampling graph embedding training;
- SIR precompiler/decompiler experiments.

## Caveats

Russian definitions are not yet populated. Russian lexical grounding currently comes from OMW/Wiktionary-derived lemmas.
This dataset is a research artifact, not a production lexical database.
It is not included in the ContextIR Python distribution. Review the upstream
terms in `THIRD_PARTY_NOTICES.md` before publishing or redistributing it.
