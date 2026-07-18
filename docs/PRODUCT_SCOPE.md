# Product Scope

## Product

ContextIR is an adaptive context and privacy gateway between applications and
language models.

```text
text or agent history
  -> local privacy masking
  -> adaptive context compiler
  -> raw | hybrid | semantic prompt
  -> language model
  -> application validation
  -> allowlisted placeholder restoration
```

## In Scope

- deterministic local preprocessing;
- inspectable semantic events and constraints;
- critical-source retention through source references;
- PII placeholders backed by a local vault;
- optional Presidio detection;
- provider-independent model prompts;
- measurable prompt reduction and semantic-preservation checks;
- RU and EN heuristics in the alpha release.

## Out of Scope

- a universal language of thought;
- replacing model reasoning;
- guaranteed fluent machine translation;
- covert steganographic transport;
- guaranteed detection of every sensitive value;
- production claims based only on WordNet overlap;
- Japanese support before RU/EN evaluation is stable.

## Alpha Exit Criteria

The project should not claim beta readiness until a representative benchmark
shows all of the following:

- at least 40% median input-token reduction on compression-eligible contexts;
- no more than 3% task-quality loss against masked raw input;
- at least 98% retention of tested negation, numbers, and hard constraints;
- measured PII precision and recall on deployment-relevant data;
- results on at least one 1-3B local model and one 7-8B model;
- bounded fallback to raw source whenever confidence is insufficient.

