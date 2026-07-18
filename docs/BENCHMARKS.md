# Benchmarks

## ContextIR Gateway Smoke

Run:

```bash
python3 scripts/evaluate_contextir.py
```

The evaluator checks:

- whether configured privacy values are replaced before rendering;
- expected event predicates, negation, and conditions;
- expected numeric entities;
- prompt characters divided by source characters;
- adaptive mode selection.
- measured product-mode selection and rejection of uneconomic compression;
- no false semantic roundtrip requirement for reasoning;
- bounded transform fallback after semantic loss;
- rejection of newly generated PII.

Current checked-in result:

```json
{
  "cases": 4,
  "pii_leaks": 0,
  "expectation_failures": 0,
  "pipeline_cases": 4,
  "pipeline_failures": 0,
  "pipeline_fallbacks": 1,
  "compression_eligible_cases": 1,
  "avg_eligible_prompt_ratio": 0.3627
}
```

Detailed, versioned cards:

- [Gateway smoke and performance](benchmarks/GATEWAY_SMOKE.md)
- [Local model A/B](benchmarks/LOCAL_MODEL_AB.md)
- [Legacy lexical roundtrip](benchmarks/LEGACY_LEXICAL.md)

This is a smoke test. The compression case contains repeated context and is
designed to prove deduplication, not estimate production quality.

## First Model A/B

The first local-model experiment now compares the same downstream tasks using
masked raw, ContextIR auto, forced hybrid, and forced semantic prompts. It uses
an official LongBench subset, RULER-style diagnostics, Ollama, and LM Studio.

The v0.3.0 result was negative for universal semantic compression. The v0.4.0
query-aware follow-up on the same cases reduced model input by about 69% and
improved aggregate quality on all three tested local backends. Qwen3 1.7B rose
from `0.5570` raw to `0.6595` auto. See the
[full card](benchmarks/LOCAL_MODEL_AB.md) and machine-readable reports under
`reports/model_ab/`.

The broader release benchmark must still add:

- a conventional text-summary baseline;
- a representative 7-8B local model;
- more official examples and bootstrap confidence intervals;
- PII precision and recall on a labelled corpus;
- application-owned tool-call and agent-history tasks.

Required measurements:

- actual tokenizer counts for every target model;
- task success and tool-call accuracy;
- preservation of numbers, negation, conditions, and constraints;
- PII precision, recall, and leakage;
- compilation latency, model latency, and memory;
- fallback rate from semantic to hybrid/raw.

Model-level compression claims remain unsupported until the quality-loss gate
is met.

## Legacy Results

The previous lexical roundtrip and graph-relation results are retained under
`reports/` and documented in [LEGACY_ROUNDTRIP.md](LEGACY_ROUNDTRIP.md). They
measure lexical concept recovery, not ContextIR v2 compression quality.
