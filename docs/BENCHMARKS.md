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
  "avg_eligible_prompt_ratio": 0.3067
}
```

Detailed, versioned cards:

- [Gateway smoke and performance](benchmarks/GATEWAY_SMOKE.md)
- [Legacy lexical roundtrip](benchmarks/LEGACY_LEXICAL.md)

This is a smoke test. The compression case contains repeated context and is
designed to prove deduplication, not estimate production quality.

## Required A/B Benchmark

The next benchmark must compare the same downstream tasks using:

1. masked raw context;
2. a conventional text summary;
3. ContextIR auto mode;
4. ContextIR forced hybrid mode.

Required measurements:

- actual tokenizer counts for every target model;
- task success and tool-call accuracy;
- preservation of numbers, negation, conditions, and constraints;
- PII precision, recall, and leakage;
- compilation latency, model latency, and memory;
- fallback rate from semantic to hybrid/raw.

Model-level claims must not be made until this evaluation exists.

## Legacy Results

The previous lexical roundtrip and graph-relation results are retained under
`reports/` and documented in [LEGACY_ROUNDTRIP.md](LEGACY_ROUNDTRIP.md). They
measure lexical concept recovery, not ContextIR v2 compression quality.
