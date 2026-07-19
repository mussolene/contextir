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
- refusal before initial or fallback prompts exceed the model budget.
- ranked retrieval packing and refusal below the best complete evidence group.
- bounded chunked retrieval and rejection of unsafe map output.

Current checked-in result:

```json
{
  "cases": 9,
  "pii_leaks": 0,
  "expectation_failures": 0,
  "pipeline_cases": 10,
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

The v0.5.0 Qwen3 8B run extends the suite to nine cases with two strict
agent/tool diagnostics. Raw and auto quality are identical at `0.7272`; auto
uses 70.0% fewer model input tokens and reduces mean latency from `53.8 s` to
`2.51 s` on this machine.

The v1.3 constrained-context run uses five `multifieldqa_en` and five
`passage_retrieval_en` examples at a 2K model context. The runtime-enabled path
uses 39.0% of raw backend input tokens. Its paired quality delta is `+0.5315`
with a 95% bootstrap interval of `[0.2086, 0.8225]` on Qwen3 8B, and `+0.1232`
with `[-0.0154, 0.3392]` on Qwen3 0.6B. This is an official-data subset, not a
full LongBench score. All official examples used packed direct retrieval; a
separate synthetic oversized-segment case verifies the map stage.

The broader release benchmark must still add:

- a conventional text-summary baseline;
- full official task sets beyond the ten-case subset;
- deployment-owned PII and agent-history tasks;
- a full official tool-calling benchmark rather than synthetic diagnostics.

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
