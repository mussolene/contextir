# Benchmark Card: ContextIR Gateway Smoke

## Identity

- Component: deterministic `contextir.v2` gateway
- Version: 1.1.0
- Cases: 9 checked-in RU/EN fixtures
- Command: `python3 scripts/evaluate_contextir.py --check`
- Hardware: developer machine; latency is not normalized across hardware

## Results

| Metric | Result |
|---|---:|
| PII leaks into public contract or prompt | 0 |
| Synthetic privacy precision / recall | 1.0000 / 1.0000 |
| Annotated synthetic PII values | 6 |
| Semantic expectation failures | 0 |
| Product pipeline cases | 6 |
| Product pipeline failures | 0 |
| Exercised bounded fallbacks | 1 |
| Compression-eligible cases | 1 |
| Eligible prompt/source character ratio | 0.3627 |
| Compile latency p50 | 0.0956 ms |
| Compile latency p95 | 0.9667 ms |
| Compile throughput | 5077.6 docs/s |

Performance uses a 100-operation warm-up followed by 5,000 repeated compilations
over the nine fixtures with Python garbage collection paused. The ratio uses
rendered characters, not a model tokenizer. Re-run on the target hardware for a
valid comparison; the checked-in latency is only a developer-machine baseline.

## Long-Context Product Path

An additional 100-iteration warm-cache run uses the same local LongBench and
synthetic inputs as the model A/B harness. These timings measure
`ContextPipeline.prepare()` with the default approximate token counter.

| Case | Input | Selected path | Prepare p50 |
|---|---:|---|---:|
| `multifieldqa_en_1` | 45,867 chars | query-aware hybrid | 17.681 ms |
| `passage_count_0` | 66,162 chars | exhaustive raw | 53.577 ms |
| `operational_privacy` | 5,036 chars | semantic/privacy | 8.018 ms |

The retrieval path avoids semantic event extraction and the pipeline no longer
compiles an unconditional raw baseline. Exhaustive counting remains the most
expensive path because correctness requires scanning and preserving the full
source. Timings are developer-machine diagnostics, not cross-machine targets.

## External Privacy Profile

The optional external run uses all 1,500 rows from
[Microsoft Presidio Research](https://github.com/microsoft/presidio-research)'s
MIT-licensed `synth_dataset_v2`. The dependency-free profile is
scored only on the kinds it claims to support in that corpus: email, phone, and
payment card.

| Metric | Result |
|---|---:|
| Expected supported spans | 277 |
| True positives | 277 |
| False positives | 50 |
| False negatives | 0 |
| Exact-value precision | 0.8471 |
| Exact-value recall | 1.0000 |

```bash
python3 scripts/evaluate_contextir.py --check --performance-iterations 0 \
  --external-privacy-dataset /path/to/presidio-research/data/synth_dataset_v2.json \
  --out reports/privacy_presidio_eval.json
```

This does not measure person names, addresses, identity documents, or Russian
PII. Use the Presidio adapter with deployment-specific recognizers for those
classes. The remaining phone false positives also make the default profile
unsuitable as a standalone compliance boundary.

## What This Establishes

The test establishes deterministic contract shape, preservation of configured
numbers/conditions/negation, basic masking, deduplication, prompt-budget
enforcement, and a fast local path for small inputs.

## What It Does Not Establish

The only long compiler compression fixture is deliberately repetitive. These
numbers do not establish general semantic fidelity, translation quality,
all-class PII recall, or savings for every tokenizer. A deployment-representative
A/B corpus is still required before production claims.
