# Benchmark Card: ContextIR Gateway Smoke

## Identity

- Component: deterministic `contextir.v2` gateway
- Version: 0.4.1
- Cases: 4 checked-in RU/EN fixtures
- Command: `python3 scripts/evaluate_contextir.py --check`
- Hardware: developer machine; latency is not normalized across hardware

## Results

| Metric | Result |
|---|---:|
| PII leaks into public contract or prompt | 0 |
| Semantic expectation failures | 0 |
| Product pipeline cases | 4 |
| Product pipeline failures | 0 |
| Exercised bounded fallbacks | 1 |
| Compression-eligible cases | 1 |
| Eligible prompt/source character ratio | 0.3627 |
| Compile latency p50 | 0.0985 ms |
| Compile latency p95 | 0.9610 ms |
| Compile throughput | 3263.8 docs/s |

Performance uses a 100-operation warm-up followed by 5,000 repeated compilations
over the four fixtures with Python garbage collection paused. The ratio uses
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

## What This Establishes

The test establishes deterministic contract shape, preservation of configured
numbers/conditions/negation, basic masking, deduplication, and a fast local path
for small inputs.

## What It Does Not Establish

The only long compression fixture is deliberately repetitive. These numbers do
not establish general semantic fidelity, translation quality, PII recall, model
task quality, or savings for any specific tokenizer. A representative A/B corpus
is required before making production claims.
