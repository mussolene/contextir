# Benchmark Card: ContextIR Gateway Smoke

## Identity

- Component: deterministic `contextir.v2` gateway
- Version: 0.3.0
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
| Eligible prompt/source character ratio | 0.3067 |
| Compile latency p50 | 0.1043 ms |
| Compile latency p95 | 0.9774 ms |
| Compile throughput | 3204.3 docs/s |

Performance uses a 100-operation warm-up followed by 5,000 repeated compilations
over the four fixtures with Python garbage collection paused. The ratio uses
rendered characters, not a model tokenizer. Re-run on the target hardware for a
valid comparison; the checked-in latency is only a developer-machine baseline.

## What This Establishes

The test establishes deterministic contract shape, preservation of configured
numbers/conditions/negation, basic masking, deduplication, and a fast local path
for small inputs.

## What It Does Not Establish

The only long compression fixture is deliberately repetitive. These numbers do
not establish general semantic fidelity, translation quality, PII recall, model
task quality, or savings for any specific tokenizer. A representative A/B corpus
is required before making production claims.
