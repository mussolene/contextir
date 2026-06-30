# SIR Roundtrip Benchmark

This benchmark checks whether SIR preserves semantic nodes through a long-text path:

1. source text -> SIR concept packet;
2. SIR packet -> bridge text in another language;
3. bridge text -> SIR concept packet;
4. packet -> reconstructed source-language text;
5. reconstructed text -> final SIR concept packet.

The current score is concept-level, not fluency-level. That is intentional: the architecture claim is that SIR can become a compact semantic intermediate representation. Surface translation quality should be measured separately after the decompiler becomes a real generator.

Metrics:

- `concept_precision`: final concepts that were present in the source packet.
- `concept_recall`: source concepts recovered after roundtrip.
- `concept_f1`: harmonic mean of precision and recall.
- `segment_coverage`: share of source segments with at least one concept hit.
- `unknown_token_rate`: source tokens not explained by matched concept aliases.
- `compression_ratio`: serialized SIR packet size divided by source text size.

Run:

```bash
python3 scripts/evaluate_roundtrip.py
```

The direct baseline keeps only one dictionary-level concept per segment, then it is scored against the full SIR source packet so its small output cannot inflate its own F1. It is a deliberately simple transpiler baseline, not a modern neural translator baseline. A FLORES/WMT adapter is the next fair external benchmark once the decompiler can generate fluent text rather than keyword streams.

Current local result:

- long-text SIR roundtrip: `concept_f1=0.5342`, `segment_coverage=0.9583`, `latency_ms=~3.2`;
- direct transpiler baseline: `concept_f1=0.2242`;
- SIR delta over direct baseline: `+0.3100`;
- existing synthetic translator bench: `exact_match=0.72`, `cycle_consistency=0.9704`, `semantic_gap=0.6380`.

Stabilization priorities exposed by the benchmark:

1. Add concept typing and relation-aware filtering before decompile.
2. Add morphology-aware Russian normalization, because many unknown tokens are inflected forms.
3. Rank aliases during decompile by language frequency/domain instead of using the first WordNet lemma.
4. Keep vector fallback out of the trusted roundtrip path until it is calibrated with hard negatives.
5. Expand the project-local SIR domain pack; the first pack improved roundtrip F1 from `0.4819` to `0.5342`.
