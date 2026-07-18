# Changelog

All notable changes to ContextIR are documented here.

## 0.4.0 - 2026-07-18

- added query-aware lexical evidence selection for long document QA;
- preserved task instructions, output format, paragraph ownership, and queries
  as normal model-facing text rather than exposing the internal IR protocol;
- routed exhaustive counting and retrieval without adequate evidence to raw;
- removed redundant protocol rendering when hybrid source spans cover every
  compiled event;
- reused placeholders for repeated occurrences of the same protected value in
  both built-in and Presidio privacy scrubbers;
- aligned retrieval and passage-count benchmark scoring with LongBench task
  semantics;
- added Ollama and LM Studio follow-up reports showing about 69% token reduction
  without aggregate quality loss on the tested subset.

## 0.3.0 - 2026-07-18

- added `ContextPipeline`, the policy-driven product entry point;
- added target-tokenizer-aware mode selection with a minimum savings gate;
- separated normal reasoning verification from strict transform retention;
- added bounded semantic-to-hybrid-to-raw fallback;
- rejected unknown, missing, and newly generated PII placeholders;
- added payload-free public traces and explicit restoration allowlists;
- fixed repeated-event semantic confidence weighting;
- promoted product pipeline scenarios into the release benchmark gate.

## 0.2.1 - 2026-07-18

- installed development dependencies before release validation;
- updated GitHub Actions to Node.js 24-compatible major versions;
- pinned Gitleaks 8.30.1 and verified its release checksum in CI.

## 0.2.0 - 2026-07-18

- renamed the public package and product from SIR Translator to ContextIR;
- added the `contextir.v2` compact contract and packaged JSON Schema;
- added adaptive raw, hybrid, semantic, and auto modes;
- separated public contracts from local PII vaults and source maps;
- added optional Presidio integration and allowlisted restoration;
- added event deduplication and structured contract comparison;
- retained earlier SIR graph and neural work as a research layer;
- added deterministic tests and a compression/privacy smoke benchmark;
- added packaged schema resources, CI, secret scanning, release automation,
  benchmark cards, and a production-oriented quick start.
