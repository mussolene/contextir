# Changelog

All notable changes to ContextIR are documented here.

## 1.0.0 - 2026-07-18

- stabilized `ContextPipeline` as the primary model-boundary API while keeping
  the existing per-call invoker compatible;
- added dependency-free callable `OllamaClient` and `OpenAICompatibleClient`
  transports with normalized usage metadata and endpoint errors;
- added `contextir run` for one-command local and OpenAI-compatible model
  invocation, including stdin and payload-free JSON trace modes;
- moved the quick start to the runnable client-plus-pipeline path and documented
  the direct release-wheel install;
- documented SemVer guarantees, stable exports, contract evolution, and the
  migration from the 0.x API;
- verified the built wheel in a clean virtual environment and exercised the
  installed CLI against local Ollama;
- retained the `contextir.v2` contract without a schema migration.

## 0.5.0 - 2026-07-18

- added strict agent tool-routing and agent-state diagnostics to the existing
  local-model A/B harness;
- added optional external privacy-corpus evaluation with precision and recall
  gates;
- added exact-value privacy annotations to the checked-in gateway benchmark;
- classified payment cards before phone numbers and added Luhn validation;
- rejected ISO dates, IPv4 values, SSN-shaped values, and short bare numbers
  from the dependency-free phone recognizer;
- measured `0.8471` precision and `1.0000` recall on 277 supported spans from
  Microsoft Presidio Research's 1,500-row MIT-licensed synthetic corpus;
- measured identical `0.7272` raw and auto quality on nine Qwen3 8B cases while
  reducing model input by 70.0% and mean latency from `53.8 s` to `2.51 s`.

## 0.4.1 - 2026-07-18

- changed the product pipeline to compile the candidate first and build a raw
  baseline only when the savings gate needs it;
- avoided duplicate compilation when auto mode already selects raw;
- skipped event and numeric-entity extraction for query-aware retrieval, where
  the model receives selected verbatim evidence instead;
- precompiled hot-path regular expressions and reused normalized token sets;
- replaced repeated paragraph-owner scans with a linear ownership pass;
- added deterministic tests for single-pass candidate and exhaustive routing;
- preserved all 16 sampled contracts and all seven saved model-facing prompt
  token counts exactly.

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
