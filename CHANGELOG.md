# Changelog

All notable changes to ContextIR are documented here.

## 1.4.0 - 2026-07-19

- added explicit `context_kind` and `query` inputs to `ContextPipeline` and
  `ContextIR` so applications can bypass language-specific task markers;
- kept external queries local to `ContextBundle` and out of the unchanged
  `contextir.v2` public contract and payload-free trace;
- scrubbed documents and external queries in one privacy namespace so distinct
  PII values receive collision-free placeholders and one local vault;
- included external queries in rendered prompts, retrieval ranking, prompt
  budgets, fallback recompilation, evidence packing, and chunk map execution;
- added `--context-kind` and `--query` to `contextir run` while preserving the
  existing automatic routing default.

## 1.3.0 - 2026-07-19

- added explicit bounded chunked retrieval for a top-ranked evidence segment
  that cannot fit as one complete group;
- split oversized evidence with configurable overlap, selected query-relevant
  chunks locally, and retained the extracted question outside the public
  contract;
- added map/reduce stages to payload-free pipeline traces without logging
  prompts, candidates, answers, source text, or vault values;
- added lexical grounding checks for text and numeric answers before accepting
  map candidates;
- rejected new PII, unknown placeholders, unsupported candidates, protocol
  output, excessive call plans, and reduce prompts exceeding the model budget;
- reserved 25% map headroom by default to reduce near-window attention
  degradation on small models;
- added `--chunked-retrieval` and bounded chunk controls to the CLI;
- expanded the release benchmark to ten pipeline cases and verified the same
  constrained retrieval answer on local Qwen3 0.6B and 8B models.
- expanded the model A/B harness with real bounded-pipeline execution, model
  call totals, compiler decisions, stage distributions, and deterministic
  paired bootstrap confidence intervals;
- measured ten official LongBench retrieval/QA examples at a 2K context:
  Qwen3 8B improved from `0.1668` raw to `0.6983` while using 39.0% of the
  backend input tokens; Qwen3 0.6B improved from `0.1889` to `0.3121`, but its
  paired confidence interval still crosses zero;
- removed Qwen-style `<think>` wrappers only after raw-response safety checks
  and excluded provider control tags from lexical grounding.
- added end-to-end neural-summary and Nomic embedding-retrieval baselines to
  the model A/B harness, including preprocessor tokens, latency, model calls,
  masked-source checks, and direct paired comparisons against ContextIR;
- measured ContextIR at `0.6983` quality versus `0.6221` for same-model neural
  summary on the ten-case Qwen3 8B subset, while summary consumed `24.95x`
  ContextIR's processed input tokens and `32.0x` its end-to-end latency;
- measured the tested Nomic segment-retrieval baseline at `0.1910` quality on
  Qwen3 8B, establishing that this baseline configuration is not competitive
  with query-aware ContextIR packing on the bounded suite.

## 1.2.0 - 2026-07-19

- added budget-aware retrieval packing that keeps the query, response format,
  and ranked evidence as complete source groups;
- retained labelled paragraph owners whenever sentence-level evidence is
  selected;
- refused retrieval when the best complete evidence group cannot fit instead
  of truncating text or silently switching to a lossy semantic summary;
- kept source priorities local to `ContextBundle` without changing the public
  `contextir.v2` JSON contract;
- removed privacy metadata for evidence excluded from the packed prompt and
  treated placeholders from pruned evidence as unknown model output;
- exposed the payload-free routing decision in `PipelineResult.public_trace()`;
- expanded the release benchmark to eight product-pipeline cases and verified
  constrained-window retrieval on local Qwen3 0.6B and 8B models.

## 1.1.0 - 2026-07-19

- connected Ollama and OpenAI-compatible context limits to the product
  pipeline, reserving requested output tokens and configurable chat-template
  overhead before invocation;
- added `PipelinePolicy.max_prompt_tokens` for custom model adapters and
  tokenizer-aware deployments;
- added the public `ContextWindowExceeded` error so oversized safe prompts fail
  before any model request instead of being silently truncated;
- stopped bounded fallback before sending a richer prompt that exceeds the
  target-model budget;
- exposed the payload-free prompt budget in prepared contexts and public
  traces, and added a concise CLI error for undersized windows;
- corrected the post-1.0 supported-version statement in the security policy.

## 1.0.1 - 2026-07-19

- added PyPI Trusted Publishing through GitHub Actions OIDC;
- split release validation, GitHub publication, and PyPI publication into
  separate least-privilege jobs;
- restricted `id-token: write` to the two-step PyPI job using the protected
  `pypi` GitHub environment;
- reused one validated artifact for both GitHub and PyPI publication.

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
