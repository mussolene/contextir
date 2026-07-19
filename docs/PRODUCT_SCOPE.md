# Product Scope

## Product

ContextIR is an adaptive context and privacy gateway between applications and
language models.

```text
text or agent history
  -> local privacy masking
  -> adaptive context compiler
  -> raw | hybrid | semantic prompt
  -> language model
  -> application validation
  -> allowlisted placeholder restoration
```

The supported product entry point is `ContextPipeline`. Direct `ContextIR`
compilation remains available for infrastructure that owns routing itself.

## In Scope

- deterministic local preprocessing;
- inspectable semantic events and constraints;
- critical-source retention through source references;
- PII placeholders backed by a local vault;
- optional Presidio detection;
- provider-independent model prompts;
- query-aware verbatim evidence selection for long document QA;
- model-budget packing of ranked complete retrieval evidence groups;
- explicitly enabled bounded chunking of one oversized top-ranked retrieval
  segment for reasoning tasks;
- raw routing for exhaustive counting and low-coverage retrieval;
- measurable prompt reduction and semantic-preservation checks;
- RU and EN heuristics in the stable 1.x package.

## Out of Scope

- a universal language of thought;
- replacing model reasoning;
- guaranteed fluent machine translation;
- generic map-reduce for transform or exhaustive tasks;
- covert steganographic transport;
- guaranteed detection of every sensitive value;
- production claims based only on WordNet overlap;
- Japanese support before RU/EN evaluation is stable.

## Production Evidence Gates

The stable API is suitable for integrations. A deployment should not claim
production readiness until its representative benchmark shows all of the
following:

- at least 40% median input-token reduction on compression-eligible contexts;
- no more than 3% task-quality loss against masked raw input;
- at least 98% retention of tested negation, numbers, and hard constraints;
- measured PII precision and recall on deployment-relevant data;
- results on at least one 1-3B local model and one 7-8B model;
- bounded fallback to raw source whenever confidence is insufficient.
- explicit refusal before a prompt or fallback exceeds the configured model
  context budget.

## 1.0 Stability Contract

ContextIR uses semantic versioning from `1.0.0`. The following public imports
are stable within the 1.x line:

- `ContextPipeline`, `PipelinePolicy`, `PipelineResult`, `PreparedContext`, and
  `ResponseVerification`, plus `ContextWindowExceeded` and
  `ChunkLimitExceeded`;
- `OllamaClient`, `OpenAICompatibleClient`, and `ModelResponse`;
- `ContextIR`, `ContextBundle`, `ContractCheck`, and `load_contextir`;
- `contextir.schemas.load_contract_schema` and the `contextir.v2` JSON shape.

New optional fields may be added to `contextir.v2` in a minor release. Removing
or changing existing fields, public call semantics, or stable imports requires
a major release. Heuristic routing decisions, confidence values, benchmark
scores, and private helpers are not compatibility guarantees.

### Migrating From 0.x

No contract migration is required: 1.0 retains `contextir.v2`. Existing code
that passes `invoke` to `ContextPipeline.run()` remains valid. New integrations
may instead provide a callable once through `ContextPipeline(invoke=...)` and
call `run(text)` repeatedly. Direct `ContextIR` compilation remains supported,
but `ContextPipeline` is the recommended model boundary.

The v0.5.0 nine-case 8B run clears its aggregate quality and token gates with no
per-case raw/auto regression. The external privacy run measures the supported
email/phone/card profile, but its `0.8471` precision and synthetic source are not
deployment evidence. The 1.0 designation stabilizes the integration surface;
it does not turn the bounded benchmark into a universal model-quality or
compliance claim. Official task coverage and deployment-specific privacy
evaluation remain required.
