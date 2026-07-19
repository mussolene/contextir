# Architecture

## Public Path

`ContextIR` is the public entry point. Its default path has no third-party
runtime dependencies and does not load the research WordNet graph.

For model-facing applications, `ContextPipeline` is the primary public entry
point. `OllamaClient` and `OpenAICompatibleClient` are optional callable
transports implemented with the Python standard library; applications can pass
any callable with the same `prompt -> text` contract instead.

```text
                    +---------------- local only ----------------+
input -> detector -> placeholders + vault -> source segment store
                    +--------------------------------------------+
                              |
                              v
                 task and query analysis
                              |
              +---------------+----------------+
              |               |                |
          exhaustive       retrieval       operational
              |               |                |
             raw      verbatim evidence   events + source
              |               |                |
              +---------------+----------------+
                              |
                        prompt renderer
                              |
                             LLM
```

## Product Algorithm

`ContextPipeline` applies one bounded decision loop:

1. compile the risk-appropriate candidate and keep its vault local;
2. classify exhaustive, retrieval, and operational context shapes during that
   single compilation pass;
3. keep exhaustive tasks raw, retrieve query-relevant evidence for document
   QA, or compile operational events and constraints;
4. count candidate tokens with the caller's target-model tokenizer;
5. pack ranked complete evidence groups for retrieval prompts that exceed the
   target model's budget;
6. enforce the budget after reserving output and chat-template tokens;
7. build a raw baseline only when the candidate does not clear the configured
   savings threshold;
8. invoke the caller-provided model adapter;
9. reject unknown placeholders and newly generated PII;
10. for transform tasks, verify numbers, negation, constraints, events, and issued
   placeholders;
11. on verification failure, retry at most once per richer source mode if that
    prompt still fits the budget;
12. restore only explicitly allowlisted placeholders after acceptance.

The fallback order is `semantic -> hybrid -> raw`; it never loops indefinitely.
Reasoning tasks use safety verification but not semantic equivalence, because a
valid answer normally does not restate the request.

`PipelineResult.public_trace()` exposes decisions and counts without prompts,
answers, source fragments, or vault values.

## Model Context Budget

`OllamaClient` and `OpenAICompatibleClient` publish a prompt budget derived
from their configured context length, output-token reserve, and configurable
chat-template overhead. The pipeline reads it automatically. A custom adapter can instead use
`PipelinePolicy.max_prompt_tokens`, while production deployments should also
provide the exact target-model tokenizer through `token_counter`.

A retrieval prompt that exceeds the budget is repacked from complete ranked
evidence groups. An initial prompt still too large raises
`ContextWindowExceeded` before invocation. If a verification fallback would
exceed the budget, the pipeline returns the rejected result with
`fallback_exceeds_prompt_budget` in its safe trace. ContextIR deliberately does
not truncate selected evidence or pretend an exhaustive task is safe to
summarize.

## Adaptive Modes

### Raw

Used for short inputs. The rendered prompt is the masked source itself, so the
gateway does not add protocol overhead.

### Hybrid

Used for query-aware document QA or when long operational input contains
critical fragments. Retrieval prompts contain normal verbatim text rather than
the internal protocol: task instructions, selected evidence, the original
query, and output-format requirements. Operational prompts use typed events
only for facts not already covered by selected source fragments.

### Semantic

Used only when the caller requests it or confidence is sufficient. The contract
contains no source text, and source references must be resolvable by the caller
if later expansion is needed.

## Privacy Boundary

`compile_private()` returns a `ContextBundle` with three separate values:

- `contract`: safe to serialize and send after application review;
- `vault`: placeholder-to-original mapping, local only;
- `sources`: source-reference map, local only unless individual fragments are
  selected by hybrid mode.

The default regex detector is a fallback. `PresidioPrivacyScrubber` accepts a
configured Presidio analyzer so deployments can provide language and domain
recognizers without coupling the compiler to one NER model.

Repeated occurrences of the same detected surface reuse one placeholder. This
keeps restoration deterministic and prevents repeated private values from
defeating context deduplication.

## Query-Aware Selection

The dependency-free selector uses lexical overlap weighted by document
frequency. It preserves the original query and edge instructions, selects up
to six evidence segments, and restores the owning `Paragraph N` segment when
sentence-level retrieval lands inside a labelled paragraph. If no evidence
clears the confidence threshold, auto mode returns masked raw input.

When the selected set exceeds the model budget, the pipeline starts with the
highest-ranked evidence group and adds lower-ranked groups only while they fit.
Each group is atomic and includes its paragraph owner. Query and output-format
segments are mandatory. PII descriptors for excluded groups are removed from
the public contract; their values remain only in the local vault and their
placeholders are not considered issued to the model.

Tasks that require exhaustive access, such as unique-passage counting, always
remain raw because top-k retrieval cannot preserve their answer space.

For accepted retrieval candidates, the compiler does not construct semantic
events or enumerate numeric entities. Those structures are unused by the plain
verbatim retrieval prompt, so omitting them reduces CPU work without changing
the model-facing text. Exhaustive and operational paths retain full extraction.

## Semantic Representation

Events are deliberately small and inspectable:

```json
{
  "predicate": "send",
  "arguments": ["payment", "again"],
  "polarity": "negative",
  "modality": "prohibition",
  "condition": "if",
  "source_ref": "s3",
  "confidence": 0.82,
  "count": 1
}
```

Identical event signatures are deduplicated and carry a count. This acts as a
session-local semantic dictionary without requiring a hidden token protocol.

## Structured Comparison

`ContextIR.compare()` evaluates retention of event polarity/modality/condition,
constraints, and numeric entities. It does not use injected concept anchors.
If recall falls below the configured threshold, `needs_source` is set and the
caller should use hybrid or raw mode.

## Research Layer

Modules prefixed with `sir_` implement previous lexical, graph, embedding, and
neural experiments. They remain useful for optional topic enrichment and
research reproduction, but they are not required by the default gateway.
