# Architecture

## Public Path

`ContextIR` is the public entry point. Its default path has no third-party
runtime dependencies and does not load the research WordNet graph.

```text
                    +---------------- local only ----------------+
input -> detector -> placeholders + vault -> source segment store
                    +--------------------------------------------+
                              |
                              v
                    event/entity compiler
                              |
                 +------------+-------------+
                 |            |             |
                raw         hybrid       semantic
                 |            |             |
                 +------------+-------------+
                              |
                        prompt renderer
                              |
                             LLM
```

## Product Algorithm

`ContextPipeline` applies one bounded decision loop:

1. build a masked raw baseline and keep its vault local;
2. select a semantic or hybrid candidate from input risk and compiler
   confidence;
3. count baseline and candidate tokens with the caller's target-model tokenizer;
4. use the candidate only when it clears the configured savings threshold;
5. invoke the caller-provided model adapter;
6. reject unknown placeholders and newly generated PII;
7. for transform tasks, verify numbers, negation, constraints, events, and issued
   placeholders;
8. on verification failure, retry at most once per richer source mode;
9. restore only explicitly allowlisted placeholders after acceptance.

The fallback order is `semantic -> hybrid -> raw`; it never loops indefinitely.
Reasoning tasks use safety verification but not semantic equivalence, because a
valid answer normally does not restate the request.

`PipelineResult.public_trace()` exposes decisions and counts without prompts,
answers, source fragments, or vault values.

## Adaptive Modes

### Raw

Used for short inputs. The rendered prompt is the masked source itself, so the
gateway does not add protocol overhead.

### Hybrid

Used when long input contains critical fragments or compiler confidence is low.
The model receives typed events plus exact fragments for negation, conditions,
numbers, quotes, code-like text, and protected placeholders.

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
