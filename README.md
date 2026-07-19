# ContextIR

ContextIR is a local-first adaptive context compiler and privacy gateway for
language models. It turns text into a compact, inspectable intermediate
representation while retaining critical source fragments when a lossy summary
would be unsafe.

Status: **stable 1.0 API**. ContextIR follows semantic versioning for the public
Python surface and the `contextir.v2` contract. Semantic compression and PII
detection still require evaluation on each deployment's data.

## Quick Start

```bash
python3 -m pip install contextir
ollama pull qwen3:0.6b
contextir run --model qwen3:0.6b --text "Reply with only READY."
```

```python
from contextir import ContextPipeline, OllamaClient

pipeline = ContextPipeline(invoke=OllamaClient("qwen3:0.6b"))
result = pipeline.run(
    "If payment 42 is complete, do not send it again.",
    source_lang="en",
    target_lang="en",
)
print(result.answer)
```

For document QA, pass the question separately instead of embedding it into the
document:

```python
result = pipeline.run(
    long_document,
    context_kind="retrieval",
    query="What is the northern deployment credential?",
)
```

The document and query are masked together inside the trusted process. The
masked query guides retrieval and is included in the model prompt and token
budget, but is omitted from the public contract and safe trace.

The release wheel avoids a source checkout and Git dependency. A pinned Git
install remains available for environments that prefer source builds.

For a two-minute integration and safe restoration example, see
[Quick start](docs/QUICKSTART.md).

## Why

Long agent histories and tool outputs consume model context with repeated facts,
formatting, and sensitive values. ContextIR provides three explicit modes:

- `raw`: masked source text, without protocol overhead;
- `hybrid`: semantic events plus source fragments containing negation,
  conditions, numbers, quotes, or protected placeholders;
- `semantic`: compact events only, for callers that accept lossy compression.

`auto` first distinguishes exhaustive tasks, document retrieval, and
operational history. Exhaustive tasks stay raw. Document QA uses local lexical
retrieval to keep the query, formatting instructions, and evidence spans
verbatim. Operational history continues to use inspectable events and critical
source fragments.

`ContextPipeline` is the product entry point. It measures candidate prompts
with a configurable target-model tokenizer, rejects compression without useful
savings, verifies output safety, and performs bounded fallback from semantic to
hybrid to raw context. The bundled model clients also reserve output tokens
and configurable chat-template overhead from their context window; ContextIR
packs ranked retrieval evidence to the remaining budget or refuses an unsafe
prompt before contacting the model.

For a query whose highest-ranked evidence segment is itself larger than the
window, enable bounded chunked retrieval explicitly:

```bash
contextir run \
  --model qwen3:0.6b \
  --context-length 4096 \
  --context-kind retrieval \
  --query "What is the release code?" \
  --chunked-retrieval \
  --text "Read the document..."
```

This path chunks only retrieved evidence for reasoning tasks, selects relevant
chunks locally, validates grounded map outputs, and performs at most one reduce
call. It does not approximate exhaustive counting or transform tasks.

## Install

From a checkout:

```bash
python3 -m pip install -e .
```

Optional Presidio integration:

```bash
python3 -m pip install -e '.[privacy]'
```

## Advanced Compiler API

```python
from contextir import ContextIR

gateway = ContextIR()
bundle = gateway.compile_private(
    "If payment 42 is complete, do not send it again. Email person@example.test.",
    source_lang="en",
    target_lang="en",
    mode="hybrid",
)

contract = bundle.contract
model_prompt = gateway.render_bundle(bundle)

# Restore only placeholders explicitly allowed by the application.
answer = gateway.restore("Contact PII_EMAIL_1", bundle, allowed={"PII_EMAIL_1"})
```

`compile()` returns only the public contract. `compile_private()` additionally
returns the local PII vault and source-reference map. Neither is inserted into
the model prompt.

For model-facing applications, prefer `ContextPipeline` over calling the
compiler directly. Use `task="transform"` for translation, rewriting, and
summarization where numbers, negation, constraints, and placeholders must
survive. Normal reasoning answers are not incorrectly required to echo the
request.

## CLI

```bash
contextir run \
  --backend ollama \
  --model qwen3:0.6b \
  --context-length 8192 \
  --max-output-tokens 256 \
  --prompt-overhead-tokens 32 \
  --text "Summarize the current agent state."

contextir compile \
  --text "Если платеж 42 выполнен, не отправляй его повторно." \
  --source-lang ru \
  --target-lang en \
  --mode hybrid \
  --out /tmp/contextir.json

contextir render --contract /tmp/contextir.json
```

The source checkout also exposes `python3 scripts/contextir.py`.

## Contract

ContextIR v2 records:

- source and target language;
- intent and confidence;
- typed entities and protected placeholders;
- events with predicate, arguments, polarity, modality, condition, and source
  reference;
- execution and privacy constraints;
- optional lexical concepts;
- included source fragments and unresolved source references;
- prompt-size and latency statistics.

The packaged JSON Schema is available through:

```python
from contextir.schemas import load_contract_schema
```

## Privacy

The default detector is deliberately small and recognizes common emails,
phones, payment-card-like values, and API-key-like tokens. Presidio can be used
as an optional detector and extended with project-specific recognizers.

Automated PII detection is not a security boundary by itself. Keep the vault
local, avoid logging prompt bodies, use explicit restoration allowlists, and
evaluate recognizers on data representative of the deployment.

## Current Evidence

The checked-in compiler smoke benchmark reports:

- 9 compiler and 12 product-pipeline cases;
- 0 expectation failures;
- 0 pipeline failures;
- 0 PII leaks into public contracts or rendered prompts;
- `0.3627` prompt/source ratio on the one compression-eligible repeated-context
  case.

The first model-level A/B exposed a large quality loss in v0.3.0. Query-aware
routing corrected that failure. On the v0.5.0 nine-case Qwen3 8B run, raw and
auto both scored `0.7272`, while auto reduced model input by 70.0% and mean
latency from `53.8 s` to `2.51 s`. An external 1,500-row Presidio Research
benchmark measured `0.8471` precision and `1.0000` recall for the dependency-free
email/phone/card profile. These are bounded results, not universal quality or
privacy claims; broader official tasks and deployment-specific PII evaluation
remain required.

A constrained-window retrieval smoke run packs the same long input to fit a
256-token Ollama context. Qwen3 0.6B and 8B both recovered `cobalt-seven`; the
backend reported 229 prompt tokens with 16 tokens reserved for output.

An oversized single-segment follow-up uses bounded chunked retrieval with a
256-token context. Qwen3 0.6B and 8B both recovered `cobalt-seven` from the
same selected chunk using one map call at 75% of the available prompt budget.

A constrained 2K-context A/B over ten official LongBench QA/retrieval examples
reduced measured backend input to 39.0% of raw for both tested models. Qwen3 8B
quality increased from `0.1668` to `0.6983` with a paired 95% bootstrap delta
of `[0.2086, 0.8225]`. Qwen3 0.6B increased from `0.1889` to `0.3121`, but its
delta interval `[-0.0154, 0.3392]` does not establish a reliable quality gain.
All ten official cases used one packed direct call; a separate synthetic
oversized-segment diagnostic activated the map path and recovered the answer
with both models.

The post-v1.4 explicit-query rerun separates application document and question
before compilation and enforces per-dataset gates. Both Qwen3 0.6B and 8B pass
the five-case `multifieldqa_en` and five-case `passage_retrieval_en` gates. The
measured backend-input ratio is `0.3439` for QA and `0.5651` for passage
retrieval; the 0.6B QA quality delta is `-0.0203`, inside the configured `-0.03`
floor, while the 8B delta is `+0.0822`.

End-to-end baselines on the same Qwen3 8B run scored `0.6221` for a 32K
same-model neural summary and `0.1910` for Nomic embedding retrieval. ContextIR
scored `0.6983`. Summary remained statistically competitive in quality on this
small subset, but processed `24.95x` as many input tokens as ContextIR and took
`32.0x` its latency. The tested embedding configuration processed `33.43x` as
many tokens and was significantly worse in paired quality. These costs include
preprocessing; they are not answer-prompt-only comparisons.

See the [local model A/B card](docs/benchmarks/LOCAL_MODEL_AB.md) and
[benchmark roadmap](docs/BENCHMARKS.md).

## Research Layer

The repository retains the earlier RU/EN lexical graph, graph embeddings,
synthetic translator, checkpoints, and roundtrip experiments. They are not
loaded by the default package path. Use the `research` extra and source checkout
for those experiments.

The old WordNet keyword roundtrip is preserved as a baseline, not presented as
the production compiler. See [RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md).

## Development

```bash
python3 -m pip install -e '.[dev,research]'
python3 -m unittest discover -s tests -v
python3 scripts/evaluate_contextir.py
python3 -m build
python3 -m twine check dist/*
```

Architecture, scope, contribution, and security guidance:

- [Architecture](docs/ARCHITECTURE.md)
- [Quick start](docs/QUICKSTART.md)
- [Product scope](docs/PRODUCT_SCOPE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

ContextIR code is licensed under Apache-2.0. Research datasets and derived
lexical artifacts retain their upstream terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing them.
