# Quick Start

ContextIR sits before an LLM or agent. It masks common sensitive values,
extracts inspectable events and constraints, and keeps critical source fragments
when semantic-only compression would be risky.

## Install

Python 3.10 or newer is required.

```bash
python3 -m pip install contextir
```

The PyPI wheel needs neither a source checkout nor Git. For a local first run
with Ollama:

```bash
ollama pull qwen3:0.6b
contextir run --model qwen3:0.6b --text "Reply with only READY."
```

## Run A Model Safely

`ContextPipeline` is the default product API. The model adapter is any callable
that accepts a prompt and returns text:

```python
from contextir import ContextPipeline, OllamaClient

pipeline = ContextPipeline(invoke=OllamaClient("qwen3:0.6b"))
result = pipeline.run(
    "If payment 42 is complete, do not send it again. Email person@example.test.",
    source_lang="en",
    target_lang="en",
    risk="standard",
    task="reasoning",
)

if not result.accepted:
    raise RuntimeError(result.public_trace())

answer = result.answer
```

For LM Studio, vLLM, LocalAI, or a hosted OpenAI-compatible endpoint:

```python
import os
from contextir import ContextPipeline, OpenAICompatibleClient

client = OpenAICompatibleClient(
    "model-name",
    base_url="http://127.0.0.1:1234/v1",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    context_length=8192,
    max_output_tokens=512,
    prompt_overhead_tokens=32,
)
result = ContextPipeline(invoke=client).run("Summarize the agent state.")
```

Existing integrations may continue passing `invoke` directly to `run()`.

Pass the real tokenizer for the target model. The built-in counter is only a
dependency-free estimate. ContextIR uses compressed context only when measured
savings exceed policy; otherwise it sends masked raw text.

The bundled clients expose a prompt budget equal to `context_length -
max_output_tokens - prompt_overhead_tokens`. The overhead reserve defaults to
32 tokens and should be calibrated for the backend's chat template.
`ContextPipeline` enforces that budget before every model request, including
richer fallback attempts. Custom adapters can set the limit explicitly:

```python
from contextir import ContextPipeline, ContextWindowExceeded, PipelinePolicy

pipeline = ContextPipeline(
    policy=PipelinePolicy(max_prompt_tokens=3584),
    token_counter=target_model_tokenizer,
    invoke=custom_model,
)

try:
    result = pipeline.run(long_context)
except ContextWindowExceeded as exc:
    route_to_chunked_workflow(exc.prompt_tokens, exc.prompt_budget)
```

ContextIR fails instead of silently truncating evidence or converting an
exhaustive task into a lossy semantic summary. For retrieval tasks it first
packs the query, output instructions, and as many ranked complete evidence
groups as fit. A labelled paragraph owner stays attached to sentence-level
evidence. If even the best complete group cannot fit, `ContextWindowExceeded`
is raised unless the explicit chunked-retrieval path is enabled. Exhaustive
inputs still require a deployment-owned domain aggregator.

For reasoning over a single oversized top-ranked retrieval segment, ContextIR
provides an explicit bounded path:

```python
result = pipeline.run(
    long_document_question,
    source_lang="en",
    target_lang="en",
    chunked_retrieval=True,
)
```

The pipeline splits that evidence with overlap, locally keeps chunks sharing
content terms with the question, invokes bounded map calls, checks that every
answer-specific term occurs in its evidence, and reduces distinct candidates
only when needed. `PipelinePolicy` controls `max_chunk_calls`,
`chunk_overlap_words`, and `chunk_prompt_ratio`. The default ratio uses only
75% of the otherwise available prompt budget because very small models can
degrade near the edge of their nominal context window.

Chunked retrieval is accepted only for `task="reasoning"`. It remains disabled
unless `chunked_retrieval=True`; exhaustive counting, translation, rewriting,
and summarization continue to fail safely when their complete required input
does not fit. Version 1.3 also requires matching source and target languages so
its lexical grounding check does not reject a valid translated answer.

Use `task="transform"` for translation, rewriting, extraction, or summarization.
That enables retention checks and bounded fallback through semantic, hybrid,
and raw representations. A normal reasoning answer is not required to repeat
the input contract.

## Compile Without Invocation

```python
from contextir import ContextIR

gateway = ContextIR()
bundle = gateway.compile_private(
    "If payment 42 is complete, do not send it again. Email person@example.test.",
    source_lang="en",
    target_lang="en",
    mode="auto",
)

prompt = gateway.render_prompt(bundle.contract)
# Send only prompt to the model. Keep bundle.vault inside your trusted process.
```

Short text remains masked raw text. Exhaustive counting stays raw. Longer
document QA uses query-aware hybrid prompts only when matching evidence and
measured token savings are available. Operational history continues to use
semantic events and critical fragments. Force `mode="hybrid"` only when the
application accepts lossy source selection.

## Restore An Answer

Restoration is an application decision. Allow only placeholders expected in the
specific output field:

```python
result = pipeline.run(
    "Send the receipt to person@example.test.",
    source_lang="en",
    target_lang="en",
    allowed_restore={"PII_EMAIL_1"},
)
answer = result.answer
```

Do not send `bundle.vault` to the model, write it to normal logs, or restore all
placeholders into an untrusted output channel.

## Validate A Returned Contract

When another component produces or modifies ContextIR, validate it before use:

```python
from jsonschema import Draft202012Validator
from contextir.schemas import load_contract_schema

Draft202012Validator(load_contract_schema()).validate(bundle.contract)
```

## Optional Presidio Detection

```bash
python3 -m pip install 'contextir[privacy]'
contextir compile --privacy presidio --text "Email person@example.test"
```

Presidio improves detector coverage but still needs recognizer and language
configuration for the deployment domain. Neither detector is a complete privacy
boundary.

## Production Checklist

- benchmark `raw`, conventional summary, `auto`, and `hybrid` on real tasks;
- measure target-model tokens rather than character ratios;
- add domain-specific PII recognizers and leakage tests;
- keep vault storage ephemeral and access controlled;
- retain raw-source fallback for low-confidence or high-impact operations;
- configure each model's real context length, output reserve, and tokenizer;
- monitor fallback rate, semantic loss, latency, and task success.
