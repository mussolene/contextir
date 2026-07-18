# ContextIR

ContextIR is a local-first adaptive context compiler and privacy gateway for
language models. It turns text into a compact, inspectable intermediate
representation while retaining critical source fragments when a lossy summary
would be unsafe.

Status: **alpha research software**. The public API and `contextir.v2` schema
are usable for experiments, but the project does not yet claim production-grade
semantic preservation or PII detection.

## Quick Start

```bash
python3 -m pip install 'contextir @ git+https://github.com/mussolene/contextir.git@v0.2.0'
contextir compile --text "If payment 42 is complete, do not send it again." \
  --source-lang en --target-lang en --mode hybrid --out context.json
contextir render --contract context.json
```

PyPI publication is planned after Trusted Publishing is configured. The tagged
GitHub install above is the current reproducible path.

For a two-minute integration and safe restoration example, see
[Quick start](docs/QUICKSTART.md).

## Why

Long agent histories and tool outputs consume model context with repeated facts,
formatting, and sensitive values. ContextIR provides three explicit modes:

- `raw`: masked source text, without protocol overhead;
- `hybrid`: semantic events plus source fragments containing negation,
  conditions, numbers, quotes, or protected placeholders;
- `semantic`: compact events only, for callers that accept lossy compression.

`auto` selects one of these modes from input length, semantic confidence, and
critical-source signals.

## Install

From a checkout:

```bash
python3 -m pip install -e .
```

Optional Presidio integration:

```bash
python3 -m pip install -e '.[privacy]'
```

## Python API

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
model_prompt = gateway.render_prompt(contract)

# Restore only placeholders explicitly allowed by the application.
answer = gateway.restore("Contact PII_EMAIL_1", bundle, allowed={"PII_EMAIL_1"})
```

`compile()` returns only the public contract. `compile_private()` additionally
returns the local PII vault and source-reference map. Neither is inserted into
the model prompt.

## CLI

```bash
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

The checked-in ContextIR smoke benchmark currently reports:

- 4 deterministic cases;
- 0 expectation failures;
- 0 PII leaks into public contracts or rendered prompts;
- `0.3067` prompt/source ratio on the one compression-eligible repeated-context
  case.

That result demonstrates the mechanism, not general compression quality. The
case is intentionally repetitive, and there is not yet a model-level A/B result
on a representative corpus. See [BENCHMARKS.md](docs/BENCHMARKS.md).

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
