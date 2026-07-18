# Quick Start

ContextIR sits before an LLM or agent. It masks common sensitive values,
extracts inspectable events and constraints, and keeps critical source fragments
when semantic-only compression would be risky.

## Install

Python 3.10 or newer is required.

```bash
python3 -m pip install 'contextir @ git+https://github.com/mussolene/contextir.git@v0.2.1'
```

PyPI publication is intentionally deferred until Trusted Publishing is
configured. Tagged GitHub releases are the current reproducible install path.

## Compile For A Model

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

Short text remains masked raw text. Longer text becomes hybrid or semantic
depending on confidence and critical fragments. Force `mode="hybrid"` for
workflows where source evidence matters.

## Restore An Answer

Restoration is an application decision. Allow only placeholders expected in the
specific output field:

```python
answer = gateway.restore(
    "Send the receipt to PII_EMAIL_1.",
    bundle,
    allowed={"PII_EMAIL_1"},
)
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
- monitor fallback rate, semantic loss, latency, and task success.
