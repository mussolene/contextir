# Contributing

ContextIR is alpha research software. Small, measurable changes are preferred
over broad architecture rewrites.

## Setup

```bash
python3 -m pip install -e '.[dev,research]'
python3 -m unittest discover -s tests -v
python3 scripts/evaluate_contextir.py
```

## Pull Requests

- explain the user-visible behavior and failure modes;
- add tests for contract or privacy changes;
- report prompt ratio and semantic expectations for compiler changes;
- keep original PII and prompt bodies out of fixtures and logs;
- preserve source and dataset license notices;
- update `CHANGELOG.md` for public API or schema changes.

Do not report model-quality improvements from synthetic or smoke data alone.

