# Benchmark Card: Legacy Lexical Roundtrip

## Identity

- Component: research WordNet concept graph and neural experiments
- Languages: Russian and English
- Primary report: `reports/roundtrip_eval.json`
- Historical documentation: [Legacy roundtrip](../LEGACY_ROUNDTRIP.md)

## Checked-In Result

The historical lexical roundtrip reports concept-level F1 of `0.5342`; the
neural smoke report records F1 of `0.4886`.

## Interpretation

These scores measure recovery of the experiment's lexical concepts. They do not
measure `contextir.v2` prompt compression, natural-language translation,
downstream model accuracy, or privacy. The research layer remains available for
architecture experiments but is not loaded by the default package.

## Distribution Constraint

The source datasets and derived checkpoints are excluded from the Python package
and retain upstream terms. Review `THIRD_PARTY_NOTICES.md` before redistributing
them independently.

