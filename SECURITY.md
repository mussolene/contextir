# Security Policy

## Supported Versions

ContextIR follows a stable 1.x API. Security fixes are applied to the latest
minor release and the default branch.

## Reporting

Do not open a public issue containing real prompts, credentials, PII, vault
contents, or exploitable details. Use the repository's private vulnerability
reporting form under **Security > Report a vulnerability**.

## Security Boundaries

- PII detection is probabilistic and incomplete.
- The local vault must never be sent to a model provider.
- Public contracts may still contain sensitive context not recognized as PII.
- Prompt injection is not neutralized by semantic compilation.
- `restore()` should be called with an explicit placeholder allowlist.
- The research checkpoints and lexical data are not security-reviewed.

Applications must apply their own authorization, retention, encryption, and
logging policies around ContextIR.
