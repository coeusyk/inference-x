# Security

InferenceX is designed for **local deployment only** — it binds to `127.0.0.1` by
default and has no authentication. It is not hardened for public internet exposure.

## Scope

Security reports are relevant for:

- Vulnerabilities in InferenceX's own code (`src/inference_x/`, `playground/`, `scripts/`)
- SSRF or injection issues in the API layer
- Unintended network exposure in default configuration

Out of scope:

- Vulnerabilities in transitive dependencies (vLLM, PyTorch, etc.) — report those
  upstream. Known accepted CVEs are documented in release notes.
- Issues that only apply to configurations explicitly unsupported (e.g. public-facing
  deployment without a reverse proxy — this is documented as unsupported)

## Reporting

For potential security issues, open a [GitHub Security Advisory](https://github.com/coeusyk/inference-x/security/advisories/new)
rather than a public issue. Include a description of the issue and steps to reproduce.

Response time is best-effort. This is a one-person project with no SLA.

## Known limitations

- No authentication (DEC-DEFER-01)
- No rate limiting (DEC-DEFER-02)
- Default bind: `127.0.0.1:8000` — do not expose without a reverse proxy and auth layer