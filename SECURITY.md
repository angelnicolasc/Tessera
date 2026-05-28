# Security Policy

## Supported versions

Tessera is pre-1.0. Security fixes are applied to the `main` branch; tagged pre-1.0
releases are supported on a best-effort basis unless a release note states otherwise.

| Version | Status |
|---|---|
| `main` | Receives security fixes |
| Tagged pre-1.0 releases | Best effort |

## Reporting a vulnerability

**Do not file a public GitHub issue for security reports.**

Email **nick.dicerutti@gmail.com** with:

1. A description of the issue.
2. Steps to reproduce or a minimal proof-of-concept.
3. Your assessment of impact (information disclosure, RCE, DoS, etc).
4. Whether you've shared the report with any other parties.

We aim to:

* Acknowledge receipt within **48 hours**.
* Provide an initial assessment within **5 business days**.
* Issue a fix or mitigation within **30 days** for high-severity issues, longer with notice
  for issues that require deeper architectural work.

We follow coordinated disclosure: please give us reasonable time to ship a fix before
public disclosure. We will credit reporters in the security advisory unless you prefer to
remain anonymous.

## Scope

In scope:

* The Rust core (`crates/tessera-core`, `crates/tessera-index`, `crates/tessera-py`).
* The Python package (`python/tessera`).
* The vLLM plugin shim.

Out of scope:

* Vulnerabilities in upstream dependencies (FlashMLA, FlashInfer, vLLM, usearch). Report
  those to their respective maintainers; we will track and bump our pins.
* CI workflow misconfigurations affecting only this repository's CI.
