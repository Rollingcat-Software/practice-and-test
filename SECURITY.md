# Security Policy

## Reporting a Vulnerability

If you believe you've found a security vulnerability in practice-and-test — the research & experiments (R&D) repository of the FIVUCSAS biometric authentication platform — please report it privately so we can fix it before disclosing publicly.

**Email:** info@fivucsas.com (primary) or rollingcat.help@gmail.com (alternate) — subject prefix: `[SECURITY] practice-and-test`

Please include:
- A clear description of the issue and its impact.
- Steps to reproduce, ideally with a minimal proof of concept.
- Affected files or commit SHAs if known.
- Whether the issue is already public.

We commit to:
- Acknowledging your report within **3 business days**.
- Providing a full assessment within **10 business days**.
- Coordinating disclosure timing with you once a fix is ready.

## Scope

This repository holds **research prototypes and experiments**, not production services. The production platform is hardened in the `identity-core-api`, `biometric-processor`, `web-app`, and `client-apps` repositories — report production-impacting issues there.

In scope here:
- Secrets, credentials, API keys, or personal/biometric data accidentally committed to this repo's history.
- Malicious or compromised dependencies introduced via experiment code.
- Sample datasets containing real (non-synthetic) personal data.

Out of scope:
- Hardening recommendations for prototype code without a concrete attack path (please open a regular issue).
- Issues that only affect a researcher's local experiment environment.

## Safe-Harbor

Good-faith research that respects user privacy, doesn't degrade service, and follows this disclosure process is welcomed. We will not pursue legal action against researchers who follow this policy.
