# Security Policy

## Supported versions

ev-flow follows [Semantic Versioning](https://semver.org). Security fixes are
released for the latest `3.0.x` line.

| Version | Supported |
|---------|-----------|
| 3.0.x   | ✅        |
| < 3.0   | ❌        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use one of these private channels:

1. **GitHub Security Advisories (preferred).** Open a private report via the
   repository's **Security** tab →
   [**Report a vulnerability**](https://github.com/bertravacca/ev-flow/security/advisories/new).
   The advisory is visible only to you and the maintainers.
2. **Email.** bertrand.travacca@gmail.com with `[ev-flow security]` in the
   subject line.

Please include a description of the issue, the affected version(s), and steps to
reproduce. You can expect an initial acknowledgement within **5 business days**,
and we will keep you updated as we work on a fix.

## Disclosure process

We follow coordinated disclosure: we will agree a fix and release timeline with
you, and credit you in the advisory and CHANGELOG unless you prefer to remain
anonymous. Please allow a reasonable window to ship a patch before any public
disclosure.

## Automated scanning

ev-flow's CI continuously scans for known issues:

- **Dependabot** alerts + security updates on the dependency graph.
- **pip-audit** audits the runtime dependency closure against the PyPA advisory
  database on every push, pull request, and weekly.
- **bandit** runs static security analysis over `src/pev_synth/` (gated on
  medium-or-higher severity).

These run in the `security` and `ci` GitHub Actions workflows. As a synthetic
data-generation library with no network services and no handling of user
secrets, ev-flow's runtime attack surface is small; the primary security
concern is the integrity of its dependency chain, which the above tooling
monitors.
