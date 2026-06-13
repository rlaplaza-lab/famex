# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No (renamed from qme; use famex) |

## Reporting a Vulnerability

Please report security issues privately:

1. Open a [GitHub Security Advisory](https://github.com/rlaplaza-lab/famex/security/advisories/new) (preferred), or
2. Email the maintainers via GitHub organization contact if you cannot use advisories.

Do not file public issues for undisclosed vulnerabilities.

We aim to acknowledge reports within 7 days and provide a fix or mitigation plan as soon as practical.

## Security Practices

- Path traversal protections are tested under `tests/security/`.
- Model downloads use sanitized filenames and validated cache paths under `~/.famex/`.
- PyPI releases are built from tagged GitHub releases via CI; API tokens are stored only in GitHub Secrets.
