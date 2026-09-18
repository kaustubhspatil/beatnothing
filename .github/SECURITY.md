# Security Policy

## Supported Versions

Security fixes are applied to the latest code on the default branch.
Please reproduce issues against the current `master` branch before reporting.

| Version | Supported |
| --- | --- |
| `master` | :white_check_mark: |
| older branches/tags | :x: |

## Reporting a Vulnerability

Please report vulnerabilities privately by opening a **private vulnerability report** in GitHub Security Advisories for this repository.

1. Go to the repository **Security** tab.
2. Select **Advisories** → **Report a vulnerability**.
3. Include impact, affected files/modules, reproduction steps, and any proof of concept.

If Security Advisories are unavailable to you, open a regular issue and include only a minimal description; maintainers will follow up with a private channel.

## Response Process

- Initial triage target: within 5 business days.
- Valid reports are acknowledged and prioritized by severity and exploitability.
- A fix is prepared on the default branch, tested, and released as soon as practical.
- Reporters are credited unless they prefer to remain anonymous.

## Disclosure Policy

Please do not publicly disclose vulnerabilities until a fix is released and maintainers confirm disclosure timing.

## Security Best Practices for Contributors

- Do not commit secrets, credentials, or private datasets.
- Validate input data paths and file formats in scripts and submissions.
- Keep dependencies reasonably up to date and avoid unpinned ad-hoc tooling in CI.
- Run `pytest -q` before opening a pull request.
