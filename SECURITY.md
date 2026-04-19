# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

This is a personal portfolio project; there is no formal security team.

If you discover a security issue, please report it privately by emailing
**marwabensalem30@gmail.com** with the subject `[SECURITY] Job_Decision_Engine`.
I will acknowledge within 7 days and aim to ship a fix or mitigation
within 30 days.

## Secrets and data handling

- `OPENAI_API_KEY` and `MONGODB_URI` are stored as HuggingFace Space
  secrets, never committed to git.
- Internal engineering artefacts (`MEMORY/`, `docs/`, `EXECUTION_RULES.md`,
  `profile*.yaml`) are gitignored and never enter the public repo or
  Docker image. A CI privacy-audit job re-checks this on every push.
- The public Space displays a labelled demo profile, not real personal data.
