# Security Policy

## Supported versions

pedotri is at an early stage of development (pre-1.0). Only the latest minor release receives security fixes; please upgrade before reporting.

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Instead, report privately via [GitHub's "Report a vulnerability" form](https://github.com/ivanfeanor/pedotri/security/advisories/new) or by emailing `ieee802@yandex.ru` with the subject `[pedotri security]`.

Include:

- A concise description of the vulnerability.
- Reproduction steps or a proof-of-concept.
- Affected versions, if known.
- Your assessment of impact and possible mitigations.

You can expect:

- An acknowledgement within 7 days.
- A status update within 30 days.
- A coordinated fix and public disclosure once a patch is available.

## Threat model in brief

pedotri is a pure-computation library: it parses TOML, runs numeric kernels, and renders SVG. Areas of practical concern:

- **Malicious classification TOML.** `pedotri.register_classification()` and the `entry_points` plugin discovery accept third-party TOML / Python. The TOML loader validates schema, but a hostile classification could declare degenerate polygons that produce expensive distance computations. Treat TOML sources like any other third-party data.
- **MCP server inputs.** `pedotri-mcp` exposes the public API over Model Context Protocol. The JSON schemas validate inputs at the protocol layer and pedotri validates again at the function layer; report any input that bypasses both.
- **SVG output.** The pure-SVG renderer escapes all user-supplied strings (locale tags, classification names, point labels) via `html.escape` before embedding them in the document. Report any input that produces an injectable SVG.

Areas explicitly out of scope:

- Numeric accuracy of pedotransfer regressions or polygon coordinates — file as a regular issue or PR.
- Denial-of-service from large inputs to vectorized functions — the library is single-process and intentionally trusts its caller for resource limits.
