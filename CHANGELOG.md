# Changelog

All notable changes to pedotri are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-15

### Added

#### Classification engine

- Vectorized point-in-polygon classifier (`pedotri.classify`) for 2-D sand-clay classifications and interval-based 1-D classifications (e.g. Kachinsky's physical-clay axis).
- TOML-based classification authoring with full validation; user-defined classifications via `register_classification()` or `pedotri.classifications` entry-point group.
- Locale fallback chain (`fr-FR` → `fr` → `en` → class key) for class names.
- Vectorized geometry: ~360k points/sec for the 2-D classifier on a single thread.

#### Built-in classifications

Fourteen classifications shipped, with class names in en / fr / de / es / ru / pt (and pl / pt / zh / ru native where applicable):

- USDA (USA, 12 classes)
- FAO (international, 3 classes)
- INTERNATIONAL (Leeper & Uren 1993, 11 classes)
- ISSS (international, 12 classes)
- JAMAGNE (France, 13 classes — Jamagne 1967)
- GEPPA (France, 14 classes — Baize & Jamagne 1995, modern refinement)
- HYPRES (Europe, 5 classes — Wösten et al. 1999)
- KA5 (Germany, full 31-class subdivision per Bodenkundliche Kartieranleitung 5)
- EMBRAPA (Brazil, 5 classes — SiBCS 5ª ed. 2018)
- KACHINSKY (Russia, 9 1-D classes — Kachinsky 1965)
- NORTHCOTE (Australia, 16 classes — Northcote 1979, fine clay subdivisions)
- PTG (Poland, 6 classes — PTG 2008)
- CHINA (national, 6 classes — GB/T 17296-2009)
- AVERY (UK, 12 classes — Avery 1980)

#### Pedotransfer functions

- `pedotri.ptf.saxton_rawls(sand, clay, organic_matter)` — Saxton & Rawls (2006). Wilting point, field capacity, saturation, plant-available water, saturated hydraulic conductivity, bulk density, air-entry tension. Optional `density_factor` keyword applies Eq. 6-8 compaction correction.
- `pedotri.ptf.wosten(sand, silt, clay, organic_matter, bulk_density, topsoil)` — HYPRES continuous PTF for Mualem-van Genuchten parameters.

#### Plotting

- Built-in pure-Python SVG renderer (`pedotri.plot.render_svg`, `pedotri.plot.TextureDiagram`). No plotting dependency required.
- Optional matplotlib backend (`pedotri[matplotlib]` extra).
- Optional plotly backend (`pedotri[plotly]` extra) using native ternary subplots.

#### Particle-size conversions

- `pedotri.psd.convert(sand, silt, clay, source=, target=)` for the USDA / FAO / ISSS / INTERNATIONAL / KA5 sand-silt cutoff conversions under a log-linear interior model.
- `pedotri.psd.interpolate_psd()` for arbitrary cutoff interpolation from a full sieve PSD.

#### Units

- `pedotri.units` module: g/kg ↔ % and g/g ↔ %, plus organic-carbon ↔ organic-matter conversion via the Van Bemmelen factor (1.724, configurable).
- `units=` keyword on `classify()`, `saxton_rawls()`, `wosten()`, and `psd.convert()` for `"%"` (default), `"g/kg"`, or `"g/g"`.

#### DataFrame accessors

- `df.soil.classify(...)` for both pandas and polars, auto-registered on `import pedotri` when the relevant DataFrame library is installed.

#### LLM integration

- `pedotri.ai` module: Anthropic-style tool schemas for 8 operations (`classify_soil`, `classify_soil_1d`, `list_classifications`, `classification_info`, `saxton_rawls`, `wosten`, `convert_particle_size`, `render_diagram`) plus a JSON-in / JSON-out dispatcher with structured error envelopes.
- `pedotri-mcp` console script (`pedotri[mcp]` extra): Model Context Protocol server speaking stdio, suitable for Claude Desktop, Cursor, and any MCP-aware client.

#### CLI

- `pedotri` console script with `list`, `info`, `classify` (single + CSV batch), and `render` subcommands. Stdlib-only (argparse).

#### Tooling and standards

- uv-managed environment, `pyproject.toml` PEP 621 metadata, src layout.
- ruff (lint + format), mypy strict, pytest + hypothesis property-based tests.
- mkdocs-material documentation site with mkdocstrings API reference.
- GitHub Actions CI matrix: Python 3.11 / 3.12 / 3.13 × Ubuntu / macOS / Windows.
- Tag-triggered release workflow with PyPI trusted publishing.

### Documentation

- README with quickstart, classifications table, plotting, accessors, CLI, AI integration, units, and PTFs.
- mkdocs site: quickstart, built-in / custom-classification / i18n guides, PTFs, plotting, CLI, AI tools, units, and full API reference.
- Polygon provenance notes (Tier 1 direct-from-literature vs. Tier 2 textbook-derived).

[Unreleased]: https://github.com/ivanfeanor/pedotri/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.1.0
