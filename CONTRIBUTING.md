# Contributing to pedotri

Thanks for considering a contribution! This document covers the practical setup, expectations, and conventions for the project.

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Bug reports and feature requests** — open an issue using one of the [templates](https://github.com/ivanfeanor/pedotri/issues/new/choose).
- **Code fixes and features** — see the development setup below.
- **New classifications** — TOML-only contributions are very welcome; see the [Custom classifications guide](https://ivanfeanor.github.io/pedotri/classifications/custom/).
- **Localizations** — add a new locale to any built-in classification's TOML.
- **Documentation, examples, and benchmarks** — README, docs site, or a notebook in `examples/`.

## Development setup

pedotri uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/ivanfeanor/pedotri
cd pedotri
uv sync --all-extras --group dev
```

That gives you a `.venv` with pedotri, every optional extra (matplotlib, plotly, pandas, polars, mcp), and all dev tools (pytest, ruff, mypy, hypothesis, mkdocs).

## Quality gates

Every PR must pass the same four checks CI runs:

```bash
uv run ruff check .         # lint
uv run ruff format --check . # format
uv run mypy                 # type-check (strict mode)
uv run pytest -q            # tests
```

To auto-fix lint and format issues:

```bash
uv run ruff check . --fix
uv run ruff format .
```

To build the docs locally:

```bash
uv run --group docs mkdocs serve
# then open http://127.0.0.1:8000/
```

## Code conventions

- **Python 3.11+** — use modern syntax (`X | None`, `match`, `tomllib`, …).
- **Type hints required** on every public function and method. Mypy runs in strict mode; new code must pass without `# type: ignore`.
- **Docstrings** on every public symbol, in Google style (matches what `mkdocstrings` already renders). Internal helpers can be one-line.
- **Tests required** for any new code path. Property-based tests with Hypothesis are encouraged for invariants.
- **No comparative framing** in commit messages, docstrings, or README ("successor to X", "better than Y", "no X dependency in the core"). State what pedotri *is*, not what it isn't.
- **Avoid mentioning third-party packages by name** when describing benefits — describe the property absolutely.

## Adding a new built-in classification

1. Author a TOML file under `src/pedotri/_data/classifications/<key>.toml` following the schema in `docs/classifications/custom.md`. Use lowercase filename; classification key is uppercase.
2. Cite a primary published reference in `[meta].reference`. Polygon vertices should come from that source, not be invented.
3. Add at least 4–6 reference-point assertions to `tests/test_builtin_classifications.py`.
4. Extend the `_BUILTIN` list and the `_2D_KEYS` / `_TILING_2D_KEYS` lists in `tests/test_properties.py` if it's a 2-D classification — the property test then enforces simplex tiling.
5. Add a row to the table in `README.md` and `docs/classifications/builtin.md`.
6. Note in the provenance section whether polygons are Tier 1 (direct from primary literature) or Tier 2 (interpreted from textbook descriptions).

## Commit conventions

- **One logical change per commit.** Don't bundle unrelated work.
- **Imperative subject line**, ≤72 characters: "Add Kachinsky classification", not "Added Kachinsky classification" or "Adding...".
- **Body explains why** the change is needed when it's not obvious from the subject. Wrap at 72 columns.
- **No `Co-Authored-By` lines** unless the named co-author actually wrote code that landed in the commit.

## Pull requests

Open the PR against `main`. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). CI runs the full quality-gate matrix on Python 3.11 / 3.12 / 3.13 across Ubuntu / macOS / Windows.

For non-trivial changes, please open an issue first to discuss the approach — saves rework if the design isn't a fit.

## Release process

Releases are tag-driven via `.github/workflows/release.yml`:

1. Update `CHANGELOG.md` with the new version's entries (move from `[Unreleased]`).
2. Bump `[project].version` in `pyproject.toml`.
3. Commit and tag: `git tag -s v0.x.y && git push --tags`.
4. The release workflow builds and publishes to PyPI via trusted publishing.

## Licence

By submitting a contribution, you agree that your work is licensed under the MIT licence (see [LICENSE](LICENSE)).
