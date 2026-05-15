<!--
Thanks for the contribution. Please fill in the sections that apply.

For substantial changes please open an issue first to discuss the design — it saves rework if the approach isn't a fit. See CONTRIBUTING.md for details.
-->

## Summary

<!-- 1-3 sentences. What changes and why. Link the issue if there is one. -->

Closes #

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] New classification (TOML + tests + docs table update)
- [ ] Documentation / examples / typo
- [ ] Internal refactor (no behavioural change)

## Checklist

- [ ] `uv run pytest -q` passes locally
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run mypy` passes (strict mode)
- [ ] `uv run --group docs mkdocs build --strict` passes if docs were touched
- [ ] Tests added or updated to cover the change
- [ ] Public-API docstrings added/updated for new or changed symbols
- [ ] README / docs site updated if user-visible behaviour changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Notes for the reviewer

<!-- Edge cases, performance implications, follow-ups, anything that helps review. -->
