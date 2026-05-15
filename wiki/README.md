# pedotri wiki source

This directory holds the source markdown for the GitHub Wiki at
<https://github.com/ivanfeanor/pedotri/wiki>. The wiki is its own
git repo, separate from the main pedotri source tree; the files here
are the canonical source that gets pushed there.

## Layout

GitHub Wikis use a flat directory: every `.md` file is a wiki page,
named by its file name (spaces become hyphens). Hyphens in URLs
become spaces in display. `_Sidebar.md` is the navigation sidebar
shown on every page.

```
Home.md
Classifications-and-when-to-use-them.md
Soil-conversions.md
Pedotransfer-functions.md
Units-and-organic-matter.md
GeoTIFF-and-SoilGrids.md
Benchmark-vs-soiltexture.md
Custom-classifications-in-practice.md
_Sidebar.md
```

## Pushing to the wiki

The wiki repo isn't created until the first manual edit on GitHub
(GitHub initializes it lazily). After the wiki is enabled and has at
least one page, sync from this directory:

```bash
# one-time: clone the wiki repo alongside the main one
git clone https://github.com/ivanfeanor/pedotri.wiki.git /tmp/pedotri-wiki

# on every update
cp wiki/*.md /tmp/pedotri-wiki/
cd /tmp/pedotri-wiki
git add -A
git commit -m "Sync wiki from main repo"
git push
```

You can automate that with a GitHub Action if updates become frequent.

## Why a wiki in addition to the docs site?

- **Wiki = field guide.** Long-form context, worked examples, "which
  classification when" decision tradeoffs. Pages here can ship loose
  references and snippets without holding up a release.
- **Docs site = reference manual.** Stable, versioned, API-shaped.
  Generated from docstrings.

Cross-links between the two are encouraged.
