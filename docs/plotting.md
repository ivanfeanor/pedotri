# Plotting

Three plotting backends share the same API and accept the same arguments. Pick whichever matches your output target.

## SVG (built-in, no dependencies)

```python
from pedotri.plot import render_svg, TextureDiagram

svg = render_svg("USDA", title="USDA Soil Texture Triangle")

# Save to file
TextureDiagram("GEPPA", locale="fr").save("geppa.svg")

# Display inline in Jupyter
TextureDiagram(
    "USDA",
    points=[(40, 25), (15, 60)],
    point_labels=["A", "B"],
    title="Two field samples",
)
```

The SVG renderer is self-contained — no matplotlib, no plotly, just string-built SVG. Files are small and embed cleanly in HTML, Markdown, GitHub READMEs, and Jupyter notebooks (via `_repr_svg_`).

## Matplotlib (`pip install pedotri[matplotlib]`)

```python
from pedotri.plot import render_mpl

fig = render_mpl(
    "USDA",
    points=[(40, 25)],
    point_labels=["sample"],
    title="USDA",
    figsize=(8, 7),
)
fig.savefig("usda.pdf")
```

Returns a `matplotlib.figure.Figure` you can further annotate, compose into a multi-panel figure, or export to any matplotlib-supported format.

## Plotly (`pip install pedotri[plotly]`)

```python
from pedotri.plot import render_plotly

fig = render_plotly(
    "USDA",
    points=[(40, 25)],
    point_labels=["sample"],
)
fig.write_html("usda.html")   # interactive standalone page
```

The Plotly backend uses Plotly's native `Scatterternary`, so you get hover tooltips per class, zoom, pan, and the ability to export to interactive HTML.

## Common arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `classification` | required | Classification key (`"USDA"`, `"GEPPA"`, …) or `Classification` instance. |
| `points` | `None` | Iterable of `(sand, clay)` pairs for 2-D, or 1-tuples / scalars for 1-D. |
| `point_labels` | `None` | Labels rendered alongside each marker. |
| `locale` | classification default | Locale tag for class names in the legend / tooltips. |
| `title` | `None` | Title above the diagram. |
| `show_legend` | `True` | Toggle the legend. (SVG / matplotlib.) |
| `show_grid` | `True` | 10 % gridlines inside the triangle. (SVG / matplotlib.) |

## 1-D classifications

Classifications with a single axis (e.g. `KACHINSKY`) render as a horizontal banded bar rather than a triangle in all three backends — class regions are coloured by group, tick labels mark each interval boundary, and point overlays are drawn as markers along the bar.
