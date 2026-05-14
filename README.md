# pedotri

[![PyPI](https://img.shields.io/pypi/v/pedotri.svg)](https://pypi.org/project/pedotri/)
[![Python](https://img.shields.io/pypi/pyversions/pedotri.svg)](https://pypi.org/project/pedotri/)
[![License](https://img.shields.io/pypi/l/pedotri.svg)](https://github.com/ikolomiets/pedotri/blob/main/LICENSE)

Modern, extensible soil texture classification and pedotransfer functions for Python.

- **Vectorized** point-in-polygon classification via numpy — fast on large arrays.
- **Extensible.** Define your own classifications in TOML and load them at runtime, or ship them as plugins via `entry_points`.
- **Multilingual.** Class names available in English, French, German, Spanish, Russian, and Portuguese, with locale fallback.
- **Comprehensive.** Built-in classifications from around the world: USDA, FAO, GEPPA/Jamagne, KA5, HYPRES, Kachinsky, Embrapa, and more.
- **Beyond classification.** Pedotransfer functions (Saxton-Rawls, Wösten / HYPRES), particle-size standard conversions, class hierarchies, distance-to-boundary, and a pure-SVG triangle renderer.

## Install

```bash
pip install pedotri                 # core only (numpy)
pip install "pedotri[matplotlib]"   # static plots
pip install "pedotri[plotly]"       # interactive plots
pip install "pedotri[pandas]"       # DataFrame accessor
pip install "pedotri[polars]"       # Polars accessor
pip install "pedotri[all]"          # everything
```

## Quick start

```python
import pedotri

# Single point — returns the class key
pedotri.classify(13, 50, "USDA")
# 'clay'

# Batch (vectorized)
pedotri.classify([13, 45, 70], [50, 24, 10], "FAO")
# ['fine', 'medium', 'coarse']

# Localized class names
pedotri.classify(13, 50, "USDA", locale="fr")
# 'argile'
pedotri.classify(13, 50, "GEPPA", locale="fr")
# 'argile lourde'

# 1-D classifications (single axis)
pedotri.classify(35, "KACHINSKY")
# 'medium_loam'

# Rich result with hierarchy and signed distance to boundary
result = pedotri.classify(13, 50, "USDA", detailed=True)
result.key       # 'clay'
result.name      # 'clay'        (localized name; identical to key for USDA en)
result.group     # 'fine'
result.distance  # positive when strictly inside the class polygon
```

## Built-in classifications

| Key             | Region          | Axes              | Classes | Reference                                          |
|-----------------|-----------------|-------------------|---------|----------------------------------------------------|
| `USDA`          | USA (global)    | sand, clay        | 12      | USDA Soil Survey Manual (Handbook 18, 1993)        |
| `FAO`           | International   | sand, clay        | 3       | Verheye & Ameryckx 1984 (FAO grouping)             |
| `INTERNATIONAL` | International   | sand, clay        | 11      | Leeper & Uren 1993                                 |
| `ISSS`          | International   | sand, clay        | 12      | ISSS / Verheye & Ameryckx 1984                     |
| `JAMAGNE`       | France          | sand, clay        | 13      | Jamagne 1967 (original)                            |
| `GEPPA`         | France          | sand, clay        | 14      | Baize & Jamagne 1995 (GEPPA-Aisne refinement)      |
| `HYPRES`        | Europe          | sand, clay        | 5       | Wösten et al. 1999                                 |
| `KA5`           | Germany         | sand, clay        | 31      | Bodenkundliche Kartieranleitung 5 (full subdivision) |
| `EMBRAPA`       | Brazil          | sand, clay        | 5       | Embrapa, SiBCS 5ª ed. 2018                         |
| `KACHINSKY`     | Russia / CIS    | physical_clay     | 9       | Качинский 1965 (1-D classification by <0.01 mm fraction) |

## API for one- and two-axis classifications

Most classifications are defined on the sand-silt-clay simplex (2-D); a few — like Kachinsky — are defined on a single axis (e.g. percent particles <0.01 mm). `classify()` adapts to the chosen classification:

```python
pedotri.classify(70, 20, "USDA")              # 2-D positional
pedotri.classify(10, "KACHINSKY")             # 1-D positional
pedotri.classify(sand=70, clay=20, classification="USDA")           # 2-D kwargs
pedotri.classify(physical_clay=10, classification="KACHINSKY")      # 1-D kwargs
```

## Custom classifications

Define your own classification in TOML and load it:

```python
pedotri.register_classification("path/to/my_classification.toml")
pedotri.classify(40, 30, classification="MY_CLASSIFICATION")
```

See [docs/custom-classifications](https://ikolomiets.github.io/pedotri/custom/) for the schema.

## Texture diagrams

Three backends share the same API and accept the same arguments. Pick whichever fits your output target.

### Built-in SVG (no plotting dependency)

```python
from pedotri.plot import render_svg, TextureDiagram

svg = render_svg("USDA", title="USDA Soil Texture Triangle")
TextureDiagram("GEPPA", locale="fr").save("geppa.svg")

# Inline in Jupyter via _repr_svg_
TextureDiagram("USDA", points=[(40, 25), (15, 60)], point_labels=["A", "B"])
```

### Matplotlib (`pip install pedotri[matplotlib]`)

```python
from pedotri.plot import render_mpl

fig = render_mpl("USDA", points=[(40, 25)], point_labels=["sample"])
fig.savefig("usda.pdf")
```

### Plotly (`pip install pedotri[plotly]`)

Uses Plotly's native ternary subplot — full interactivity, hover tooltips, export to interactive HTML.

```python
from pedotri.plot import render_plotly

fig = render_plotly("USDA", points=[(40, 25)], point_labels=["sample"])
fig.write_html("usda.html")
```

1-D classifications (e.g. Kachinsky) render as a horizontal banded axis in all three backends.

## DataFrame accessors

When `pandas` or `polars` is installed, `import pedotri` registers a `.soil` accessor on DataFrames:

```python
import pandas as pd
import pedotri  # registers df.soil

df = pd.DataFrame({"sand": [60, 20, 70], "clay": [10, 50, 5]})
df["texture"] = df.soil.classify("USDA")

# Localized names
df["fr"] = df.soil.classify("GEPPA", locale="fr")

# Different column names
df2 = pd.DataFrame({"S": [60], "C": [10]})
df2.soil.classify("USDA", columns={"sand": "S", "clay": "C"})
```

Polars works identically:

```python
import polars as pl
df = pl.DataFrame({"sand": [60, 20], "clay": [10, 50]})
df.with_columns(texture=df.soil.classify("USDA"))
```

## Command-line interface

```bash
pedotri list                                              # list registered classifications
pedotri info USDA                                          # show details for one
pedotri classify --sand 60 --clay 20 -c USDA               # single sample
pedotri classify -c KACHINSKY --physical-clay 35           # 1-D
pedotri classify --csv samples.csv --output out.csv -c USDA # batch
pedotri render -c USDA --title "USDA" -o usda.svg          # write an SVG
```

## Pedotransfer functions

Two PTFs ship in `pedotri.ptf`:

### Saxton-Rawls (2006)

Water retention + saturated conductivity from sand, clay, and organic matter.

```python
from pedotri.ptf import saxton_rawls

r = saxton_rawls(sand=40, clay=20, organic_matter=2.0)
r.wilting_point             # theta_1500 (m³/m³)
r.field_capacity            # theta_33   (m³/m³)
r.saturation                # theta_s    (m³/m³)
r.available_water           # FC - WP
r.saturated_conductivity    # K_s (mm/h)
r.bulk_density              # g/cm³
r.air_entry_tension         # psi_e (kPa)
```

### Wösten (1999) — HYPRES Mualem-van Genuchten

```python
from pedotri.ptf import wosten

w = wosten(sand=40, silt=40, clay=20, organic_matter=2.0, bulk_density=1.4, topsoil=True)
w.theta_s                   # saturated water content
w.alpha                     # van Genuchten alpha (1/cm)
w.n                         # van Genuchten n
w.saturated_conductivity    # K_s (cm/day)
w.l                         # Mualem pore-connectivity parameter
```

Both functions accept either scalar inputs (returning one result object) or array-like inputs (returning a list).

## License

MIT
