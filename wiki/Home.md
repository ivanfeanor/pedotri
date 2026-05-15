# pedotri wiki

This wiki holds the **long-form context** that doesn't fit in the [API docs](https://ivanfeanor.github.io/pedotri/) or the README: when to use which classification, what the pedotransfer functions actually compute, how to wire the library into a GeoTIFF pipeline, and how it stacks up against alternatives.

If you're new, start with the [README](https://github.com/ivanfeanor/pedotri/blob/main/README.md) — it's enough to classify your first sample. Then come back here when one of the following questions shows up:

| If you want to… | Read |
|---|---|
| See the kind of maps pedotri actually produces | [Examples-gallery](Examples-gallery) |
| Pick the right classification for your region | [Classifications & when to use them](Classifications-and-when-to-use-them) |
| Understand why classifications conflict and how pedotri tries to help | [Classification-challenges](Classification-challenges) |
| Classify with SoilGrids-style Q05/Q95 uncertainty and get class probabilities | [Uncertainty-aware classification](Uncertainty-aware-classification) |
| Aggregate a region (village / commune / farm) into a posterior — incl. SOC stock | [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock) |
| Classify the same sample under USDA + FAO + regional schemes at once | [Classifications & when to use them](Classifications-and-when-to-use-them#classifying-once-against-every-scheme) |
| Convert sand/silt/clay between USDA, ISSS, KA5 | [Soil-conversions](Soil-conversions) |
| Predict water retention, K&#x209B;, bulk density | [Pedotransfer-functions](Pedotransfer-functions) |
| Plug lab reports (`g/kg`, organic carbon) into pedotri | [Units-and-organic-matter](Units-and-organic-matter) |
| Classify a whole GeoTIFF (e.g. SoilGrids) | [GeoTIFF-and-SoilGrids](GeoTIFF-and-SoilGrids) |
| Build a soil-texture map (regional grids or sparse field samples) | [Map-products-and-field-sampling](Map-products-and-field-sampling) |
| See how pedotri compares to `soiltexture` | [Benchmark-vs-soiltexture](Benchmark-vs-soiltexture) |
| Add a custom regional classification | [Custom-classifications-in-practice](Custom-classifications-in-practice) |

## Why a wiki?

The mkdocs site is the reference manual: function signatures, argument
tables, the canonical class list. This wiki is the **field guide** —
worked examples, tradeoffs, and the why-do-it-this-way explanations
that age faster than the API does. Pages here can ship snippets,
quote the literature loosely, and link out to data sources without
adding maintenance load to the main docs.

## Versioning

Wiki pages target the latest minor release on PyPI. Where an example
depends on a feature that landed in a specific version (e.g.
`pedotri.raster` arrived in 0.2.0), the page calls that out explicitly.
