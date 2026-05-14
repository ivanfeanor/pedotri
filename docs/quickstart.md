# Quick start

## Installation

```bash
pip install pedotri                 # core only (numpy)
pip install "pedotri[matplotlib]"   # static plots
pip install "pedotri[plotly]"       # interactive plots
pip install "pedotri[pandas]"       # DataFrame accessor
pip install "pedotri[polars]"       # polars accessor
pip install "pedotri[all]"          # everything
```

## Classifying a single sample

```python
import pedotri

pedotri.classify(60, 20, "USDA")
# 'sandy_clay_loam'
```

The first two arguments are the percentages of sand and clay; silt is implicit (100 − sand − clay). The third argument is the classification key.

## Localized class names

Pass `locale="fr"` (or any other registered locale) to get the localized display name instead of the class key:

```python
pedotri.classify(60, 20, "USDA", locale="fr")
# 'limon argilo-sableux'

pedotri.classify(13, 50, "GEPPA", locale="fr")
# 'argile'
```

If a class has no name for the requested locale, the resolver falls back through the locale chain (`fr-CA` → `fr` → `en`) and ultimately to the class key.

## Batch classification

All inputs are vectorized — pass lists, numpy arrays, pandas Series, or any array-like:

```python
pedotri.classify([13, 45, 70], [50, 24, 10], "FAO")
# ['fine', 'medium', 'coarse']
```

Performance is on the order of hundreds of thousands of points per second on a single thread.

## Detailed results

`detailed=True` returns a `ClassifyResult` with the class key, localized name, hierarchical group, parent class, and signed distance to the class boundary:

```python
result = pedotri.classify(13, 50, "USDA", detailed=True)
result.key       # 'clay'
result.name      # 'clay'
result.group     # 'fine'
result.distance  # ~2.12 — positive means inside the polygon
```

Distance is in the (sand %, clay %) plane; the larger the value, the deeper the sample sits inside its class.

## One-dimensional classifications

Some systems (e.g. Kachinsky) classify by a single axis rather than the sand–silt–clay simplex. The API adapts automatically:

```python
pedotri.classify(35, "KACHINSKY")
# 'medium_loam'

pedotri.classify(physical_clay=35, classification="KACHINSKY")
# 'medium_loam'
```

## Listing what is available

```python
pedotri.list_classifications()
# ['EMBRAPA', 'FAO', 'GEPPA', 'HYPRES', 'INTERNATIONAL', 'ISSS',
#  'JAMAGNE', 'KA5', 'KACHINSKY', 'USDA']
```

Or, from the shell:

```bash
pedotri list
pedotri info USDA
```

See [Custom classifications](classifications/custom.md) for how to add your own.
