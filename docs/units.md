# Units

## Default

Every fraction input in pedotri is **percent** (0-100 range) by default. This matches what classification references and PTF regressions publish.

```python
pedotri.classify(60, 20, "USDA")
# sand=60%, clay=20%, silt=20% (implicit)
```

## Other units via `units=`

Lab reports often use other conventions. `classify()`, `saxton_rawls()`, `wosten()`, and `psd.convert()` accept a `units=` keyword that applies the conversion at the function boundary:

```python
# Same sample, three input conventions — identical result
pedotri.classify(60, 20, "USDA")                       # %  (default)
pedotri.classify(600, 200, "USDA", units="g/kg")       # g/kg
pedotri.classify(0.6, 0.2, "USDA", units="g/g")        # mass fraction

# Saxton-Rawls with OM in g/kg
saxton_rawls(sand=40, clay=20, organic_matter=20, units="g/kg")
```

Valid identifiers:

| `units=` | Multiplier to % | Typical use |
|----------|----------------:|---|
| `"%"`    | 1               | Default; what classification references publish |
| `"g/kg"` | × 0.1           | European soil chemistry lab reports |
| `"g/g"`  | × 100           | Mass fraction (0-1) in modelling pipelines |

`bulk_density` in `wosten()` is always g/cm³ regardless of the `units` keyword — it's a density, not a mass fraction.

## Organic carbon vs. organic matter

This is the single most common pitfall when feeding lab data to a PTF. Saxton-Rawls and Wösten both expect organic **matter** (OM), but most lab reports give organic **carbon** (OC). They differ by the Van Bemmelen factor (~1.724).

```python
from pedotri.units import organic_carbon_to_organic_matter
from pedotri.ptf import saxton_rawls

oc_pct = 1.16  # from lab report (% organic carbon)
om_pct = float(organic_carbon_to_organic_matter(oc_pct)[0])  # → ~2.0 %

r = saxton_rawls(sand=40, clay=20, organic_matter=om_pct)
```

The factor is configurable:

```python
# Pribyl 2010 — less-humified material
organic_carbon_to_organic_matter(oc_pct, factor=1.9)

# Some forest-soil references use 2.0
organic_carbon_to_organic_matter(oc_pct, factor=2.0)
```

## Helper functions

`pedotri.units` exposes the conversions directly when you need to convert at a different point in your pipeline:

```python
from pedotri.units import (
    g_per_kg_to_percent,
    percent_to_g_per_kg,
    g_per_g_to_percent,
    percent_to_g_per_g,
    organic_carbon_to_organic_matter,
    organic_matter_to_organic_carbon,
    to_percent,
    from_percent,
)

# Generic
to_percent(250, "g/kg")     # → array([25.])
from_percent(25, "g/g")     # → array([0.25])

# Specific
g_per_kg_to_percent([100, 200, 300])  # → array([10., 20., 30.])
```

All helpers accept scalars or array-likes and return a 1-D `numpy.float64` array.

## Validation

Inputs are validated **after** unit conversion — i.e. pedotri checks that the converted percent value is in `[0, 100]`. Out-of-range values raise `InvalidInputError`:

```python
pedotri.classify(60, 1500, "USDA", units="g/kg")
# InvalidInputError: 'clay' must be a percentage in [0, 100].
# (1500 g/kg = 150% — impossible.)
```

Unknown unit identifiers also raise:

```python
pedotri.classify(60, 20, "USDA", units="ppm")
# InvalidInputError: Unknown units 'ppm'. Valid: '%', 'g/kg', 'g/g'.
```
