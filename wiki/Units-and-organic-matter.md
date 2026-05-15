# Units and organic matter

Pedotri's public functions all take percentages by default — sand,
silt, clay, organic matter as numbers in [0, 100]. Real-world lab
reports almost never come in that form. This page lists the conversions
you'll actually need and shows how to plug them in.

## What units do labs use?

| Convention | Range | Where you see it |
|---|---|---|
| **percent** | 0-100 | Most US / international literature; pedotri default |
| **g/kg** | 0-1000 | European labs, especially for organic matter and exchangeable cations |
| **g/g (mass fraction)** | 0-1 | Modelling pipelines, internal MD scripts |
| **% w/w** | 0-100 | Synonym for "percent"; nothing special |
| **% OC** vs **% OM** | 0-100 | PTFs want organic matter; labs report organic carbon |

## The `units=` keyword

Four public functions accept a `units=` keyword that converts at the
boundary:

```python
import pedotri

# Same soil, three conventions, identical answer
pedotri.classify(60, 20, "USDA")                       # default %
pedotri.classify(600, 200, "USDA", units="g/kg")       # g/kg
pedotri.classify(0.60, 0.20, "USDA", units="g/g")      # mass fraction
```

The same keyword works on:

- `pedotri.classify`
- `pedotri.ptf.saxton_rawls`
- `pedotri.ptf.wosten`
- `pedotri.psd.convert`

When you mix sand/clay in one unit and organic matter in another,
convert by hand using the `pedotri.units` helpers (next section). The
`units=` keyword applies uniformly to all fraction arguments.

## OC ↔ OM (Van Bemmelen)

The Van Bemmelen factor 1.724 assumes organic matter is ~58 % carbon
by mass:

```
OM = OC × 1.724
OC = OM / 1.724
```

```python
from pedotri.units import organic_carbon_to_organic_matter

oc_pct = 1.16              # % organic carbon from lab
om_pct = float(organic_carbon_to_organic_matter(oc_pct)[0])
print(om_pct)              # ~2.0
```

### When to override the factor

The 1.724 default works for typical agricultural mineral soils. For
specific contexts, consider:

| Factor | Source | When to use |
|---|---|---|
| **1.724** | Van Bemmelen 1890 (default) | Generic mineral agricultural soils |
| **1.9** | Pribyl 2010 | Less-decomposed organic material, surface horizons |
| **2.0** | Various forest-soil studies | Forest litter, O horizons |

```python
om_forest = float(organic_carbon_to_organic_matter(oc_pct, factor=2.0)[0])
```

## g/kg ↔ %

Trivial but everyone gets it wrong eventually:

```python
from pedotri.units import g_per_kg_to_percent, percent_to_g_per_kg

g_per_kg_to_percent(20)          # → array([2.0])
percent_to_g_per_kg(2)           # → array([20.0])
```

## A real lab report → pedotri pipeline

Lab gives you (in their preferred units):

```
sand:    420 g/kg
silt:    380 g/kg
clay:    200 g/kg
OC:      11.5 g/kg
BD:      1.42 g/cm³
```

Classify with USDA and run Saxton-Rawls:

```python
import pedotri
from pedotri.ptf import saxton_rawls
from pedotri.units import g_per_kg_to_percent, organic_carbon_to_organic_matter

sand_pct = float(g_per_kg_to_percent(420)[0])    # 42.0
clay_pct = float(g_per_kg_to_percent(200)[0])    # 20.0
oc_pct   = float(g_per_kg_to_percent(11.5)[0])   # 1.15
om_pct   = float(organic_carbon_to_organic_matter(oc_pct)[0])  # ~1.98

print(pedotri.classify(sand_pct, clay_pct, "USDA"))
# 'loam'

r = saxton_rawls(sand=sand_pct, clay=clay_pct, organic_matter=om_pct)
print(r.field_capacity, r.wilting_point, r.available_water)
```

Or, equivalently, push the conversion to the `units=` boundary and
convert OC→OM upstream:

```python
om_g_per_kg = 11.5 * 1.724
pedotri.classify(420, 200, "USDA", units="g/kg")
saxton_rawls(sand=420, clay=200, organic_matter=om_g_per_kg, units="g/kg")
```

## Things that look like units but are not

- **"Texture index"** is not a unit; it's the output of a classification.
- **"% &lt; 2 µm"** = clay; **"% &lt; 50 µm"** = silt + clay (USDA). Don't
  hand pedotri a "% &lt; 50 µm" number as `silt` — it's not silt.
- **"Mechanical analysis"** = a sand/silt/clay PSD, but the cutoffs
  depend on the country; see [Soil-conversions](Soil-conversions).
- **EC, pH, CEC** are not pedotri inputs at all; the library is
  texture-and-OM only at the moment.

## Related

- API docs: [`pedotri.units`](https://ivanfeanor.github.io/pedotri/units/)
- [Pedotransfer-functions](Pedotransfer-functions) — PTFs expect OM, not OC
- [Soil-conversions](Soil-conversions) — converts triples between standards, with the same `units=` keyword
