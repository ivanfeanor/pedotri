# Pedotransfer functions

The `pedotri.ptf` subpackage estimates soil hydraulic properties from texture (and optional organic matter, bulk density, topsoil indicator).

## Saxton-Rawls (2006)

The most widely used PTF for agronomy. Takes sand, clay, and organic matter; returns wilting point, field capacity, saturation, available water, saturated hydraulic conductivity, bulk density, and air-entry tension.

```python
from pedotri.ptf import saxton_rawls

r = saxton_rawls(sand=40, clay=20, organic_matter=2.0)
r.wilting_point             # theta at 1500 kPa (m³/m³)
r.field_capacity            # theta at  33 kPa (m³/m³)
r.saturation                # theta at saturation (m³/m³)
r.available_water           # field_capacity - wilting_point
r.saturated_conductivity    # K_s (mm/h)
r.bulk_density              # g/cm³
r.air_entry_tension         # kPa

# Compaction adjustment per Saxton-Rawls 2006 Eq. 6-7:
r_compacted = saxton_rawls(40, 20, 2.0, density_factor=1.10)
r_compacted.bulk_density    # ~10 % denser than `r.bulk_density`
r_compacted.saturation      # reduced porosity under compaction
```

**Reference:** Saxton, K.E. & Rawls, W.J. (2006). *Soil water characteristic estimates by texture and organic matter for hydrologic solutions.* Soil Science Society of America Journal 70(5): 1569–1578.

## Wösten 1999 / HYPRES

A continuous PTF for Mualem-van Genuchten parameters, used as the basis for European HYPRES soil-hydraulic database.

```python
from pedotri.ptf import wosten

w = wosten(
    sand=40, silt=40, clay=20,
    organic_matter=2.0,
    bulk_density=1.4,
    topsoil=True,
)
w.theta_r                  # residual water content (fixed at 0.01 in HYPRES)
w.theta_s                  # saturated water content
w.alpha                    # van Genuchten alpha (1/cm)
w.n                        # van Genuchten n
w.saturated_conductivity   # K_s (cm/day)
w.l                        # Mualem pore-connectivity parameter
```

**Reference:** Wösten, J.H.M., Lilly, A., Nemes, A., Le Bas, C. (1999). *Development and use of a database of hydraulic properties of European soils.* Geoderma 90(3-4): 169–185.

## Vectorized inputs

Both functions accept either scalar or array-like inputs and adapt the return type accordingly:

```python
results = saxton_rawls(
    sand=[60, 20, 70],
    clay=[10, 50, 5],
    organic_matter=2.0,         # broadcast across the array
)
# returns list[SaxtonRawlsResult] of length 3
```

## Choosing between Saxton-Rawls and Wösten

- **Saxton-Rawls** is the easier API (sand + clay + OM only) and gives point estimates of the agronomic water-balance quantities directly.
- **Wösten 1999** returns the parameters of the *continuous* water-retention curve, which you can then evaluate at any matric potential. Use it when you need the full curve or are coupling pedotri to a hydrological model that expects van Genuchten parameters.

## Validation

The implementations follow the published regression equations literally. Outputs are in the expected ranges across the full sand-silt-clay simplex (water contents in [0, 1], K_s strictly positive). For specific lab samples, expect deviations of a few hundredths of a unit from published lookup tables — both PTFs were fit to large but finite training sets and remain approximate by construction.
