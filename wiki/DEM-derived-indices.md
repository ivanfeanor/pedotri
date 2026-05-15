# DEM-derived topographic indices

Soil properties don't just depend on what's in the soil — they also
depend on where it sits in the landscape. A property at the bottom of
a long shallow hillslope tends to be wetter than the same property on
a ridge; a profile-curved foot-slope accumulates organic matter that a
profile-curved shoulder loses. `pedotri.dem` gives you the four
classical primitives that encode this for downstream pedotransfer
functions and ML feature engineering: **slope**, **aspect**,
**curvature** (profile and plan), and **topographic wetness index**.

## When you need this

Three common use cases:

1. **Feature engineering for digital soil mapping** — feeding a
   regression model that predicts soil properties from elevation
   covariates (the SoilGrids approach).
2. **Hydrological proxies** — TWI as a surrogate for soil moisture
   redistribution, slope as a runoff predictor.
3. **Sanity checks** — visualising a DEM to confirm it's oriented the
   way you expect before trusting downstream computations.

## Quick example

```python
import numpy as np
from pedotri.dem import slope, aspect, topographic_wetness_index

# A simple linear ramp: descends 0.5 m per pixel eastward
y, x = np.mgrid[0:50, 0:50]
dem = 0.5 * x  # 50 × 50 grid, elevation in metres

# Slope as a percent (rise / run × 100), pixel size 10 m
beta_pct = slope(dem, pixel_size=10.0, units="percent")
beta_pct.mean()   # ≈ 5.0 — 0.5 m rise over 10 m run

# Aspect — compass bearing of the downhill direction
alpha = aspect(dem, pixel_size=10.0)
alpha[25, 25]     # ≈ 90° — descends east, faces east

# Topographic wetness index (slope-only surrogate; see below)
twi = topographic_wetness_index(dem, pixel_size=10.0)
```

All four functions take a 2-D DEM, a pixel size (single float for
square pixels or a `(dx, dy)` tuple), and return an array of matching
shape. Boundary pixels are filled by reflecting the DEM, so the output
isn't shrunk relative to the input.

## Picking the right function

| You want to know… | Call |
|---|---|
| How steep is each pixel? | `slope` |
| Which compass direction does the slope face? | `aspect` |
| Does water accelerate or decelerate downhill here? | `profile_curvature` |
| Does water converge (valley) or diverge (ridge) here? | `plan_curvature` |
| How wet would this pixel be relative to its neighbours? | `topographic_wetness_index` |

## Slope and aspect — Horn (1981)

Horn's 3×3 Sobel-weighted finite-difference scheme is what GDAL's
`gdaldem` and the SAGA-GIS terrain suite both ship. We match it
exactly. For pixel `(i, j)`, with its eight neighbours labelled like
the cardinal points:

```
NW  N  NE
W   z  E
SW  S  SE
```

the partial derivatives are

```
∂z/∂x = ((NE + 2·E + SE) − (NW + 2·W + SW)) / (8 · Δx)
∂z/∂y = ((SW + 2·S + SE) − (NW + 2·N + NE)) / (8 · Δy)
```

(`∂z/∂y` is south-positive in pedotri's internal convention, matching
the raster array convention where the row index grows southward.)

**Slope angle** is the arctangent of the gradient magnitude:

```
β = arctan( √( (∂z/∂x)² + (∂z/∂y)² ) )
```

reported in `degrees`, `radians`, or `percent` (rise / run × 100 — the
convention used in USDA soil-survey reports and most agronomic
decision tables).

**Aspect** (compass bearing of the downhill direction, clockwise from
north): pedotri uses

```
α = atan2(−∂z/∂x, ∂z/∂y)
```

with the result wrapped into `[0°, 360°)`. North = 0°, East = 90°,
South = 180°, West = 270°. Pixels with negligible slope (gradient
magnitude < 1e-12) return `NaN` — aspect is undefined on a flat patch.

### Why Sobel weights, not central differences?

A central difference uses only the four directly-adjacent neighbours
(N, S, E, W) and ignores the diagonals. It's noisier on real DEMs:
SRTM / ALOS / Copernicus rasters all have salt-and-pepper artefacts
at the metre scale, and central differences amplify them. Horn's
weights average over the whole 3×3 neighbourhood with double weight
on the cardinal neighbours, which damps that noise out. The 1981
paper has the full argument; the practical effect is that pedotri's
slope numbers agree with GDAL's to floating-point error.

## Curvature — Zevenbergen & Thorne (1987)

Z&T fit a 9-coefficient polynomial through the 3×3 neighbourhood:

```
z ≈ A·x²·y² + B·x²·y + C·x·y² + D·x² + E·y² + F·x·y + G·x + H·y + I
```

The six second-derivative coefficients are what we need. With `L` =
pixel side:

```
D = ((z_W + z_E)/2 − z) / L²               ∂²z/∂x²
E = ((z_N + z_S)/2 − z) / L²               ∂²z/∂y²
F = (−z_NW + z_NE + z_SW − z_SE) / (4 L²)  ∂²z/∂x∂y
G = (z_E − z_W) / (2 L)                    ∂z/∂x
H = (z_N − z_S) / (2 L)                    ∂z/∂y
```

**Profile curvature** — bending in the steepest-descent direction:

```
k_profile = −2 · (D·G² + E·H² + F·G·H) / (G² + H²)
```

Sign convention:

- Negative → water *decelerates* downhill (foot of a slope, water
  accumulates, soil tends to be wetter and richer in organic matter).
- Positive → water *accelerates* downhill (shoulder, drier soil).

Pedotransfer functions fitted to grade-distinguishing data sets often
use profile curvature as a soil-moisture predictor.

**Plan curvature** — bending in the across-contour direction:

```
k_plan = 2 · (D·H² + E·G² − F·G·H) / (G² + H²)
```

- Negative → diverging flow lines (ridge-like, water spreads out).
- Positive → converging flow lines (valley-like, water concentrates).

Both have units of `1/length`, so for a DEM in metres they come out
in `1/m`. Pixels with truly zero denominator (flat patches) return
zero curvature.

## Topographic wetness index — Beven & Kirkby (1979)

```
TWI = ln(a / tan β)
```

where `a` is the upstream contributing area per unit contour length
(metres) and `β` is the local slope angle (radians). TWI is a
non-dimensional proxy for soil moisture redistribution: a pixel at
the foot of a long, shallow hillslope ends up with a much higher TWI
than one on a steep ridge.

Empirically TWI tracks measured soil-moisture patterns surprisingly
well at the field scale, which is why it's the default
topography-as-covariate feature in most digital-soil-mapping work
(Sørensen et al. 2006 review the literature).

### Why pedotri doesn't ship a flow-accumulation engine

Computing `a` correctly is a hydrology problem in its own right: you
need a flow-routing algorithm (D8, D∞, MFD, …) plus depression
filling, and the choice has significant downstream impact. There's a
perfectly good ecosystem — `richdem`, `pysheds`, GRASS GIS's
`r.watershed`, SAGA's `ta_channels` — that does this much better than
pedotri ever could.

So `topographic_wetness_index` takes an optional `flow_accumulation=`
ndarray (number of upstream cells per pixel). When you supply it,
TWI is computed for real. When you don't, pedotri falls back to a
**one-cell-per-pixel surrogate** that turns TWI into a function of
slope alone. The surrogate is useful as:

- A fast ML feature when slope-only signal is enough.
- A sanity check on the DEM before plugging in a hydrology pipeline.

It is **not** a hydrology product. The docstring says so out loud.

### The flat-pixel floor

Strictly zero slope makes `1/tan β` diverge. Following SAGA-GIS
convention, pedotri floors `tan β` at `min_slope = 1e-4` (≈ 0.006°),
which is below the noise level of any practical DEM but above the
representable-as-positive-float zero. Override with `min_slope=` if
your DEM has hand-tuned flat patches you want preserved.

## All five functions in one mental model

For a pixel `z[i, j]`:

1. **Slope and aspect** look at the **first** derivative of elevation
   — the gradient vector. Magnitude → slope; direction → aspect.
2. **Curvature** looks at the **second** derivative — how the
   gradient changes. Two natural directions: along the gradient
   (profile) and orthogonal to it (plan).
3. **TWI** combines slope with upstream area — a kinematic statement
   about how water that's reached this pixel will sit there.

## References

- Beven, K. J. & Kirkby, M. J. (1979). A physically based, variable
  contributing area model of basin hydrology. *Hydrological Sciences
  Bulletin* 24(1): 43–69.
- Horn, B. K. P. (1981). Hill shading and the reflectance map.
  *Proceedings of the IEEE* 69(1): 14–47.
- Sørensen, R., Zinko, U. & Seibert, J. (2006). On the calculation
  of the topographic wetness index: evaluation of different methods
  based on field observations. *Hydrology and Earth System Sciences*
  10(1): 101–112.
- Zevenbergen, L. W. & Thorne, C. R. (1987). Quantitative analysis of
  land surface topography. *Earth Surface Processes and Landforms*
  12(1): 47–56.

## See also

- [Reprojection & target grids](Reprojection-and-target-grids) — when
  your DEM is on a different grid than your soil rasters, line them up
  first.
- [Pedotransfer functions](Pedotransfer-functions) — most PTFs
  consume slope or TWI as a covariate; this module produces them.
