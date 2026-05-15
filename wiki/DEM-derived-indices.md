# DEM-derived topographic indices

`pedotri.dem` ships four primitives that show up in every pedology /
agronomy workflow that wants to relate a soil property to local
terrain:

- `slope(dem, pixel_size, units="degrees")`
- `aspect(dem, pixel_size)`
- `profile_curvature(dem, pixel_size)`
- `plan_curvature(dem, pixel_size)`
- `topographic_wetness_index(dem, pixel_size, flow_accumulation=...)`

All five are pure numpy, broadcast over a 2-D DEM, and pad with
reflection at the edges so the output retains the input shape.

## Slope and aspect — Horn (1981)

Horn's classic 3×3 Sobel-weighted finite-difference scheme is the
algorithm GDAL's `gdaldem hillshade/slope/aspect` uses, the one
expected by every covariate-engineering library for digital soil
mapping (SoilGrids' own covariate stack, the SAGA-GIS terrain
analysis suite, …). Pedotri matches it exactly.

For pixel `(i, j)`, with neighbours labelled like the cardinal
points:

```
NW  N  NE
W   z  E
SW  S  SE
```

the partial derivatives are

```
∂z/∂x = ((NE + 2·E + SE) − (NW + 2·W + SW)) / (8 · Δx)

∂z/∂y_south_positive
      = ((SW + 2·S + SE) − (NW + 2·N + NE)) / (8 · Δy)
```

(pedotri's internal `dzdy` is south-positive, mirroring the raster
array convention where row index grows southward — the alternative
GDAL also implements). Then:

**Slope** (the angle the surface makes with horizontal):

```
β = arctan( √( (∂z/∂x)² + (∂z/∂y)² ) )
```

reported in degrees, radians, or percent (rise/run × 100) — the
last being the convention used by USDA soil-survey reports and most
agronomic decision tables.

**Aspect** (compass direction the slope faces, clockwise from north):
the downhill direction in geographic (east, north) coordinates is
`(−∂z/∂x, ∂z/∂y_south_positive)` (note the sign flip on the y-axis
because of the raster convention). The compass bearing is then
`atan2(east_component, north_component)`, wrapped into `[0, 2π]`.
Pixels with negligible slope (gradient magnitude `< 1e-12`) are
returned as `NaN` — aspect is undefined there.

### Why Sobel weights and not simple central differences?

A central difference uses only the four directly-adjacent neighbours
(N, S, E, W), ignoring the four diagonals. That gets noisier as soon
as the DEM has any salt-and-pepper artefact, which is the rule rather
than the exception for SRTM / ALOS / Copernicus DEMs. Horn's weights
average over the entire 3×3 neighbourhood, with double weight on the
direct neighbours, so the gradient estimate is well-defined under
realistic noise. The 1981 paper has the full noise-rejection
argument.

## Curvature — Zevenbergen & Thorne (1987)

Z&T fit a 9-coefficient polynomial through the 3×3 neighbourhood:

```
z ≈ A·x²·y² + B·x²·y + C·x·y² + D·x² + E·y² + F·x·y + G·x + H·y + I
```

Six of those nine coefficients are needed for the second-derivative
curvatures pedology cares about. Let `L` be the pixel side:

```
D = ((z_W + z_E)/2 − z) / L²            ← second derivative ∂²z/∂x²
E = ((z_N + z_S)/2 − z) / L²            ← second derivative ∂²z/∂y²
F = (−z_NW + z_NE + z_SW − z_SE) / (4 L²)  ← cross term ∂²z/∂x∂y
G = (z_E − z_W) / (2 L)                  ← ∂z/∂x
H = (z_N − z_S) / (2 L)                  ← ∂z/∂y
```

Then:

**Profile curvature** (in the steepest-descent direction):

```
k_profile = −2 · (D·G² + E·H² + F·G·H) / (G² + H²)
```

Negative → water *decelerates* downhill (foot of a slope, accumulates
moisture). Positive → water *accelerates* (shoulder, dries out).
Pedotransfer functions fitted to grade-distinguishing data sets often
use profile curvature as a predictor for soil moisture distribution.

**Plan curvature** (across-contour, orthogonal to flow):

```
k_plan = 2 · (D·H² + E·G² − F·G·H) / (G² + H²)
```

Negative → diverging flow lines (ridge-like, water spreads out).
Positive → converging flow lines (valley-like, water concentrates).

Both have units of `1/length`, so for a DEM in metres they come out
in `1/m`. Boundary pixels use reflected DEM values; pixels with zero
denominator (truly flat patches) are returned as zero curvature.

## Topographic wetness index — Beven & Kirkby (1979)

```
TWI = ln(a / tan β)
```

where `a` is the upstream contributing area per unit contour width
(in metres, when the DEM and pixel size are in metres) and `β` is the
local slope angle (radians). TWI is a non-dimensional proxy for soil
moisture redistribution: a pixel at the foot of a long, shallow
hillslope ends up with a much higher TWI than one on a steep ridge,
and empirically this tracks the soil moisture pattern measured at the
field scale (Beven & Kirkby's original 1979 paper validates against
catchment hydrology; Sørensen et al. 2006 review the literature).

### Why pedotri does not ship a flow-accumulation engine

Computing `a` correctly is a hydrology problem: you need a
flow-routing algorithm (D8, D∞, MFD…) plus depression filling, and
the choice has significant downstream impact on the TWI map. There's
a perfectly good ecosystem — `richdem`, `pysheds`, GRASS GIS's
`r.watershed`, SAGA's `ta_channels` — that does this much better than
pedotri ever could.

So `topographic_wetness_index` takes an optional
`flow_accumulation=` array (number of upstream cells per pixel). When
you don't supply it, the function falls back to a one-cell-per-pixel
surrogate that makes TWI a function of slope alone. This is **not a
hydrology product**; it's a fast covariate generator suitable for
ML feature engineering and as a quick sanity check before you wire
in a proper hydrology pipeline. The docstring says so out loud.

### The flat-pixel floor

Strictly zero slope makes `1/tan β` diverge. Following SAGA-GIS
convention, pedotri floors `tan β` at `min_slope = 1e-4` (≈ 0.006°),
which is below the noise level of any practical DEM but above the
representable-as-positive-float zero.

## References

- Beven, K. J., & Kirkby, M. J. (1979). A physically based,
  variable contributing area model of basin hydrology. *Hydrological
  Sciences Bulletin* 24(1): 43–69.
- Horn, B. K. P. (1981). Hill shading and the reflectance map.
  *Proceedings of the IEEE* 69(1): 14–47.
- Sørensen, R., Zinko, U., & Seibert, J. (2006). On the
  calculation of the topographic wetness index: evaluation of
  different methods based on field observations. *Hydrology and
  Earth System Sciences* 10(1): 101–112.
- Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis
  of land surface topography. *Earth Surface Processes and
  Landforms* 12(1): 47–56.
