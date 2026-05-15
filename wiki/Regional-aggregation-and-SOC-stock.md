# Regional aggregation and SOC stock

You have a polygon — a village, a commune, a farm AOI, a delineated
field — and a stack of SoilGrids-style rasters (mean + Q0.05 + Q0.95
per property). You need a single number per property, *with error
bars*, restricted to the relevant land use. This is exactly the kind
of analysis that gets done badly when libraries only operate on the
mean band: somebody takes the spatial average, calls it the SOC, and
loses the distributional content that made the upstream model worth
buying.

`pedotri.zonal.zonal_aggregate` handles the full workflow in sample
space.

## What it does, in three steps

1. **Restrict to the relevant pixels.** The region polygon defines
   *where you want an answer*. An optional mask raster (e.g. ESA
   WorldCover, or any user-supplied raster) plus a `mask_include`
   class allowlist defines *which pixels are valid*. The intersection
   gives the effective mask.
2. **Sample per pixel.** For each property, draw `n_samples` per pixel
   from the (mean, Q05, Q95) distribution implied by the raster bands.
3. **Aggregate per draw.** For each draw, compute the regional area
   mean. The resulting `n_samples` values form a posterior distribution
   over the regional aggregate.

The samples are kept verbatim on the returned object, so any further
derivation — SOC stock, depth-weighted means, anything you can write
as a function of the property means — propagates the full uncertainty
through one call to `combine()`.

## Quick start

```python
import numpy as np
from pedotri.uncertainty import Quantiles
from pedotri.zonal import zonal_aggregate

# Two properties on a synthetic 100×100 grid.
shape = (100, 100)
soc_mean = np.full(shape, 22.0)        # g / kg
bd_mean  = np.full(shape, 1.32)        # g / cm³

# Region: bottom-right quadrant of the grid.
region = np.zeros(shape, dtype=bool)
region[50:, 50:] = True

# Land cover: every other pixel is "cropland" (class 40).
mask = np.zeros(shape, dtype=np.int32)
mask[::2, ::2] = 40

agg = zonal_aggregate(
    region=region,
    properties={
        "soc": {"mean": soc_mean, "uncertainty": Quantiles(soc_mean - 4, soc_mean + 4)},
        "bd":  {"mean": bd_mean,  "uncertainty": Quantiles(bd_mean - 0.10, bd_mean + 0.10)},
    },
    mask=mask,
    mask_include=[40],
    n_samples=2000,
    seed=42,
)

agg["soc"].mean, agg["soc"].q05, agg["soc"].q95
# (≈ 22.0, ≈ 21.8, ≈ 22.2)

agg.mask_coverage     # 0.25 — quarter of the region's pixels survived the mask
agg.n_pixels_used     # 625 — quarter of 2500 region pixels
```

### Other accepted property-spec shapes

Each property's value is the spec for that property. Any of these
work and produce identical results:

```python
# Recommended — dict with explicit Quantiles
properties={"soc": {"mean": m, "uncertainty": Quantiles(q05, q95)}}

# Equivalent — dict with bare q05/q95 keys
properties={"soc": {"mean": m, "q05": q05, "q95": q95}}

# Concise 2-tuple form
properties={"soc": (m, Quantiles(q05, q95))}

# Single σ (when you've already fit a distribution)
properties={"soc": {"mean": m, "sigma": sigma_array}}
properties={"soc": (m, sigma_array)}

# Deprecated — emits DeprecationWarning; use Quantiles instead
properties={"soc": (m, q05, q95)}
```

`mean` / `q05` / `q95` / `sigma` may themselves be ndarrays *or* paths
to single-band GeoTIFFs (read with rasterio at call time).

The per-property Q05/Q95 (here `21.8`/`22.2`) is much tighter than the
per-pixel Q05/Q95 (`18`/`26`). That's correct: averaging across 625
pixels under an independent-pixel assumption shrinks uncertainty by
about √625 ≈ 25 ×. (See the caveat below — this number is a *lower
bound* on the real regional uncertainty.)

## SOC stock — propagating through a formula

`AggregateDistribution.samples` and `ZonalAggregate.combine()` exist
so you can express derived quantities without leaving sample space:

```python
AREA_M2 = 1_000_000.0    # 100 ha of cropland under our AOI

stock = agg.combine(
    lambda soc, bd, depth=0.30, area=AREA_M2:
        soc * 0.001        # g/kg → kg/kg
        * bd * 1000.0      # g/cm³ → kg/m³
        * depth            # m
        * area,            # m²  →  kg total
    name="soc_stock_kg",
)

stock.mean, stock.q05, stock.q95
# (≈ 8.7e6 kg, ≈ 8.6e6 kg, ≈ 8.8e6 kg)
```

The lambda is applied element-wise across the 2000 paired draws.
Defaults on the lambda — `depth=0.30`, `area=AREA_M2` — let you bake
in constants without leaking them through the aggregation API. The
returned `AggregateDistribution` carries `.mean`, `.std`, `.q05`,
`.q50`, `.q95`, `.quantile(q)`, and the raw `.samples` for further
analysis or plotting.

## Region inputs

`region=` accepts three forms:

- A **boolean ndarray** of the property raster's shape (the simplest
  case — no georeferencing involved).
- A **shapely geometry** (Polygon or MultiPolygon) — rasterized onto
  the property grid via `rasterio.features.rasterize`. Requires the
  `[raster]` extra and a `profile=` argument carrying the target grid
  (`crs`, `transform`, `width`, `height`).
- Anything exposing `__geo_interface__` (GeoJSON-like dict, geopandas
  rows, etc.) — same path as the shapely case.

When you read property rasters from disk by passing paths to
`zonal_aggregate`, the function picks up the profile automatically;
`profile=` is only needed when you pass pre-loaded ndarrays *and* a
geometry region.

## Multi-depth properties: aggregating 0–30 cm with uncertainty

SoilGrids 2.0 publishes its predictions at six standard depth intervals
— 0–5, 5–15, 15–30, 30–60, 60–100, 100–200 cm — each with its own
mean + Q05 + Q95. The conventional "topsoil" layer for agronomic
work is 0–30 cm, which means weighting the first three bands
proportionally to their interval width (5, 10, 15) for a single
0–30 cm raster *per quantile band*. Most production pipelines hand-roll
this with numpy and quietly hope the quantile arithmetic is correct.

`pedotri.zonal.aggregate_depths` does it explicitly:

```python
from pedotri.zonal import aggregate_depths
from pedotri.uncertainty import Quantiles

soc_topsoil_mean, soc_topsoil_q = aggregate_depths(
    samples={
        "0-5cm":  {"mean": soc_0_5_mean,  "uncertainty": Quantiles(soc_0_5_q05,  soc_0_5_q95)},
        "5-15cm": {"mean": soc_5_15_mean, "uncertainty": Quantiles(soc_5_15_q05, soc_5_15_q95)},
        "15-30cm": {"mean": soc_15_30_mean,"uncertainty": Quantiles(soc_15_30_q05,soc_15_30_q95)},
    },
    weights={"0-5cm": 5, "5-15cm": 10, "15-30cm": 15},
)

# Drop straight into zonal_aggregate.
agg = zonal_aggregate(
    region=region,
    properties={"soc": {"mean": soc_topsoil_mean, "uncertainty": soc_topsoil_q}},
    n_samples=1000,
    seed=0,
)
```

Weights need not sum to 1; they're renormalised internally. Inputs can
be scalars or arrays of any shape — every band is element-wise
combined under the same weight vector. Mean / Q05 / Q95 are aggregated
**independently** with the same weights (the standard "average the
quantile bands" recipe — see the caveat at the bottom of this page for
the statistical fine print).

Per-depth specs accept the same shapes as a `zonal_aggregate` property
spec: dict with `uncertainty=` / `sigma=` / `q05+q95`, 2-tuple, or
deprecated 3-tuple.

## Masks

`mask=` accepts either an ndarray or a path. Two modes:

- **Boolean mask** (any boolean ndarray, or a categorical with
  `mask_include=None`): `True` cells are kept.
- **Categorical mask** with `mask_include=[…]`: the mask is treated as
  integer class codes; cells matching any value in the allowlist are
  kept. The ESA WorldCover convention is `mask_include=[40]` for
  cropland; combine multiple classes by adding their codes.

The mask is intersected with the region mask *and* a validity mask
that drops non-finite property pixels and pixels with `Q95 < Q05`.

### Pulling a WorldCover mask just for the AOI

`pedotri.sources.worldcover` pulls the ESA WorldCover land-cover
raster for any bounding box without downloading whole 3°×3° tiles.
Internally it uses GDAL's `/vsicurl/` driver to read only the
windows it needs from the published Cloud-Optimized GeoTIFFs — a
1 km² village pulls a few hundred KB instead of the ~1 GB tile —
and caches the cropped result on disk:

```python
from pedotri.sources import worldcover
from pedotri.zonal import zonal_aggregate

aoi = worldcover.fetch_aoi((2.50, 46.95, 2.65, 47.05), year=2021)
aoi.class_fraction(worldcover.CROPLAND)     # e.g. 0.61

agg = zonal_aggregate(
    region=village_polygon,
    properties={
        "soc": {
            "mean": "soc_mean.tif",
            "q05":  "soc_q05.tif",
            "q95":  "soc_q95.tif",
        }
    },
    mask=aoi.array,
    mask_include=[worldcover.CROPLAND],
    profile=aoi.profile,                    # only needed if region is a geometry
    n_samples=1000,
    seed=0,
)
```

Pass any shapely polygon to `fetch_aoi_from_polygon(geom, …)` instead
of a bbox. The fetched array shares WorldCover's native EPSG:4326 grid
at ~10 m resolution; if your SoilGrids inputs live on a coarser grid
you'll need a resampling step before the masks line up — that's part
of the 0.4 reprojection work.

## The independent-pixel caveat

`zonal_aggregate` draws each pixel's value independently of its
neighbours. SoilGrids residuals are strongly spatially autocorrelated
— neighbouring 250 m pixels share most of their predictive uncertainty
— so the regional Q05/Q95 reported here is a **lower bound** on the
true posterior width. In practical terms:

- The **mean** is unaffected.
- The **standard deviation** can underestimate by an integer factor —
  three to five is realistic for SoilGrids at typical commune extents.
- For decisions where the headline number is the regional *expectation*
  (carbon accounting, agronomic targeting), this is usually acceptable.
- For decisions that explicitly weight downside risk (compliance,
  liability, scenario planning), treat the reported Q05 as optimistic.

A spatially-correlated sampling mode is on the [0.4 roadmap](0.3-roadmap)
— probably an isotropic Gaussian field with a configurable range
parameter, with the option to use SoilGrids' per-property correlation
where available.

## Pre-flight cost guard

The MC workload scales as `n_pixels × n_samples × n_properties`. A
soft warning fires above 1e8 pixel-sample-properties; the function
aborts above 1e9 unless you pass `confirm=True`. For a 1 ha SOC
assessment at 250 m SoilGrids resolution that's not even close — for a
European-country-scale aggregate you'll hit it and probably want to
chunk by sub-region instead.

## End-to-end example

`examples/aoi_soc_stock.py` chains every piece introduced in 0.3 —
WorldCover land-cover mask → SoilGrids-style mean+Q05+Q95 stack →
`aggregate_depths` to 0–30 cm → `zonal_aggregate` restricted to
cropland → `combine` for the SOC stock — on synthetic inputs so it
runs in under a second with no network access:

```text
$ python examples/aoi_soc_stock.py
AOI region pixels:   10,000
Cropland pixels:     5,000
Mask coverage:       50.0%
Cropland area:       50,000,000 m² = 5000.0 ha

SOC mean  (g/kg):    mean= 20.83  Q05= 20.78  Q95= 20.89
BD  mean  (g/cm³):   mean=  1.364 Q05=  1.362 Q95=  1.365

SOC stock (t/AOI):   mean=    426.2  Q05=    425.0  Q95=    427.4  σ=    0.74
```

Swap the `build_synthetic_*` helpers for real data loaders
(`worldcover.fetch_aoi(...)`, rasterio reads of your SoilGrids tiles)
and the same script becomes a production SOC-stock pipeline.

## See also

- [Uncertainty-aware classification](Uncertainty-aware-classification)
  — the per-pixel building blocks `zonal_aggregate` is built on.
- [0.3 roadmap](0.3-roadmap) — what comes next (spatial correlation,
  reprojection, multi-resolution alignment).
