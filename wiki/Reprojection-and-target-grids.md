# Reprojection and target grids

0.3 required every raster input to share the same CRS, transform, and
shape. 0.4 introduces `pedotri.grid.TargetGrid` so consumers can stitch
heterogeneous sources — SoilGrids at ~250 m in EPSG:4326, ESA
WorldCover at 10 m in the same CRS, a local UTM DEM at 30 m — onto a
single working grid before they hit the analytical primitives.

## API in one paragraph

```python
from pedotri.grid import TargetGrid, reproject_to_grid, align_to_grid
from pedotri.zonal import zonal_aggregate

target = TargetGrid.from_bounds((2.5, 46.95, 2.65, 47.05), resolution=0.0025)

# Manual reproject:
arr, prof = reproject_to_grid(soc_array, soc_profile, target)

# Or just hand zonal_aggregate the target grid and let it do the work:
agg = zonal_aggregate(
    region=village_polygon,
    properties={"soc": "soc_mean.tif", "bd": "bd_mean.tif"},  # both reprojected
    mask="worldcover.tif",                                    # also reprojected
    target_grid=target,
    profile=target.to_profile(),
    n_samples=1000,
    seed=0,
)
```

## What rasterio.warp actually does

`reproject_to_grid` delegates to `rasterio.warp.reproject`, which
implements the standard inverse-mapping algorithm GDAL has used since
the 1990s. For every destination pixel `(i, j)`:

1. Compute its world coordinate
   `(x_dst, y_dst) = T_dst · (j + 0.5, i + 0.5)`
   where `T_dst` is the affine transform of the target grid and the
   ` + 0.5` puts us at the pixel centre.
2. Transform `(x_dst, y_dst)` from `dst_crs` to `src_crs` via PROJ —
   accounting for any datum shift, geoid model, etc.
3. Invert the source affine transform to land at floating-point
   source-pixel coordinates `(u, v) = T_src⁻¹ · (x_src, y_src)`.
4. Sample the source raster around `(u, v)` using the chosen
   **resampling kernel**:

| Kernel | Formula | Use it for |
|---|---|---|
| `nearest` | `z = src[round(v), round(u)]` | Categorical (class codes, modal output) |
| `bilinear` | 2×2 window with bilinear weights | Continuous values you're upsampling |
| `cubic` / `cubic_spline` / `lanczos` | 4×4 / 6×6 / larger windows with higher-order kernels | Continuous values where you care about smoothness |
| `average` | Mean of every source pixel whose centre falls inside the destination pixel | Continuous values you're **down**sampling |
| `mode` | Most common source value inside the destination pixel | Categorical downsampling (rare; usually nearest is preferred) |
| `min` / `max` / `sum` | Self-explanatory | Statistical reductions |

The math underlying these kernels is standard signal-processing fare;
the GDAL documentation has the full reference table at
<https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r>.

## Why the per-dtype resampling defaults matter

When you pass `resampling=None` (the default), pedotri picks for you:

- Integer dtype → `nearest`. Averaging WorldCover class codes
  `[10, 40, 50]` would yield a meaningless `33.3`.
- Float dtype, target pixel ≥ 1.5× source pixel (i.e. downsampling) →
  `average`. Preserves the mean of every source value the destination
  footprint covers — essential for "regional mean SOC" not picking up
  one stray pixel as a representative.
- Float dtype, target pixel < 1.5× source pixel (upsampling or
  near-equal) → `bilinear`. Smooth enough that polygon classification
  in the next step doesn't fall on resampling artefacts.

If your input is a Q0.05 / Q0.95 quantile band you want to treat
exactly like its mean band (the standard recipe — see
[Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)),
this heuristic does the right thing because the bands share the same
dtype + grid scale.

Override with `resampling="..."` when the workflow needs something
else — e.g. `resampling="cubic"` if you want smooth gradients for
downstream slope computation, or `resampling="min"` if you're
producing a worst-case map.

## Affine transform refresher

A rasterio `Affine` is a 6-tuple `(a, b, c, d, e, f)` that maps
pixel `(col, row)` to world `(x, y)` via:

```
x = a · col + b · row + c
y = d · col + e · row + f
```

For an axis-aligned, north-up raster (the common case), `b = d = 0`,
`a > 0` is the x-pixel size, `e < 0` is `-y-pixel size`, and `(c, f)`
is the world coordinate of the **top-left pixel corner**. The
`TargetGrid.from_bounds` constructor builds this for you given a
bounding box and a square pixel size; `TargetGrid.from_profile` reads
it out of a rasterio profile.

## Caching reprojections

0.4 deliberately does *not* ship a reprojection cache. The cost of one
warp call is small (single Mpix tiles complete in milliseconds); the
cost of an incorrect cache hit (silent staleness) is much higher.
Users with hot reprojection paths should cache at the *workflow* level
— e.g. write the reprojected raster to disk and pass the path back —
or wrap `reproject_to_grid` themselves with a sha-256-keyed cache.

## See also

- [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)
  — `target_grid=` in the context of zonal aggregation.
- [Uncertainty-aware classification](Uncertainty-aware-classification)
  — for the per-pixel uncertainty story that this grid sits under.
