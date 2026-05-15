# Reprojection and target grids

You almost never get to pick the grids your soil data comes on. SoilGrids
publishes at ~250 m in EPSG:4326. ESA WorldCover is 10 m in the same CRS
but a different transform. Your local DEM is probably 30 m in a UTM zone.
A village boundary you got from the cadastre is a polygon in yet another
projection. Mixing them in one calculation means picking a common grid
and reprojecting everything onto it — which is what `pedotri.grid` does.

## When you need this

If you've ever hit `InvalidInputError: Raster shape mismatch` or
`Raster CRS mismatch` from pedotri 0.3, you needed this. The 0.3 raster
helpers required every input to share grid + CRS exactly. 0.4 lets you
pick a single target grid and have pedotri align the rest.

The three most common scenarios:

1. **Two soil bands on different grids.** SoilGrids ships SOC mean at
   one resolution but Q0.05 / Q0.95 quantiles sometimes at a coarser
   one. You want them lined up for `zonal_aggregate`.
2. **A coarse property + a fine mask.** SoilGrids at 250 m and ESA
   WorldCover at 10 m. The mask needs to be downsampled to the
   property grid (or the property bands upsampled — your choice).
3. **A polygon AOI from a different CRS.** Reprojecting the polygon
   geometry is shapely / pyproj territory; reprojecting the rasters
   it gets compared against is what this module does.

## Quick example

```python
from pedotri.grid import TargetGrid
from pedotri.zonal import zonal_aggregate

# Pick a common grid — here, 250 m square pixels covering the AOI.
target = TargetGrid.from_bounds((2.50, 46.95, 2.65, 47.05), resolution=0.0025)

agg = zonal_aggregate(
    region=village_polygon,
    properties={
        "soc": "soc_mean.tif",     # SoilGrids native ~250 m
        "bd":  "bd_mean.tif",      # also ~250 m
    },
    mask="worldcover.tif",          # ESA WorldCover native 10 m
    mask_include=[40],              # cropland only
    target_grid=target,             # ← everything gets aligned here
    n_samples=1000,
    seed=0,
)
```

Each `.tif` is opened, checked against `target`, and reprojected with a
sensible default kernel if it doesn't already match. Pre-loaded numpy
arrays passed directly to `properties=` are assumed to be on the target
grid already — pedotri won't reproject them silently, you're expected
to do that yourself if needed.

## What `TargetGrid` actually carries

Just four fields, mirroring a rasterio profile's geometry side:

- `crs` — coordinate reference system (anything rasterio accepts).
- `transform` — `affine.Affine`, the pixel-to-world mapping.
- `width` — number of columns.
- `height` — number of rows.

Constructors:

```python
# From an existing rasterio profile (the most common case)
target = TargetGrid.from_profile(soc_profile)

# From a bounding box + square pixel size
target = TargetGrid.from_bounds((west, south, east, north), resolution=0.0025)

# Hand-built
target = TargetGrid(crs="EPSG:4326", transform=affine, width=200, height=200)
```

Useful properties: `.bounds` returns `(left, bottom, right, top)` in CRS
units, `.pixel_size` returns `(|x|, |y|)`, `.shape` returns `(height,
width)`, and `.matches(profile)` is the cheap equality check
`reproject_to_grid` uses to short-circuit no-op reprojections.

## Per-data-type resampling defaults

When you don't pass `resampling=` explicitly, pedotri picks based on the
input dtype and whether you're upsampling or downsampling:

| Input dtype | Direction | Default kernel | Why |
|---|---|---|---|
| Integer (uint8, int16, …) | Either | `nearest` | Averaging class codes 10 + 40 → 25 is meaningless. |
| Float | Downsample (target pixel ≥ 1.5× source) | `average` | Preserves the regional mean — what you want for aggregation. |
| Float | Upsample or near-equal | `bilinear` | Smooth enough that downstream polygon classification doesn't see artefacts. |

Override with `resampling="..."` when the workflow needs something
different. Available kernels: `nearest`, `bilinear`, `cubic`,
`cubic_spline`, `lanczos`, `average`, `mode`, `min`, `max`, `sum` — the
same set GDAL ships.

## Calling reprojection by hand

When you want explicit control (e.g. caching the result, or running on
arrays that didn't come from disk):

```python
from pedotri.grid import reproject_to_grid, align_to_grid

# Single raster
warped, profile = reproject_to_grid(source_array, source_profile, target)

# Several rasters onto a shared target (auto-derived from the first if
# you don't pass target=)
aligned, target = align_to_grid([(a, prof_a), (b, prof_b), (c, prof_c)])
```

Both functions are pure — no caching, no side effects. If you call them
repeatedly with the same inputs, you'll pay the warp cost every time.
For hot paths, cache at the workflow level.

## How `rasterio.warp` actually works

Under the hood we delegate to `rasterio.warp.reproject`, which uses the
standard inverse-mapping algorithm GDAL has shipped since the 1990s.
For every destination pixel `(i, j)`:

1. Compute its world coordinate
   `(x_dst, y_dst) = T_dst · (j + 0.5, i + 0.5)` — the `+ 0.5` lands
   on the pixel centre.
2. Transform `(x_dst, y_dst)` from `dst_crs` to `src_crs` via PROJ —
   handling datum shifts, geoid models, and so on.
3. Invert the source affine to land at floating-point source-pixel
   coordinates `(u, v) = T_src⁻¹ · (x_src, y_src)`.
4. Sample the source raster around `(u, v)` with the chosen resampling
   kernel.

The destination pixel value is then a weighted combination of source
pixels in a window around `(u, v)`:

- `nearest`: just `src[round(v), round(u)]`.
- `bilinear`: 2×2 window with linear weights.
- `cubic` / `cubic_spline` / `lanczos`: larger windows with higher-order
  kernels — more expensive, smoother for upsampling.
- `average`: mean over every source pixel whose centre falls within the
  destination pixel's footprint. The right answer when you want to
  preserve the local mean across a downsample.

The full GDAL documentation table is at
<https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r>.

## Why pedotri doesn't ship a reprojection cache

The cost of one warp call is small (a 1 Mpix tile completes in
milliseconds); the cost of an incorrect cache hit (silent staleness as
a source file is updated under you) is much higher. Cache at the
*workflow* level — write the reprojected raster to disk and pass the
path back — or wrap `reproject_to_grid` with your own sha-256-keyed
cache layer.

## Affine transform refresher

A rasterio `Affine` is a 6-tuple `(a, b, c, d, e, f)` that maps
pixel `(col, row)` to world `(x, y)` via:

```
x = a · col + b · row + c
y = d · col + e · row + f
```

For an axis-aligned, north-up raster (the common case), `b = d = 0`,
`a > 0` is the x-pixel size, `e < 0` is `-y-pixel size`, and `(c, f)`
is the world coordinate of the **top-left pixel corner**.

## See also

- [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)
  — where `target_grid=` plugs into the zonal workflow.
- [Spatial correlation](Spatial-correlation) — uses the target grid's
  pixel size to convert correlation ranges from CRS units to pixels.
