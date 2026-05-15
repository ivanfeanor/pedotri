# GeoTIFF and SoilGrids

This page walks through classifying a real GeoTIFF dataset — the
ISRIC **SoilGrids 2.0** sand/silt/clay maps for central France at
0-5 cm depth — into a USDA texture-class raster you can plot, share,
or feed into a downstream GIS workflow.

The full runnable script is at
[`examples/soilgrids_france.py`](https://github.com/ivanfeanor/pedotri/blob/main/examples/soilgrids_france.py).
This page tells you what's going on inside it.

> Requires pedotri ≥ 0.2 (for the `pedotri.raster` module) and the
> `[raster]` extra: `pip install "pedotri[raster,matplotlib]"`.

## What is SoilGrids?

[SoilGrids](https://www.isric.org/explore/soilgrids) is ISRIC's
global gridded soil-property dataset. The 2.0 version provides
sand / silt / clay (plus several other properties) at six depths
(0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm) on a 250 m grid in
Goode Homolosine projection. The data is served via WCS in WGS 84
when you ask for it that way.

Native units: **g/kg** (so values are typically 0-1000, not 0-100).

## Downloading a tile

You can pull a 2° × 2° tile from the WCS endpoint with `curl` or
`urllib`:

```
https://maps.isric.org/mapserv
  ?map=/map/sand.map
  &SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage
  &COVERAGEID=sand_0-5cm_mean
  &FORMAT=image/tiff
  &SUBSET=long(2,4)
  &SUBSET=lat(46,48)
  &SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
  &OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326
```

Swap `sand` → `silt` → `clay` (in both `map=` and `COVERAGEID=`) to
fetch all three bands. For a 2° × 2° box at central-France latitudes,
each tile is ~600 kB compressed GeoTIFF (≈ 885 × 837 Int16 pixels at
~250 m).

The example script does this automatically on first run and caches the
files under `tests/data/soilgrids_france/`.

## Classifying with `pedotri.raster`

The `pedotri.raster` module gives you a one-shot helper:

```python
from pedotri.raster import classify_geotiff, write_classified_geotiff

codes, keys, profile = classify_geotiff(
    sand="tests/data/soilgrids_france/sand_0-5cm_mean.tif",
    clay="tests/data/soilgrids_france/clay_0-5cm_mean.tif",
    classification="USDA",
    units="g/kg",  # SoilGrids native unit
)
```

What just happened:

- The two GeoTIFFs are read via `rasterio` and checked for alignment
  (same CRS, transform, shape) — mismatched grids raise
  `InvalidInputError` so you don't silently classify the wrong pair
  of pixels.
- Every pixel goes through the vectorized point-in-polygon test for
  the USDA 12-class triangle. The whole 740 745-pixel tile classifies
  in ≈ 0.6 s on a 2024 M-series Mac.
- `codes` is an `int16` numpy array, same shape as the input, where
  each value is an index into `keys` (the list of class keys in
  classification order). `-1` (`pedotri.raster.NODATA_CODE`) marks
  pixels where input data was missing or outside [0, 100] %.
- `profile` is a rasterio profile already set up with `dtype="int16"`,
  `nodata=-1`, and the source raster's CRS/transform — ready to write.

### Pure-numpy variant

If you don't want a `rasterio` dependency (e.g. you're using
`rioxarray`, `zarr`, or your own loader) call `classify_array`
directly:

```python
import rioxarray, pedotri.raster as pr

sand = rioxarray.open_rasterio("sand_0-5cm_mean.tif").squeeze()
clay = rioxarray.open_rasterio("clay_0-5cm_mean.tif").squeeze()

codes, keys = pr.classify_array(
    sand.values, clay.values,
    classification="USDA", units="g/kg",
)
```

`classify_array` takes any 2-D (or n-D) numpy array of the same shape
and returns an array of the same shape. It has no I/O dependency.

## Writing the result back out

```python
write_classified_geotiff("usda_0-5cm.tif", codes, profile=profile, keys=keys)
```

The output is a single-band `int16` GeoTIFF with `nodata=-1`. Class
keys are stored as band-level GDAL metadata:

```
$ gdalinfo usda_0-5cm.tif | grep class_
  class_count=12
  class_0=clay
  class_1=silty_clay
  ...
  class_11=sandy_loam
```

So downstream tools (`gdalinfo`, QGIS via "Show classes from raster
metadata") can recover the meaning of each integer code without an
external lookup table.

## Plotting

A categorical colormap with one band per class:

```python
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

n = len(keys)
cmap = ListedColormap([plt.get_cmap("tab20")(i) for i in range(n)])
norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
disp = np.ma.masked_equal(codes, -1)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(disp, cmap=cmap, norm=norm, origin="upper",
               extent=(2, 4, 46, 48))
fig.colorbar(im, ax=ax, ticks=range(n)).ax.set_yticklabels(keys)
```

## What pedotri does *not* do

- **No reprojection.** Pedotri assumes the sand and clay rasters
  already live on the same grid. Reproject upstream with rasterio
  / gdalwarp / rioxarray.
- **No DEM-aware land masks.** If your tile crosses an ocean or a
  region where SoilGrids has no data, the corresponding pixels will
  be set to `NODATA_CODE` based on the input rasters' nodata flag,
  but pedotri itself doesn't pull in a coastline.
- **No depth-aware logic.** Each depth slice is independent. Want the
  depth-weighted average soil texture? Compute the weighted average
  of sand/clay across depth bands *before* calling `classify_array`.

## A worked example: depth-weighted "topsoil" texture

```python
import numpy as np, rasterio
from pedotri.raster import classify_array

def read(p): 
    return rasterio.open(p).read(1).astype(np.float64)

depths = [(0, 5), (5, 15), (15, 30)]
weights = np.array([5, 10, 15], dtype=np.float64)  # cm
weights /= weights.sum()

sand_topsoil = sum(
    w * read(f"sand_{lo}-{hi}cm_mean.tif")
    for w, (lo, hi) in zip(weights, depths)
)
clay_topsoil = sum(
    w * read(f"clay_{lo}-{hi}cm_mean.tif")
    for w, (lo, hi) in zip(weights, depths)
)

codes, keys = classify_array(
    sand_topsoil, clay_topsoil,
    classification="USDA", units="g/kg",
)
```

The result is the USDA classification of the 0-30 cm depth-weighted
average — sometimes a better answer than classifying each layer
separately and picking the dominant class.

## Related

- [Benchmark-vs-soiltexture](Benchmark-vs-soiltexture) — what
  "vectorized 740k pixels/s" actually buys you
- API docs: [`pedotri.raster`](https://ivanfeanor.github.io/pedotri/) (added in 0.2)
- Example script: [`examples/soilgrids_france.py`](https://github.com/ivanfeanor/pedotri/blob/main/examples/soilgrids_france.py)
