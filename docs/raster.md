# Raster (GeoTIFF) classification

The `pedotri.raster` module classifies whole rasters of sand / clay
fractions into class-coded grids in one call. It works on plain numpy
arrays (no I/O dependency) or, with the optional `pedotri[raster]`
extra, on GeoTIFFs via rasterio.

> Added in 0.2.0.

## Install

```bash
pip install "pedotri[raster]"     # rasterio for the GeoTIFF helpers
```

`classify_array` works without the extra — bring your own loader.

## End-to-end: classify a SoilGrids tile

```python
from pedotri.raster import classify_geotiff, write_classified_geotiff

codes, keys, profile = classify_geotiff(
    sand="sand_0-5cm_mean.tif",
    clay="clay_0-5cm_mean.tif",
    classification="USDA",
    units="g/kg",  # SoilGrids native units (0-1000)
)

# `codes` is an int16 array, same shape as the input rasters. Each
# value is an index into `keys` (the class-key list) or NODATA_CODE
# (-1) for masked / out-of-range pixels.
print(keys[codes[0, 0]])
# 'loam'

write_classified_geotiff("usda.tif", codes, profile=profile, keys=keys)
```

The output is a single-band `int16` GeoTIFF with `nodata=-1` and class
names embedded as band metadata. `gdalinfo` will show them under
`class_0`, `class_1`, … so downstream tools can recover the mapping
without a separate lookup table.

## Numpy-only path

If you already have sand and clay as numpy arrays from `rioxarray`,
`xarray`, `zarr`, or your own GDAL loader, call `classify_array`
directly:

```python
import rioxarray
from pedotri.raster import classify_array

sand = rioxarray.open_rasterio("sand_0-5cm_mean.tif").squeeze()
clay = rioxarray.open_rasterio("clay_0-5cm_mean.tif").squeeze()

codes, keys = classify_array(
    sand.values, clay.values,
    classification="USDA", units="g/kg",
)
```

## Masking and validity

Pixels are skipped (coded `NODATA_CODE`) when:

- Input is NaN or outside `[0, 100]` % (after units conversion).
- `sand + clay > 100 + sum_tolerance` (default tolerance 5 pp,
  matches the typical SoilGrids per-pixel inconsistency).
- A user-supplied `mask` argument is `False` at that pixel.

For SoilGrids, where the three independently-modelled fractions don't
always sum to exactly 1000 g/kg, the default tolerance is what you
want. Pass `sum_tolerance=float("inf")` to disable the check entirely
if you have tightly-summed data.

## Performance

`classify_array` skips the per-edge signed-distance computation that
`classify(..., detailed=False)` also skips, so a 740 745-pixel
SoilGrids tile (885 × 837) classifies in about 0.6 s on a 2024
M-series Mac. See [benchmarking](https://github.com/ivanfeanor/pedotri/wiki/Benchmark-vs-soiltexture).

## What it does not do

- **No reprojection.** Sand and clay rasters must already be on the
  same grid (CRS, transform, shape). The helpers raise
  `InvalidInputError` rather than silently classifying misaligned
  pixels — reproject upstream with `rasterio.warp` / `gdalwarp` /
  `rioxarray.reproject`.
- **No file format detection beyond rasterio's.** Anything rasterio
  can open works (GeoTIFF, COG, JP2, …); anything it can't, you
  have to convert first.
- **No 1-D classification on rasters.** Kachinsky has a single axis
  (`physical_clay`), which is just a normal 1-D `classify()` call on a
  flat array. Reshape afterwards if you need a raster back.

## Related

- [Wiki: GeoTIFF and SoilGrids](https://github.com/ivanfeanor/pedotri/wiki/GeoTIFF-and-SoilGrids) — full SoilGrids walkthrough
- Example script: [`examples/soilgrids_france.py`](https://github.com/ivanfeanor/pedotri/blob/main/examples/soilgrids_france.py)
- API reference: [`pedotri.raster`](api.md#raster-geotiff-classification)
