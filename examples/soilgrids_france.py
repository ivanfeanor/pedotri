"""Classify a SoilGrids tile of central France into USDA texture classes.

This example walks through the full pipeline:

1. Read SoilGrids 2.0 sand / clay rasters (0-5 cm topsoil).
2. Classify every pixel with :func:`pedotri.raster.classify_geotiff`.
3. Write the result as a class-coded GeoTIFF with class names embedded
   as GDAL metadata (so QGIS / rasterio downstream sees them).
4. Render a categorical map with matplotlib.

The input tiles are downloaded from the ISRIC SoilGrids 2.0 WCS
endpoint at first run and cached under ``tests/data/soilgrids_france/``.
The bounding box covers ``long(2..4)`` × ``lat(46..48)`` —
approximately the Allier / Cher / Nièvre region.

Run with::

    pip install 'pedotri[raster,matplotlib]'
    python examples/soilgrids_france.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from pedotri.raster import (
    NODATA_CODE,
    classify_geotiff,
    write_classified_geotiff,
)

# 2° × 2° bounding box, central France
BBOX = {"lon_min": 2, "lon_max": 4, "lat_min": 46, "lat_max": 48}
DEPTH = "0-5cm"
DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "soilgrids_france"


def soilgrids_url(variable: str) -> str:
    """Build a SoilGrids 2.0 WCS GetCoverage URL for one variable."""
    return (
        "https://maps.isric.org/mapserv"
        f"?map=/map/{variable}.map"
        "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={variable}_{DEPTH}_mean"
        "&FORMAT=image/tiff"
        f"&SUBSET=long({BBOX['lon_min']},{BBOX['lon_max']})"
        f"&SUBSET=lat({BBOX['lat_min']},{BBOX['lat_max']})"
        "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    )


def ensure_inputs(out_dir: Path) -> dict[str, Path]:
    """Download sand / silt / clay tiles if not already cached locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for var in ("sand", "silt", "clay"):
        target = out_dir / f"{var}_{DEPTH}_mean.tif"
        if not target.exists():
            url = soilgrids_url(var)
            print(f"downloading {var}: {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, target)
        paths[var] = target
    return paths


def render_map(codes: np.ndarray, keys: list[str], out_png: Path) -> None:
    """Render the classified raster as a categorical map.

    Each class gets its own qualitative color; nodata pixels are
    transparent. The colorbar is built from ``keys`` so the legend
    always matches the actual class set rather than a hardcoded list.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n = len(keys)
    cmap_base = plt.get_cmap("tab20", n)
    cmap = ListedColormap([cmap_base(i) for i in range(n)])
    cmap.set_bad(alpha=0.0)
    display = np.ma.masked_equal(codes, NODATA_CODE)

    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        display,
        cmap=cmap,
        norm=norm,
        extent=(BBOX["lon_min"], BBOX["lon_max"], BBOX["lat_min"], BBOX["lat_max"]),
        origin="upper",
        interpolation="nearest",
    )
    ax.set_title(f"USDA soil texture — central France, {DEPTH} (SoilGrids 2.0)")
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")

    cbar = fig.colorbar(im, ax=ax, ticks=range(n))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("USDA texture class")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}", file=sys.stderr)


def main() -> None:
    inputs = ensure_inputs(DATA_DIR)
    print(f"input tiles: {list(inputs)}", file=sys.stderr)

    codes, keys, profile = classify_geotiff(
        sand=inputs["sand"],
        clay=inputs["clay"],
        classification="USDA",
        units="g/kg",  # SoilGrids native unit
    )

    out_tif = DATA_DIR / f"usda_{DEPTH}.tif"
    write_classified_geotiff(out_tif, codes, profile=profile, keys=keys)
    print(f"wrote {out_tif}", file=sys.stderr)

    valid = codes != NODATA_CODE
    unique, counts = np.unique(codes[valid], return_counts=True)
    total = valid.sum()
    print("\nUSDA class breakdown:")
    for code, count in sorted(zip(unique, counts), key=lambda kv: -kv[1]):
        print(f"  {keys[code]:>18s}  {count:>7,d} px  ({count / total * 100:5.2f} %)")

    out_png = DATA_DIR / f"usda_{DEPTH}.png"
    try:
        render_map(codes, keys, out_png)
    except ImportError:
        print("matplotlib not installed; skipping map render", file=sys.stderr)


if __name__ == "__main__":
    main()
