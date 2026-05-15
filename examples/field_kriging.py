"""Field-scale soil mapping from sparse in-situ samples.

Synthesizes a realistic case: a 100 m x 100 m field with 20 soil
sampling locations on a jittered grid, each with measured sand %,
silt %, clay %. Pedotri's :mod:`pedotri.interp` ordinary-krige's
sand and clay onto a 1 m grid, optionally masks to the field
boundary, classifies under USDA, smooths the result with a 3x3
majority filter, writes a paletted GeoTIFF + PNG + Shapefile.

Run with::

    pip install 'pedotri[raster,vector,interp,matplotlib]'
    python examples/field_kriging.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from pedotri.interp import krige_sand_clay
from pedotri.raster import (
    classified_to_features,
    classify_array,
    render_classified_png,
    smooth_codes,
    write_classified_geotiff,
    write_features_shapefile,
)

OUT = Path(__file__).resolve().parent / "field_output"


def synthesize_samples(n: int = 20, seed: int = 0) -> list[dict[str, float]]:
    """Sample N points within a 100m x 100m field with a SE→NW gradient."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(5, 95, size=n)
    ys = rng.uniform(5, 95, size=n)
    # Sand ramps from ~30 % in the SE to ~60 % in the NW
    sand = 30.0 + 0.30 * (100 - xs) + 0.10 * ys + rng.normal(0, 3, size=n)
    clay = 30.0 - 0.20 * (100 - xs) - 0.05 * ys + rng.normal(0, 2, size=n)
    sand = np.clip(sand, 5, 90)
    clay = np.clip(clay, 5, 70)
    return [
        {"x": float(x), "y": float(y), "sand": float(s), "clay": float(c)}
        for x, y, s, c in zip(xs, ys, sand, clay)
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    samples = synthesize_samples(n=20, seed=42)
    print(f"synthesized {len(samples)} samples (x, y in [0, 100]; sand, clay in %)")

    # Field boundary: not a square — a hexagon-ish shape
    boundary = Polygon(
        [
            (10, 5),
            (90, 5),
            (95, 50),
            (90, 95),
            (10, 95),
            (5, 50),
        ]
    )

    print(f"kriging on a 1m grid covering {boundary.bounds}...")
    sand_grid, clay_grid, profile = krige_sand_clay(
        samples,
        bbox=(0, 0, 100, 100),
        resolution=1.0,
        variogram="spherical",
        mask_polygon=boundary,
        crs="EPSG:32631",  # any projected CRS in metres works for a synthetic example
    )
    print(f"  sand range: {np.nanmin(sand_grid):.1f}..{np.nanmax(sand_grid):.1f} %")
    print(f"  clay range: {np.nanmin(clay_grid):.1f}..{np.nanmax(clay_grid):.1f} %")

    print("classifying under USDA...")
    codes, keys = classify_array(sand_grid, clay_grid, classification="USDA")

    print("smoothing (3x3 majority, 2 passes)...")
    smoothed = smooth_codes(codes, window=3, iterations=2)

    print("writing outputs:")
    write_classified_geotiff(OUT / "field_usda.tif", smoothed, profile=profile, keys=keys)
    print(f"  -> {OUT / 'field_usda.tif'}")
    render_classified_png(
        OUT / "field_usda.png",
        smoothed,
        keys=keys,
        title="Synthesized field — USDA texture (kriged + smoothed)",
        extent=(0, 100, 0, 100),
        xlabel="easting (m)",
        ylabel="northing (m)",
    )
    print(f"  -> {OUT / 'field_usda.png'}")

    features = classified_to_features(
        smoothed,
        keys=keys,
        transform=profile["transform"],
        simplify_tolerance=0.5,  # ~half a metre — preserves clear boundaries
    )
    write_features_shapefile(features, OUT / "field_usda.shp", crs="EPSG:32631")
    print(f"  -> {OUT / 'field_usda.shp'} ({len(features)} polygons)")


if __name__ == "__main__":
    main()
