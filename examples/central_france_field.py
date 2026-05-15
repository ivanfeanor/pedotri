"""Soil map of a random ~10 ha field in central France from 3 samples.

A worked field-survey example representative of what a small soil-
sampling campaign on a single farm plot actually looks like: an
irregular field of about 10 hectares, a *minimal* sampling plan of
three boreholes placed in visibly distinct sub-areas, and the
classified soil-texture map you can produce from them.

The field falls inside the same 2°-4°E × 46°-48°N central-France
bounding box used by ``examples/soilgrids_france.py``, so you can
overlay this plot against the regional SoilGrids classification if
you want to compare scales.

Coordinates are EPSG:32631 (UTM zone 31N, metres). The field
polygon is generated from a fixed random seed so the example is
fully reproducible.

The output is a USDA-coded GeoTIFF + PNG with the three sample
locations overlaid + a Shapefile of the per-class regions.

Three samples is the *minimum* for ordinary kriging — pykrige will
not fit a meaningful variogram beyond a linear model with so few
points. The resulting map is honest about that: between samples it's
essentially a smooth interpolation. For production work, aim for
8-15 samples per field; see :mod:`pedotri.interp` for the
variance-as-confidence-map trick.

Run::

    pip install 'pedotri[raster,vector,interp,matplotlib]'
    python examples/central_france_field.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from pedotri.interp import krige_sand_clay
from pedotri.raster import (
    NODATA_CODE,
    classified_to_features,
    classify_array,
    smooth_codes,
    write_classified_geotiff,
    write_features_shapefile,
)

OUT = Path(__file__).resolve().parent / "central_france_output"

# Field center in EPSG:32631 (UTM 31N) — inside the SoilGrids demo
# bounding box (2°-4°E, 46°-48°N). Picked so the field is well
# clear of the box boundary and lands on agricultural-looking terrain
# at the rough lat/lon of (47°N, 3°E).
CENTER_E, CENTER_N = 502_000.0, 5_207_500.0


def random_field_polygon(target_ha: float = 10.0, seed: int = 2024) -> Polygon:
    """Build an irregular convex polygon of approximately ``target_ha`` hectares.

    Seven random points are drawn around the field center in UTM
    coordinates, then their convex hull is taken so the result is a
    well-formed simple polygon. Areas are evaluated in m² (UTM is
    metric) and the polygon is rescaled to hit the target hectarage.
    """
    rng = np.random.default_rng(seed)
    radius = 200.0  # roughly the half-extent for a 10 ha field after rescaling
    offsets_e = rng.uniform(-radius, radius, size=7)
    offsets_n = rng.uniform(-radius, radius, size=7)
    pts = [Point(CENTER_E + e, CENTER_N + n) for e, n in zip(offsets_e, offsets_n)]
    raw = Polygon([(p.x, p.y) for p in pts]).convex_hull
    actual_ha = raw.area / 10_000.0
    factor = (target_ha / actual_ha) ** 0.5
    coords = [
        (CENTER_E + (x - CENTER_E) * factor, CENTER_N + (y - CENTER_N) * factor)
        for x, y in raw.exterior.coords
    ]
    return Polygon(coords)


def sample_layout(field: Polygon) -> list[dict[str, float]]:
    """Place 3 in-situ samples in distinct sub-areas of the field.

    The samples are chosen to span three USDA classes (sandy loam in
    the SW, silty clay in the NE, loam in the centre) so the kriged
    map has visible texture-class structure rather than a single
    uniform region. Real surveys would place samples on a stratified
    grid; the spread here is what you might get from a quick scout.
    """
    minx, miny, maxx, maxy = field.bounds
    width = maxx - minx
    height = maxy - miny

    candidates = [
        # (relative east, relative north, sand %, clay %)
        (0.20, 0.20, 60.0, 15.0),  # SW — sandy loam
        (0.80, 0.80, 15.0, 40.0),  # NE — silty clay
        (0.50, 0.50, 35.0, 25.0),  # centre — loam
    ]
    samples: list[dict[str, float]] = []
    for fx, fy, sand, clay in candidates:
        x = minx + fx * width
        y = miny + fy * height
        # Pull the sample toward the field centroid until it lands
        # inside the (irregular) polygon — robust to seed variation.
        cx, cy = field.centroid.x, field.centroid.y
        for _ in range(20):
            if field.contains(Point(x, y)):
                break
            x = 0.5 * (x + cx)
            y = 0.5 * (y + cy)
        samples.append({"x": x, "y": y, "sand": sand, "clay": clay})
    return samples


def render_field_png(
    path: Path,
    codes: np.ndarray,
    keys: list[str],
    field: Polygon,
    samples: list[dict[str, float]],
    extent: tuple[float, float, float, float],
) -> None:
    """PNG renderer that overlays the field boundary + sample points.

    Lives in the example (not in :mod:`pedotri.raster`) because
    overlaying domain-specific markers is a per-project concern.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n = len(keys)
    base = plt.get_cmap("tab20", n)
    palette = ListedColormap([base(i) for i in range(n)])
    palette.set_bad(alpha=0.0)
    display = np.ma.masked_equal(codes, NODATA_CODE)
    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        display, cmap=palette, norm=norm, extent=extent, origin="upper", interpolation="nearest"
    )
    xs, ys = field.exterior.xy
    ax.plot(xs, ys, "-", color="black", linewidth=1.5, alpha=0.8)
    for i, s in enumerate(samples, start=1):
        ax.plot(
            s["x"],
            s["y"],
            "o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"#{i}  {s['sand']:.0f}/{s['clay']:.0f}",
            (s["x"], s["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
        )

    ax.set_xlabel("easting (m, UTM 31N)")
    ax.set_ylabel("northing (m, UTM 31N)")
    ax.set_title(
        f"USDA soil texture — {field.area / 1e4:.1f} ha field, central France\n"
        f"(3 in-situ samples, ordinary kriging, 3×3 majority smoothing)"
    )
    cbar = fig.colorbar(im, ax=ax, ticks=range(n))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("USDA class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    field = random_field_polygon(target_ha=10.0, seed=2024)
    samples = sample_layout(field)

    print(f"field area: {field.area / 1e4:.2f} ha (target 10.0)")
    print(f"field bbox: {field.bounds}")
    print("samples:")
    for i, s in enumerate(samples, start=1):
        print(f"  #{i}  ({s['x']:.0f}, {s['y']:.0f})  sand={s['sand']:.0f}%  clay={s['clay']:.0f}%")

    minx, miny, maxx, maxy = field.bounds
    resolution = 2.0  # 2 m grid is fine for visible class structure on 10 ha

    print(f"kriging on a {resolution} m grid...")
    sand_grid, clay_grid, profile = krige_sand_clay(
        samples,
        bbox=(minx, miny, maxx, maxy),
        resolution=resolution,
        variogram="linear",  # most robust for n=3 samples
        n_lags=2,
        mask_polygon=field,
        crs="EPSG:32631",
    )
    valid = ~np.isnan(sand_grid)
    print(f"  sand inside field: {sand_grid[valid].min():.1f}..{sand_grid[valid].max():.1f}")
    print(f"  clay inside field: {clay_grid[valid].min():.1f}..{clay_grid[valid].max():.1f}")

    print("classifying under USDA...")
    codes, keys = classify_array(sand_grid, clay_grid, classification="USDA")

    print("smoothing (3x3 majority, 2 passes)...")
    smoothed = smooth_codes(codes, window=3, iterations=2)

    print("writing outputs:")
    write_classified_geotiff(OUT / "field_usda.tif", smoothed, profile=profile, keys=keys)
    print(f"  -> {OUT / 'field_usda.tif'}")

    render_field_png(
        OUT / "field_usda.png",
        smoothed,
        keys,
        field,
        samples,
        extent=(minx, maxx, miny, maxy),
    )
    print(f"  -> {OUT / 'field_usda.png'}")

    features = classified_to_features(
        smoothed,
        keys=keys,
        transform=profile["transform"],
        simplify_tolerance=1.0,
    )
    write_features_shapefile(features, OUT / "field_usda.shp", crs="EPSG:32631")
    print(f"  -> {OUT / 'field_usda.shp'} ({len(features)} polygons)")


if __name__ == "__main__":
    main()
