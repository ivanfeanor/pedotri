"""Raster (GeoTIFF) soil-texture classification.

This module brings :func:`pedotri.classify` to whole rasters: pass in
sand and clay grids of any shape and get back a class-coded integer
grid you can write to disk, plot, or post-process. Internally it
re-uses the vectorized point-in-polygon machinery so the same
hundreds-of-thousands-of-points-per-second throughput applies to every
pixel of a typical SoilGrids tile.

Two entry points are provided:

- :func:`classify_array` — pure-numpy. Takes ``ndarray`` inputs of any
  shape, returns an ``int16`` code array plus an index→class-key list.
  No GeoTIFF dependency.
- :func:`classify_geotiff` — convenience wrapper that reads sand /
  clay GeoTIFFs from disk (via the optional ``rasterio`` extra),
  applies units conversion, runs :func:`classify_array`, and returns
  the codes together with a rasterio profile ready to feed into
  :func:`write_classified_geotiff`.

The ``rasterio`` dependency is **only** needed for the on-disk helpers;
``classify_array`` works with any ``numpy`` array source (your own
GDAL loader, ``xarray``, ``rioxarray``, ``zarr``, …).

Example::

    from pedotri.raster import classify_geotiff, write_classified_geotiff

    codes, keys, profile = classify_geotiff(
        sand="sand_0-5cm_mean.tif",
        clay="clay_0-5cm_mean.tif",
        classification="USDA",
        units="g/kg",  # SoilGrids native units
    )
    write_classified_geotiff("usda.tif", codes, profile=profile, keys=keys)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pedotri.errors import InvalidInputError, PedotriError
from pedotri.geometry import points_in_polygon
from pedotri.registry import get_classification
from pedotri.schema import Classification
from pedotri.units import _convert_inputs

if TYPE_CHECKING:
    from pathlib import Path

    from pedotri._types import FloatArray


#: Sentinel value written into the class-code array for pixels that
#: were either masked out, contained NaN / out-of-range data, or did
#: not match any class polygon. ``-1`` keeps the codes addressable by
#: ``keys[code]`` for all *valid* pixels.
NODATA_CODE: int = -1


def classify_array(
    sand: FloatArray | np.ndarray,
    clay: FloatArray | np.ndarray,
    *,
    classification: str | Classification,
    units: str = "%",
    mask: np.ndarray | None = None,
    sum_tolerance: float = 5.0,
) -> tuple[np.ndarray, list[str]]:
    """Classify a raster of sand / clay fractions into integer codes.

    Args:
        sand: Sand fraction raster (any shape). Default units are
            percent; pass ``units="g/kg"`` for SoilGrids-native rasters.
        clay: Clay fraction raster, same shape as ``sand``.
        classification: A registered 2-axis classification key
            (``"USDA"``, ``"GEPPA"``, …) or a :class:`Classification`
            instance. Must use the ``(sand, clay)`` axis order — every
            built-in 2-D classification does.
        units: Units of the input rasters. ``"%"`` (default), ``"g/kg"``,
            or ``"g/g"``. Converted internally to percent before the
            point-in-polygon test.
        mask: Optional boolean array, same shape as ``sand``. ``False``
            cells are written as :data:`NODATA_CODE` and skipped during
            classification. Useful for a water / land mask or
            externally-supplied nodata mask from rasterio.
        sum_tolerance: A pixel whose ``sand + clay`` exceeds
            ``100 + sum_tolerance`` is treated as missing data (codes to
            :data:`NODATA_CODE`). The default of 5 percentage points
            accommodates SoilGrids' independently-modelled fractions,
            whose sum can deviate from 100 % by a few percent per pixel.
            Pass ``float("inf")`` to disable the check.

    Returns:
        ``(codes, keys)`` —

        - ``codes``: int16 array, same shape as ``sand``. Each entry is
          either an index into ``keys`` (matched class) or
          :data:`NODATA_CODE`.
        - ``keys``: list of class keys in the order used by the codes;
          ``keys[codes[i, j]]`` is the class key for pixel ``(i, j)``
          when ``codes[i, j] != NODATA_CODE``.

    Raises:
        InvalidInputError: For shape mismatches or unknown units.
        PedotriError: If the chosen classification has a single axis
            (1-D classifications are out of scope for raster
            classification — they would need a different input shape).
    """
    cls_obj = _resolve_classification(classification)
    if len(cls_obj.axes) != 2:
        raise PedotriError(
            f"classify_array supports 2-axis classifications only; "
            f"{cls_obj.key!r} has axes {cls_obj.axes}. For 1-D classifications "
            f"such as KACHINSKY use pedotri.classify() directly on a flat array."
        )

    sand_arr = np.asarray(sand, dtype=np.float64)
    clay_arr = np.asarray(clay, dtype=np.float64)
    if sand_arr.shape != clay_arr.shape:
        raise InvalidInputError(
            f"sand and clay rasters must have the same shape; "
            f"got {sand_arr.shape} and {clay_arr.shape}."
        )

    if units != "%":
        sand_arr = np.asarray(_convert_inputs(sand_arr, units), dtype=np.float64)
        clay_arr = np.asarray(_convert_inputs(clay_arr, units), dtype=np.float64)

    valid = (
        np.isfinite(sand_arr)
        & np.isfinite(clay_arr)
        & (sand_arr >= 0.0)
        & (sand_arr <= 100.0)
        & (clay_arr >= 0.0)
        & (clay_arr <= 100.0)
    )
    if sum_tolerance != float("inf"):
        valid &= (sand_arr + clay_arr) <= (100.0 + sum_tolerance)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != sand_arr.shape:
            raise InvalidInputError(
                f"mask shape {mask_arr.shape} does not match raster shape {sand_arr.shape}."
            )
        valid &= mask_arr

    codes = np.full(sand_arr.shape, NODATA_CODE, dtype=np.int16)
    if not valid.any():
        return codes, [cls.key for cls in cls_obj.classes]

    flat_sand = sand_arr[valid]
    flat_clay = clay_arr[valid]
    points = np.column_stack([flat_sand, flat_clay])

    # First-match-wins, mirroring _classify_polygons in classifier.py
    # but emitting int codes directly so we never materialize a list of
    # Python strings the size of the raster.
    flat_codes = np.full(points.shape[0], NODATA_CODE, dtype=np.int16)
    remaining = np.ones(points.shape[0], dtype=bool)
    for idx, cls in enumerate(cls_obj.classes):
        if not remaining.any():
            break
        assert cls.vertices is not None  # guarded by Classification._validate
        sub_idx = np.flatnonzero(remaining)
        inside = points_in_polygon(points[sub_idx], cls.vertices)
        if not inside.any():
            continue
        hit_idx = sub_idx[inside]
        flat_codes[hit_idx] = idx
        remaining[hit_idx] = False

    codes[valid] = flat_codes
    return codes, [cls.key for cls in cls_obj.classes]


def classify_geotiff(
    sand: Path | str,
    clay: Path | str,
    *,
    classification: str | Classification,
    units: str = "%",
    sum_tolerance: float = 5.0,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Read two GeoTIFFs, classify pixel-wise, return codes + profile.

    Args:
        sand: Path to a single-band sand-fraction GeoTIFF.
        clay: Path to a single-band clay-fraction GeoTIFF in the same
            CRS, resolution, and extent as ``sand``. (Mismatches raise.)
        classification: Same as :func:`classify_array`.
        units: Units of both input rasters. Default ``"%"``. Pass
            ``"g/kg"`` for SoilGrids 2.0 tiles.
        sum_tolerance: Forwarded to :func:`classify_array`.

    Returns:
        ``(codes, keys, profile)``. ``profile`` is a rasterio profile
        dict with ``dtype="int16"``, ``nodata=NODATA_CODE``, and the
        sand raster's CRS / transform / size, ready to pass to
        :func:`write_classified_geotiff`.

    Raises:
        ImportError: If the optional ``rasterio`` dependency is not
            installed (``pip install pedotri[raster]``).
        InvalidInputError: If the two rasters disagree on shape, CRS,
            or transform.
    """
    rio = _require_rasterio()

    with rio.open(sand) as src_sand, rio.open(clay) as src_clay:
        _check_rasters_aligned(src_sand, src_clay)
        sand_arr = src_sand.read(1).astype(np.float64)
        clay_arr = src_clay.read(1).astype(np.float64)
        sand_mask = (
            np.asarray(src_sand.dataset_mask(), dtype=bool)
            if src_sand.nodata is not None
            else np.ones(sand_arr.shape, dtype=bool)
        )
        clay_mask = (
            np.asarray(src_clay.dataset_mask(), dtype=bool)
            if src_clay.nodata is not None
            else np.ones(clay_arr.shape, dtype=bool)
        )
        profile = src_sand.profile

    combined_mask = sand_mask & clay_mask
    codes, keys = classify_array(
        sand_arr,
        clay_arr,
        classification=classification,
        units=units,
        mask=combined_mask,
        sum_tolerance=sum_tolerance,
    )

    profile = dict(profile)
    profile.update(
        dtype="int16",
        nodata=NODATA_CODE,
        count=1,
        compress=profile.get("compress", "deflate"),
    )
    return codes, keys, profile


def write_classified_geotiff(
    path: Path | str,
    codes: np.ndarray,
    *,
    profile: dict[str, Any],
    keys: list[str],
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
) -> None:
    """Write a class-code raster as a GeoTIFF with class names + colors.

    Output is a paletted ``uint8`` GeoTIFF (the only dtype on which
    GDAL/QGIS persist a color table). The in-memory ``NODATA_CODE``
    sentinel (``-1``, int16) is remapped to ``255`` on disk so the
    file still has a clean nodata value. Class keys are written both
    as band-level GDAL tags (``class_<index>``) and through rasterio's
    color-table API, so QGIS and ``gdalinfo`` automatically pick up
    the legend.

    Args:
        path: Output path.
        codes: int16 array from :func:`classify_array` or
            :func:`classify_geotiff`.
        profile: rasterio profile to use for georeferencing. Typically
            the third element returned by :func:`classify_geotiff`.
        keys: Class-key list returned alongside ``codes``. Up to 255
            classes are supported (every pedotri built-in fits).
        colormap: Optional mapping ``{class_index: (R, G, B, A)}``
            attached as the band's color table. When omitted, a
            qualitative palette is generated from matplotlib's
            ``tab20``; falls back to a hand-built 12-tone palette if
            matplotlib isn't installed. The nodata entry (255) is
            forced transparent.

    Raises:
        ImportError: If ``rasterio`` is not installed
            (``pip install pedotri[raster]``).
        InvalidInputError: If ``len(keys) > 255``.
    """
    if len(keys) > 255:
        raise InvalidInputError(
            f"write_classified_geotiff supports up to 255 classes "
            f"(uint8 + nodata=255); got {len(keys)}."
        )
    rio = _require_rasterio()
    profile = dict(profile)
    profile.update(dtype="uint8", nodata=255, count=1)
    write_codes = np.where(codes == NODATA_CODE, 255, codes).astype(np.uint8)
    with rio.open(path, "w", **profile) as dst:
        dst.write(write_codes, 1)
        dst.set_band_description(1, "soil texture class index")
        dst.update_tags(1, **{f"class_{i}": key for i, key in enumerate(keys)})
        dst.update_tags(class_count=str(len(keys)))
        cmap = colormap if colormap is not None else _default_colormap(len(keys))
        cmap.setdefault(255, (0, 0, 0, 0))  # nodata: transparent
        dst.write_colormap(1, cmap)


def _default_colormap(n: int) -> dict[int, tuple[int, int, int, int]]:
    """Build a default RGBA palette for class codes.

    Uses matplotlib's qualitative ``tab20`` when available so that
    adjacent class indices land on visually distinct colors; falls
    back to a fixed 20-tone palette if matplotlib isn't installed
    (so the ``[raster]`` extra alone is sufficient for a valid
    color table).
    """
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")
        return {
            i: tuple(int(255 * c) for c in cmap(i % cmap.N))  # type: ignore[misc]
            for i in range(n)
        }
    except ImportError:
        # Hand-built 20-tone palette mirroring tab20, so the GeoTIFF
        # always carries a legible color table even without matplotlib.
        fallback = [
            (31, 119, 180),
            (174, 199, 232),
            (255, 127, 14),
            (255, 187, 120),
            (44, 160, 44),
            (152, 223, 138),
            (214, 39, 40),
            (255, 152, 150),
            (148, 103, 189),
            (197, 176, 213),
            (140, 86, 75),
            (196, 156, 148),
            (227, 119, 194),
            (247, 182, 210),
            (127, 127, 127),
            (199, 199, 199),
            (188, 189, 34),
            (219, 219, 141),
            (23, 190, 207),
            (158, 218, 229),
        ]
        return {i: (*fallback[i % len(fallback)], 255) for i in range(n)}


def smooth_codes(
    codes: np.ndarray,
    *,
    window: int = 3,
    iterations: int = 1,
    n_classes: int | None = None,
) -> np.ndarray:
    """Apply a majority filter to a class-coded raster.

    Replaces each pixel with the modal class index in a ``window x
    window`` neighborhood (so isolated mis-classifications get
    absorbed into their dominant surroundings — the standard
    salt-and-pepper cleanup for remote-sensing class maps).

    Implementation: one-hot expand the codes, run a uniform-box filter
    on each class channel via :func:`scipy.ndimage.uniform_filter`,
    and take the per-pixel ``argmax`` across classes. All vectorized
    in C — orders of magnitude faster than a Python-callback mode
    filter on large rasters.

    The filter preserves :data:`NODATA_CODE` (``-1``) pixels: any
    pixel that was nodata stays nodata; nodata pixels do not vote
    for their neighbors' new class.

    Args:
        codes: int array of class codes (same convention as
            :func:`classify_array`). ``-1`` is treated as nodata.
        window: Side length of the square neighborhood (odd, ``>= 3``).
        iterations: How many smoothing passes to apply. Each pass
            re-applies the same filter; two or three passes converge
            quickly for typical noise patterns.
        n_classes: Number of distinct class indices to consider.
            Defaults to ``codes.max() + 1`` (or 1 if the raster is
            empty / all-nodata). Pass explicitly when the raster
            might not contain every class (e.g. for an aligned
            ensemble where you want stable class indices across tiles).

    Returns:
        Smoothed code array, same shape and dtype as ``codes``.

    Raises:
        ImportError: If ``scipy`` is not installed
            (``pip install pedotri[raster]``).
        InvalidInputError: For invalid ``window`` (must be odd, ``>= 3``).
    """
    if window < 3 or window % 2 == 0:
        raise InvalidInputError(f"window must be odd and >= 3, got {window}.")
    if iterations < 1:
        raise InvalidInputError(f"iterations must be >= 1, got {iterations}.")

    ndi = _require_scipy_ndimage()
    out = codes.astype(np.int16, copy=True)
    nodata_mask = out == NODATA_CODE
    if nodata_mask.all():
        return out

    if n_classes is None:
        n_classes = int(max(out.max(), 0)) + 1

    for _ in range(iterations):
        # Per-class indicator stack: count of class k pixels in the
        # window centered at each cell. uniform_filter returns the
        # *mean*, so multiply by window**2 to recover counts (the
        # argmax is unchanged either way; keeping it int-comparable
        # for clarity).
        counts = np.empty((n_classes, *out.shape), dtype=np.float32)
        for k in range(n_classes):
            indicator = ((out == k) & ~nodata_mask).astype(np.float32)
            counts[k] = ndi.uniform_filter(indicator, size=window, mode="constant")
        new_codes = counts.argmax(axis=0).astype(np.int16)
        # If no class had any presence in a pixel's window, leave it
        # as nodata. Also, never overwrite original nodata pixels.
        all_zero = (counts.sum(axis=0) == 0) | nodata_mask
        new_codes[all_zero] = NODATA_CODE
        out = new_codes
    return out


def render_classified_png(
    path: Path | str,
    codes: np.ndarray,
    *,
    keys: list[str],
    title: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str | None = "longitude",
    ylabel: str | None = "latitude",
    figsize: tuple[float, float] = (8, 7),
    dpi: int = 150,
    cmap: str = "tab20",
) -> None:
    """Render a class-coded raster as a PNG map with a side legend.

    Args:
        path: Output PNG path.
        codes: int array from :func:`classify_array`. ``-1`` pixels
            are rendered transparent.
        keys: Class-key list returned alongside ``codes``. Each class
            index becomes a labeled tick on the colorbar.
        title: Optional figure title.
        extent: ``(left, right, bottom, top)`` in the data CRS, used
            for axis ticks. If ``None``, axes show pixel coordinates.
        xlabel, ylabel: Axis labels. Default to ``"longitude"`` /
            ``"latitude"`` for WGS 84 / EPSG:4326 maps. Set to e.g.
            ``"easting (m)" / "northing (m)"`` for projected CRSs,
            or ``None`` to omit the label entirely.
        figsize: matplotlib figure size in inches.
        dpi: Output DPI; 150 is good for ~1500-px web images, 300 for
            print.
        cmap: A qualitative matplotlib colormap. ``tab20`` matches the
            default GeoTIFF color table.

    Raises:
        ImportError: If matplotlib is not installed
            (``pip install pedotri[matplotlib]``).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as exc:
        raise ImportError(
            "render_classified_png requires matplotlib. "
            "Install with: pip install 'pedotri[matplotlib]'"
        ) from exc

    n = len(keys)
    base = plt.get_cmap(cmap, n)
    palette = ListedColormap([base(i) for i in range(n)])
    palette.set_bad(alpha=0.0)
    display = np.ma.masked_equal(codes, NODATA_CODE)
    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        display, cmap=palette, norm=norm, extent=extent, origin="upper", interpolation="nearest"
    )
    if title is not None:
        ax.set_title(title)
    if extent is not None:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("class")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def classified_to_features(
    codes: np.ndarray,
    *,
    keys: list[str],
    transform: Any,
    simplify_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    """Polygonize a class-coded raster into per-class vector features.

    Adjacent pixels sharing the same class index are merged into a
    single polygon (with holes where appropriate). nodata pixels
    (``-1``) are excluded.

    Args:
        codes: int array of class codes.
        keys: Class-key list returned alongside ``codes``.
        transform: rasterio ``Affine`` transform giving pixel→world
            coordinates. Typically ``profile["transform"]``.
        simplify_tolerance: If set, apply
            :meth:`shapely.geometry.base.BaseGeometry.simplify` to
            each polygon with this tolerance (units of the data CRS).
            Useful for big SoilGrids tiles where un-simplified
            polygons can have hundreds of thousands of vertices.

    Returns:
        List of feature dicts ``{"geometry": shapely geom,
        "properties": {"class_id": int, "class_key": str}}``. Pass
        to :func:`write_features_shapefile` to write to disk, or
        feed directly into ``geopandas.GeoDataFrame.from_features``.

    Raises:
        ImportError: If rasterio or shapely is not installed
            (``pip install 'pedotri[raster,vector]'``).
    """
    rio_features = _require_rasterio_features()
    shapely_geometry = _require_shapely_geometry()

    valid_mask = (codes != NODATA_CODE).astype(np.uint8)
    raster = codes.astype(np.int32)
    features: list[dict[str, Any]] = []
    for geom_dict, value in rio_features.shapes(raster, mask=valid_mask, transform=transform):
        idx = int(value)
        if idx < 0 or idx >= len(keys):
            continue
        geom = shapely_geometry.shape(geom_dict)
        if simplify_tolerance is not None and simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        features.append(
            {
                "geometry": geom,
                "properties": {"class_id": idx, "class_key": keys[idx]},
            }
        )
    return features


def write_features_shapefile(
    features: list[dict[str, Any]],
    path: Path | str,
    *,
    crs: str = "EPSG:4326",
) -> None:
    """Write features from :func:`classified_to_features` to a Shapefile.

    Uses ``fiona`` directly so no GeoPandas dependency is required.
    For GeoPackage or GeoJSON output, change the file extension —
    fiona infers the driver from the path suffix.

    Args:
        features: List of feature dicts from
            :func:`classified_to_features`.
        path: Output path. ``.shp`` writes ESRI Shapefile;
            ``.gpkg`` writes GeoPackage; ``.geojson`` writes GeoJSON.
        crs: Coordinate reference system as an authority string
            (e.g. ``"EPSG:4326"``). Pass the CRS of the source
            raster ``profile["crs"]``.

    Raises:
        ImportError: If shapely or fiona is not installed
            (``pip install 'pedotri[vector]'``).
    """
    fiona = _require_fiona()
    shapely_geometry = _require_shapely_geometry()

    schema = {
        "geometry": "MultiPolygon",
        "properties": {"class_id": "int", "class_key": "str:64"},
    }
    driver = _driver_for_path(path)
    with fiona.open(path, "w", driver=driver, schema=schema, crs=crs) as out:
        for feat in features:
            geom = feat["geometry"]
            if geom.geom_type == "Polygon":
                geom = shapely_geometry.MultiPolygon([geom])
            out.write(
                {
                    "geometry": shapely_geometry.mapping(geom),
                    "properties": feat["properties"],
                }
            )


def _driver_for_path(path: Path | str) -> str:
    """Map a file extension to a fiona / OGR driver name."""
    suffix = str(path).lower().rsplit(".", 1)[-1]
    return {
        "shp": "ESRI Shapefile",
        "gpkg": "GPKG",
        "geojson": "GeoJSON",
        "json": "GeoJSON",
    }.get(suffix, "ESRI Shapefile")


def _require_rasterio_features() -> Any:
    """Import rasterio.features lazily for the polygonize helper."""
    try:
        import rasterio.features
    except ImportError as exc:
        raise ImportError(
            "pedotri.raster.classified_to_features requires rasterio. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc
    return rasterio.features


def _require_scipy_ndimage() -> Any:
    """Import scipy.ndimage lazily for the smoothing helper."""
    try:
        import scipy.ndimage as ndi
    except ImportError as exc:
        raise ImportError(
            "pedotri.raster.smooth_codes requires scipy. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc
    return ndi


def _require_shapely_geometry() -> Any:
    """Import shapely.geometry lazily for the vectorize / write helpers."""
    try:
        from shapely import geometry
    except ImportError as exc:
        raise ImportError(
            "pedotri.raster's vector helpers require shapely. "
            "Install with: pip install 'pedotri[vector]'"
        ) from exc
    return geometry


def _require_fiona() -> Any:
    """Import fiona lazily for the Shapefile / GeoPackage writer."""
    try:
        import fiona
    except ImportError as exc:
        raise ImportError(
            "pedotri.raster.write_features_shapefile requires fiona. "
            "Install with: pip install 'pedotri[vector]'"
        ) from exc
    return fiona


def _resolve_classification(c: str | Classification) -> Classification:
    if isinstance(c, str):
        return get_classification(c)
    if isinstance(c, Classification):
        return c
    raise TypeError(f"classification must be a str or Classification, got {type(c).__name__}.")


def _require_rasterio() -> Any:
    """Import rasterio lazily so the core package has no GDAL dependency."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "pedotri.raster's GeoTIFF helpers require rasterio. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc
    return rasterio


def _check_rasters_aligned(src_a: Any, src_b: Any) -> None:
    """Verify two rasterio datasets share CRS, transform, and shape."""
    if src_a.shape != src_b.shape:
        raise InvalidInputError(
            f"Raster shape mismatch: {src_a.shape} vs {src_b.shape}. "
            f"Reproject / resample to a common grid before classification."
        )
    if src_a.crs != src_b.crs:
        raise InvalidInputError(
            f"Raster CRS mismatch: {src_a.crs} vs {src_b.crs}. "
            f"Reproject to a common CRS before classification."
        )
    if src_a.transform != src_b.transform:
        raise InvalidInputError(
            f"Raster transform mismatch:\n  {src_a.transform}\nvs\n  {src_b.transform}\n"
            f"Resample to a common grid before classification."
        )


__all__ = [
    "NODATA_CODE",
    "classified_to_features",
    "classify_array",
    "classify_geotiff",
    "render_classified_png",
    "smooth_codes",
    "write_classified_geotiff",
    "write_features_shapefile",
]
