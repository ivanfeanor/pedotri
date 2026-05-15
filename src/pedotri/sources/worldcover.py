"""ESA WorldCover 10 m land-cover fetcher (AOI-only).

ESA WorldCover publishes its 10 m global land-cover raster as
Cloud-Optimized GeoTIFFs on S3, organised into 3°×3° tiles
(``ESA_WorldCover_10m_2021_v200_<TILE>.tif``). GDAL's ``/vsicurl/``
driver lets rasterio do **windowed reads** over HTTP, so pulling an
AOI for a village is a few hundred kilobytes rather than the
multi-gigabyte tile.

This module exposes one entry point, :func:`fetch_aoi`, that:

1. Computes which 3°×3° tiles intersect the requested bounding box.
2. Opens each intersecting tile remotely via GDAL ``/vsicurl/``.
3. Reads only the AOI window from each tile.
4. Mosaics the windows into a single raster when the AOI spans tile
   boundaries.
5. Caches the cropped result on disk so subsequent calls are
   immediate and offline.

A small convenience, :func:`fetch_aoi_from_polygon`, derives the bbox
from any shapely geometry. The fetched raster is meant to feed
:func:`pedotri.zonal.zonal_aggregate`'s ``mask=`` parameter together
with a class allowlist (e.g. ``mask_include=[CROPLAND]``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

from pedotri.errors import InvalidInputError, PedotriError

#: ESA WorldCover v200 (2021) class codes. v100 (2020) lacks 95
#: (Mangroves) but is otherwise identical.
TREE_COVER: int = 10
SHRUBLAND: int = 20
GRASSLAND: int = 30
CROPLAND: int = 40
BUILT_UP: int = 50
BARE: int = 60
SNOW_ICE: int = 70
WATER: int = 80
HERBACEOUS_WETLAND: int = 90
MANGROVES: int = 95
MOSS_LICHEN: int = 100

#: Class-code → human-readable label, useful when rendering legends.
CLASS_LABELS: dict[int, str] = {
    TREE_COVER: "Tree cover",
    SHRUBLAND: "Shrubland",
    GRASSLAND: "Grassland",
    CROPLAND: "Cropland",
    BUILT_UP: "Built-up",
    BARE: "Bare / sparse vegetation",
    SNOW_ICE: "Snow and ice",
    WATER: "Permanent water bodies",
    HERBACEOUS_WETLAND: "Herbaceous wetland",
    MANGROVES: "Mangroves",
    MOSS_LICHEN: "Moss and lichen",
}

#: Tile size in degrees (3° per side).
TILE_DEGREES: int = 3

#: Supported WorldCover versions.
_VERSION_INFO: dict[int, tuple[str, str]] = {
    2020: ("v100", "v100/2020/map"),
    2021: ("v200", "v200/2021/map"),
}

#: S3 bucket hosting the public WorldCover dataset.
_BUCKET_BASE: str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"


@dataclass(frozen=True, slots=True)
class WorldCoverAOI:
    """Fetched WorldCover AOI raster + rasterio profile.

    ``array`` is a 2-D ``uint8`` array; ``255`` is the nodata sentinel
    used for pixels outside any intersecting tile (mosaic gaps over
    ocean, for example). ``profile`` is rasterio-shaped and ready to
    feed :func:`pedotri.raster.write_classified_geotiff` or
    :func:`pedotri.zonal.zonal_aggregate`'s ``mask=`` parameter.
    """

    array: np.ndarray
    profile: dict[str, Any]
    bbox: tuple[float, float, float, float]
    year: int
    cached: bool
    provenance: Any | None = None

    def class_fraction(self, class_codes: int | list[int]) -> float:
        """Fraction of pixels matching the given class codes."""
        if isinstance(class_codes, int):
            class_codes = [class_codes]
        valid = self.array != 255
        if not valid.any():
            return 0.0
        match = np.isin(self.array, list(class_codes)) & valid
        return float(match.sum() / valid.sum())


def fetch_aoi(
    bbox: tuple[float, float, float, float],
    *,
    year: int = 2021,
    cache_dir: Path | str | None = None,
    cache_ttl_days: float = 365.0,
    max_pixels: int = 25_000_000,
) -> WorldCoverAOI:
    """Fetch the WorldCover land-cover raster for a bounding box.

    Args:
        bbox: ``(west, south, east, north)`` in EPSG:4326 (decimal
            degrees). Must satisfy ``west < east`` and ``south < north``.
        year: Vintage. 2020 (v100) or 2021 (v200, default).
        cache_dir: Override the on-disk cache root. Defaults to
            ``$XDG_CACHE_HOME/pedotri/worldcover/``.
        cache_ttl_days: Entries older than this are refetched.
            WorldCover is annual, so 365 days is conservative; pass
            ``float("inf")`` to never invalidate.
        max_pixels: Refuse AOIs that would materialize more than this
            many pixels at native 10 m resolution (default ~25 Mpix,
            roughly a 50×50 km tile). Pass ``float("inf")`` if you
            really mean it.

    Returns:
        :class:`WorldCoverAOI` with the cropped raster (``uint8``,
        ``255`` = nodata) plus a rasterio profile.

    Raises:
        InvalidInputError: For malformed bboxes, unsupported years,
            or AOIs larger than ``max_pixels``.
        PedotriError: If a remote read fails after retries.
        ImportError: When rasterio is not installed.
    """
    west, south, east, north = _validate_bbox(bbox)
    if year not in _VERSION_INFO:
        raise InvalidInputError(
            f"Unsupported WorldCover year {year}. Available: {sorted(_VERSION_INFO)!r}."
        )

    pixel_size = 10.0 / 111_320.0  # WorldCover native ≈ 10 m → degrees at equator
    width = round((east - west) / pixel_size)
    height = round((north - south) / pixel_size)
    n_pixels = width * height
    if n_pixels > max_pixels:
        raise InvalidInputError(
            f"AOI would materialize {n_pixels:,} pixels at WorldCover "
            f"native resolution; cap is {max_pixels:,}. Shrink the bbox "
            "or raise max_pixels= explicitly."
        )

    # The max_pixels guard is request-scoped (caller might tighten it
    # between runs), so include it in the cache key so an over-budget
    # cached AOI can't sneak past a stricter follow-up call.
    cache_path = _cache_path((bbox, year, max_pixels), cache_dir)
    cached_result = _load_cached_tif(cache_path, cache_ttl_days)
    if cached_result is not None:
        array, profile = cached_result
        prov = _build_provenance(bbox, year, cached=True)
        return WorldCoverAOI(
            array=array,
            profile=profile,
            bbox=bbox,
            year=year,
            cached=True,
            provenance=prov,
        )

    array, profile = _fetch_and_mosaic(bbox, year, width, height)
    _save_cached_tif(cache_path, array, profile)
    prov = _build_provenance(bbox, year, cached=False)
    return WorldCoverAOI(
        array=array,
        profile=profile,
        bbox=bbox,
        year=year,
        cached=False,
        provenance=prov,
    )


def _build_provenance(bbox: tuple[float, float, float, float], year: int, *, cached: bool) -> Any:
    """ISO 14067 provenance record for a WorldCover AOI fetch."""
    from pedotri.audit import Provenance, _utc_iso_now, make_source

    version_id, _ = _VERSION_INFO[year]
    source = make_source(
        name="ESA WorldCover",
        version=version_id,
        url=f"{_BUCKET_BASE}/{_VERSION_INFO[year][1]}/",
        accessed_utc=_utc_iso_now(),
        cached=cached,
        bbox=list(bbox),
    )
    return Provenance(
        operation="pedotri.sources.worldcover.fetch_aoi",
        parameters={"bbox": list(bbox), "year": year},
        sources=[source],
    )


def fetch_aoi_from_polygon(geom: Any, *, year: int = 2021, **kwargs: Any) -> WorldCoverAOI:
    """Convenience: derive the bbox from a shapely geometry and forward to :func:`fetch_aoi`.

    The geometry's CRS is assumed to be EPSG:4326. Reproject upstream
    if you're working in a projected CRS.
    """
    if hasattr(geom, "bounds"):
        bounds = tuple(geom.bounds)
    elif hasattr(geom, "__geo_interface__"):
        bounds = _bounds_from_geo_interface(geom.__geo_interface__)
    elif isinstance(geom, dict):
        bounds = _bounds_from_geo_interface(geom)
    else:
        raise InvalidInputError(
            "fetch_aoi_from_polygon expects a shapely geometry, "
            "a __geo_interface__ object, or a GeoJSON dict."
        )
    return fetch_aoi(bounds, year=year, **kwargs)


def clear_cache(*, cache_dir: Path | str | None = None) -> int:
    """Delete every cached WorldCover crop in ``cache_dir`` (or the default)."""
    cache_root = _cache_root(cache_dir)
    if not cache_root.exists():
        return 0
    removed = 0
    for entry in cache_root.glob("*.tif"):
        entry.unlink()
        removed += 1
    return removed


# --- tile arithmetic ----------------------------------------------------


def tile_id(lon_origin: int, lat_origin: int) -> str:
    """Format a WorldCover tile id from a (lower-left) origin in whole degrees.

    Tile origins are multiples of :data:`TILE_DEGREES`; e.g.
    ``tile_id(3, 48) == "N48E003"`` for the 3°×3° tile covering
    ``[3..6°E, 48..51°N]``.
    """
    if lon_origin % TILE_DEGREES != 0 or lat_origin % TILE_DEGREES != 0:
        raise InvalidInputError(
            f"Tile origins must be multiples of {TILE_DEGREES}°; got ({lon_origin}, {lat_origin})."
        )
    ns = "N" if lat_origin >= 0 else "S"
    ew = "E" if lon_origin >= 0 else "W"
    return f"{ns}{abs(lat_origin):02d}{ew}{abs(lon_origin):03d}"


def intersecting_tiles(
    bbox: tuple[float, float, float, float],
) -> list[tuple[str, int, int]]:
    """Return ``[(tile_id, lon_origin, lat_origin), …]`` for ``bbox``.

    Each origin is a lower-left corner in whole degrees (multiple of
    :data:`TILE_DEGREES`). The list is sorted ``(lon, lat)``.
    """
    west, south, east, north = _validate_bbox(bbox)
    lon_start = (int(west) // TILE_DEGREES) * TILE_DEGREES
    lon_stop = (int(np.floor(east - 1e-9)) // TILE_DEGREES) * TILE_DEGREES
    lat_start = (int(south) // TILE_DEGREES) * TILE_DEGREES
    lat_stop = (int(np.floor(north - 1e-9)) // TILE_DEGREES) * TILE_DEGREES
    if east <= west or north <= south:  # pragma: no cover — handled by _validate_bbox
        return []
    return [
        (tile_id(lon0, lat0), lon0, lat0)
        for lon0 in range(lon_start, lon_stop + 1, TILE_DEGREES)
        for lat0 in range(lat_start, lat_stop + 1, TILE_DEGREES)
    ]


def tile_url(tile: str, year: int) -> str:
    """Public S3 URL for a given tile id + year."""
    if year not in _VERSION_INFO:
        raise InvalidInputError(f"Unsupported year {year}.")
    version, path = _VERSION_INFO[year]
    return f"{_BUCKET_BASE}/{path}/ESA_WorldCover_10m_{year}_{version}_{tile}.tif"


# --- internals -----------------------------------------------------------


def _validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise InvalidInputError(f"bbox must be a (west, south, east, north) 4-tuple; got {bbox!r}.")
    west, south, east, north = (float(v) for v in bbox)
    if not (-180.0 <= west < east <= 180.0):
        raise InvalidInputError(
            f"Longitudes must satisfy -180 ≤ west < east ≤ 180; got {west=}, {east=}."
        )
    if not (-90.0 <= south < north <= 90.0):
        raise InvalidInputError(
            f"Latitudes must satisfy -90 ≤ south < north ≤ 90; got {south=}, {north=}."
        )
    return (west, south, east, north)


def _bounds_from_geo_interface(gi: dict[str, Any]) -> tuple[float, float, float, float]:
    """Crude bounding-box extraction from a __geo_interface__ dict.

    For polygons / multipolygons we walk the coordinate sequences and
    take min/max. Sufficient for kicking off a fetch; geopandas /
    shapely have far better implementations for real workflows.
    """
    geom_type = gi.get("type")
    if geom_type == "Feature":
        return _bounds_from_geo_interface(gi.get("geometry", {}))
    coords = gi.get("coordinates")
    if coords is None:
        raise InvalidInputError(f"GeoJSON object has no coordinates: {gi!r}")

    def _iter_xy(seq: Any) -> Iterator[tuple[float, float]]:
        if isinstance(seq, (list, tuple)) and seq and isinstance(seq[0], (int, float)):
            yield float(seq[0]), float(seq[1])
            return
        for sub in seq:
            yield from _iter_xy(sub)

    xs: list[float] = []
    ys: list[float] = []
    for x, y in _iter_xy(coords):
        xs.append(x)
        ys.append(y)
    if not xs:
        raise InvalidInputError(f"Empty coordinate sequence in {gi!r}")
    return (min(xs), min(ys), max(xs), max(ys))


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pedotri" / "worldcover"


def _cache_path(key: tuple[Any, ...], cache_dir: Path | str | None) -> Path:
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()
    return _cache_root(cache_dir) / f"{digest}.tif"


def _load_cached_tif(path: Path, ttl_days: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not path.exists():
        return None
    if ttl_days != float("inf"):
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > ttl_days * 86400.0:
            return None
    try:
        import rasterio
    except ImportError:  # pragma: no cover
        return None
    try:
        with rasterio.open(path) as src:
            return src.read(1), dict(src.profile)
    except OSError:
        return None


def _save_cached_tif(path: Path, array: np.ndarray, profile: dict[str, Any]) -> None:
    try:
        import rasterio
    except ImportError:  # pragma: no cover
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = dict(profile)
    write_profile.update(driver="GTiff", count=1, dtype="uint8", compress="deflate")
    tmp = path.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **write_profile) as dst:
        dst.write(array, 1)
    tmp.replace(path)


def _fetch_and_mosaic(
    bbox: tuple[float, float, float, float],
    year: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read AOI windows from every intersecting WorldCover tile and stitch."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.windows import from_bounds as window_from_bounds
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pedotri.sources.worldcover requires rasterio. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc

    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    mosaic = np.full((height, width), 255, dtype=np.uint8)
    tiles = intersecting_tiles(bbox)
    if not tiles:
        raise InvalidInputError("No WorldCover tiles intersect the given bbox.")

    for tile, _lon0, _lat0 in tiles:
        url = tile_url(tile, year)
        url_for_gdal = url if url.startswith("/vsi") else f"/vsicurl/{url}"
        try:
            tile_window = _read_tile_window(
                url_for_gdal,
                bbox=bbox,
                rasterio=rasterio,
                window_from_bounds=window_from_bounds,
            )
        except (rasterio.errors.RasterioIOError, OSError) as exc:
            # Tiles over ocean don't exist; treat missing-file errors
            # as "all 255" and continue.
            if _looks_like_missing(exc):
                continue
            raise PedotriError(f"Failed to read WorldCover tile {tile} ({url}): {exc}") from exc
        if tile_window is None:
            continue
        tile_array, tile_bbox = tile_window
        _paste_into_mosaic(mosaic, tile_array, tile_bbox, bbox, width, height)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    return mosaic, profile


def _read_tile_window(
    url: str,
    *,
    bbox: tuple[float, float, float, float],
    rasterio: Any,
    window_from_bounds: Any,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Read just the AOI intersection of one remote tile."""
    with rasterio.open(url) as src:
        tile_bbox = (
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )
        west = max(bbox[0], tile_bbox[0])
        south = max(bbox[1], tile_bbox[1])
        east = min(bbox[2], tile_bbox[2])
        north = min(bbox[3], tile_bbox[3])
        if east <= west or north <= south:
            return None
        window = window_from_bounds(west, south, east, north, transform=src.transform)
        arr = src.read(1, window=window)
    return arr, (west, south, east, north)


def _paste_into_mosaic(
    mosaic: np.ndarray,
    tile_array: np.ndarray,
    tile_bbox: tuple[float, float, float, float],
    aoi_bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Place a windowed read into the right slice of the output mosaic."""
    west_a, south_a, east_a, north_a = aoi_bbox
    west_t, _south_t, _east_t, north_t = tile_bbox
    px = (east_a - west_a) / width
    py = (north_a - south_a) / height
    col_start = round((west_t - west_a) / px)
    col_stop = col_start + tile_array.shape[1]
    row_start = round((north_a - north_t) / py)
    row_stop = row_start + tile_array.shape[0]
    col_start_c = max(col_start, 0)
    row_start_c = max(row_start, 0)
    col_stop_c = min(col_stop, width)
    row_stop_c = min(row_stop, height)
    if col_stop_c <= col_start_c or row_stop_c <= row_start_c:
        return
    src_col_start = col_start_c - col_start
    src_row_start = row_start_c - row_start
    src_col_stop = src_col_start + (col_stop_c - col_start_c)
    src_row_stop = src_row_start + (row_stop_c - row_start_c)
    mosaic[row_start_c:row_stop_c, col_start_c:col_stop_c] = tile_array[
        src_row_start:src_row_stop, src_col_start:src_col_stop
    ]


def _looks_like_missing(exc: BaseException) -> bool:
    """Identify "tile doesn't exist on the bucket" errors so we can ignore them.

    WorldCover only publishes tiles where there is actually land — every
    ocean tile is a 404. We tolerate those silently and leave the
    matching mosaic pixels as nodata. Everything else is re-raised by
    the caller.

    Detection prefers the typed signal (an ``HTTPError`` with status
    404 / 403 wrapped inside rasterio's exception chain) and falls back
    to string-matching for older rasterio/GDAL versions where the
    typed error is lost in translation.
    """
    import urllib.error

    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, urllib.error.HTTPError) and cur.code in (403, 404):
            return True
        cur = cur.__cause__ or cur.__context__
    msg = str(exc).lower()
    return any(
        marker in msg for marker in ("404", "403", "not found", "no such file", "access denied")
    )


__all__ = [
    "BARE",
    "BUILT_UP",
    "CLASS_LABELS",
    "CROPLAND",
    "GRASSLAND",
    "HERBACEOUS_WETLAND",
    "MANGROVES",
    "MOSS_LICHEN",
    "SHRUBLAND",
    "SNOW_ICE",
    "TILE_DEGREES",
    "TREE_COVER",
    "WATER",
    "WorldCoverAOI",
    "clear_cache",
    "fetch_aoi",
    "fetch_aoi_from_polygon",
    "intersecting_tiles",
    "tile_id",
    "tile_url",
]
