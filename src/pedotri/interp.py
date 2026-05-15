"""Sparse-sample interpolation for the field-scale soil mapping case.

When a survey has a handful of in-situ samples plus a bounding polygon
(a single field, a research plot, a soil-profile transect), pedotri's
raster classifier wants gridded sand and clay rasters to consume. This
module fills that gap with **ordinary kriging** — the geostatistical
standard for soil-property interpolation — and returns rasters ready
to hand to :func:`pedotri.raster.classify_array`.

The flow is:

1. :func:`krige_samples` — interpolate a single property (e.g. clay %)
   on a regular grid covering the bounding box, with one of
   pykrige's variogram models.
2. :func:`krige_sand_clay` — convenience wrapper that kriges sand and
   clay independently from a list of samples, optionally masks the
   result to a bounding polygon (so cells outside the field are
   nodata), and returns ``(sand_grid, clay_grid, profile)`` ready for
   :func:`pedotri.raster.classify_array` and friends.

This is the "few samples" workflow. For the "tile of SoilGrids
rasters" workflow, use :mod:`pedotri.raster` directly — no
interpolation needed because the input is already gridded.

Why ordinary kriging
--------------------

Kriging is the gold-standard interpolator for soil sampling because
it accounts for the **spatial autocorrelation** present in soil
properties (nearby samples are more similar than distant ones, and
that similarity decays at a measurable rate). The variogram fit step
produces a per-cell *kriging variance* alongside the prediction —
useful as a confidence map.

Alternatives like IDW or RBF are faster but don't model
autocorrelation explicitly and have no built-in uncertainty estimate.

Caveats
-------

- Kriging assumes some degree of stationarity in the underlying
  process. For very small N (< ~10 samples) or highly heterogeneous
  fields, the variogram fit can be unreliable and the resulting map
  is little better than IDW. Inspect the variance grid if in doubt.
- Variogram model choice (``"linear"``, ``"spherical"``,
  ``"gaussian"``, ``"exponential"``) matters; pykrige will auto-fit
  parameters but you should sanity-check the resulting model for any
  serious work.
- pykrige's pure-Python kriging is fine for ~tens-of-thousands of
  output cells; for larger grids consider tiling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pedotri.errors import InvalidInputError, PedotriError

if TYPE_CHECKING:
    from collections.abc import Sequence


VariogramModel = Literal[
    "linear",
    "power",
    "gaussian",
    "spherical",
    "exponential",
    "hole-effect",
]


def krige_samples(
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    resolution: float | tuple[float, float],
    variogram: VariogramModel = "spherical",
    n_lags: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Ordinary-krige a single property on a regular grid.

    Args:
        xs, ys: Sample coordinates. Length-N each.
        values: Property values at each sample. Length-N, same order
            as ``xs`` / ``ys``.
        bbox: Output extent ``(x_min, y_min, x_max, y_max)``. Should
            comfortably enclose all samples (the kriging prediction
            outside the convex hull of the samples is an
            extrapolation — pykrige will produce a value, but its
            kriging variance will be large there).
        resolution: Output cell size. Either a single float (square
            cells) or a tuple ``(dx, dy)``. Units match ``xs`` / ``ys``
            (typically degrees in WGS 84 or metres in a projected CRS).
        variogram: Variogram model name passed to pykrige. ``"spherical"``
            is the most common default for soil texture; ``"linear"``
            is robust on tiny sample sets.
        n_lags: Number of lag bins used for empirical variogram
            estimation. pykrige's default is 6; raise for >50 samples,
            lower for very small N.

    Returns:
        ``(grid, variance, profile)``:

        - ``grid`` — kriging prediction, shape ``(rows, cols)``,
          ``float64``.
        - ``variance`` — kriging variance at each cell, same shape.
          Larger means lower confidence.
        - ``profile`` — rasterio-style profile dict (``transform``,
          ``crs=None``, ``width``, ``height``, ``dtype="float64"``,
          ``count=1``) suitable to feed
          :func:`pedotri.raster.write_classified_geotiff` after the
          grid is classified. ``crs`` is left as ``None`` because the
          interpolator doesn't know what CRS the coordinates are in —
          set it on the returned profile before writing if you care.

    Raises:
        ImportError: If pykrige is not installed
            (``pip install 'pedotri[interp]'``).
        InvalidInputError: For mismatched array lengths, fewer than
            three samples, or zero-area bbox.
    """
    OrdinaryKriging = _require_pykrige_ok()  # noqa: N806 — pykrige class
    from rasterio.transform import from_origin

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    vals_arr = np.asarray(values, dtype=np.float64)
    if not (xs_arr.shape == ys_arr.shape == vals_arr.shape) or xs_arr.ndim != 1:
        raise InvalidInputError(
            f"xs, ys, values must be 1-D arrays of equal length; "
            f"got shapes {xs_arr.shape}, {ys_arr.shape}, {vals_arr.shape}."
        )
    if xs_arr.size < 3:
        raise InvalidInputError(f"Kriging needs at least 3 samples; got {xs_arr.size}.")

    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        raise InvalidInputError(f"bbox must satisfy x_max > x_min, y_max > y_min; got {bbox!r}.")

    if isinstance(resolution, tuple):
        dx, dy = float(resolution[0]), float(resolution[1])
    else:
        dx = dy = float(resolution)
    if dx <= 0 or dy <= 0:
        raise InvalidInputError(f"resolution must be positive; got dx={dx}, dy={dy}.")

    # Grid construction. ``xs_target`` runs west→east; ``ys_target``
    # runs south→north (pykrige's natural convention). We flip the
    # row axis at the end so the returned array matches the raster
    # convention (row 0 = top).
    xs_target = np.arange(x_min + dx / 2, x_max, dx, dtype=np.float64)
    ys_target = np.arange(y_min + dy / 2, y_max, dy, dtype=np.float64)

    ok = OrdinaryKriging(
        xs_arr,
        ys_arr,
        vals_arr,
        variogram_model=variogram,
        nlags=n_lags,
        verbose=False,
        enable_plotting=False,
    )
    z, ss = ok.execute("grid", xs_target, ys_target)
    grid = np.asarray(z, dtype=np.float64)[::-1, :]  # flip to row-0-on-top
    variance = np.asarray(ss, dtype=np.float64)[::-1, :]

    transform = from_origin(x_min, y_min + dy * len(ys_target), dx, dy)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": "float64",
        "count": 1,
        "height": grid.shape[0],
        "width": grid.shape[1],
        "transform": transform,
        "crs": None,
        "nodata": None,
    }
    return grid, variance, profile


def krige_sand_clay(
    samples: Sequence[dict[str, float]] | np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    resolution: float | tuple[float, float],
    variogram: VariogramModel = "spherical",
    n_lags: int = 6,
    mask_polygon: Any | None = None,
    crs: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Krige sand and clay independently from a single sample table.

    This is the field-scale convenience function: given a list of in-
    situ samples (each with x, y, sand %, clay %), it kriges both
    fractions, optionally masks the result to a bounding polygon
    (so cells outside the field become nodata), and returns rasters
    suitable for :func:`pedotri.raster.classify_array`.

    Args:
        samples: Either:

            - A list of dicts ``[{"x": ..., "y": ..., "sand": ...,
              "clay": ...}, ...]`` (key names must be ``x``, ``y``,
              ``sand``, ``clay``).
            - A 2-D numpy array of shape ``(N, 4)`` with columns
              ``(x, y, sand, clay)``.

        bbox, resolution, variogram, n_lags: Same as
            :func:`krige_samples`.
        mask_polygon: Optional shapely Polygon / MultiPolygon
            defining the field boundary. Cells whose centers fall
            outside the polygon are set to ``nan`` in both output
            rasters. Pass ``None`` to skip masking. Polygon
            coordinates must be in the same CRS as the samples.
        crs: Optional CRS authority string (e.g. ``"EPSG:4326"``) to
            stamp on the returned profile.

    Returns:
        ``(sand_grid, clay_grid, profile)``. Hand directly to
        :func:`pedotri.raster.classify_array`:

        .. code-block:: python

            codes, keys = classify_array(sand_grid, clay_grid, classification="USDA")

    Raises:
        ImportError: If pykrige (or shapely, when ``mask_polygon`` is
            given) is not installed.
        InvalidInputError: For malformed inputs (see
            :func:`krige_samples`).
    """
    xs, ys, sand_v, clay_v = _unpack_samples(samples)
    sand_grid, _sand_var, profile = krige_samples(
        xs,
        ys,
        sand_v,
        bbox=bbox,
        resolution=resolution,
        variogram=variogram,
        n_lags=n_lags,
    )
    clay_grid, _clay_var, _ = krige_samples(
        xs,
        ys,
        clay_v,
        bbox=bbox,
        resolution=resolution,
        variogram=variogram,
        n_lags=n_lags,
    )
    if mask_polygon is not None:
        mask = _polygon_mask(mask_polygon, profile)
        sand_grid = np.where(mask, sand_grid, np.nan)
        clay_grid = np.where(mask, clay_grid, np.nan)
    if crs is not None:
        profile = dict(profile)
        profile["crs"] = crs
    return sand_grid, clay_grid, profile


def _unpack_samples(
    samples: Sequence[dict[str, float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce dict-of-list or NxN array inputs into separate column arrays."""
    if isinstance(samples, np.ndarray):
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise InvalidInputError(
                f"samples ndarray must have shape (N, 4) with columns "
                f"(x, y, sand, clay); got {arr.shape}."
            )
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    if not samples:
        raise InvalidInputError("samples is empty.")
    keys = {"x", "y", "sand", "clay"}
    for i, s in enumerate(samples):
        if not isinstance(s, dict) or not keys.issubset(s):
            got_keys = sorted(s.keys()) if isinstance(s, dict) else "?"
            raise InvalidInputError(
                f"samples[{i}] must be a dict with keys {sorted(keys)!r}; "
                f"got {type(s).__name__} with keys {got_keys}."
            )
    xs = np.array([float(s["x"]) for s in samples], dtype=np.float64)
    ys = np.array([float(s["y"]) for s in samples], dtype=np.float64)
    sand = np.array([float(s["sand"]) for s in samples], dtype=np.float64)
    clay = np.array([float(s["clay"]) for s in samples], dtype=np.float64)
    return xs, ys, sand, clay


def _polygon_mask(polygon: Any, profile: dict[str, Any]) -> np.ndarray:
    """Return a bool mask: True where pixel center is inside ``polygon``."""
    try:
        from rasterio.features import geometry_mask
        from shapely import geometry as sgeo
    except ImportError as exc:
        raise ImportError(
            "krige_sand_clay's polygon masking requires rasterio + shapely. "
            "Install with: pip install 'pedotri[raster,vector]'"
        ) from exc
    geom = sgeo.mapping(polygon) if hasattr(polygon, "geom_type") else polygon
    inverted = geometry_mask(
        [geom],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        invert=True,
    )
    return np.asarray(inverted, dtype=bool)


def _require_pykrige_ok() -> Any:
    """Lazy-import pykrige.OrdinaryKriging with a helpful error."""
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise PedotriError(
            "pedotri.interp requires pykrige. Install with: pip install 'pedotri[interp]'"
        ) from exc
    return OrdinaryKriging


__all__ = [
    "VariogramModel",
    "krige_samples",
    "krige_sand_clay",
]
