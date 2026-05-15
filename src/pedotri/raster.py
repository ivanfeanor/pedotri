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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pedotri.errors import InvalidInputError, PedotriError
from pedotri.geometry import points_in_polygon
from pedotri.registry import get_classification
from pedotri.schema import Classification
from pedotri.units import _convert_inputs

if TYPE_CHECKING:
    from pedotri._types import FloatArray


#: Sentinel value written into the class-code array for pixels that
#: were either masked out, contained NaN / out-of-range data, or did
#: not match any class polygon. ``-1`` keeps the codes addressable by
#: ``keys[code]`` for all *valid* pixels.
NODATA_CODE: int = -1


#: Default number of top-ranked classes kept per pixel by the Monte
#: Carlo raster path. Five is a reasonable compromise between resolving
#: realistic ambiguity (a boundary pixel in USDA rarely splits across
#: more than 4–5 classes in practice) and keeping output GeoTIFFs small.
DEFAULT_TOP_K: int = 5

#: Soft / hard pre-flight thresholds for the Monte Carlo raster path,
#: in units of "classification work" = ``valid_pixels × n_samples``.
#: Above the soft threshold the function emits a warning; above the
#: hard threshold it raises unless ``confirm=True``.
_MC_WORK_WARN: int = 100_000_000
_MC_WORK_BLOCK: int = 1_000_000_000


@dataclass(slots=True, frozen=True)
class RasterClassification:
    """Uncertainty-aware classification of a raster.

    The :attr:`codes` and :attr:`keys` fields mirror the legacy tuple
    returned by :func:`classify_array`: ``codes`` is an ``int16``
    array of class indices into ``keys`` (or :data:`NODATA_CODE`).
    The remaining fields are populated by whichever method ran.

    Distance method fills :attr:`confidence` only. Monte Carlo fills
    :attr:`top_k_codes`, :attr:`top_k_probs`, :attr:`entropy`, and
    :attr:`unclassified_probability`.

    Use the :attr:`has_confidence` / :attr:`has_probabilities` boolean
    properties to dispatch on which method ran without reaching for
    the raw :attr:`method` string.
    """

    codes: np.ndarray
    keys: list[str]
    method: str
    confidence: np.ndarray | None = None
    top_k_codes: np.ndarray | None = None
    top_k_probs: np.ndarray | None = None
    entropy: np.ndarray | None = None
    unclassified_probability: np.ndarray | None = None
    profile: dict[str, Any] | None = field(default=None)

    @property
    def has_confidence(self) -> bool:
        """True when :attr:`confidence` is populated (distance / MC methods)."""
        return self.confidence is not None

    @property
    def has_probabilities(self) -> bool:
        """True when :attr:`top_k_probs` is populated (Monte Carlo method)."""
        return self.top_k_probs is not None

    @property
    def n_classes(self) -> int:
        """Number of registered classes in the active classification."""
        return len(self.keys)

    def modal_confidence(self) -> np.ndarray:
        """Per-pixel confidence in a single scalar form.

        For the distance method this is :attr:`confidence` verbatim;
        for the Monte-Carlo method it's the modal-class probability
        (band 1 of :attr:`top_k_probs`). Pixels where the method
        produced no confidence value get ``NaN``.

        Raises :class:`InvalidInputError` if neither was populated
        (e.g. when called on a deterministic result that was wrapped
        in a :class:`RasterClassification` artificially).
        """
        if self.confidence is not None:
            return self.confidence
        if self.top_k_probs is not None:
            modal = self.top_k_probs[0].astype(np.float32, copy=True)
            # Where no class had any probability mass, surface that as NaN.
            modal[modal == 0] = np.nan
            return np.asarray(modal)
        raise InvalidInputError(
            "RasterClassification has no confidence data; the active method "
            f"({self.method!r}) populated neither .confidence nor .top_k_probs."
        )

    def class_probability(self, class_key: str) -> np.ndarray:
        """Per-pixel probability of a named class (Monte-Carlo only).

        Walks the top-k stack and returns the probability stored under
        the matching rank band, or ``0.0`` for pixels where the class
        didn't make the top-k cut. Useful for "fraction of cropland
        risk being clay" kinds of queries.

        Raises :class:`InvalidInputError` on non-MC results or on an
        unknown class key.
        """
        if not self.has_probabilities or self.top_k_codes is None or self.top_k_probs is None:
            raise InvalidInputError(
                "class_probability() requires a Monte-Carlo result. "
                f"This RasterClassification was produced by method={self.method!r}."
            )
        try:
            target = self.keys.index(class_key)
        except ValueError as exc:
            raise InvalidInputError(
                f"Unknown class key {class_key!r}. Available: {self.keys!r}."
            ) from exc
        out = np.zeros(self.codes.shape, dtype=np.float32)
        for k in range(self.top_k_codes.shape[0]):
            match = self.top_k_codes[k] == target
            if match.any():
                out = np.where(match, self.top_k_probs[k], out)
        return out


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
        A ``(codes, keys)`` tuple. ``codes`` is an int16 array of the
        same shape as ``sand``: each entry is either an index into
        ``keys`` (matched class) or :data:`NODATA_CODE`. ``keys`` is
        the list of class keys in the order used by the codes — so
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


def classify_array_with_uncertainty(  # noqa: PLR0912, PLR0915 — single-entry dispatcher
    sand_mean: FloatArray | np.ndarray,
    clay_mean: FloatArray | np.ndarray,
    *,
    sand_uncertainty: Any | None = None,
    clay_uncertainty: Any | None = None,
    classification: str | Classification,
    method: str = "distance",
    units: str = "%",
    mask: np.ndarray | None = None,
    sum_tolerance: float = 5.0,
    n_samples: int = 1000,
    seed: Any = None,
    top_k: int = DEFAULT_TOP_K,
    chunk_pixels: int = 50_000,
    confirm: bool = False,
) -> RasterClassification:
    """Classify a raster with per-pixel uncertainty bands.

    Per axis you express uncertainty through a single keyword in one
    of three shapes (matching :func:`pedotri.classify`):

    - :class:`~pedotri.uncertainty.Quantiles` of two rasters
      (``Quantiles(q05_array, q95_array)``) — the recommended form
      when you have SoilGrids-style bands.
    - A scalar or array σ.
    - ``None`` — that axis is treated as deterministic.

    Mixing forms across axes is allowed (Quantiles for sand,
    fitted σ for clay).

    Args:
        sand_mean: Sand fraction mean raster.
        clay_mean: Clay fraction mean raster, same shape as ``sand_mean``.
        sand_uncertainty: Sand-axis uncertainty in any of the three
            forms above. Default ``None`` (axis is deterministic).
        clay_uncertainty: Clay-axis uncertainty, same conventions.
        classification: A registered 2-axis classification key or
            :class:`Classification` instance.
        method: ``"distance"`` (default, cheap) or ``"monte_carlo"``.
        units: Input units. Default ``"%"``; pass ``"g/kg"`` for SoilGrids
            tiles. Applied to the mean rasters; quantile / σ rasters are
            assumed to already share the mean's units.
        mask: Optional boolean array, ``False`` cells are excluded.
        sum_tolerance: Forwarded to the underlying classification.
        n_samples: Number of Monte Carlo draws per pixel.
        seed: Optional int / Generator for reproducible MC.
        top_k: Number of top-ranked classes kept per pixel for the MC
            method. Default 5.
        chunk_pixels: MC processes valid pixels in chunks of this size to
            cap peak memory; larger values trade RAM for fewer Python
            loop iterations.
        confirm: Set to ``True`` to bypass the MC pre-flight guard when
            ``valid_pixels × n_samples`` exceeds the hard threshold.

    Returns:
        A :class:`RasterClassification` with ``codes`` and ``keys``
        populated, plus method-specific extras:

        - **distance**: ``confidence`` (float32, ``[0, 1]``, NaN on
          unclassified / zero-σ pixels).
        - **monte_carlo**: ``top_k_codes`` (int16, shape
          ``(top_k, H, W)``, ``-1`` in empty slots), ``top_k_probs``
          (float32, shape ``(top_k, H, W)``), ``entropy`` (float32,
          nats), ``unclassified_probability`` (float32, fraction of
          draws that fell outside every class).

    Raises:
        InvalidInputError: For shape mismatches, contradictory σ inputs,
            or an MC workload above the hard threshold without
            ``confirm=True``.
        PedotriError: For 1-axis classifications (use
            :func:`pedotri.classify` for those).
    """
    cls_obj = _resolve_classification(classification)
    if len(cls_obj.axes) != 2:
        raise PedotriError(
            f"classify_array_with_uncertainty supports 2-axis classifications only; "
            f"{cls_obj.key!r} has axes {cls_obj.axes}."
        )
    if method not in ("distance", "monte_carlo"):
        raise InvalidInputError(f"Unknown method {method!r}. Use 'distance' or 'monte_carlo'.")

    sand_arr = np.asarray(sand_mean, dtype=np.float64)
    clay_arr = np.asarray(clay_mean, dtype=np.float64)
    if sand_arr.shape != clay_arr.shape:
        raise InvalidInputError(
            f"sand_mean and clay_mean must share shape; got {sand_arr.shape} vs {clay_arr.shape}."
        )

    sand_sigma_arr = _resolve_axis_uncertainty("sand", sand_uncertainty, sand_arr.shape)
    clay_sigma_arr = _resolve_axis_uncertainty("clay", clay_uncertainty, sand_arr.shape)

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

    keys = [cls.key for cls in cls_obj.classes]
    n_classes = len(keys)
    codes = np.full(sand_arr.shape, NODATA_CODE, dtype=np.int16)

    if not valid.any():
        return RasterClassification(codes=codes, keys=keys, method=method)

    flat_sand = sand_arr[valid]
    flat_clay = clay_arr[valid]
    flat_sand_sigma = sand_sigma_arr[valid]
    flat_clay_sigma = clay_sigma_arr[valid]
    flat_codes = _classify_flat(flat_sand, flat_clay, cls_obj)
    codes[valid] = flat_codes

    if method == "distance":
        confidence_flat = _distance_confidence_flat(
            flat_sand, flat_clay, flat_sand_sigma, flat_clay_sigma, flat_codes, cls_obj
        )
        confidence = np.full(sand_arr.shape, np.nan, dtype=np.float32)
        confidence[valid] = confidence_flat.astype(np.float32)
        return RasterClassification(codes=codes, keys=keys, method=method, confidence=confidence)

    n_valid = int(valid.sum())
    work = n_valid * n_samples
    if work > _MC_WORK_BLOCK and not confirm:
        raise InvalidInputError(
            f"Monte Carlo workload {work:,} exceeds the hard threshold "
            f"({_MC_WORK_BLOCK:,}). Lower n_samples, mask out more pixels, "
            "or pass confirm=True to override."
        )
    if work > _MC_WORK_WARN and not confirm:
        # ``confirm=True`` opts the caller out of both the abort and the
        # soft warning — they've already acknowledged the cost.
        import warnings

        warnings.warn(
            f"Monte Carlo workload is large ({work:,} classifications). "
            "Consider chunking the raster or lowering n_samples.",
            UserWarning,
            stacklevel=2,
        )

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    probs_flat = np.zeros((n_valid, n_classes), dtype=np.float32)
    unclass_flat = np.zeros(n_valid, dtype=np.float32)
    for start in range(0, n_valid, chunk_pixels):
        stop = min(start + chunk_pixels, n_valid)
        chunk_probs, chunk_unclass = _monte_carlo_chunk(
            flat_sand[start:stop],
            flat_sand_sigma[start:stop],
            flat_clay[start:stop],
            flat_clay_sigma[start:stop],
            cls_obj,
            n_samples,
            n_classes,
            rng,
        )
        probs_flat[start:stop] = chunk_probs
        unclass_flat[start:stop] = chunk_unclass

    top_k_eff = min(top_k, n_classes)
    top_k_codes_flat, top_k_probs_flat = _top_k_from_probs(probs_flat, top_k_eff)

    entropy_flat = _entropy_from_probs(probs_flat)

    top_k_codes = np.full((top_k_eff, *sand_arr.shape), NODATA_CODE, dtype=np.int16)
    top_k_probs = np.zeros((top_k_eff, *sand_arr.shape), dtype=np.float32)
    entropy_arr = np.full(sand_arr.shape, np.nan, dtype=np.float32)
    unclass_arr = np.full(sand_arr.shape, np.nan, dtype=np.float32)
    for k in range(top_k_eff):
        top_k_codes[k][valid] = top_k_codes_flat[:, k]
        top_k_probs[k][valid] = top_k_probs_flat[:, k]
    entropy_arr[valid] = entropy_flat
    unclass_arr[valid] = unclass_flat
    return RasterClassification(
        codes=codes,
        keys=keys,
        method=method,
        top_k_codes=top_k_codes,
        top_k_probs=top_k_probs,
        entropy=entropy_arr,
        unclassified_probability=unclass_arr,
    )


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


def classify_geotiff_with_uncertainty(
    sand_mean: Path | str,
    clay_mean: Path | str,
    *,
    sand_uncertainty: Any | None = None,
    clay_uncertainty: Any | None = None,
    classification: str | Classification,
    method: str = "distance",
    units: str = "%",
    sum_tolerance: float = 5.0,
    n_samples: int = 1000,
    seed: Any = None,
    top_k: int = DEFAULT_TOP_K,
    chunk_pixels: int = 50_000,
    confirm: bool = False,
) -> RasterClassification:
    """Read SoilGrids-style mean + uncertainty GeoTIFFs, classify, return result.

    All input rasters must be georeferenced to the same grid (CRS,
    transform, shape). Each axis takes a single uncertainty kwarg in
    one of these forms:

    - ``(q05_path, q95_path)`` — a 2-tuple of GeoTIFF paths bracketing
      a SoilGrids-style 90 % credible interval. The two rasters are
      read with the same alignment check as the mean band, then wrapped
      in :class:`~pedotri.uncertainty.Quantiles` internally.
    - A single GeoTIFF path or array of σ values.
    - ``None`` — that axis is deterministic.

    The returned :class:`RasterClassification` carries:

    - ``profile`` — a rasterio profile aligned with ``sand_mean``, ready
      to feed into :func:`write_classified_geotiff` for the modal class
      output.
    - method-specific outputs as documented on
      :func:`classify_array_with_uncertainty`.
    """
    rio = _require_rasterio()

    with rio.open(sand_mean) as src_sand_mean, rio.open(clay_mean) as src_clay_mean:
        _check_rasters_aligned(src_sand_mean, src_clay_mean)
        profile = src_sand_mean.profile
        sand_arr = src_sand_mean.read(1).astype(np.float64)
        clay_arr = src_clay_mean.read(1).astype(np.float64)
        sand_mask_band = np.asarray(src_sand_mean.dataset_mask(), dtype=bool)
        clay_mask_band = np.asarray(src_clay_mean.dataset_mask(), dtype=bool)

    sand_unc = _read_uncertainty_paths(rio, sand_mean, sand_uncertainty, "sand")
    clay_unc = _read_uncertainty_paths(rio, sand_mean, clay_uncertainty, "clay")

    combined_mask = sand_mask_band & clay_mask_band

    result = classify_array_with_uncertainty(
        sand_arr,
        clay_arr,
        sand_uncertainty=sand_unc,
        clay_uncertainty=clay_unc,
        classification=classification,
        method=method,
        units=units,
        mask=combined_mask,
        sum_tolerance=sum_tolerance,
        n_samples=n_samples,
        seed=seed,
        top_k=top_k,
        chunk_pixels=chunk_pixels,
        confirm=confirm,
    )
    out_profile = dict(profile)
    out_profile.update(
        dtype="int16",
        nodata=NODATA_CODE,
        count=1,
        compress=profile.get("compress", "deflate"),
    )
    return RasterClassification(
        codes=result.codes,
        keys=result.keys,
        method=result.method,
        confidence=result.confidence,
        top_k_codes=result.top_k_codes,
        top_k_probs=result.top_k_probs,
        entropy=result.entropy,
        unclassified_probability=result.unclassified_probability,
        profile=out_profile,
    )


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


def write_confidence_geotiff(
    path: Path | str,
    confidence: np.ndarray,
    *,
    profile: dict[str, Any],
    method: str = "distance",
) -> None:
    """Write a single-band float32 confidence raster.

    NaN pixels are preserved as the GeoTIFF nodata value so QGIS and
    downstream tools render them transparent. The band gets a tag
    recording which method produced the values (``distance`` returns
    ``Φ(d/σ)``; ``monte_carlo`` returns the modal-class probability).
    """
    rio = _require_rasterio()
    out = np.asarray(confidence, dtype=np.float32)
    out_profile = dict(profile)
    out_profile.update(
        dtype="float32",
        nodata=float("nan"),
        count=1,
        compress=out_profile.get("compress", "deflate"),
    )
    out_profile.pop("photometric", None)
    with rio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, f"classification confidence ({method})")
        dst.update_tags(method=method)


def write_probability_stack_geotiff(
    path: Path | str,
    top_k_codes: np.ndarray,
    top_k_probs: np.ndarray,
    *,
    profile: dict[str, Any],
    keys: list[str],
) -> None:
    """Write the top-k class codes and probabilities as a single multi-band GeoTIFF.

    Output layout: ``2 × top_k`` bands. Bands ``1..top_k`` hold the
    class indices (``uint8``, ``255`` reserved for nodata, so up to
    255 classes are addressable). Bands ``top_k+1..2*top_k`` hold the
    probabilities as ``uint8`` scaled by ``255`` (``0 = 0 %``,
    ``255 = 100 %``). Pixels with a NODATA_CODE entry in the codes
    band carry probability ``0`` in the matching probability band.

    Dataset-level tags record the class-key mapping (``class_0``,
    ``class_1``, …) and the active top-k value. Each band's
    description identifies what it represents (``rank_1_class``,
    ``rank_1_prob``, …) so QGIS and ``gdalinfo`` make the schema
    legible without external metadata.

    Args:
        path: Output GeoTIFF path.
        top_k_codes: Shape ``(top_k, H, W)``, int16, with NODATA_CODE
            (-1) in empty slots.
        top_k_probs: Shape ``(top_k, H, W)``, float32 in ``[0, 1]``.
        profile: Rasterio profile to use for georeferencing. Use the
            ``profile`` field from
            :func:`classify_geotiff_with_uncertainty`.
        keys: Class-key list returned alongside ``codes``. Up to 255
            classes supported.

    Raises:
        ImportError: If rasterio is not installed.
        InvalidInputError: For shape mismatch between codes and probs,
            or more than 255 classes.
    """
    if top_k_codes.shape != top_k_probs.shape:
        raise InvalidInputError(
            f"top_k_codes and top_k_probs must share shape; "
            f"got {top_k_codes.shape} vs {top_k_probs.shape}."
        )
    if top_k_codes.ndim != 3:
        raise InvalidInputError(
            f"top_k arrays must be 3-D (top_k, H, W); got shape {top_k_codes.shape}."
        )
    if len(keys) > 255:
        raise InvalidInputError(
            f"write_probability_stack_geotiff supports up to 255 classes; got {len(keys)}."
        )
    rio = _require_rasterio()
    top_k = top_k_codes.shape[0]
    n_bands = 2 * top_k

    code_bands = np.where(top_k_codes == NODATA_CODE, 255, top_k_codes).astype(np.uint8)
    prob_bands = np.clip(top_k_probs * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out_profile = dict(profile)
    out_profile.update(
        dtype="uint8",
        nodata=255,
        count=n_bands,
        compress=out_profile.get("compress", "deflate"),
    )
    out_profile.pop("photometric", None)
    with rio.open(path, "w", **out_profile) as dst:
        for k in range(top_k):
            dst.write(code_bands[k], k + 1)
            dst.set_band_description(k + 1, f"rank_{k + 1}_class")
        for k in range(top_k):
            dst.write(prob_bands[k], top_k + k + 1)
            dst.set_band_description(top_k + k + 1, f"rank_{k + 1}_prob_x255")
        dst.update_tags(
            top_k=str(top_k),
            class_count=str(len(keys)),
            prob_scale="uint8 / 255",
            **{f"class_{i}": key for i, key in enumerate(keys)},
        )


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
        xlabel: X-axis label. Defaults to ``"longitude"`` for WGS 84
            maps; set to e.g. ``"easting (m)"`` for a projected CRS,
            or ``None`` to omit.
        ylabel: Y-axis label. Defaults to ``"latitude"``; same
            convention as ``xlabel``.
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


def _read_uncertainty_paths(
    rio: Any,
    reference_path: Path | str,
    value: Any | None,
    axis: str,
) -> Any | None:
    """Translate path-shaped uncertainty inputs into in-memory arrays / Quantiles.

    Used by :func:`classify_geotiff_with_uncertainty` to support the
    convenient ``sand_uncertainty=(q05_path, q95_path)`` shorthand
    alongside in-memory σ arrays and :class:`Quantiles` objects.
    Each read is alignment-checked against the mean raster.
    """
    from pedotri.uncertainty import Quantiles

    if value is None:
        return None

    def _read(path: Path | str) -> np.ndarray:
        with rio.open(path) as src_q, rio.open(reference_path) as src_ref:
            _check_rasters_aligned(src_q, src_ref)
            return np.asarray(src_q.read(1).astype(np.float64))

    # (q05_path, q95_path) — common SoilGrids shape.
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, (str, Path)) for v in value)
    ):
        q05_arr = _read(value[0])
        q95_arr = _read(value[1])
        return Quantiles(q05_arr, q95_arr)
    # Single GeoTIFF path → σ raster.
    if isinstance(value, (str, Path)):
        return _read(value)
    # Already a Quantiles / array / scalar / deprecated bare 2-tuple of
    # numbers — pass through unchanged. _resolve_axis_uncertainty handles
    # the rest (and emits the DeprecationWarning for bare numeric tuples).
    _ = axis  # documented for symmetry with sibling helpers
    return value


def _resolve_axis_uncertainty(
    axis: str,
    value: Any | None,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Translate any supported uncertainty kwarg form to a per-pixel σ raster.

    Accepts the same forms as :func:`pedotri.classify`:

    - :class:`~pedotri.uncertainty.Quantiles` (recommended).
    - Scalar or array σ.
    - ``None`` — axis is deterministic (returns all-zero σ).
    - Bare 2-tuple ``(q05, q95)`` — deprecated; routes through
      :func:`_parse_uncertainty` which emits the DeprecationWarning.

    The result is broadcast against ``target_shape`` so downstream
    machinery sees a uniform per-pixel σ regardless of the input form.
    """
    from pedotri.uncertainty import _parse_uncertainty

    if value is None:
        return np.zeros(target_shape, dtype=np.float64)
    parsed = _parse_uncertainty(value)
    if parsed is None:
        return np.zeros(target_shape, dtype=np.float64)
    try:
        return np.broadcast_to(parsed, target_shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise InvalidInputError(
            f"{axis} σ raster shape {parsed.shape} cannot broadcast to {target_shape}."
        ) from exc


def _classify_flat(
    sand_flat: np.ndarray,
    clay_flat: np.ndarray,
    cls_obj: Classification,
) -> np.ndarray:
    """First-match-wins classification on flat arrays.

    Mirrors the inline loop in :func:`classify_array` so the uncertainty
    path can call it on chunks without rebuilding the masking logic.
    """
    points = np.column_stack([sand_flat, clay_flat])
    flat_codes = np.full(points.shape[0], NODATA_CODE, dtype=np.int16)
    remaining = np.ones(points.shape[0], dtype=bool)
    for idx, cls in enumerate(cls_obj.classes):
        if not remaining.any():
            break
        assert cls.vertices is not None
        sub_idx = np.flatnonzero(remaining)
        inside = points_in_polygon(points[sub_idx], cls.vertices)
        if not inside.any():
            continue
        hit_idx = sub_idx[inside]
        flat_codes[hit_idx] = idx
        remaining[hit_idx] = False
    return flat_codes


def _distance_confidence_flat(
    sand_flat: np.ndarray,
    clay_flat: np.ndarray,
    sand_sigma: np.ndarray,
    clay_sigma: np.ndarray,
    codes: np.ndarray,
    cls_obj: Classification,
) -> np.ndarray:
    """Per-pixel confidence from |distance to assigned class boundary|.

    Pixels with no matched class (``codes == NODATA_CODE``) get NaN,
    same as zero-σ pixels.
    """
    from pedotri.geometry import signed_distance_to_polygon

    sigma_eff = np.sqrt((sand_sigma * sand_sigma + clay_sigma * clay_sigma) / 2.0)
    confidence = np.full(sand_flat.shape, np.nan, dtype=np.float64)
    classified = codes != NODATA_CODE
    if not classified.any():
        return confidence
    points = np.column_stack([sand_flat, clay_flat])
    for idx, cls in enumerate(cls_obj.classes):
        assigned = classified & (codes == idx)
        if not assigned.any():
            continue
        assert cls.vertices is not None
        sub_points = points[assigned]
        d = signed_distance_to_polygon(sub_points, cls.vertices)
        local_sigma = sigma_eff[assigned]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = d / local_sigma
        c = 0.5 * (1.0 + _erf_vec(z / np.sqrt(2.0)))
        c = np.where(local_sigma > 0, c, np.nan)
        confidence[assigned] = c
    return confidence


def _monte_carlo_chunk(
    sand_mean: np.ndarray,
    sand_sigma: np.ndarray,
    clay_mean: np.ndarray,
    clay_sigma: np.ndarray,
    cls_obj: Classification,
    n_samples: int,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_samples`` compositions per pixel, classify, tally per-class probabilities."""
    from pedotri.uncertainty import _aggregate_class_probabilities, sample_compositional

    sand_samples, clay_samples = sample_compositional(
        sand_mean, sand_sigma, clay_mean, clay_sigma, n_samples, rng=rng
    )
    flat_sand = sand_samples.reshape(-1)
    flat_clay = clay_samples.reshape(-1)
    flat_codes = _classify_flat(flat_sand, flat_clay, cls_obj)
    grid = flat_codes.reshape(sand_mean.shape[0], n_samples)
    probs, unclass = _aggregate_class_probabilities(grid, n_classes)
    return probs.astype(np.float32), unclass.astype(np.float32)


def _top_k_from_probs(probs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(top_k_codes, top_k_probs)`` of shape ``(n_points, top_k)``."""
    order = np.argsort(-probs, axis=1)[:, :top_k]
    top_probs = np.take_along_axis(probs, order, axis=1)
    top_codes = np.where(top_probs > 0, order.astype(np.int16), np.int16(NODATA_CODE))
    return top_codes, top_probs.astype(np.float32)


def _entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy in nats."""
    p = probs.astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return np.asarray((-terms.sum(axis=-1)).astype(np.float32))


def _erf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized ``erf`` — delegated to :mod:`pedotri.uncertainty`."""
    from pedotri.uncertainty import _erf_array

    return _erf_array(x)


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
    "DEFAULT_TOP_K",
    "NODATA_CODE",
    "RasterClassification",
    "classified_to_features",
    "classify_array",
    "classify_array_with_uncertainty",
    "classify_geotiff",
    "classify_geotiff_with_uncertainty",
    "render_classified_png",
    "smooth_codes",
    "write_classified_geotiff",
    "write_confidence_geotiff",
    "write_features_shapefile",
    "write_probability_stack_geotiff",
]
