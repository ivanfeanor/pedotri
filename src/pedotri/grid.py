"""Target-grid reprojection and multi-resolution alignment.

The pedotri 0.3 raster surface required every input to share the same
CRS, transform, and shape — that's enough for "I have one SoilGrids
tile, classify it" workflows, but as soon as you cross datasets it
breaks: SoilGrids ships in EPSG:4326 at ~250 m, ESA WorldCover is also
EPSG:4326 but at 10 m, a digital terrain model might be in a local UTM
zone at 30 m, and the AOI polygon might come from a national cadastre
in yet another projection.

This module introduces a single :class:`TargetGrid` concept (CRS +
``affine.Affine`` transform + ``width`` × ``height``) that pedotri can
reproject every input onto. The implementation defers to
``rasterio.warp.reproject`` — the same code path GDAL itself uses —
so the numerics match every other geospatial toolchain in the wild.

Two entry points:

- :func:`reproject_to_grid` — warp one ``(array, profile)`` pair onto a
  :class:`TargetGrid`. Picks a per-dtype default resampling kernel
  (bilinear for continuous floats, nearest for categorical ints,
  average for explicit downsampling).
- :func:`align_to_grid` — convenience for the common pattern: take a
  bag of heterogeneously-gridded sources, reproject every one onto a
  common target (either supplied explicitly or derived from the first
  source), return the aligned arrays.

Mathematical primer (full derivation in
``wiki/Reprojection-and-target-grids.md``):

For each *destination* pixel ``(i, j)`` on a target grid with affine
transform ``T_dst``, GDAL computes its world coordinate
``(x, y) = T_dst · (j + 0.5, i + 0.5)``, transforms it through the
``src_crs → dst_crs`` reprojector to land in the source CRS, then
inverts the source transform ``T_src⁻¹`` to find the floating-point
source-pixel coordinate ``(u, v)``. The destination value is then a
weighted combination of source pixels around ``(u, v)`` chosen by the
resampling kernel — for nearest-neighbour it's the single closest
pixel, for bilinear it's a 2×2 window with bilinear weights, for
average (used when downsampling) it's the mean over every source
pixel whose centre falls within the destination footprint.

Picking the right kernel matters for soil texture work:

- Continuous **fractions** (clay %, SOC g/kg) want **bilinear** when
  upsampling so polygon classification doesn't fall onto boundary
  artifacts, or **average** when downsampling so the mean is
  preserved.
- **Categorical** rasters (WorldCover class codes, modal-class
  outputs) want **nearest** — averaging class codes is meaningless.
- **Quantile** bands (Q0.05, Q0.95) follow the same rule as their
  mean: average / bilinear, never nearest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pedotri.errors import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


#: Resampling kernels accepted by :func:`reproject_to_grid`. These map
#: 1-to-1 onto the rasterio / GDAL enum of the same name; we keep them
#: as strings so callers don't need to import rasterio just to pass an
#: option.
RESAMPLING_METHODS: tuple[str, ...] = (
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "min",
    "max",
    "sum",
)


@dataclass(frozen=True, slots=True)
class TargetGrid:
    """A georeferenced raster grid that other inputs can be aligned to.

    Attributes mirror the ``crs`` / ``transform`` / ``width`` / ``height``
    fields of a rasterio profile, which is what :func:`reproject_to_grid`
    forwards to ``rasterio.warp.reproject``. The class is frozen so a
    grid can be used as a cache key once it's been built.

    The grid is fully described by four fields — there is no implicit
    extra state. ``bounds`` (left, bottom, right, top in CRS units),
    ``shape`` (height, width), and ``pixel_size`` (x-res, |y-res|) are
    derived properties for convenience.
    """

    crs: Any
    transform: Any
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise InvalidInputError(
                f"TargetGrid must have positive width / height; "
                f"got width={self.width}, height={self.height}."
            )

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> TargetGrid:
        """Build a :class:`TargetGrid` from a rasterio profile dict.

        The profile must contain ``crs``, ``transform``, ``width``,
        ``height``. Any other fields (dtype, nodata, compression, …)
        are ignored — the grid only encodes geometry.
        """
        missing = [k for k in ("crs", "transform", "width", "height") if k not in profile]
        if missing:
            raise InvalidInputError(
                f"profile is missing required keys for TargetGrid: {missing!r}."
            )
        return cls(
            crs=profile["crs"],
            transform=profile["transform"],
            width=int(profile["width"]),
            height=int(profile["height"]),
        )

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        *,
        resolution: float,
        crs: str = "EPSG:4326",
    ) -> TargetGrid:
        """Build a grid with a square pixel size from a bounding box.

        ``bounds`` is ``(left, bottom, right, top)`` in CRS units;
        ``resolution`` is the pixel side in the same units. The grid
        is sized to cover the bounds, rounded up to whole pixels.
        """
        try:
            from rasterio.transform import from_origin
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TargetGrid.from_bounds requires rasterio. "
                "Install with: pip install 'pedotri[raster]'"
            ) from exc

        left, bottom, right, top = bounds
        if right <= left or top <= bottom:
            raise InvalidInputError(
                f"bounds must satisfy left<right and bottom<top; got {bounds!r}."
            )
        if resolution <= 0:
            raise InvalidInputError(f"resolution must be positive; got {resolution}.")
        width = int(np.ceil((right - left) / resolution))
        height = int(np.ceil((top - bottom) / resolution))
        transform = from_origin(left, top, resolution, resolution)
        return cls(crs=crs, transform=transform, width=width, height=height)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """``(|x-pixel|, |y-pixel|)`` in CRS units."""
        t = self.transform
        return (float(abs(t.a)), float(abs(t.e)))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in CRS units."""
        t = self.transform
        left = float(t.c)
        top = float(t.f)
        right = left + self.width * float(t.a)
        bottom = top + self.height * float(t.e)
        # transform.e is negative for north-up rasters; normalise so
        # callers always get min/max-ordered tuples.
        if bottom > top:
            bottom, top = top, bottom
        if right < left:
            left, right = right, left
        return (left, bottom, right, top)

    def to_profile(self, *, dtype: str = "float32", nodata: Any = None) -> dict[str, Any]:
        """Return a rasterio profile dict aligned with this grid.

        ``dtype`` / ``nodata`` are convenience defaults for writers;
        every other rasterio profile field is left to the caller to
        set explicitly.
        """
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": 1,
            "dtype": dtype,
            "nodata": nodata,
        }

    def matches(self, profile: dict[str, Any]) -> bool:
        """True when ``profile`` already describes this grid.

        Compares CRS, transform, width, and height with rasterio's own
        equality so callers can short-circuit no-op reprojections.
        """
        try:
            return (
                self.crs == profile.get("crs")
                and self.transform == profile.get("transform")
                and int(profile.get("width", -1)) == self.width
                and int(profile.get("height", -1)) == self.height
            )
        except (TypeError, ValueError):
            return False


def reproject_to_grid(
    array: np.ndarray,
    source_profile: dict[str, Any],
    target: TargetGrid,
    *,
    resampling: str | None = None,
    src_nodata: Any = None,
    dst_nodata: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Warp one raster onto ``target``.

    Args:
        array: 2-D ndarray of source pixel values.
        source_profile: Rasterio profile describing ``array`` (must
            include ``crs`` and ``transform``).
        target: Destination grid.
        resampling: Override the default resampling kernel — one of
            :data:`RESAMPLING_METHODS`. When ``None`` (default),
            :func:`_default_resampling` picks ``bilinear`` for float
            inputs, ``nearest`` for integer inputs, and ``average``
            when the target pixel is materially coarser than the
            source pixel (downsampling).
        src_nodata: Override the source nodata sentinel. Defaults to
            ``source_profile.get("nodata")``.
        dst_nodata: Override the destination nodata sentinel. Defaults
            to ``src_nodata``.

    Returns:
        ``(warped, profile)`` — the warped array on the target grid
        and a profile aligned with it (``dtype`` preserved from the
        source).

    Raises:
        InvalidInputError: For unknown resampling kernel or mismatched
            source array / profile shapes.
        ImportError: When the optional ``[raster]`` extra (rasterio)
            isn't installed.
    """
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reproject_to_grid requires rasterio. Install with: pip install 'pedotri[raster]'"
        ) from exc

    array = np.asarray(array)
    if array.ndim != 2:
        raise InvalidInputError(f"reproject_to_grid expects a 2-D array; got shape {array.shape}.")

    expected_h = int(source_profile.get("height") or array.shape[0])
    expected_w = int(source_profile.get("width") or array.shape[1])
    if (expected_h, expected_w) != array.shape:
        raise InvalidInputError(
            f"source_profile shape {(expected_h, expected_w)} does not "
            f"match array shape {array.shape}."
        )

    if target.matches(source_profile):
        # Already on the target grid — return verbatim. Saves the
        # rasterio.warp round-trip and preserves byte identity for
        # cache hits on the audit-trail path.
        out_profile = dict(source_profile)
        out_profile.update(target.to_profile(dtype=str(array.dtype)))
        return array.copy(), out_profile

    resampling_name = resampling or _default_resampling(array.dtype, source_profile, target)
    if resampling_name not in RESAMPLING_METHODS:
        raise InvalidInputError(
            f"Unknown resampling kernel {resampling_name!r}. "
            f"Pick one of {list(RESAMPLING_METHODS)!r}."
        )
    rs_enum = getattr(Resampling, resampling_name)

    src_nodata_eff = src_nodata if src_nodata is not None else source_profile.get("nodata")
    # Drop the nodata sentinel if the profile is inconsistent with the
    # array dtype (e.g. an old profile carrying ``-9999.0`` got reused
    # with a uint8 categorical input). Better to silently ignore than
    # to make rasterio.warp reject the call.
    if src_nodata_eff is not None and not _is_nodata_compatible(src_nodata_eff, array.dtype):
        src_nodata_eff = None
    dst_nodata_eff = dst_nodata if dst_nodata is not None else src_nodata_eff

    out = np.empty((target.height, target.width), dtype=array.dtype)
    reproject(
        source=array,
        destination=out,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=src_nodata_eff,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=dst_nodata_eff,
        resampling=rs_enum,
    )
    out_profile = target.to_profile(dtype=str(array.dtype), nodata=dst_nodata_eff)
    return out, out_profile


def align_to_grid(
    sources: Iterable[tuple[np.ndarray, dict[str, Any]]],
    *,
    target: TargetGrid | None = None,
    resampling: Sequence[str | None] | None = None,
) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], TargetGrid]:
    """Reproject every source onto a common :class:`TargetGrid`.

    Args:
        sources: Iterable of ``(array, profile)`` pairs.
        target: Destination grid. If ``None``, derive it from the
            first source's profile.
        resampling: Optional per-source resampling override. When
            ``None`` (default), each source picks its own default via
            :func:`_default_resampling`. Otherwise must have the same
            length as ``sources``.

    Returns:
        ``(aligned, target)`` — list of ``(array, profile)`` pairs all
        on the same grid, plus the resolved :class:`TargetGrid`.
    """
    source_list = list(sources)
    if not source_list:
        raise InvalidInputError("align_to_grid: no sources supplied.")
    if target is None:
        target = TargetGrid.from_profile(source_list[0][1])
    resampling_list: list[str | None]
    if resampling is None:
        resampling_list = [None] * len(source_list)
    else:
        resampling_list = list(resampling)
        if len(resampling_list) != len(source_list):
            raise InvalidInputError(
                f"resampling length {len(resampling_list)} must match "
                f"number of sources ({len(source_list)})."
            )
    aligned: list[tuple[np.ndarray, dict[str, Any]]] = []
    for (arr, prof), method in zip(source_list, resampling_list, strict=True):
        warped, out_prof = reproject_to_grid(arr, prof, target, resampling=method)
        aligned.append((warped, out_prof))
    return aligned, target


# --- internals -----------------------------------------------------------


def _is_nodata_compatible(nodata: Any, dtype: np.dtype) -> bool:
    """True when ``nodata`` is representable in the given numpy dtype."""
    try:
        if np.issubdtype(dtype, np.floating):
            float(nodata)
            return True
        nodata_int = int(nodata)
    except (TypeError, ValueError, OverflowError):
        return False
    info = np.iinfo(dtype)
    return info.min <= nodata_int <= info.max


def _default_resampling(
    dtype: np.dtype,
    source_profile: dict[str, Any],
    target: TargetGrid,
) -> str:
    """Pick a resampling kernel based on dtype + scale change.

    The conservative choices for soil-texture / SoilGrids workloads:

    - Categorical (integer) inputs always use **nearest** — averaging
      class codes is nonsense.
    - Continuous (float) inputs use **average** when the destination
      pixel is materially coarser than the source (≥ 1.5× area-wise)
      so the regional mean is preserved; otherwise **bilinear** so
      upsampling doesn't introduce blocky artifacts.

    Override by passing ``resampling=`` explicitly when this heuristic
    doesn't match your workflow (e.g. you want bilinear on a Q0.95
    band even when downsampling because you want the local maximum
    behaviour).
    """
    if np.issubdtype(dtype, np.integer):
        return "nearest"
    src_t = source_profile.get("transform")
    if src_t is None:
        return "bilinear"
    src_px_x = abs(float(src_t.a))
    src_px_y = abs(float(src_t.e))
    dst_px_x, dst_px_y = target.pixel_size
    # Areal scale ratio: dst pixel size relative to source pixel size,
    # in both axes. Larger than 1 means we're downsampling.
    ratio = (dst_px_x / src_px_x) * (dst_px_y / src_px_y) if src_px_x and src_px_y else 1.0
    if ratio >= 1.5:
        return "average"
    return "bilinear"


__all__ = [
    "RESAMPLING_METHODS",
    "TargetGrid",
    "align_to_grid",
    "reproject_to_grid",
]
