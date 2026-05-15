"""Regional aggregation with uncertainty propagation.

Workflows that need a regional answer — SOC stock per village, mean
clay over a commune for an agronomic decision, average bulk density on
a farm — typically aggregate pixel values from a SoilGrids-style
raster over an area of interest. Doing this on the *mean* band alone
throws away the published per-pixel uncertainty and gives a single
point estimate that nobody can put error bars on.

This module performs the aggregation in sample space:

1. Restrict to the relevant pixels (region polygon ∩ optional land-cover
   mask ∩ valid raster values).
2. For each property, draw ``n_samples`` realizations per pixel from
   the per-pixel distribution implied by the (mean, Q05, Q95) bands.
3. For each draw, compute the regional aggregate (currently the area
   mean).
4. Report mean / std / Q05 / Q50 / Q95 of the resulting *distribution
   over the regional aggregate*.

The output keeps the raw samples (:attr:`AggregateDistribution.samples`)
so derived quantities — SOC stock, depth-weighted averages, anything
the caller cares to express as a function of the property aggregates —
inherit the full uncertainty via :meth:`ZonalAggregate.combine`.

Caveat: 0.3.x treats pixels as independent draws. SoilGrids residuals
are strongly spatially autocorrelated, so the reported regional
uncertainty is a **lower bound** on the true posterior width. A
spatial-correlation mode is on the 0.4 roadmap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pedotri.errors import InvalidInputError
from pedotri.grid import TargetGrid, reproject_to_grid
from pedotri.uncertainty import Quantiles, sigma_from_quantiles

if TYPE_CHECKING:
    from collections.abc import Callable, KeysView, Mapping

_AGG_WORK_WARN: int = 100_000_000
_AGG_WORK_BLOCK: int = 1_000_000_000


@dataclass(slots=True, frozen=True)
class AggregateDistribution:
    """Posterior distribution over a regional aggregate of one property.

    Samples are stored verbatim so callers can compute any further
    statistic (custom quantiles, histograms, covariance with another
    distribution sharing the same draw indices).
    """

    samples: np.ndarray
    n_pixels_used: int
    name: str | None = None
    provenance: Any | None = None

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        if self.samples.size < 2:
            return 0.0
        return float(self.samples.std(ddof=1))

    @property
    def q05(self) -> float:
        return float(np.quantile(self.samples, 0.05))

    @property
    def q50(self) -> float:
        return float(np.quantile(self.samples, 0.50))

    @property
    def q95(self) -> float:
        return float(np.quantile(self.samples, 0.95))

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    def summary(self) -> dict[str, float | int | str | None]:
        return {
            "name": self.name,
            "n_pixels_used": self.n_pixels_used,
            "mean": self.mean,
            "std": self.std,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
        }


@dataclass(slots=True, frozen=True)
class ZonalAggregate:
    """Bundle of per-property :class:`AggregateDistribution`s sharing a mask.

    Distributions inside this bundle share both the same effective mask
    *and* the same draw indices (``samples[i]`` of property A
    corresponds to draw ``i`` of property B), so element-wise products
    in :meth:`combine` propagate uncertainty correctly across the
    aggregate distributions. Note that within a single pixel, the draws
    across properties are independent — SoilGrids' published quantiles
    are marginal, so any cross-property correlation in the raw data is
    not preserved.
    """

    properties: dict[str, AggregateDistribution]
    n_pixels_used: int
    mask_coverage: float
    region_pixels: int = field(default=0)
    provenance: Any | None = field(default=None)

    def __getitem__(self, key: str) -> AggregateDistribution:
        return self.properties[key]

    def __contains__(self, key: object) -> bool:
        return key in self.properties

    def keys(self) -> KeysView[str]:
        return self.properties.keys()

    def combine(
        self,
        fn: Callable[..., np.ndarray | float],
        *,
        name: str | None = None,
    ) -> AggregateDistribution:
        """Propagate the samples through an arbitrary user formula.

        The function is called with one keyword argument per property
        name *that the function actually asks for*. Any extra keyword
        parameters with defaults — typically scalar physical constants
        like soil depth or area — are used as-is. Numpy broadcasts
        scalars naturally across the sample arrays. Properties on the
        aggregate that the function doesn't accept are silently ignored.

        Example::

            # Aggregate has properties {soc, bd, ph} but the formula
            # only consumes soc and bd — that's fine.
            stock = agg.combine(
                lambda soc, bd, depth=0.30, area=village_area_ha: (
                    soc * bd * depth * area * 0.1
                ),  # convert to t/ha
            )
            stock.mean, stock.q05, stock.q95

        Functions that use ``**kwargs`` to absorb everything also work;
        they receive every property as a keyword argument.

        The returned distribution carries the same ``n_pixels_used`` as
        the parent aggregate; ``name`` defaults to ``None`` so callers
        can tag the derived quantity explicitly.
        """
        sample_kwargs = _filter_kwargs_for_fn(
            fn, {nm: d.samples for nm, d in self.properties.items()}
        )
        result = fn(**sample_kwargs)
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim == 0:
            raise InvalidInputError(
                "combine() formula must return an array of samples, not a scalar. "
                "Make sure the function operates element-wise on its inputs."
            )
        derived_prov: Any | None = None
        if self.provenance is not None:
            from pedotri.audit import Provenance

            derived_prov = Provenance(
                operation="pedotri.zonal.ZonalAggregate.combine",
                parameters={
                    "fn": getattr(fn, "__name__", repr(fn)),
                    "name": name,
                    "consumed_properties": sorted(sample_kwargs.keys()),
                },
                upstream=[self.provenance],
            )
        return AggregateDistribution(
            samples=arr,
            n_pixels_used=self.n_pixels_used,
            name=name,
            provenance=derived_prov,
        )


def zonal_aggregate(
    *,
    region: Any,
    properties: dict[str, Any],
    mask: Any | None = None,
    mask_include: list[int] | None = None,
    profile: dict[str, Any] | None = None,
    target_grid: TargetGrid | None = None,
    n_samples: int = 1000,
    seed: Any = None,
    confirm: bool = False,
    correlation_range: float | None = None,
    correlation_model: str = "exponential",
) -> ZonalAggregate:
    """Aggregate property rasters over an AOI with uncertainty propagation.

    All arguments are keyword-only — there is no natural positional
    ordering for the inputs and explicit names are kinder to readers of
    the resulting call sites.

    Args:
        region: One of:

            - a boolean ``ndarray`` of the same shape as the property
              rasters (``True`` = inside AOI). No georeferencing needed.
            - a shapely Polygon / MultiPolygon (or any object exposing
              ``__geo_interface__``). Rasterized onto the property grid
              via ``rasterio.features.rasterize``; requires the
              ``[raster,vector]`` extras and a valid ``profile``.
        properties: Mapping ``name -> spec`` where ``spec`` describes the
            property's mean band and (optional) uncertainty. Accepted
            shapes per spec, in order of preference:

            - ``dict``: ``{"mean": …, "uncertainty": Quantiles(q05, q95)}``
              (or ``"sigma"`` instead of ``"uncertainty"``). The
              recommended form — explicit and forward-compatible.
            - ``dict`` legacy: ``{"mean": …, "q05": …, "q95": …}``.
              Equivalent to the ``uncertainty=Quantiles(...)`` form.
            - 2-tuple ``(mean, uncertainty)`` where ``uncertainty`` is a
              :class:`Quantiles` or a σ ndarray.
            - 3-tuple ``(mean, q05, q95)`` — deprecated; emits a
              :class:`DeprecationWarning`. Use the dict or 2-tuple
              forms in new code.

            Each value (mean / q05 / q95 / sigma) may itself be an
            ``ndarray`` or a path to a single-band GeoTIFF.
        mask: Optional land-cover or any other classification raster
            (ndarray or path) restricting the aggregation to a subset
            of the AOI pixels. When ``mask_include`` is set, the mask
            is treated as categorical and only pixels whose value is
            in the allowlist are kept. When ``mask_include`` is
            ``None``, the mask is treated as boolean.
        mask_include: Class allowlist for categorical masks (e.g.
            ``[40]`` for the cropland class in ESA WorldCover).
        profile: Rasterio profile (must contain ``crs``, ``transform``,
            ``width``, ``height``). Required when ``region`` is a
            geometry; ignored when ``region`` is already a boolean array.
        target_grid: Optional :class:`~pedotri.grid.TargetGrid` to
            align every path-based input onto. When ``None``, the
            working grid is taken from the first property's mean band
            (or ``profile=`` if provided), matching the 0.3 behaviour.
            When set, *path-based* property bands and the mask raster
            are reprojected onto this grid via
            :func:`pedotri.grid.reproject_to_grid` with per-data-type
            resampling defaults. Pre-loaded ``ndarray`` inputs are
            still expected to already be on the working grid.
        n_samples: Number of Monte-Carlo draws per pixel per property.
        seed: Optional int or ``np.random.Generator`` for reproducibility.
        confirm: Pass ``True`` to bypass the pre-flight cost guard.
        correlation_range: Spatial correlation length for the per-pixel
            uncertainty. When ``None`` (default), pixels are sampled
            **independently** — same behaviour as 0.3, regional
            Q05/Q95 is a documented lower bound. When ``> 0``, the
            sampler is replaced by the FFT-based correlated-field
            sampler in :mod:`pedotri.uncertainty`, which draws whole
            2-D Gaussian fields with the chosen correlation function.
            Expressed in **CRS units** when a profile / target grid
            is in play (e.g. metres for a UTM grid, degrees for
            EPSG:4326), or pixels when neither is. The conversion uses
            the target grid's pixel size.
        correlation_model: ``"exponential"`` (default — matches the
            Matérn ν=½ assumption typical in digital-soil-mapping
            kriging), ``"gaussian"`` (smooth fields), or
            ``"spherical"`` (compactly supported beyond
            ``correlation_range``). See
            :func:`pedotri.uncertainty.sample_correlated_field` for
            the kernel formulas and references.

    Returns:
        A :class:`ZonalAggregate` with one
        :class:`AggregateDistribution` per property, plus
        :attr:`mask_coverage` (fraction of region pixels kept after the
        valid + mask intersection) and :attr:`n_pixels_used`.

    Raises:
        InvalidInputError: For shape / profile mismatches, missing
            ``profile`` with a geometry region, contradictory mask
            arguments, or a workload above the hard threshold without
            ``confirm=True``.
    """
    if not properties:
        raise InvalidInputError("properties must be a non-empty mapping.")
    if n_samples < 1:
        raise InvalidInputError(f"n_samples must be >= 1, got {n_samples}.")

    prop_arrays, raster_profile = _load_property_stack(properties, profile, target_grid)
    raster_shape = next(iter(prop_arrays.values()))[0].shape

    region_mask = _resolve_region_mask(region, raster_shape, raster_profile)
    n_region = int(region_mask.sum())

    mask_arr = _resolve_mask_raster(mask, mask_include, raster_shape, raster_profile)
    valid_data = np.ones(raster_shape, dtype=bool)
    for mean, sigma in prop_arrays.values():
        valid_data &= np.isfinite(mean) & np.isfinite(sigma) & (sigma >= 0)

    effective = region_mask & mask_arr & valid_data
    n_eff = int(effective.sum())
    coverage = (n_eff / n_region) if n_region > 0 else 0.0

    if n_eff == 0:
        empty_samples = np.array([], dtype=np.float64)
        return ZonalAggregate(
            properties={
                nm: AggregateDistribution(samples=empty_samples, n_pixels_used=0, name=nm)
                for nm in prop_arrays
            },
            n_pixels_used=0,
            mask_coverage=coverage,
            region_pixels=n_region,
        )

    work = n_eff * n_samples * len(prop_arrays)
    if work > _AGG_WORK_BLOCK and not confirm:
        raise InvalidInputError(
            f"Zonal aggregation workload {work:,} exceeds the hard threshold "
            f"({_AGG_WORK_BLOCK:,}). Reduce n_samples, mask more aggressively, "
            "or pass confirm=True to override."
        )
    if work > _AGG_WORK_WARN and not confirm:
        warnings.warn(
            f"Zonal aggregation workload is large ({work:,} pixel-sample-properties). "
            "Consider lowering n_samples or chunking the AOI.",
            UserWarning,
            stacklevel=2,
        )

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    # Translate ``correlation_range`` from CRS units to pixels using the
    # target grid's pixel size, if available. Falls back to the raw
    # number (interpreted as pixels) when no grid info is in play.
    range_pixels: float | None = None
    if correlation_range is not None and correlation_range > 0:
        range_pixels = _correlation_range_in_pixels(correlation_range, raster_profile)
    out_props: dict[str, AggregateDistribution] = {}
    for name, (mean_arr, sigma_arr) in prop_arrays.items():
        if range_pixels is None:
            sub_mean = mean_arr[effective]
            sub_sigma = sigma_arr[effective]
            regional_samples = _aggregate_property(sub_mean, sub_sigma, n_samples, rng)
        else:
            regional_samples = _aggregate_property_correlated(
                mean_arr,
                sigma_arr,
                effective,
                n_samples,
                rng,
                range_pixels=range_pixels,
                model=correlation_model,
            )
        out_props[name] = AggregateDistribution(
            samples=regional_samples,
            n_pixels_used=n_eff,
            name=name,
        )

    aggregate_prov = _build_zonal_provenance(
        properties_spec=properties,
        mask=mask,
        mask_include=mask_include,
        n_samples=n_samples,
        seed=seed,
        correlation_range=correlation_range,
        correlation_model=correlation_model,
        n_eff=n_eff,
        n_region=n_region,
    )
    # Plumb the same Provenance back into each AggregateDistribution so
    # downstream ``combine()`` propagation stays traceable.
    out_props = {
        name: AggregateDistribution(
            samples=dist.samples,
            n_pixels_used=dist.n_pixels_used,
            name=dist.name,
            provenance=aggregate_prov,
        )
        for name, dist in out_props.items()
    }
    return ZonalAggregate(
        properties=out_props,
        n_pixels_used=n_eff,
        mask_coverage=coverage,
        region_pixels=n_region,
        provenance=aggregate_prov,
    )


# --- internals -----------------------------------------------------------


def _load_property_stack(
    properties: dict[str, Any],
    profile: dict[str, Any] | None,
    target_grid: TargetGrid | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any] | None]:
    """Coerce each property's spec into a canonical ``(mean, sigma)`` pair.

    Supported spec shapes (per :func:`zonal_aggregate`):

    - ``dict`` with ``"mean"`` plus one of ``"uncertainty"`` /
      ``"sigma"`` / (``"q05"`` and ``"q95"``).
    - 2-tuple ``(mean, uncertainty)``.
    - 3-tuple ``(mean, q05, q95)`` — deprecated, emits a warning.

    Each value may be an ``ndarray`` or a path to a single-band GeoTIFF.
    The first encountered path supplies the working profile if the
    caller didn't pass one.
    """
    raster_profile = dict(profile) if profile is not None else None
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reference_shape: tuple[int, ...] | None = None
    # ``effective_grid`` is the working grid every path-based input
    # gets aligned to. When the caller passed ``target_grid=`` we use
    # it from the start; otherwise the first path-loaded profile sets
    # it (matching 0.3 behaviour). ndarray inputs are assumed to be
    # already aligned.
    effective_grid: TargetGrid | None = target_grid
    if effective_grid is None and raster_profile is not None:
        try:
            effective_grid = TargetGrid.from_profile(raster_profile)
        except InvalidInputError:
            effective_grid = None

    def _to_array(name: str, kind: str, item: Any) -> np.ndarray:
        nonlocal raster_profile, reference_shape, effective_grid
        if isinstance(item, np.ndarray):
            arr = np.asarray(item, dtype=np.float64)
        elif isinstance(item, (str, Path)):
            raw_arr, raw_prof = _read_raster_band(item)
            if effective_grid is None:
                # First path encountered sets the working grid.
                effective_grid = TargetGrid.from_profile(raw_prof)
                raster_profile = dict(raw_prof)
                arr = raw_arr
            elif effective_grid.matches(raw_prof):
                arr = raw_arr
                if raster_profile is None:
                    raster_profile = dict(raw_prof)
            else:
                # Reproject this band onto the working grid. The
                # default resampling kernel is picked by
                # ``pedotri.grid._default_resampling`` based on dtype
                # (categorical → nearest, continuous + downsample →
                # average, otherwise → bilinear) so SoilGrids quantile
                # bands behave like their mean.
                arr, out_prof = reproject_to_grid(raw_arr, raw_prof, effective_grid)
                arr = np.asarray(arr, dtype=np.float64)
                if raster_profile is None:
                    raster_profile = dict(out_prof)
        else:
            raise InvalidInputError(
                f"properties[{name!r}].{kind} must be an ndarray or a GeoTIFF path; "
                f"got {type(item).__name__}."
            )
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise InvalidInputError(
                f"All property rasters must share shape; "
                f"{name!r} {kind} has shape {arr.shape}, expected {reference_shape}."
            )
        return arr

    for name, spec in properties.items():
        arrays[name] = _spec_to_mean_sigma(name, spec, _to_array)

    return arrays, raster_profile


def _spec_to_mean_sigma(
    name: str,
    spec: Any,
    to_array: Callable[[str, str, Any], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on the four supported spec shapes."""
    if isinstance(spec, dict):
        if "mean" not in spec:
            raise InvalidInputError(
                f"properties[{name!r}]: dict spec needs a 'mean' key. Got {list(spec)!r}."
            )
        mean_arr = to_array(name, "mean", spec["mean"])
        if "uncertainty" in spec:
            sigma_arr = _resolve_property_uncertainty(
                name,
                spec["uncertainty"],
                mean_arr.shape,
                to_array,
            )
        elif "sigma" in spec:
            sigma_arr = to_array(name, "sigma", spec["sigma"])
        elif "q05" in spec and "q95" in spec:
            q05_arr = to_array(name, "q05", spec["q05"])
            q95_arr = to_array(name, "q95", spec["q95"])
            sigma_arr = np.asarray(sigma_from_quantiles(q05_arr, q95_arr))
        else:
            sigma_arr = np.zeros(mean_arr.shape, dtype=np.float64)
        return mean_arr, sigma_arr

    if isinstance(spec, tuple):
        if len(spec) == 3:
            warnings.warn(
                "properties[name] = (mean, q05, q95) is deprecated; use "
                "{'mean': ..., 'uncertainty': Quantiles(q05, q95)} or "
                "(mean, Quantiles(q05, q95)) instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            mean_arr = to_array(name, "mean", spec[0])
            q05_arr = to_array(name, "q05", spec[1])
            q95_arr = to_array(name, "q95", spec[2])
            sigma_arr = np.asarray(sigma_from_quantiles(q05_arr, q95_arr))
            return mean_arr, sigma_arr
        if len(spec) == 2:
            mean_arr = to_array(name, "mean", spec[0])
            sigma_arr = _resolve_property_uncertainty(name, spec[1], mean_arr.shape, to_array)
            return mean_arr, sigma_arr

    raise InvalidInputError(
        f"properties[{name!r}] has unsupported spec shape {type(spec).__name__}; "
        "expected a dict, 2-tuple (mean, uncertainty), or deprecated "
        "3-tuple (mean, q05, q95)."
    )


def _resolve_property_uncertainty(
    name: str,
    value: Any,
    target_shape: tuple[int, ...],
    to_array: Callable[[str, str, Any], np.ndarray],
) -> np.ndarray:
    """Translate a property's ``uncertainty`` spec into a σ array."""
    if isinstance(value, Quantiles):
        q05_arr = to_array(name, "uncertainty.q05", value.q05)
        q95_arr = to_array(name, "uncertainty.q95", value.q95)
        return np.asarray(sigma_from_quantiles(q05_arr, q95_arr))
    if isinstance(value, np.ndarray):
        return to_array(name, "uncertainty", value)
    if isinstance(value, (str, Path)):
        return to_array(name, "uncertainty", value)
    if isinstance(value, (int, float)):
        sigma = float(value)
        if sigma < 0:
            raise InvalidInputError(
                f"properties[{name!r}].uncertainty σ must be non-negative; got {sigma}."
            )
        return np.full(target_shape, sigma, dtype=np.float64)
    raise InvalidInputError(
        f"properties[{name!r}].uncertainty must be Quantiles, ndarray, "
        f"path, or scalar σ; got {type(value).__name__}."
    )


def _read_raster_band(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band GeoTIFF into (array, profile)."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover — exercised in CI matrix
        raise ImportError(
            "Reading property rasters from disk requires rasterio. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), dict(src.profile)


def _resolve_region_mask(
    region: Any, raster_shape: tuple[int, ...], profile: dict[str, Any] | None
) -> np.ndarray:
    """Turn any supported ``region`` input into a boolean raster-shape mask."""
    if isinstance(region, np.ndarray) and region.dtype == bool:
        if region.shape != raster_shape:
            raise InvalidInputError(
                f"Boolean region shape {region.shape} does not match property raster "
                f"shape {raster_shape}."
            )
        return region
    if isinstance(region, np.ndarray):
        raise InvalidInputError(
            "ndarray region must be of dtype bool; pass an explicit boolean mask."
        )
    if profile is None:
        raise InvalidInputError(
            "Rasterizing a geometry region requires a rasterio profile "
            "(pass profile= or use file-based property inputs)."
        )
    return _rasterize_geometry(region, raster_shape, profile)


def _rasterize_geometry(
    geom: Any, raster_shape: tuple[int, ...], profile: dict[str, Any]
) -> np.ndarray:
    """Burn a polygon onto the property raster grid."""
    try:
        from rasterio import features
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Rasterizing a polygon region requires rasterio. "
            "Install with: pip install 'pedotri[raster]'"
        ) from exc
    if hasattr(geom, "__geo_interface__"):
        shapes = [(geom.__geo_interface__, 1)]
    elif isinstance(geom, dict):
        shapes = [(geom, 1)]
    else:
        raise InvalidInputError(
            "region must be a boolean ndarray, a shapely geometry, or a "
            "__geo_interface__ / GeoJSON dict."
        )
    if "transform" not in profile:
        raise InvalidInputError("profile is missing a 'transform' for rasterization.")
    mask = features.rasterize(
        shapes,
        out_shape=raster_shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return np.asarray(mask.astype(bool))


def _resolve_mask_raster(
    mask: Any | None,
    mask_include: list[int] | None,
    raster_shape: tuple[int, ...],
    target_profile: dict[str, Any] | None,
) -> np.ndarray:
    """Turn the user's mask input into a boolean keep-mask.

    When ``mask`` is supplied as a GeoTIFF path *and* its native grid
    differs from the working ``target_profile``, the raster is
    reprojected onto the property grid (with categorical / nearest
    resampling by default — see :func:`pedotri.grid._default_resampling`).
    ndarray masks are still expected to already be on the property
    grid; pass them pre-aligned.
    """
    if mask is None:
        return np.ones(raster_shape, dtype=bool)

    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        raw_arr, raw_prof = _read_raster_band(mask)
        if target_profile is not None:
            try:
                grid = TargetGrid.from_profile(target_profile)
            except InvalidInputError:
                grid = None
            if grid is not None and not grid.matches(raw_prof):
                # Preserve the source dtype so integer class codes stay
                # categorical — the helper will pick `nearest` from the
                # dtype heuristic.
                src = np.asarray(raw_arr, dtype=np.int64 if mask_include else raw_arr.dtype)
                src_prof = dict(raw_prof)
                src_prof["dtype"] = str(src.dtype)
                arr, _ = reproject_to_grid(src, src_prof, grid)
            else:
                arr = raw_arr
        else:
            arr = raw_arr

    arr = np.asarray(arr)
    if arr.shape != raster_shape:
        raise InvalidInputError(
            f"mask shape {arr.shape} does not match property raster shape {raster_shape}."
        )

    if mask_include is None:
        return arr.astype(bool)

    allowed = np.asarray(list(mask_include))
    return np.isin(arr, allowed)


def _filter_kwargs_for_fn(fn: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    """Return only the candidates that ``fn``'s signature can accept.

    Uses :func:`inspect.signature` to enumerate ``fn``'s named parameters
    and pass through anything matching. If ``fn`` declares ``**kwargs``,
    every candidate is passed (the function can decide what to do).
    Functions with unintrospectable signatures (some C extensions) are
    treated as accepting everything.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(candidates)
    has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return dict(candidates)
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in candidates.items() if k in accepted}


def _aggregate_property(
    mean: np.ndarray,
    sigma: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-property regional-mean sampler.

    Draws ``(n_pixels, n_samples)`` independent normals — clipped at
    zero from below to keep stocks / concentrations physical — then
    averages across pixels for each draw.
    """
    n_pixels = mean.size
    samples = rng.normal(
        loc=mean[:, None],
        scale=sigma[:, None],
        size=(n_pixels, n_samples),
    )
    np.maximum(samples, 0.0, out=samples)
    return np.asarray(samples.mean(axis=0), dtype=np.float64)


def _aggregate_property_correlated(
    mean_grid: np.ndarray,
    sigma_grid: np.ndarray,
    effective_mask: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    range_pixels: float,
    model: str,
) -> np.ndarray:
    """Correlated regional-mean sampler — draws full 2-D fields, averages over mask.

    The independent sampler in :func:`_aggregate_property` shrinks
    σ_regional by ``√n_pixels`` (CLT) because every draw is independent
    of its neighbours; that's the documented "lower bound on regional
    uncertainty" caveat from 0.3. With a correlated sampler the
    neighbours share predictive uncertainty, so the regional mean
    inherits a much larger spread — closer to the published
    SoilGrids-residual structure.

    We use the FFT-based circulant-embedding sampler from
    :mod:`pedotri.uncertainty` to draw whole ``(H, W)`` Gaussian fields,
    clip at zero, then take the mean over the AOI's effective pixels.
    """
    from pedotri.uncertainty import sample_correlated_field

    fields = sample_correlated_field(
        mean_grid,
        sigma_grid,
        correlation_range=range_pixels,
        n_samples=n_samples,
        rng=rng,
        model=model,
    )
    np.maximum(fields, 0.0, out=fields)
    flat = fields.reshape(n_samples, -1)
    mask_flat = effective_mask.reshape(-1)
    return np.asarray(flat[:, mask_flat].mean(axis=1), dtype=np.float64)


def _build_zonal_provenance(
    *,
    properties_spec: dict[str, Any],
    mask: Any,
    mask_include: list[int] | None,
    n_samples: int,
    seed: Any,
    correlation_range: float | None,
    correlation_model: str,
    n_eff: int,
    n_region: int,
) -> Any:
    """ISO 14067 provenance for one ``zonal_aggregate`` invocation."""
    from pedotri.audit import Provenance

    # Collect upstream provenance from any source-derived inputs the
    # caller threaded into the call: an ndarray with a .provenance
    # attribute (e.g. WorldCoverAOI.array → no, that's still a raw
    # ndarray) — for now we just record the spec shape and the
    # numerical parameters. Downstream callers can attach explicit
    # upstream records via Provenance.upstream at the audit-trail level.
    seed_int: int | None
    try:
        seed_int = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed_int = None
    return Provenance(
        operation="pedotri.zonal.zonal_aggregate",
        parameters={
            "properties": sorted(properties_spec.keys()),
            "mask_supplied": mask is not None,
            "mask_include": list(mask_include) if mask_include else None,
            "n_samples": int(n_samples),
            "correlation_range": correlation_range,
            "correlation_model": correlation_model if correlation_range is not None else None,
            "n_pixels_used": int(n_eff),
            "region_pixels": int(n_region),
        },
        seed=seed_int,
    )


def _correlation_range_in_pixels(
    correlation_range: float, raster_profile: dict[str, Any] | None
) -> float:
    """Translate ``correlation_range`` (CRS units) to pixels.

    When no profile is available (pure-ndarray inputs), the value is
    taken as already-in-pixels. With a profile, we divide by the
    target grid's mean pixel side (x and y averaged), so a 1 km
    correlation range on a 250 m grid becomes ~4 pixels regardless of
    which axis the user thinks in.
    """
    if raster_profile is None:
        return float(correlation_range)
    transform = raster_profile.get("transform")
    if transform is None:
        return float(correlation_range)
    try:
        px_x = abs(float(transform.a))
        px_y = abs(float(transform.e))
    except (AttributeError, TypeError, ValueError):
        return float(correlation_range)
    avg_px = (px_x + px_y) * 0.5
    if avg_px <= 0:
        return float(correlation_range)
    return float(correlation_range) / avg_px


def aggregate_depths(
    samples: dict[str, Any],
    weights: Mapping[str, float],
) -> tuple[Any, Any]:
    """Depth-weighted aggregation of per-depth (mean, Q05, Q95) bands.

    Recipe baked in by every multi-depth SoilGrids ingest pipeline:
    you want one 0–30 cm topsoil layer derived from the three native
    bands (0–5 cm, 5–15 cm, 15–30 cm) weighted by interval width.
    Doing it right means applying the same weights to **each**
    published band — mean, Q0.05, and Q0.95 — so the resulting
    aggregate still carries an uncertainty envelope.

    Args:
        samples: Mapping from depth label to a per-depth spec. Each
            spec accepts the same shapes as :func:`zonal_aggregate`'s
            ``properties`` values: dict with ``"mean"`` and either
            ``"uncertainty"`` (a :class:`Quantiles`) /
            ``"sigma"`` / (``"q05"`` + ``"q95"``); or a 2-tuple
            ``(mean, uncertainty)``; or a deprecated 3-tuple
            ``(mean, q05, q95)``. Values may be scalars or numpy
            arrays of identical shape.
        weights: Mapping from depth label to weight. Need not sum to
            1 — they are renormalised internally. Every depth in
            ``samples`` must appear here; extra weights are an error
            so callers don't silently drop a depth they meant to
            include.

    Returns:
        ``(mean, Quantiles(q05, q95))`` — both either scalars or
        arrays, matching the input shape.

    Raises:
        InvalidInputError: For mismatched depth labels, non-positive
            total weight, or mixed-shape inputs.

    Example::

        from pedotri.zonal import aggregate_depths
        from pedotri.uncertainty import Quantiles

        topsoil_mean, topsoil_q = aggregate_depths(
            samples={
                "0-5cm": {"mean": 24.0, "uncertainty": Quantiles(18.0, 30.0)},
                "5-15cm": {"mean": 22.0, "uncertainty": Quantiles(16.0, 28.0)},
                "15-30cm": {"mean": 19.0, "uncertainty": Quantiles(13.0, 25.0)},
            },
            weights={"0-5cm": 5, "5-15cm": 10, "15-30cm": 15},
        )
    """
    if not samples:
        raise InvalidInputError("aggregate_depths(samples=) is empty.")
    if set(samples) != set(weights):
        only_in_samples = sorted(set(samples) - set(weights))
        only_in_weights = sorted(set(weights) - set(samples))
        raise InvalidInputError(
            "Depth label mismatch between samples and weights. "
            f"In samples only: {only_in_samples!r}. In weights only: {only_in_weights!r}."
        )
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise InvalidInputError(f"Depth weights must sum to a positive value; got {weight_sum}.")

    reference_shape: tuple[int, ...] | None = None

    def _coerce(value: Any, kind: str, depth: str) -> np.ndarray:
        nonlocal reference_shape
        arr = np.asarray(value, dtype=np.float64)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise InvalidInputError(
                f"aggregate_depths: {depth}.{kind} has shape {arr.shape}; "
                f"expected {reference_shape}."
            )
        return arr

    mean_acc: np.ndarray | None = None
    q05_acc: np.ndarray | None = None
    q95_acc: np.ndarray | None = None

    for depth, spec in samples.items():
        weight = float(weights[depth]) / weight_sum
        mean_arr, q05_arr, q95_arr = _extract_mean_q05_q95(depth, spec, _coerce)
        if mean_acc is None:
            mean_acc = weight * mean_arr
            q05_acc = weight * q05_arr
            q95_acc = weight * q95_arr
        else:
            mean_acc = mean_acc + weight * mean_arr
            q05_acc = q05_acc + weight * q05_arr  # type: ignore[operator]
            q95_acc = q95_acc + weight * q95_arr  # type: ignore[operator]

    assert mean_acc is not None
    assert q05_acc is not None
    assert q95_acc is not None
    if mean_acc.ndim == 0:
        return float(mean_acc), Quantiles(float(q05_acc), float(q95_acc))
    return mean_acc, Quantiles(q05_acc, q95_acc)


def _extract_mean_q05_q95(  # noqa: PLR0912 — dispatcher over four spec shapes
    depth: str,
    spec: Any,
    coerce: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (mean, q05, q95) out of any supported per-depth spec shape."""
    if isinstance(spec, dict):
        if "mean" not in spec:
            raise InvalidInputError(f"aggregate_depths[{depth!r}]: dict spec needs a 'mean' key.")
        mean = coerce(spec["mean"], "mean", depth)
        if "uncertainty" in spec:
            unc = spec["uncertainty"]
            if isinstance(unc, Quantiles):
                q05 = coerce(unc.q05, "uncertainty.q05", depth)
                q95 = coerce(unc.q95, "uncertainty.q95", depth)
            else:
                sigma = coerce(unc, "uncertainty (σ)", depth)
                q05 = mean - 1.6448536269514722 * sigma
                q95 = mean + 1.6448536269514722 * sigma
        elif "sigma" in spec:
            sigma = coerce(spec["sigma"], "sigma", depth)
            q05 = mean - 1.6448536269514722 * sigma
            q95 = mean + 1.6448536269514722 * sigma
        elif "q05" in spec and "q95" in spec:
            q05 = coerce(spec["q05"], "q05", depth)
            q95 = coerce(spec["q95"], "q95", depth)
        else:
            raise InvalidInputError(
                f"aggregate_depths[{depth!r}]: dict needs one of 'uncertainty', "
                f"'sigma', or ('q05' and 'q95'). Got {list(spec)!r}."
            )
        return mean, q05, q95

    if isinstance(spec, tuple):
        if len(spec) == 2:
            mean = coerce(spec[0], "mean", depth)
            unc = spec[1]
            if isinstance(unc, Quantiles):
                q05 = coerce(unc.q05, "uncertainty.q05", depth)
                q95 = coerce(unc.q95, "uncertainty.q95", depth)
            else:
                sigma = coerce(unc, "uncertainty (σ)", depth)
                q05 = mean - 1.6448536269514722 * sigma
                q95 = mean + 1.6448536269514722 * sigma
            return mean, q05, q95
        if len(spec) == 3:
            warnings.warn(
                "aggregate_depths spec (mean, q05, q95) is deprecated; pass "
                "{'mean': ..., 'uncertainty': Quantiles(q05, q95)} or "
                "(mean, Quantiles(q05, q95)) instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            mean = coerce(spec[0], "mean", depth)
            q05 = coerce(spec[1], "q05", depth)
            q95 = coerce(spec[2], "q95", depth)
            return mean, q05, q95

    raise InvalidInputError(
        f"aggregate_depths[{depth!r}]: unsupported spec shape {type(spec).__name__}."
    )


__all__ = [
    "AggregateDistribution",
    "ZonalAggregate",
    "aggregate_depths",
    "zonal_aggregate",
]
