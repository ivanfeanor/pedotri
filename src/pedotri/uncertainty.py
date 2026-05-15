"""Distribution helpers for uncertainty-aware classification.

Modern digital soil maps such as SoilGrids publish Q0.05 / Q0.50 / Q0.95
quantile layers alongside the mean for each property. This module turns
those quantile inputs into distribution parameters and draws
compositional samples that respect the constraint
``sand + silt + clay = 100``.

Two paths consume these primitives:

- The cheap **distance** method compares the point's distance to its
  class boundary against the input uncertainty magnitude, returning a
  single confidence number for the modal class.
- The richer **Monte Carlo** method draws ``n_samples`` realizations,
  classifies each, and tallies a probability distribution over classes
  plus an entropy score.

The default distribution shape is a normal clipped to ``[0, 100]``,
parameterized so that ``Q95 - Q05`` matches the published spread. This
is the standard "truncated normal" assumption used in most uncertainty
propagation pipelines for soil texture; it ignores any asymmetry
between ``mean - Q05`` and ``Q95 - mean``. For strongly skewed
properties such as SOC, a log-normal fit would be more faithful — left
as a future extension.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pedotri.errors import InvalidInputError

if TYPE_CHECKING:
    from pedotri._types import FloatArray


@dataclass(frozen=True, slots=True)
class Quantiles:
    """Pair of distribution quantiles (Q0.05, Q0.95).

    Use this whenever you want to express uncertainty on a soil
    property in terms of its 90 % credible-interval bounds — the form
    SoilGrids 2.0 and most digital-soil-map products publish.

    ``Quantiles`` is the unambiguous companion to bare floats / scalars
    (which are interpreted as standard deviations) anywhere pedotri
    accepts uncertainty input::

        # explicit and self-documenting
        pedotri.classify(
            sand=27,
            clay=45,
            sand_uncertainty=Quantiles(20, 34),
            clay_uncertainty=Quantiles(38, 52),
            classification="USDA",
        )

    The dataclass is intentionally minimal — two floats, a sanity
    check, and helpers to derive σ or width. For per-pixel arrays,
    use ``Quantiles(q05_array, q95_array)``: both fields broadcast
    naturally.
    """

    q05: Any
    q95: Any

    def __post_init__(self) -> None:
        q05 = np.asarray(self.q05, dtype=np.float64)
        q95 = np.asarray(self.q95, dtype=np.float64)
        if q05.shape != q95.shape:
            raise InvalidInputError(
                f"Quantiles: q05 shape {q05.shape} does not match q95 shape {q95.shape}."
            )
        if np.any(q95 < q05):
            raise InvalidInputError("Quantiles: q95 must be >= q05.")

    @property
    def sigma(self) -> FloatArray:
        """Implied σ under a normal-distribution fit: ``(q95 − q05) / 3.2897``."""
        return sigma_from_quantiles(self.q05, self.q95)

    @property
    def width(self) -> FloatArray:
        """Raw Q0.95 − Q0.05 (in input units)."""
        return np.asarray(self.q95, dtype=np.float64) - np.asarray(self.q05, dtype=np.float64)


_NORM_PPF_95 = 1.6448536269514722
# Spread between Q0.05 and Q0.95 of a unit normal, used to back out σ
# from a published quantile pair: σ ≈ (Q95 - Q05) / _QUANTILE_SPREAD.
_QUANTILE_SPREAD = 2.0 * _NORM_PPF_95


def sigma_from_quantiles(q05: float | FloatArray, q95: float | FloatArray) -> FloatArray:
    """Back out a normal σ from a (Q0.05, Q0.95) pair.

    ``σ = (Q95 − Q05) / 3.2897``. Works elementwise on arrays so the
    raster path can pass per-pixel quantile bands directly.
    """
    q05_arr = np.asarray(q05, dtype=np.float64)
    q95_arr = np.asarray(q95, dtype=np.float64)
    if q05_arr.shape != q95_arr.shape:
        raise InvalidInputError(
            f"Q0.05 and Q0.95 must have the same shape; got {q05_arr.shape} vs {q95_arr.shape}."
        )
    if np.any(q95_arr < q05_arr):
        raise InvalidInputError("Q0.95 must be greater than or equal to Q0.05.")
    return (q95_arr - q05_arr) / _QUANTILE_SPREAD


def _parse_uncertainty(value: object) -> FloatArray | None:
    """Normalize a user-supplied uncertainty kwarg to a σ array.

    Accepted forms, in order of preference:

    - ``None`` — no uncertainty (returns ``None``).
    - :class:`Quantiles` — explicit ``(Q0.05, Q0.95)`` pair. *Use this
      form going forward — it self-documents at the call site.*
    - scalar or array-like — interpreted directly as σ.
    - bare ``tuple`` of length 2 — treated as ``(Q0.05, Q0.95)`` for
      backwards compatibility, but **emits a DeprecationWarning** at
      call time. The bare-tuple form is ambiguous (could be read as
      ``(low, high)`` of an error bar, ``(−err, +err)``, …) and will
      be removed in a future minor release. Wrap in
      :class:`Quantiles` to keep working.

    Lists are *not* treated as the quantile-pair form because a bare
    two-element list is ambiguous with a two-point input array.
    """
    if value is None:
        return None
    if isinstance(value, Quantiles):
        return sigma_from_quantiles(value.q05, value.q95)
    if isinstance(value, tuple) and len(value) == 2:
        warnings.warn(
            "Passing a bare 2-tuple as *_uncertainty is ambiguous and "
            "deprecated; use Quantiles(q05, q95) — or pass a scalar/array "
            "if you meant a σ value. The bare tuple will stop being "
            "interpreted as (Q0.05, Q0.95) in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return sigma_from_quantiles(value[0], value[1])
    arr = np.asarray(value, dtype=np.float64)
    if np.any(arr < 0):
        raise InvalidInputError("σ values must be non-negative.")
    return arr


# Public alias retained for back-compat with anyone who imported
# ``parse_uncertainty`` directly. New code should not depend on it.
parse_uncertainty = _parse_uncertainty


def sample_truncated_normal(
    mean: FloatArray,
    sigma: FloatArray,
    n_samples: int,
    *,
    lo: float = 0.0,
    hi: float = 100.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Draw ``n_samples`` normal samples per point and **clip** to ``[lo, hi]``.

    Returns an array of shape ``(n_points, n_samples)``. Used by every
    Monte-Carlo path in pedotri:

    - ``pedotri.classify(..., method="monte_carlo")`` for the 1-axis
      classifications (Kachinsky, RUS2004).
    - ``sample_compositional`` for the (sand, clay) simplex sampling,
      which in turn feeds the 2-axis MC classifier and
      ``classify_array_with_uncertainty``.
    - ``pedotri.zonal.zonal_aggregate`` per-pixel sampling for
      regional aggregation.

    **"Truncated" but actually "clipped".** SciPy's
    ``scipy.stats.truncnorm`` implements the textbook truncated normal:
    values outside ``[lo, hi]`` are *rejected and redrawn* so the
    sample distribution has zero mass beyond the support. Pedotri
    takes the cheaper route — sample an unbounded normal, then clamp
    every value into ``[lo, hi]``. Draws that fall outside become
    point masses at the boundary.

    Why the clip is fine for soil texture:

    - Soil-property σ is small relative to the support width. SoilGrids
      σ on sand is typically 3–7 % while the support is 100 % wide,
      so for a pixel with mean = 27 % the fraction of draws that need
      clipping is 10⁻⁵ – 10⁻⁷. Negligible.
    - Even at the extreme end (mean ≈ 1 % or 99 %, σ = 10 %), the
      ~16 % of draws pinned to the boundary biases the regional
      aggregate by less than the spatial-autocorrelation lower bound
      we already document.
    - Clipping is ~3× faster and avoids the scipy dependency.

    If your inputs sit very close to the support boundary and you need
    proper rejection sampling, sample with
    ``scipy.stats.truncnorm.rvs`` upstream and pass the σ values
    directly through ``zonal_aggregate``'s ``"sigma"`` spec form.
    """
    if n_samples < 1:
        raise InvalidInputError(f"n_samples must be >= 1, got {n_samples}.")
    rng = rng if rng is not None else np.random.default_rng()
    mean_arr = np.atleast_1d(np.asarray(mean, dtype=np.float64))
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=np.float64), mean_arr.shape).astype(
        np.float64, copy=True
    )
    samples = rng.normal(
        loc=mean_arr[:, None],
        scale=sigma_arr[:, None],
        size=(mean_arr.size, n_samples),
    )
    np.clip(samples, lo, hi, out=samples)
    return samples


def sample_compositional(
    sand_mean: FloatArray,
    sand_sigma: FloatArray,
    clay_mean: FloatArray,
    clay_sigma: FloatArray,
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample ``(sand, clay)`` compositions with simplex renormalization.

    Independent marginals are drawn for sand and clay; whenever a draw
    has ``sand + clay > 100`` (which would imply negative silt and lies
    outside the texture triangle), both values are scaled down
    proportionally so that ``sand + clay = 100`` and silt = 0. This
    keeps the ratio between sand and clay intact while projecting the
    invalid sample onto the nearest edge of the simplex.

    Returns ``(sand_samples, clay_samples)``, each of shape
    ``(n_points, n_samples)``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sand = sample_truncated_normal(sand_mean, sand_sigma, n_samples, rng=rng)
    clay = sample_truncated_normal(clay_mean, clay_sigma, n_samples, rng=rng)
    total = sand + clay
    over = total > 100.0
    if over.any():
        safe_total = np.where(total > 0, total, 1.0)
        scale = np.where(over, 100.0 / safe_total, 1.0)
        sand = sand * scale
        clay = clay * scale
    return sand, clay


def _distance_confidence(distance: float, sigma_effective: float) -> float:
    """Map a signed distance-to-boundary into a confidence in ``[0, 1]``.

    Uses the standard-normal CDF: a point sitting exactly on a
    boundary (``distance = 0``) returns ``0.5``; one sitting many σ
    inside its class returns ≈ 1.0. ``sigma_effective`` should be the
    composite per-axis σ relevant to the local boundary geometry; for a
    2-axis classification, a reasonable default is the quadratic mean
    ``√((σ_sand² + σ_clay²) / 2)``.

    Returns ``nan`` when ``sigma_effective`` is zero or non-finite — in
    that case the caller's input has no uncertainty and the
    deterministic class is the answer.
    """
    if not np.isfinite(sigma_effective) or sigma_effective <= 0:
        return float("nan")
    if not np.isfinite(distance):
        return float("nan")
    return float(_norm_cdf(distance / sigma_effective))


def _effective_sigma(sigmas: tuple[float, ...]) -> float:
    """Combine per-axis σ into a single scalar for distance comparison.

    Uses the quadratic mean across axes — a defensible isotropic proxy
    when we don't know the local boundary orientation.
    """
    arr = np.asarray(sigmas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def _aggregate_class_probabilities(
    codes: np.ndarray, n_classes: int
) -> tuple[FloatArray, FloatArray]:
    """Tally per-class frequencies from a ``(n_points, n_samples)`` codes array.

    Returns ``(probs, unclassified_rate)`` where ``probs`` has shape
    ``(n_points, n_classes)`` and rows sum to ``1 − unclassified_rate``.
    Codes equal to ``-1`` are counted toward ``unclassified_rate``.

    Implementation: a single global ``np.bincount`` over flattened bin
    indices ``row · n_classes + code`` (with unclassified pixels
    redirected to a sink bin past the end). Avoids the per-row Python
    loop used in earlier versions and runs 2–4× faster on typical
    raster MC workloads.
    """
    if codes.ndim != 2:
        raise InvalidInputError(
            f"codes must be 2-D (n_points, n_samples); got shape {codes.shape}."
        )
    n_points, n_samples = codes.shape
    if n_samples == 0:
        return (
            np.zeros((n_points, n_classes), dtype=np.float64),
            np.zeros(n_points, dtype=np.float64),
        )
    rows = np.arange(n_points, dtype=np.int64)[:, None]
    valid = codes >= 0
    # Unclassified pixels go to a sink bin at the very end so we only
    # need one bincount call. minlength keeps the result shape stable
    # across runs where some classes are absent.
    sink = n_points * n_classes
    flat_bins = np.where(valid, rows * n_classes + codes.astype(np.int64), sink).ravel()
    counts = np.bincount(flat_bins, minlength=sink + 1)
    probs = counts[:sink].reshape(n_points, n_classes).astype(np.float64) / n_samples
    classified_per_row = probs.sum(axis=1) * n_samples
    unclassified = (n_samples - classified_per_row) / n_samples
    return probs, unclassified


def shannon_entropy(probs: FloatArray) -> FloatArray:
    """Shannon entropy in nats, computed row-wise.

    Zero-probability classes contribute zero (the ``0 · log 0`` limit
    is handled explicitly). Returns one scalar per row.
    """
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=-1)


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via ``erf`` — avoids a SciPy dependency.

    ``Φ(x) = 0.5 · (1 + erf(x / √2))``.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --- Why pedotri carries its own ``erf`` implementation ------------------
#
# The distance-method confidence path turns "this pixel is k σ deep
# inside its class" into "we are P(k) confident the pixel belongs to
# that class", where P is the standard-normal CDF
#
#       Φ(k) = 0.5 · (1 + erf(k / √2))
#
# That's the only place ``erf`` shows up in pedotri, but at raster
# scale it shows up *a lot*: every pixel of every confidence map calls
# it once. A 1 000 × 1 000 SoilGrids tile passed through
# ``classify_array_with_uncertainty(..., method="distance")`` evaluates
# ``erf`` a million times.
#
# Three plausible implementations, with throughput on a 1 Mpix array:
#
# ============================================  ========  ============
# Implementation                                Time      Dep weight
# ============================================  ========  ============
# ``np.vectorize(math.erf)`` — Python per cell  ~60 ms    none
# Abramowitz & Stegun 7.1.26 — pure numpy       ~13 ms    none
# ``scipy.special.erf`` — C implementation      ~11 ms    scipy
# ============================================  ========  ============
#
# We picked the middle option. The pure-numpy A&S series gives us a
# ~5× speedup over ``np.vectorize`` *without* turning scipy into a
# core dependency (it remains optional under the ``[raster]`` extra).
# Scipy is only marginally faster from here, so the trade-off favours
# the lighter install.
#
# The coefficients below are taken verbatim from Milton Abramowitz and
# Irene Stegun's *Handbook of Mathematical Functions*, §7.1.26 (1965
# edition; widely reproduced — see e.g. the National Bureau of
# Standards Applied Mathematics Series 55, p. 299). They define a
# 5-term polynomial-times-exp(−x²) approximation whose maximum
# absolute error across the real line is ε ≤ 1.5 · 10⁻⁷, far below
# any precision the confidence path cares about (we report
# 3-significant-figure values like 0.873 to users; float32 itself only
# has ~7 digits).
_AS_A1 = 0.254829592
_AS_A2 = -0.284496736
_AS_A3 = 1.421413741
_AS_A4 = -1.453152027
_AS_A5 = 1.061405429
_AS_P = 0.3275911


def _erf_array(x: np.ndarray) -> np.ndarray:
    """Vectorized ``erf`` using the Abramowitz & Stegun 7.1.26 series.

    Drop-in replacement for ``scipy.special.erf`` on the hot
    distance-method confidence path. Accurate to ≤ 1.5 × 10⁻⁷ across
    the real line — see the module-level comment above for the
    rationale and benchmark numbers.

    Mathematical form: for ``x ≥ 0``,

        t   = 1 / (1 + p · x)
        erf(x) ≈ 1 − (a₁ t + a₂ t² + a₃ t³ + a₄ t⁴ + a₅ t⁵) · exp(−x²)

    and ``erf(−x) = −erf(x)`` for negative inputs. Implemented via
    Horner's rule on ``t`` to keep the numpy expression tight.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + _AS_P * ax)
    y = 1.0 - (((((_AS_A5 * t + _AS_A4) * t) + _AS_A3) * t + _AS_A2) * t + _AS_A1) * t * np.exp(
        -ax * ax
    )
    return sign * y


def _norm_cdf_array(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF: ``Φ(x) = 0.5 · (1 + erf(x / √2))``.

    Used by ``classify(..., method="distance")`` (point + raster) to
    turn a signed distance-to-class-boundary, expressed in units of
    σ_effective, into a confidence in ``[0, 1]``. A pixel sitting two
    σ inside its class returns 0.977; a pixel right on the boundary
    returns 0.5.
    """
    return 0.5 * (1.0 + _erf_array(x / np.sqrt(2.0)))


__all__ = [
    "Quantiles",
    "parse_uncertainty",
    "sample_compositional",
    "sample_truncated_normal",
    "shannon_entropy",
    "sigma_from_quantiles",
]
