"""Particle-size distribution (PSD) conversions across cutoff standards.

Soil texture classifications use different particle-size cutoffs to
separate sand from silt — USDA puts the boundary at 0.05 mm, ISSS at
0.02 mm, KA5 at 0.063 mm — while the silt/clay boundary is universally
at 0.002 mm. A soil sample analyzed against one standard cannot be
classified against another without converting its (sand, silt, clay)
fractions.

The conversion is approximate: it requires assumptions about the
particle-size distribution *within* the source standard's silt or sand
fraction. This module uses a **log-linear** assumption — i.e. the
cumulative-mass curve is treated as a straight line in log-particle-size
space within each source fraction. This is the simplest reasonable
model and is widely used when only three-point textural data is
available; for higher accuracy, supply a full sieve PSD and call
:func:`interpolate_psd` directly.

Supported standards (sand-silt cutoff):

============== ==================
``USDA``       0.05 mm
``FAO``        0.05 mm
``ISSS``       0.02 mm
``INTERNATIONAL`` 0.02 mm
``KA5``        0.063 mm
============== ==================

**Kachinsky is not convertible from / to this set.** The Russian
Kachinsky classification (Качинский 1965) keys on *physical clay* —
the mass fraction of particles < 0.01 mm. That cutoff combines what
USDA / ISSS / KA5 split between clay (< 0.002 mm) and the fine half of
silt, so a single sand-silt cutoff move is insufficient — you'd need a
full sieve PSD that resolves the 0.002-0.01 mm range. Use
:func:`interpolate_psd` directly with explicit sieve data if you need
to derive a Kachinsky physical-clay value, then classify with
``pedotri.classify(value, "KACHINSKY")``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pedotri.errors import InvalidInputError, PedotriError
from pedotri.units import _convert_inputs

if TYPE_CHECKING:
    from pedotri._types import FloatArray, ScalarOrArrayLike


# Sand-silt boundary by standard, in millimetres. Clay-silt boundary is
# 0.002 mm in every system covered here. The upper bound of sand
# (gravel boundary) is 2.0 mm universally.
SAND_SILT_CUTOFF_MM: dict[str, float] = {
    "USDA": 0.05,
    "FAO": 0.05,
    "ISSS": 0.02,
    "INTERNATIONAL": 0.02,
    "KA5": 0.063,
}
_CLAY_CUTOFF_MM = 0.002
_MAX_PARTICLE_MM = 2.0
_SUM_TOLERANCE = 0.5  # percent — covers rounding in lab reports


def convert(
    sand: ScalarOrArrayLike,
    silt: ScalarOrArrayLike,
    clay: ScalarOrArrayLike,
    *,
    source: str,
    target: str,
    units: str = "%",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert (sand, silt, clay) fractions between particle-size standards.

    Args:
        sand: Sand fraction under the ``source`` standard. Default
            units: percent in [0, 100]. See ``units``.
        silt: Silt fraction under the ``source`` standard.
        clay: Clay fraction. Unchanged across all supported standards
            (the silt/clay boundary is universally 0.002 mm).
        source: Particle-size standard name of the input, one of
            :data:`SAND_SILT_CUTOFF_MM`'s keys.
        target: Standard to convert *to*, same key set.
        units: Units of the three fraction arguments. One of ``"%"``
            (default), ``"g/kg"``, or ``"g/g"``. The conversion is
            performed internally in percent and the output arrays are
            *always returned in percent* — call
            :func:`pedotri.units.from_percent` if you need to round-
            trip to the source units.

    Returns:
        ``(sand', silt', clay)`` arrays in the target standard, in
        percent. Outputs are always 1-D numpy arrays; pass
        single-element arrays through ``float(...)`` if you need
        scalars.

    Raises:
        InvalidInputError: For unknown standards, out-of-range
            percentages, or fractions whose sum deviates from 100 % by
            more than 0.5 percentage points.

    Note:
        Conversion is *not* exactly reversible: ``convert(usda → isss → usda)``
        generally differs from the input by a percent or two, reflecting
        the limitations of a single log-linear interior model.
    """
    _validate_standard(source, role="source")
    _validate_standard(target, role="target")

    if units != "%":
        sand = _convert_inputs(sand, units)
        silt = _convert_inputs(silt, units)
        clay = _convert_inputs(clay, units)

    s = np.atleast_1d(np.asarray(sand, dtype=np.float64))
    si = np.atleast_1d(np.asarray(silt, dtype=np.float64))
    c = np.atleast_1d(np.asarray(clay, dtype=np.float64))
    shape = np.broadcast_shapes(s.shape, si.shape, c.shape)
    s = np.broadcast_to(s, shape).copy()
    si = np.broadcast_to(si, shape).copy()
    c = np.broadcast_to(c, shape).copy()

    _validate_fractions(s, si, c)

    src_cut = SAND_SILT_CUTOFF_MM[source]
    tgt_cut = SAND_SILT_CUTOFF_MM[target]

    if math.isclose(src_cut, tgt_cut, rel_tol=1e-9):
        return s, si, c

    if src_cut > tgt_cut:
        # Target boundary is finer: part of the source silt fraction
        # (between target_cut and source_cut) is reclassified as
        # target-standard sand.
        frac_stays_silt = (math.log10(tgt_cut) - math.log10(_CLAY_CUTOFF_MM)) / (
            math.log10(src_cut) - math.log10(_CLAY_CUTOFF_MM)
        )
        new_silt = si * frac_stays_silt
        new_sand = s + si * (1.0 - frac_stays_silt)
    else:
        # Target boundary is coarser: part of the source sand fraction
        # (between source_cut and target_cut) is reclassified as
        # target-standard silt.
        frac_stays_sand = (math.log10(_MAX_PARTICLE_MM) - math.log10(tgt_cut)) / (
            math.log10(_MAX_PARTICLE_MM) - math.log10(src_cut)
        )
        new_sand = s * frac_stays_sand
        new_silt = si + s * (1.0 - frac_stays_sand)

    return new_sand, new_silt, c


def interpolate_psd(
    sieve_sizes_mm: FloatArray,
    cumulative_pct_passing: FloatArray,
    target_size_mm: float,
) -> float:
    """Interpolate a particle-size distribution onto a target sieve.

    Given a measured PSD as (sieve_sizes, % passing each sieve), return
    the % of mass that would pass a sieve at ``target_size_mm``,
    interpolating log-linearly between the nearest measured points.

    This is the building block for arbitrary cutoff conversions when
    a richer PSD than the three-point sand/silt/clay split is available.

    Args:
        sieve_sizes_mm: Strictly-increasing sieve sizes in mm.
        cumulative_pct_passing: % of sample mass finer than each sieve,
            same length as ``sieve_sizes_mm``. Must be non-decreasing
            with respect to sieve size.
        target_size_mm: Sieve size to interpolate onto, in mm.

    Returns:
        % of sample mass finer than ``target_size_mm``, in [0, 100].

    Raises:
        InvalidInputError: For mismatched shapes, non-monotonic inputs,
            or a target outside the measured range.
    """
    sizes = np.asarray(sieve_sizes_mm, dtype=np.float64)
    passing = np.asarray(cumulative_pct_passing, dtype=np.float64)

    if sizes.shape != passing.shape or sizes.ndim != 1:
        raise InvalidInputError(
            "sieve_sizes_mm and cumulative_pct_passing must be 1-D arrays "
            f"of the same length; got {sizes.shape} and {passing.shape}."
        )
    if sizes.size < 2:
        raise InvalidInputError("Need at least two measured sieve points to interpolate.")
    if np.any(np.diff(sizes) <= 0):
        raise InvalidInputError("sieve_sizes_mm must be strictly increasing.")
    if np.any(np.diff(passing) < 0):
        raise InvalidInputError("cumulative_pct_passing must be non-decreasing with sieve size.")
    if not sizes[0] <= target_size_mm <= sizes[-1]:
        raise InvalidInputError(
            f"target_size_mm {target_size_mm} is outside the measured "
            f"range [{sizes[0]}, {sizes[-1]}]."
        )
    log_sizes = np.log10(sizes)
    log_target = math.log10(target_size_mm)
    return float(np.interp(log_target, log_sizes, passing))


_NON_TRIANGLE_STANDARDS: dict[str, str] = {
    "KACHINSKY": (
        "KACHINSKY uses physical clay (< 0.01 mm) rather than a sand-silt "
        "cutoff, so it cannot be converted via sand/silt/clay re-binning. "
        "Use a full sieve PSD with interpolate_psd() to derive a Kachinsky "
        "physical-clay value directly."
    ),
}


def _validate_standard(standard: str, *, role: str) -> None:
    """Raise a useful error when the user picks an unsupported standard."""
    if standard in SAND_SILT_CUTOFF_MM:
        return
    if standard.upper() in _NON_TRIANGLE_STANDARDS:
        raise InvalidInputError(
            f"{role} standard {standard!r} is not convertible: "
            f"{_NON_TRIANGLE_STANDARDS[standard.upper()]}"
        )
    raise InvalidInputError(
        f"Unknown {role} standard {standard!r}. "
        f"Known: {sorted(SAND_SILT_CUTOFF_MM)!r}."
    )


def _validate_fractions(sand: FloatArray, silt: FloatArray, clay: FloatArray) -> None:
    """Range + sum validation, matching the classify() conventions."""
    for name, arr in (("sand", sand), ("silt", silt), ("clay", clay)):
        if np.isnan(arr).any():
            raise InvalidInputError(f"{name!r} must not contain NaN.")
        if ((arr < 0) | (arr > 100)).any():
            raise InvalidInputError(f"{name!r} must be a percentage in [0, 100].")
    totals = sand + silt + clay
    if np.any(np.abs(totals - 100.0) > _SUM_TOLERANCE):
        bad = totals[np.abs(totals - 100.0) > _SUM_TOLERANCE]
        raise InvalidInputError(
            f"sand + silt + clay must sum to 100 % (within "
            f"{_SUM_TOLERANCE} %); got {bad.tolist()!r}."
        )


def supported_standards() -> list[str]:
    """Return the sorted list of supported particle-size standards."""
    return sorted(SAND_SILT_CUTOFF_MM)


def cutoff_mm(standard: str) -> float:
    """Return the sand-silt cutoff (mm) for a given standard."""
    if standard not in SAND_SILT_CUTOFF_MM:
        raise PedotriError(
            f"Unknown standard {standard!r}. Known: {sorted(SAND_SILT_CUTOFF_MM)!r}."
        )
    return SAND_SILT_CUTOFF_MM[standard]
