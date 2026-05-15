"""Unit conversions for soil composition values.

Most of pedotri's public API takes percentages (mass fraction * 100,
0-100 range) because that is what the underlying classification
references and PTF regressions use. Real-world lab reports use other
units constantly:

- **g/kg** (grams per kilogram) — common in European and African soil
  labs, *especially* for organic matter and organic carbon.
- **g/g** (mass fraction, 0-1 range) — used in physical chemistry and
  modelling pipelines.
- **Organic carbon (OC)** vs **organic matter (OM)** — most labs report
  OC; PTFs like Saxton-Rawls 2006 expect OM. The Van Bemmelen factor
  (1.724) converts between them.

This module provides the conversions. The four public functions that
accept percentage inputs (:func:`pedotri.classify`,
:func:`pedotri.ptf.saxton_rawls`, :func:`pedotri.ptf.wosten`,
:func:`pedotri.psd.convert`) also accept a ``units=`` keyword that
applies the same conversions at the function boundary, so you rarely
need to call these helpers directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from pedotri.errors import InvalidInputError

if TYPE_CHECKING:
    from pedotri._types import FloatArray, ScalarOrArrayLike


#: Valid string identifiers for the ``units=`` keyword.
UnitLiteral: TypeAlias = Literal["%", "g/kg", "g/g"]

#: Canonical "no conversion" default used across pedotri.
DEFAULT_UNITS: UnitLiteral = "%"

#: Van Bemmelen factor — the standard empirical OC -> OM ratio.
#: Originally derived from the assumption that organic matter is
#: ~58 % carbon by mass, so OM = OC * (100 / 58) ~ 1.724.
VAN_BEMMELEN_FACTOR = 1.724


def to_percent(value: ScalarOrArrayLike, units: UnitLiteral = "%") -> FloatArray:
    """Convert ``value`` from the named units to percent.

    Args:
        value: Scalar or array-like in the source units.
        units: Source units. One of ``"%"`` (no conversion, the default),
            ``"g/kg"`` (divide by 10), or ``"g/g"`` (multiply by 100).

    Returns:
        A 1-D float array of percentages.

    Raises:
        InvalidInputError: If ``units`` is not recognised.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    factor = _factor_to_percent(units)
    return arr * factor


def from_percent(value: ScalarOrArrayLike, units: UnitLiteral = "%") -> FloatArray:
    """Convert ``value`` from percent to the named units.

    The inverse of :func:`to_percent`.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    factor = _factor_to_percent(units)
    return arr / factor


def g_per_kg_to_percent(value: ScalarOrArrayLike) -> FloatArray:
    """g/kg → %. (Divide by 10.)"""
    return to_percent(value, "g/kg")


def percent_to_g_per_kg(value: ScalarOrArrayLike) -> FloatArray:
    """% → g/kg. (Multiply by 10.)"""
    return from_percent(value, "g/kg")


def g_per_g_to_percent(value: ScalarOrArrayLike) -> FloatArray:
    """g/g (mass fraction) → %. (Multiply by 100.)"""
    return to_percent(value, "g/g")


def percent_to_g_per_g(value: ScalarOrArrayLike) -> FloatArray:
    """% → g/g (mass fraction). (Divide by 100.)"""
    return from_percent(value, "g/g")


def organic_carbon_to_organic_matter(
    oc: ScalarOrArrayLike, factor: float = VAN_BEMMELEN_FACTOR
) -> FloatArray:
    """Convert organic carbon to organic matter via the Van Bemmelen factor.

    Args:
        oc: Organic-carbon value in any units. The conversion is a
            multiplicative scalar, so units are preserved (% → %,
            g/kg → g/kg, g/g → g/g).
        factor: Conversion factor. Default 1.724 (Van Bemmelen, ~58 %
            carbon in organic matter). Common alternatives in the
            literature are 1.9 (Pribyl 2010) for less-decomposed
            material and 2.0 for some forest soils.

    Returns:
        Organic matter, in the same units as the input.
    """
    arr = np.atleast_1d(np.asarray(oc, dtype=np.float64))
    if factor <= 0:
        raise InvalidInputError(f"OC-to-OM factor must be strictly positive, got {factor}.")
    return arr * factor


def organic_matter_to_organic_carbon(
    om: ScalarOrArrayLike, factor: float = VAN_BEMMELEN_FACTOR
) -> FloatArray:
    """Convert organic matter to organic carbon (inverse of Van Bemmelen)."""
    arr = np.atleast_1d(np.asarray(om, dtype=np.float64))
    if factor <= 0:
        raise InvalidInputError(f"OC-to-OM factor must be strictly positive, got {factor}.")
    return arr / factor


# --- Internal helpers (used by classify / PTFs / psd.convert) ------------


def _factor_to_percent(units: str) -> float:
    """Multiplicative factor to convert from ``units`` to percent."""
    if units == "%":
        return 1.0
    if units == "g/kg":
        return 0.1
    if units == "g/g":
        return 100.0
    raise InvalidInputError(f"Unknown units {units!r}. Valid: '%', 'g/kg', 'g/g'.")


def _convert_inputs(value: ScalarOrArrayLike, units: str) -> ScalarOrArrayLike:
    """Apply a units conversion to a scalar or array, preserving shape.

    Used internally by classify / PTFs / psd.convert to apply ``units=``
    at the function boundary before existing percentage-based logic
    runs. Returns the original input unchanged when ``units == "%"``.
    """
    if units == "%":
        return value
    factor = _factor_to_percent(units)
    if isinstance(value, (int, float)):
        return float(value) * factor
    return np.asarray(value, dtype=np.float64) * factor


__all__ = [
    "DEFAULT_UNITS",
    "VAN_BEMMELEN_FACTOR",
    "UnitLiteral",
    "from_percent",
    "g_per_g_to_percent",
    "g_per_kg_to_percent",
    "organic_carbon_to_organic_matter",
    "organic_matter_to_organic_carbon",
    "percent_to_g_per_g",
    "percent_to_g_per_kg",
    "to_percent",
]
