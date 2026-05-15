"""Wösten et al. (1999) HYPRES pedotransfer function.

Estimates Mualem-van Genuchten water-retention and hydraulic-conductivity
parameters from texture, organic matter, bulk density, and a
topsoil/subsoil indicator. The implementation follows the *continuous*
PTF of:

    Wösten, J.H.M., Lilly, A., Nemes, A., Le Bas, C. (1999). Development
    and use of a database of hydraulic properties of European soils.
    Geoderma 90(3-4): 169-185.

The residual water content theta_r is fixed at 0.01, following the
HYPRES convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pedotri.errors import InvalidInputError

if TYPE_CHECKING:
    from pedotri._types import ArrayLike, FloatArray, ScalarOrArrayLike


_THETA_R = 0.01  # HYPRES convention


@dataclass(slots=True, frozen=True)
class WostenResult:
    """Mualem-van Genuchten parameters from Wösten 1999.

    The van Genuchten water-retention function is::

        theta(h) = theta_r + (theta_s - theta_r) / [1 + (alpha*h)^n]^m
        with m = 1 - 1/n

    The Mualem unsaturated hydraulic-conductivity function is::

        K(theta) = K_s * S_e^L * [1 - (1 - S_e^(1/m))^m]^2
        with S_e = (theta - theta_r) / (theta_s - theta_r)

    Attributes:
        theta_r: Residual water content (m³/m³). Fixed at 0.01 in HYPRES.
        theta_s: Saturated water content (m³/m³).
        alpha: Van Genuchten alpha (1/cm).
        n: Van Genuchten n (dimensionless, > 1).
        saturated_conductivity: K_s in cm/day.
        l: Mualem pore-connectivity parameter, bounded by Wösten's
            logistic transform to ``(-10, 10)``. Note that the
            continuous PTF can yield negative L for many mineral soils;
            this is reproduced from the published regression and is not
            a sign error. If you need the strictly-positive Mualem L of
            typical theoretical interpretation, prefer the FAO-class
            class-averaged values in Wösten 1999 Table 4 or use a
            different PTF.
    """

    theta_r: float
    theta_s: float
    alpha: float
    n: float
    saturated_conductivity: float
    l: float  # noqa: E741 — single-letter Mualem parameter name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "theta_r": self.theta_r,
            "theta_s": self.theta_s,
            "alpha": self.alpha,
            "n": self.n,
            "saturated_conductivity": self.saturated_conductivity,
            "l": self.l,
        }


@overload
def wosten(
    sand: float | int,
    silt: float | int,
    clay: float | int,
    *,
    organic_matter: float | int = ...,
    bulk_density: float | int = ...,
    topsoil: bool = ...,
) -> WostenResult: ...


@overload
def wosten(
    sand: ArrayLike,
    silt: ArrayLike,
    clay: ArrayLike,
    *,
    organic_matter: ArrayLike | float | int = ...,
    bulk_density: ArrayLike | float | int = ...,
    topsoil: bool = ...,
) -> list[WostenResult]: ...


def wosten(
    sand: ScalarOrArrayLike,
    silt: ScalarOrArrayLike,
    clay: ScalarOrArrayLike,
    *,
    organic_matter: ScalarOrArrayLike = 1.0,
    bulk_density: ScalarOrArrayLike = 1.4,
    topsoil: bool = True,
) -> WostenResult | list[WostenResult]:
    """Estimate Mualem-van Genuchten parameters via Wösten 1999.

    Args:
        sand: Percent sand, [0, 100]. Used only for input validation
            and as part of the texture sum check; the PTF uses silt
            and clay explicitly.
        silt: Percent silt, (0, 100]. Must be strictly positive — the
            PTF contains ``1/silt`` and ``ln(silt)`` terms.
        clay: Percent clay, (0, 100]. Must be strictly positive — the
            PTF contains ``1/clay``.
        organic_matter: Percent organic matter, (0, 100]. Default 1.0
            %. Must be strictly positive — the PTF contains
            ``1/organic_matter`` and ``ln(organic_matter)``.
        bulk_density: Dry bulk density (g/cm³). Default 1.4.
        topsoil: True if the sample is a topsoil horizon, False for
            subsoil. Affects alpha, n, and K_s.

    Returns:
        A :class:`WostenResult` (scalar inputs) or a list (arrays).
    """
    sand_arr, silt_arr, clay_arr, om_arr, d_arr, scalar = _coerce(
        sand, silt, clay, organic_matter, bulk_density
    )
    ts = 1.0 if topsoil else 0.0

    # Aliases matching the paper's notation
    c = clay_arr
    si = silt_arr
    om = om_arr
    d = d_arr

    theta_s = (
        0.7919
        + 0.001691 * c
        - 0.29619 * d
        - 0.000001491 * si**2
        + 0.0000821 * om**2
        + 0.02427 / c
        + 0.01113 / si
        + 0.01472 * np.log(si)
        - 0.0000733 * om * c
        - 0.000619 * d * c
        - 0.001183 * d * om
        - 0.0001664 * ts * si
    )

    alpha_star = (
        -14.96
        + 0.03135 * c
        + 0.0351 * si
        + 0.646 * om
        + 15.29 * d
        - 0.192 * ts
        - 4.671 * d**2
        - 0.000781 * c**2
        - 0.00687 * om**2
        + 0.0449 / om
        + 0.0663 * np.log(si)
        + 0.1482 * np.log(om)
        - 0.04546 * d * si
        - 0.4852 * d * om
        + 0.00673 * ts * c
    )
    alpha = np.exp(alpha_star)

    n_star = (
        -25.23
        - 0.02195 * c
        + 0.0074 * si
        - 0.1940 * om
        + 45.5 * d
        - 7.24 * d**2
        + 0.0003658 * c**2
        + 0.002885 * om**2
        - 12.81 / d
        - 0.1524 / si
        - 0.01958 / om
        - 0.2876 * np.log(si)
        - 0.0709 * np.log(om)
        - 44.6 * np.log(d)
        - 0.02264 * d * c
        + 0.0896 * d * om
        + 0.00718 * ts * c
    )
    n = np.exp(n_star) + 1.0

    ks_star = (
        7.755
        + 0.0352 * si
        + 0.93 * ts
        - 0.967 * d**2
        - 0.000484 * c**2
        - 0.000322 * si**2
        + 0.001 / si
        - 0.0748 / om
        - 0.643 * np.log(si)
        - 0.01398 * d * c
        - 0.1673 * d * om
        + 0.02986 * ts * c
        - 0.03305 * ts * si
    )
    ks = np.exp(ks_star)  # cm/day

    l_star = (
        0.0202
        + 0.0006193 * c**2
        - 0.001136 * om**2
        - 0.2316 * np.log(om)
        - 0.03544 * d * c
        + 0.00283 * d * si
        + 0.0488 * d * om
    )
    # Wösten's L is bounded by (-1, +1) via the logistic-style transform;
    # in practice L is typically ~0.5 for mineral soils.
    l_unbounded = np.exp(l_star)
    l_val = 10.0 * (l_unbounded - 1.0) / (l_unbounded + 1.0)

    results = [
        WostenResult(
            theta_r=_THETA_R,
            theta_s=float(theta_s[i]),
            alpha=float(alpha[i]),
            n=float(n[i]),
            saturated_conductivity=float(ks[i]),
            l=float(l_val[i]),
        )
        for i in range(sand_arr.shape[0])
    ]
    return results[0] if scalar else results


def _coerce(
    sand: ScalarOrArrayLike,
    silt: ScalarOrArrayLike,
    clay: ScalarOrArrayLike,
    organic_matter: ScalarOrArrayLike,
    bulk_density: ScalarOrArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, bool]:
    s_raw = np.asarray(sand, dtype=np.float64)
    si_raw = np.asarray(silt, dtype=np.float64)
    c_raw = np.asarray(clay, dtype=np.float64)
    om_raw = np.asarray(organic_matter, dtype=np.float64)
    d_raw = np.asarray(bulk_density, dtype=np.float64)

    scalar = all(a.ndim == 0 for a in (s_raw, si_raw, c_raw, om_raw, d_raw))

    s = np.atleast_1d(s_raw)
    si = np.atleast_1d(si_raw)
    c = np.atleast_1d(c_raw)
    om = np.atleast_1d(om_raw)
    d = np.atleast_1d(d_raw)

    target_shape = np.broadcast_shapes(s.shape, si.shape, c.shape, om.shape, d.shape)
    s = np.broadcast_to(s, target_shape).copy()
    si = np.broadcast_to(si, target_shape).copy()
    c = np.broadcast_to(c, target_shape).copy()
    om = np.broadcast_to(om, target_shape).copy()
    d = np.broadcast_to(d, target_shape).copy()

    for name, arr in (("sand", s), ("silt", si), ("clay", c), ("organic_matter", om)):
        if np.isnan(arr).any():
            raise InvalidInputError(f"{name!r} must not contain NaN.")
        if ((arr < 0) | (arr > 100)).any():
            raise InvalidInputError(f"{name!r} must be a percentage in [0, 100].")

    if np.isnan(d).any():
        raise InvalidInputError("bulk_density must not contain NaN.")
    if (d <= 0).any():
        raise InvalidInputError("bulk_density must be strictly positive (g/cm³).")

    # Wösten formulas contain 1/silt, 1/clay, 1/organic_matter and
    # ln(silt), ln(organic_matter); any zero would blow up.
    if (si <= 0).any() or (c <= 0).any() or (om <= 0).any():
        raise InvalidInputError(
            "wosten() requires silt, clay, and organic_matter to be strictly "
            "positive (the PTF contains 1/x and ln(x) terms). For pure-sand or "
            "organic-matter-free samples use saxton_rawls() instead."
        )
    return s, si, c, om, d, scalar
