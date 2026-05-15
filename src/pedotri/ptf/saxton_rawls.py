"""Saxton & Rawls (2006) pedotransfer function.

Estimates soil-water retention and saturated hydraulic conductivity
from sand %, clay %, and organic matter %. Optionally adjusts for
bulk-density compaction via a density factor and for gravel content.

The implementation follows the equations in:

    Saxton, K.E. & Rawls, W.J. (2006). Soil water characteristic
    estimates by texture and organic matter for hydrologic solutions.
    Soil Science Society of America Journal, 70(5): 1569-1578.

Inputs are *percentages* (0-100) to match :func:`pedotri.classify`;
the formulas use fractions internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np

from pedotri.errors import InvalidInputError

if TYPE_CHECKING:
    from pedotri._types import ArrayLike, FloatArray, ScalarOrArrayLike


@dataclass(slots=True, frozen=True)
class SaxtonRawlsResult:
    """Soil-water characteristics estimated by Saxton-Rawls (2006).

    All water-content quantities are *volumetric* (cm³ water / cm³ bulk
    soil) — i.e. fractions in [0, 1].

    Attributes:
        wilting_point: Water content at 1500 kPa suction (θ_1500), the
            permanent wilting point.
        field_capacity: Water content at 33 kPa suction (θ_33), the
            conventional drained-equilibrium field capacity for
            agronomic soils.
        saturation: Water content at saturation (θ_s).
        available_water: Plant-available water = ``field_capacity -
            wilting_point``.
        saturated_conductivity: Saturated hydraulic conductivity (K_s)
            in mm/h.
        bulk_density: Normal-density bulk density (g/cm³).
        air_entry_tension: Air-entry tension (bubbling pressure) ψ_e in
            kPa.
    """

    wilting_point: float
    field_capacity: float
    saturation: float
    available_water: float
    saturated_conductivity: float
    bulk_density: float
    air_entry_tension: float


@overload
def saxton_rawls(
    sand: float | int,
    clay: float | int,
    organic_matter: float | int = ...,
    *,
    density_factor: float = ...,
) -> SaxtonRawlsResult: ...


@overload
def saxton_rawls(
    sand: ArrayLike,
    clay: ArrayLike,
    organic_matter: ArrayLike | float | int = ...,
    *,
    density_factor: float = ...,
) -> list[SaxtonRawlsResult]: ...


def saxton_rawls(
    sand: ScalarOrArrayLike,
    clay: ScalarOrArrayLike,
    organic_matter: ScalarOrArrayLike = 1.0,
    *,
    density_factor: float = 1.0,
) -> SaxtonRawlsResult | list[SaxtonRawlsResult]:
    """Compute soil hydraulic properties from texture and organic matter.

    Args:
        sand: Percent sand, [0, 100]. Scalar or array-like.
        clay: Percent clay, [0, 100]. Scalar or array-like.
        organic_matter: Percent organic matter (mass), [0, 100]. Default
            1.0 % which is typical for agricultural mineral soils.
        density_factor: Compaction-adjustment multiplier on the
            regression's normal bulk density (Saxton & Rawls 2006,
            Eq. 6-7). Default ``1.0`` matches the original regression
            (no compaction). Values > 1.0 model compacted soils:
            saturation and field capacity are reduced via the paper's
            density correction. Wilting point is unaffected per the
            paper. Reasonable range is ``[0.9, 1.3]``.

    Returns:
        A :class:`SaxtonRawlsResult` for scalar inputs, or a list of
        results for array inputs.

    Raises:
        InvalidInputError: If inputs are out of range or have mismatched
            shapes.
    """
    sand_arr, clay_arr, om_arr, scalar = _coerce(sand, clay, organic_matter)

    s = sand_arr / 100.0
    c = clay_arr / 100.0
    om = om_arr / 100.0

    # --- θ_1500: water content at 1500 kPa (permanent wilting point) ---
    theta_1500t = (
        -0.024 * s
        + 0.487 * c
        + 0.006 * om
        + 0.005 * s * om
        - 0.013 * c * om
        + 0.068 * s * c
        + 0.031
    )
    theta_1500 = theta_1500t + (0.14 * theta_1500t - 0.02)

    # --- θ_33: water content at 33 kPa (field capacity) ---
    theta_33t = (
        -0.251 * s
        + 0.195 * c
        + 0.011 * om
        + 0.006 * s * om
        - 0.027 * c * om
        + 0.452 * s * c
        + 0.299
    )
    theta_33 = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

    # --- θ_(s-33): saturation minus 33 kPa water content ---
    theta_s33t = (
        0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om - 0.584 * s * c + 0.078
    )
    theta_s33 = theta_s33t + (0.636 * theta_s33t - 0.107)

    # --- θ_s: saturation, with sand-only adjustment for coarse soils ---
    theta_s_normal = theta_33 + theta_s33 - 0.097 * s + 0.043

    # --- Bulk density (normal) from saturation ---
    bulk_density_normal = (1.0 - theta_s_normal) * 2.65

    # --- Apply density-factor compaction correction (Eq. 6-8) ---
    bulk_density = bulk_density_normal * density_factor
    theta_s = 1.0 - bulk_density / 2.65
    # θ_33 shifts proportionally to the saturation loss (Eq. 8).
    theta_33 = theta_33 - 0.2 * (theta_s_normal - theta_s)

    # --- Saturated hydraulic conductivity K_s (mm/h) ---
    # B is the slope of the moisture-tension curve in log-log space
    # between θ_33 (at 33 kPa) and θ_1500 (at 1500 kPa).
    b = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_33) - np.log(theta_1500))
    lam = 1.0 / b
    k_s = 1930.0 * np.power(theta_s - theta_33, 3.0 - lam)

    # --- Air-entry tension ψ_e (kPa) ---
    psi_et = (
        -21.67 * s
        - 27.93 * c
        - 81.97 * theta_s33
        + 71.12 * s * theta_s33
        + 8.29 * c * theta_s33
        + 14.05 * s * c
        + 27.16
    )
    psi_e = psi_et + (0.02 * psi_et**2 - 0.113 * psi_et - 0.7)

    available_water = theta_33 - theta_1500

    results = [
        SaxtonRawlsResult(
            wilting_point=float(theta_1500[i]),
            field_capacity=float(theta_33[i]),
            saturation=float(theta_s[i]),
            available_water=float(available_water[i]),
            saturated_conductivity=float(k_s[i]),
            bulk_density=float(bulk_density[i]),
            air_entry_tension=float(psi_e[i]),
        )
        for i in range(sand_arr.shape[0])
    ]
    return results[0] if scalar else results


def _coerce(
    sand: ScalarOrArrayLike,
    clay: ScalarOrArrayLike,
    organic_matter: ScalarOrArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    s_raw = np.asarray(sand, dtype=np.float64)
    c_raw = np.asarray(clay, dtype=np.float64)
    om_raw = np.asarray(organic_matter, dtype=np.float64)
    scalar = s_raw.ndim == 0 and c_raw.ndim == 0 and om_raw.ndim == 0

    s = np.atleast_1d(s_raw)
    c = np.atleast_1d(c_raw)
    om = np.atleast_1d(om_raw)

    target_shape = np.broadcast_shapes(s.shape, c.shape, om.shape)
    s = np.broadcast_to(s, target_shape).copy()
    c = np.broadcast_to(c, target_shape).copy()
    om = np.broadcast_to(om, target_shape).copy()

    for name, arr in (("sand", s), ("clay", c), ("organic_matter", om)):
        if np.isnan(arr).any():
            raise InvalidInputError(f"{name!r} must not contain NaN.")
        if ((arr < 0) | (arr > 100)).any():
            raise InvalidInputError(f"{name!r} must be a percentage in [0, 100].")
    return s, c, om, scalar
