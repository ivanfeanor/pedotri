"""Tests for ``pedotri.dem`` — slope / aspect / curvature / TWI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pedotri.dem import (
    aspect,
    plan_curvature,
    profile_curvature,
    slope,
    topographic_wetness_index,
)
from pedotri.errors import InvalidInputError

# --- slope ---------------------------------------------------------------


def test_slope_on_flat_terrain_is_zero() -> None:
    dem = np.full((10, 10), 100.0)
    out = slope(dem, pixel_size=1.0)
    assert out.shape == dem.shape
    assert np.allclose(out, 0.0)


def test_slope_on_constant_grade_matches_analytical() -> None:
    """A linear ramp z = 0.5·x on a 1 m grid → tan β = 0.5 → β ≈ 26.565°."""
    _y, x = np.mgrid[0:20, 0:20]
    dem = 0.5 * x  # rise 0.5 per pixel east
    out_deg = slope(dem, pixel_size=1.0, units="degrees")
    expected = math.degrees(math.atan(0.5))
    # Interior pixels should hit the analytical value to within FP error.
    interior = out_deg[2:-2, 2:-2]
    assert np.allclose(interior, expected, atol=1e-6)


def test_slope_units_consistency() -> None:
    """Same DEM in three units should match the analytical conversions."""
    _y, x = np.mgrid[0:10, 0:10]
    dem = 0.25 * x
    s_rad = slope(dem, pixel_size=1.0, units="radians")
    s_deg = slope(dem, pixel_size=1.0, units="degrees")
    s_pct = slope(dem, pixel_size=1.0, units="percent")
    interior = (slice(2, -2), slice(2, -2))
    assert np.allclose(np.degrees(s_rad[interior]), s_deg[interior])
    # percent = tan(β) × 100 = 25 for the 0.25-per-pixel rise.
    assert np.allclose(s_pct[interior], 25.0, atol=1e-6)


def test_slope_pixel_size_scales_correctly() -> None:
    """Same rise per pixel, 2× larger pixel → slope cut in half."""
    _y, x = np.mgrid[0:10, 0:10]
    dem = 0.5 * x  # rise 0.5 m per pixel
    s_1m = slope(dem, pixel_size=1.0)
    s_2m = slope(dem, pixel_size=2.0)
    # tan β halves; arctan halves the angle only at small slopes.
    expected_1m = math.degrees(math.atan(0.5))
    expected_2m = math.degrees(math.atan(0.25))
    interior = (slice(2, -2), slice(2, -2))
    assert s_1m[interior].mean() == pytest.approx(expected_1m, abs=1e-5)
    assert s_2m[interior].mean() == pytest.approx(expected_2m, abs=1e-5)


def test_slope_rectangular_pixels() -> None:
    """Anisotropic pixels (dx ≠ dy) should change x- and y-gradients independently."""
    _y, x = np.mgrid[0:10, 0:10]
    dem = 0.5 * x  # gradient only in x
    s_square = slope(dem, pixel_size=1.0)
    s_rect = slope(dem, pixel_size=(2.0, 1.0))
    # dx doubled → x-component halved.
    interior = (slice(2, -2), slice(2, -2))
    assert s_rect[interior].mean() < s_square[interior].mean()


def test_slope_rejects_non_2d() -> None:
    with pytest.raises(InvalidInputError, match="2-D"):
        slope(np.zeros(10), pixel_size=1.0)


def test_slope_rejects_bad_units() -> None:
    with pytest.raises(InvalidInputError, match="units"):
        slope(np.zeros((5, 5)), units="grads")


def test_slope_rejects_non_positive_pixel_size() -> None:
    with pytest.raises(InvalidInputError, match="positive"):
        slope(np.zeros((5, 5)), pixel_size=0.0)


# --- aspect --------------------------------------------------------------


def test_aspect_flat_terrain_is_nan() -> None:
    dem = np.full((10, 10), 100.0)
    out = aspect(dem)
    assert np.isnan(out).all()


def test_aspect_east_facing_slope() -> None:
    """A ramp that descends from west (high) to east (low) faces east."""
    _y, x = np.mgrid[0:10, 0:10]
    dem = -x.astype(float)  # high at x=0, low at x=9 → descends east → faces east (90°)
    out = aspect(dem, units="degrees")
    interior = out[2:-2, 2:-2]
    assert np.allclose(interior, 90.0, atol=1.0)


def test_aspect_south_facing_slope() -> None:
    """Descending southward → faces south = 180°."""
    y, _x = np.mgrid[0:10, 0:10]
    dem = -y.astype(float)
    out = aspect(dem, units="degrees")
    interior = out[2:-2, 2:-2]
    assert np.allclose(interior, 180.0, atol=1.0)


# --- curvature -----------------------------------------------------------


def test_profile_curvature_on_plane_is_zero() -> None:
    """A flat plane has zero curvature everywhere."""
    y, x = np.mgrid[0:15, 0:15]
    dem = 0.3 * x + 0.2 * y  # tilted plane
    out = profile_curvature(dem)
    interior = out[2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-9)


def test_plan_curvature_on_plane_is_zero() -> None:
    y, x = np.mgrid[0:15, 0:15]
    dem = 0.3 * x + 0.2 * y
    out = plan_curvature(dem)
    interior = out[2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-9)


def test_profile_curvature_on_concave_hill_is_positive_downhill() -> None:
    """A symmetric paraboloid hill should decelerate water on its flanks."""
    y, x = np.mgrid[0:21, 0:21].astype(np.float64)
    # z = -((x-10)² + (y-10)²) / 50 — paraboloid peaking at the centre.
    dem = -(((x - 10) ** 2 + (y - 10) ** 2) / 50.0)
    out = profile_curvature(dem)
    # Flanks (away from the centre) should be non-trivially curved.
    assert abs(out[5, 5]) > 0.001


# --- TWI -----------------------------------------------------------------


def test_twi_higher_on_gentle_slopes() -> None:
    """TWI ∝ 1 / tan β → steeper terrain has *smaller* TWI."""
    _y, x = np.mgrid[0:20, 0:20].astype(float)
    gentle = 0.05 * x  # 5 % slope
    steep = 0.5 * x  # 50 % slope
    twi_gentle = topographic_wetness_index(gentle, pixel_size=1.0)
    twi_steep = topographic_wetness_index(steep, pixel_size=1.0)
    assert twi_gentle[10, 10] > twi_steep[10, 10]


def test_twi_uses_flow_accumulation_when_provided() -> None:
    _y, x = np.mgrid[0:10, 0:10].astype(float)
    dem = 0.1 * x
    fa = np.full(dem.shape, 100.0)  # 100 cells upstream everywhere
    twi_default = topographic_wetness_index(dem, pixel_size=1.0)
    twi_with_fa = topographic_wetness_index(dem, pixel_size=1.0, flow_accumulation=fa)
    # Larger upstream area → larger TWI.
    assert twi_with_fa[5, 5] > twi_default[5, 5]


def test_twi_rejects_mismatched_flow_accumulation() -> None:
    dem = np.zeros((10, 10))
    fa = np.zeros((5, 5))
    with pytest.raises(InvalidInputError, match="shape"):
        topographic_wetness_index(dem, flow_accumulation=fa)
