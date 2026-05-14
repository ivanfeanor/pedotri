"""Tests for particle-size standard conversions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.psd import (
    SAND_SILT_CUTOFF_MM,
    convert,
    cutoff_mm,
    interpolate_psd,
    supported_standards,
)


def test_supported_standards() -> None:
    assert set(supported_standards()) == {
        "USDA",
        "FAO",
        "ISSS",
        "INTERNATIONAL",
        "KA5",
    }


def test_cutoff_mm_lookup() -> None:
    assert cutoff_mm("USDA") == 0.05
    assert cutoff_mm("ISSS") == 0.02
    assert cutoff_mm("KA5") == 0.063


def test_convert_same_standard_is_noop() -> None:
    s, si, c = convert(60.0, 30.0, 10.0, source="USDA", target="USDA")
    assert s.tolist() == [60.0]
    assert si.tolist() == [30.0]
    assert c.tolist() == [10.0]


def test_convert_same_cutoff_different_name() -> None:
    # USDA and FAO share the 0.05 mm sand/silt boundary
    s, si, _ = convert(60, 30, 10, source="USDA", target="FAO")
    assert s.tolist() == [60.0]
    assert si.tolist() == [30.0]


def test_convert_usda_to_isss_moves_silt_to_sand() -> None:
    """USDA cutoff (0.05) > ISSS cutoff (0.02); part of USDA silt becomes ISSS sand."""
    s, si, c = convert(60, 30, 10, source="USDA", target="ISSS")
    # ISSS sand should be strictly greater than USDA sand
    assert s[0] > 60.0
    # ISSS silt should be strictly less than USDA silt
    assert si[0] < 30.0
    # Clay unchanged
    assert c[0] == 10.0
    # Total must still equal 100
    assert s[0] + si[0] + c[0] == pytest.approx(100.0, abs=0.01)


def test_convert_isss_to_usda_moves_sand_to_silt() -> None:
    """ISSS cutoff (0.02) < USDA cutoff (0.05); part of ISSS sand becomes USDA silt."""
    s, si, c = convert(68.54, 21.46, 10, source="ISSS", target="USDA")
    assert s[0] < 68.54
    assert si[0] > 21.46
    assert c[0] == 10.0


def test_convert_loglinear_factor_for_usda_to_isss() -> None:
    """The log-linear silt-retention fraction is fully determined by the
    fixed cutoffs (0.002 / 0.02 / 0.05 mm)."""
    expected_frac_stays_silt = (math.log10(0.02) - math.log10(0.002)) / (
        math.log10(0.05) - math.log10(0.002)
    )
    # ≈ 0.7153
    assert expected_frac_stays_silt == pytest.approx(0.7153, abs=1e-3)

    _, si, _ = convert(60, 30, 10, source="USDA", target="ISSS")
    assert si[0] == pytest.approx(30 * expected_frac_stays_silt, abs=1e-2)


def test_convert_arrays() -> None:
    s, si, _ = convert(sand=[60, 20], silt=[30, 50], clay=[10, 30], source="USDA", target="ISSS")
    assert s.shape == (2,)
    assert si.shape == (2,)
    # Both should have moved some silt into sand
    assert s[0] > 60
    assert s[1] > 20


def test_convert_broadcasting() -> None:
    """Mixed scalar + array inputs should broadcast cleanly."""
    # Both rows must sum to 100 after broadcasting clay=10.
    s, _, c = convert(
        sand=np.array([60, 20]),
        silt=np.array([30, 70]),
        clay=10,
        source="USDA",
        target="ISSS",
    )
    assert c.tolist() == [10.0, 10.0]
    assert s.shape == (2,)


def test_convert_kachinsky_unsupported_raises() -> None:
    """KACHINSKY isn't a particle-size standard in the table."""
    with pytest.raises(InvalidInputError, match="Unknown source"):
        convert(60, 30, 10, source="KACHINSKY", target="USDA")
    with pytest.raises(InvalidInputError, match="Unknown target"):
        convert(60, 30, 10, source="USDA", target="GEPPA")


def test_convert_rejects_bad_sum() -> None:
    with pytest.raises(InvalidInputError, match="sum to 100"):
        convert(60, 60, 10, source="USDA", target="ISSS")


def test_convert_rejects_out_of_range() -> None:
    with pytest.raises(InvalidInputError, match=r"\[0, 100\]"):
        convert(-5, 95, 10, source="USDA", target="ISSS")


def test_convert_rejects_nan() -> None:
    with pytest.raises(InvalidInputError, match="NaN"):
        convert([float("nan"), 60], [30, 30], [10, 10], source="USDA", target="ISSS")


# --- interpolate_psd -----------------------------------------------------


def test_interpolate_psd_at_measured_point_returns_exact() -> None:
    sizes = np.array([0.002, 0.02, 0.05, 2.0])
    passing = np.array([15.0, 35.0, 50.0, 100.0])
    # Evaluating at a measured sieve returns the measured value
    assert interpolate_psd(sizes, passing, 0.02) == pytest.approx(35.0)
    assert interpolate_psd(sizes, passing, 0.05) == pytest.approx(50.0)


def test_interpolate_psd_between_measured_points() -> None:
    """Log-linear interpolation across [0.02, 0.05] mm sieves."""
    sizes = np.array([0.02, 0.05])
    passing = np.array([35.0, 50.0])
    # At the geometric mean of the bounds, expect the linear midpoint.
    geom_mid = math.sqrt(0.02 * 0.05)
    result = interpolate_psd(sizes, passing, geom_mid)
    assert result == pytest.approx(42.5)


def test_interpolate_psd_rejects_out_of_range() -> None:
    sizes = np.array([0.02, 0.05])
    passing = np.array([35.0, 50.0])
    with pytest.raises(InvalidInputError, match="outside the measured range"):
        interpolate_psd(sizes, passing, 1.0)


def test_interpolate_psd_rejects_non_monotonic_sizes() -> None:
    with pytest.raises(InvalidInputError, match="strictly increasing"):
        interpolate_psd(np.array([0.05, 0.02]), np.array([50, 35]), 0.03)


def test_interpolate_psd_rejects_non_monotonic_passing() -> None:
    with pytest.raises(InvalidInputError, match="non-decreasing"):
        interpolate_psd(np.array([0.02, 0.05]), np.array([50, 35]), 0.03)


def test_constants_match_expected() -> None:
    """Cutoff values shouldn't drift silently."""
    assert SAND_SILT_CUTOFF_MM == {
        "USDA": 0.05,
        "FAO": 0.05,
        "ISSS": 0.02,
        "INTERNATIONAL": 0.02,
        "KA5": 0.063,
    }
