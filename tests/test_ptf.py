"""Tests for pedotransfer functions."""

from __future__ import annotations

import pytest

from pedotri.errors import InvalidInputError
from pedotri.ptf import SaxtonRawlsResult, WostenResult, saxton_rawls, wosten

# --- Saxton-Rawls --------------------------------------------------------


def test_saxton_rawls_returns_result_for_scalar() -> None:
    r = saxton_rawls(60, 10, 2.5)
    assert isinstance(r, SaxtonRawlsResult)


def test_saxton_rawls_returns_list_for_arrays() -> None:
    rs = saxton_rawls([60, 20], [10, 50], [2.5, 2.5])
    assert isinstance(rs, list)
    assert len(rs) == 2
    assert all(isinstance(r, SaxtonRawlsResult) for r in rs)


def test_saxton_rawls_broadcasts_organic_matter() -> None:
    """A scalar organic_matter should broadcast across an array of textures."""
    rs = saxton_rawls([60, 20], [10, 50], organic_matter=2.5)
    assert len(rs) == 2


def test_saxton_rawls_sandy_loam_in_expected_range() -> None:
    """A typical sandy loam (S=65, C=10, OM=2.5) should fall in the
    well-known ballpark: low WP, moderate FC, high K_s.

    Tolerances are deliberately loose. Saxton-Rawls 2006 Table 1 is
    derived from the same regressions implemented here, but the table
    values are rounded and use representative class centroids, so
    point-by-point comparison drifts by a few hundredths.
    """
    r = saxton_rawls(sand=65, clay=10, organic_matter=2.5)
    assert 0.03 < r.wilting_point < 0.10
    assert 0.12 < r.field_capacity < 0.22
    assert 0.35 < r.saturation < 0.50
    assert r.available_water > 0.05
    assert r.saturated_conductivity > 10  # mm/h, well-drained sandy


def test_saxton_rawls_clay_in_expected_range() -> None:
    """Heavy clay (S=20, C=60, OM=2.5) — high retention, low K_s."""
    r = saxton_rawls(sand=20, clay=60, organic_matter=2.5)
    assert 0.25 < r.wilting_point < 0.40
    assert 0.40 < r.field_capacity < 0.55
    assert 0.45 < r.saturation < 0.60
    assert r.saturated_conductivity < 5  # mm/h, slow drainage


def test_saxton_rawls_monotonic_across_texture() -> None:
    """Field capacity and wilting point must increase with clay content."""
    sandy = saxton_rawls(80, 5, 2.0)
    loamy = saxton_rawls(40, 20, 2.0)
    clayey = saxton_rawls(20, 50, 2.0)
    assert sandy.wilting_point < loamy.wilting_point < clayey.wilting_point
    assert sandy.field_capacity < loamy.field_capacity < clayey.field_capacity
    # And conductivity must decrease with clay
    assert sandy.saturated_conductivity > loamy.saturated_conductivity
    assert loamy.saturated_conductivity > clayey.saturated_conductivity


def test_saxton_rawls_invariants() -> None:
    """Water-content monotonicity must hold across textures."""
    for sand, clay in [(80, 5), (50, 20), (20, 50), (10, 80)]:
        r = saxton_rawls(sand, clay)
        assert 0 < r.wilting_point < r.field_capacity < r.saturation < 1
        assert r.available_water > 0
        assert r.bulk_density > 0
        assert r.saturated_conductivity > 0


def test_saxton_rawls_rejects_out_of_range() -> None:
    with pytest.raises(InvalidInputError, match=r"\[0, 100\]"):
        saxton_rawls(150, 10)
    with pytest.raises(InvalidInputError, match=r"\[0, 100\]"):
        saxton_rawls(50, 10, organic_matter=-1)


def test_saxton_rawls_rejects_nan() -> None:
    with pytest.raises(InvalidInputError, match="NaN"):
        saxton_rawls([60, float("nan")], [10, 10])


# --- Wösten 1999 ---------------------------------------------------------


def test_wosten_returns_result_for_scalar() -> None:
    w = wosten(sand=60, silt=30, clay=10, organic_matter=2.5, bulk_density=1.4)
    assert isinstance(w, WostenResult)
    assert w.theta_r == 0.01


def test_wosten_returns_list_for_arrays() -> None:
    ws = wosten([60, 20], [30, 50], [10, 30], organic_matter=2.5, bulk_density=1.4)
    assert isinstance(ws, list)
    assert len(ws) == 2


def test_wosten_basic_invariants() -> None:
    """Output ranges must be physically sensible."""
    w = wosten(sand=60, silt=30, clay=10, organic_matter=2.5, bulk_density=1.4)
    assert 0 < w.theta_s < 1
    assert w.alpha > 0
    assert w.n > 1  # van Genuchten n is strictly > 1 by construction
    assert w.saturated_conductivity > 0
    assert -10 <= w.l <= 10  # Wösten's logistic bound


def test_saxton_rawls_density_factor_increases_bulk_density() -> None:
    """Saxton-Rawls 2006 Eq. 6-7: DF > 1 compacts the soil."""
    normal = saxton_rawls(65, 10, 2.5)
    compacted = saxton_rawls(65, 10, 2.5, density_factor=1.10)
    assert compacted.bulk_density == pytest.approx(normal.bulk_density * 1.10, rel=1e-6)
    # Compaction reduces porosity → lower saturation.
    assert compacted.saturation < normal.saturation
    # Field capacity moves slightly under compaction (Eq. 8).
    assert compacted.field_capacity <= normal.field_capacity
    # Wilting point is unaffected by compaction in Saxton-Rawls.
    assert compacted.wilting_point == pytest.approx(normal.wilting_point)


def test_saxton_rawls_default_density_factor_matches_normal() -> None:
    """density_factor=1.0 (default) must reproduce the original regression."""
    explicit = saxton_rawls(65, 10, 2.5, density_factor=1.0)
    default = saxton_rawls(65, 10, 2.5)
    assert explicit.bulk_density == pytest.approx(default.bulk_density)
    assert explicit.saturation == pytest.approx(default.saturation)
    assert explicit.field_capacity == pytest.approx(default.field_capacity)


def test_wosten_topsoil_vs_subsoil_differ() -> None:
    """The topsoil flag must materially change the prediction."""
    top = wosten(60, 30, 10, organic_matter=2.5, bulk_density=1.4, topsoil=True)
    sub = wosten(60, 30, 10, organic_matter=2.5, bulk_density=1.4, topsoil=False)
    assert top.saturated_conductivity != sub.saturated_conductivity
    assert top.alpha != sub.alpha


def test_wosten_rejects_zero_silt_clay_om() -> None:
    """Wösten's 1/x and ln(x) terms require strictly positive inputs."""
    with pytest.raises(InvalidInputError, match="strictly positive"):
        wosten(95, 0, 5)
    with pytest.raises(InvalidInputError, match="strictly positive"):
        wosten(60, 30, 10, organic_matter=0)
    with pytest.raises(InvalidInputError, match="strictly positive"):
        wosten(60, 30, 10, bulk_density=0)


def test_wosten_rejects_nan() -> None:
    with pytest.raises(InvalidInputError, match="NaN"):
        wosten([60, float("nan")], [30, 30], [10, 10], organic_matter=2.5)


def test_wosten_rejects_out_of_range_fraction() -> None:
    with pytest.raises(InvalidInputError, match=r"\[0, 100\]"):
        wosten(sand=120, silt=30, clay=10, organic_matter=2.5, bulk_density=1.4)


def test_wosten_rejects_bulk_density_nan() -> None:
    with pytest.raises(InvalidInputError, match="bulk_density"):
        wosten(sand=60, silt=30, clay=10, organic_matter=2.5, bulk_density=float("nan"))
