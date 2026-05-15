"""Property-based tests for pedotri invariants.

These use Hypothesis to generate random inputs and check properties that
should hold regardless of which classification or point is chosen. They
complement the table-driven golden tests by surfacing edge cases the
hand-written tests miss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pedotri
from pedotri.geometry import points_in_polygon, signed_distance_to_polygon

# Built-in 2-D classifications used in the parametric properties.
_2D_KEYS = [
    "USDA",
    "FAO",
    "INTERNATIONAL",
    "ISSS",
    "GEPPA",
    "JAMAGNE",
    "HYPRES",
    "EMBRAPA",
    "KA5",
    "NORTHCOTE",
    "PTG",
    "CHINA",
    "AVERY",
]


@st.composite
def _sand_clay(draw: st.DrawFn) -> tuple[float, float]:
    """Generate (sand, clay) pairs that satisfy sand + clay <= 100."""
    sand = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    max_clay = 100.0 - sand
    clay = draw(st.floats(min_value=0.0, max_value=max_clay, allow_nan=False))
    return sand, clay


@pytest.mark.parametrize("key", _2D_KEYS)
@given(point=_sand_clay())
@settings(
    max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_classify_never_raises_in_simplex(key: str, point: tuple[float, float]) -> None:
    """Every point on the simplex classifies to either a class key or None."""
    sand, clay = point
    result = pedotri.classify(sand, clay, key)
    assert result is None or isinstance(result, str)


# Classifications whose polygons are *designed* to tile the entire
# sand-silt-clay simplex with no gaps. (Any classification could in
# principle have intentional gaps; the ones listed here promise
# complete coverage.)
_TILING_2D_KEYS = [
    "USDA",
    "FAO",
    "ISSS",
    "HYPRES",
    "EMBRAPA",
    "KA5",
    "GEPPA",
    "JAMAGNE",
    "NORTHCOTE",
    "PTG",
    "CHINA",
    "AVERY",
]


@pytest.mark.parametrize("key", _TILING_2D_KEYS)
@given(point=_sand_clay())
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_tiling_classifications_cover_simplex(key: str, point: tuple[float, float]) -> None:
    """Tiling classifications never return None on any interior point.

    Surfaces accidental coverage gaps introduced by hand-authored polygon
    sets (KA5's 31-class transcription and Northcote's clay subdivisions
    were the historically risky additions).
    """
    sand, clay = point
    result = pedotri.classify(sand, clay, key)
    assert result is not None, f"{key} has a coverage gap at ({sand}, {clay})"


@pytest.mark.parametrize("key", _2D_KEYS)
@given(point=_sand_clay())
@settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_classify_matches_expected_classes(key: str, point: tuple[float, float]) -> None:
    """If a point classifies, its key is one of the classification's classes."""
    sand, clay = point
    result = pedotri.classify(sand, clay, key)
    if result is not None:
        expected = set(pedotri.get_classification(key).class_keys())
        assert result in expected


@pytest.mark.parametrize("key", _2D_KEYS)
@given(point=_sand_clay())
@settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_scalar_and_array_classify_match(key: str, point: tuple[float, float]) -> None:
    """Classifying [(s, c)] and (s, c) must give the same answer."""
    sand, clay = point
    scalar = pedotri.classify(sand, clay, key)
    batched = pedotri.classify([sand], [clay], key)
    assert batched[0] == scalar


@given(point=_sand_clay())
@settings(max_examples=100, deadline=None)
def test_kachinsky_pure_clay_axis(point: tuple[float, float]) -> None:
    """The 1-D Kachinsky classifier covers [0, 100] without gaps for the
    'physical_clay' axis (i.e. always classifies, never returns None)."""
    pc = point[1]  # use the clay component as the physical-clay value
    result = pedotri.classify(pc, "KACHINSKY")
    assert result is not None
    assert isinstance(result, str)


# --- Geometry properties -------------------------------------------------


@st.composite
def _convex_polygon(draw: st.DrawFn) -> np.ndarray:
    """Generate a small triangle in [0, 100]^2 for geometry tests."""
    pts = [
        (
            draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
            draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
        )
        for _ in range(3)
    ]
    return np.array(pts)


@given(poly=_convex_polygon(), point=_sand_clay())
@settings(max_examples=80, deadline=None)
def test_signed_distance_sign_agrees_with_inside_test(
    poly: np.ndarray, point: tuple[float, float]
) -> None:
    """The sign of ``signed_distance_to_polygon`` must match
    ``points_in_polygon`` (modulo the zero / on-boundary case)."""
    # Skip degenerate triangles where vertices are too close to be a polygon.
    a, b, c_ = poly[0], poly[1], poly[2]
    edge_area = abs((b[0] - a[0]) * (c_[1] - a[1]) - (c_[0] - a[0]) * (b[1] - a[1]))
    if edge_area < 1e-6:
        return
    pts = np.array([point])
    inside = points_in_polygon(pts, poly)
    dist = signed_distance_to_polygon(pts, poly)
    if not math.isclose(float(dist[0]), 0.0, abs_tol=1e-6):
        assert bool(inside[0]) == bool(dist[0] > 0)


@given(point=_sand_clay())
@settings(max_examples=50, deadline=None)
def test_classify_locale_preserves_class(
    point: tuple[float, float],
) -> None:
    """Switching locale must never change which class a point belongs to —
    only the label it is rendered with."""
    sand, clay = point
    key = pedotri.classify(sand, clay, "USDA")
    name_fr = pedotri.classify(sand, clay, "USDA", locale="fr")
    if key is None:
        assert name_fr is None
        return
    # Detailed result with French locale must agree with both.
    detailed = pedotri.classify(sand, clay, "USDA", detailed=True, locale="fr")
    assert detailed.key == key
