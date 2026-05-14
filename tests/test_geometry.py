"""Tests for the vectorized point-in-polygon geometry."""

from __future__ import annotations

import numpy as np
import pytest

from pedotri.geometry import points_in_polygon, signed_distance_to_polygon

# A simple unit square for sanity tests
_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_points_inside_square() -> None:
    pts = np.array([[0.5, 0.5], [0.1, 0.9], [0.99, 0.01]])
    assert points_in_polygon(pts, _SQUARE).tolist() == [True, True, True]


def test_points_outside_square() -> None:
    pts = np.array([[1.5, 0.5], [-0.1, 0.5], [0.5, 1.5], [0.5, -0.1]])
    assert points_in_polygon(pts, _SQUARE).tolist() == [False] * 4


def test_points_on_edges_are_inside() -> None:
    pts = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
    assert points_in_polygon(pts, _SQUARE).all()


def test_vertices_are_inside() -> None:
    assert points_in_polygon(_SQUARE, _SQUARE).all()


def test_concave_polygon_works() -> None:
    # A "C" / pacman-ish polygon — the bite makes (0.5, 0.5) lie outside
    poly = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.4],
            [0.4, 0.4],
            [0.4, 0.6],
            [1.0, 0.6],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    pts = np.array([[0.5, 0.5], [0.2, 0.5], [0.7, 0.5]])
    assert points_in_polygon(pts, poly).tolist() == [False, True, False]


def test_signed_distance_inside_is_positive() -> None:
    # Center of the unit square is distance 0.5 from each edge
    d = signed_distance_to_polygon(np.array([[0.5, 0.5]]), _SQUARE)
    assert d[0] == pytest.approx(0.5)


def test_signed_distance_outside_is_negative() -> None:
    d = signed_distance_to_polygon(np.array([[2.0, 0.5]]), _SQUARE)
    assert d[0] == pytest.approx(-1.0)


def test_signed_distance_on_boundary_is_zero() -> None:
    d = signed_distance_to_polygon(np.array([[0.0, 0.5]]), _SQUARE)
    assert d[0] == pytest.approx(0.0)


def test_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="points"):
        points_in_polygon(np.zeros((3, 3)), _SQUARE)
    with pytest.raises(ValueError, match="polygon"):
        points_in_polygon(np.zeros((3, 2)), np.zeros((2, 2)))
