"""Vectorized point-in-polygon for soil texture classification.

Uses the crossing-number (ray casting) test of Sunday (2001), adapted to
numpy so a single call classifies many points against one polygon
without Python-level loops over points. The package depends only on
numpy for this core operation.

Reference:
    Sunday, D. (2001). Inclusion of a point in a polygon.
    https://web.archive.org/web/20210806044010/http://geomalgorithms.com/a03-_inclusion.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pedotri._types import FloatArray


def points_in_polygon(points: FloatArray, polygon: FloatArray) -> np.ndarray:
    """Return a boolean mask indicating which points are inside ``polygon``.

    Args:
        points: (M, 2) array of (x, y) query points.
        polygon: (N, 2) array of polygon vertices in order. Closure is
            implicit (the last vertex is connected back to the first).

    Returns:
        Boolean (M,) array; ``True`` where the corresponding query point
        is inside the polygon. Points exactly on an edge are considered
        inside (a common convention for soil-classification triangles,
        where boundary lines are typically attributed to the more
        clay-rich class by lookup order).
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (M, 2), got {points.shape}")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError(f"polygon must have shape (>=3, 2), got {polygon.shape}")

    # Close the polygon by appending the first vertex.
    closed = np.vstack([polygon, polygon[:1]])
    x = points[:, 0:1]  # (M, 1)
    y = points[:, 1:2]  # (M, 1)
    x1 = closed[:-1, 0]  # (N,)
    y1 = closed[:-1, 1]
    x2 = closed[1:, 0]
    y2 = closed[1:, 1]

    # For each (point, edge), is the point's y strictly between the edge endpoints?
    # The standard ray-casting trick: a horizontal ray to the right of the point
    # crosses the edge iff the y-spans overlap and the x at that y is to the right.
    crosses_y = ((y1 <= y) & (y < y2)) | ((y2 <= y) & (y < y1))  # (M, N)

    # x of intersection of horizontal ray y = point.y with the edge segment:
    # x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1).
    # Horizontal or near-horizontal edges (tiny dy) would overflow the
    # divide; we filter those out via crosses_y above, so the slope value
    # itself doesn't matter for the final result and we can swallow all
    # the numerical complaints.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dy = y2 - y1
        slope = np.where(dy != 0, (x2 - x1) / dy, 0.0)
        x_int = x1 + (y - y1) * slope  # (M, N) via broadcasting

    crosses = crosses_y & (x < x_int)
    inside = (crosses.sum(axis=1) % 2).astype(bool)

    # Also include points that lie exactly on a polygon edge — the
    # ray-cast test is half-open and would otherwise drop them.
    on_edge = _points_on_edge(points, closed)
    return np.asarray(inside | on_edge)


def _points_on_edge(points: FloatArray, closed_poly: FloatArray) -> np.ndarray:
    """Boolean mask of points lying exactly on a polygon edge.

    ``closed_poly`` is the polygon with its first vertex repeated as the
    last vertex (length N+1 for an N-gon).
    """
    x = points[:, 0:1]
    y = points[:, 1:2]
    x1 = closed_poly[:-1, 0]
    y1 = closed_poly[:-1, 1]
    x2 = closed_poly[1:, 0]
    y2 = closed_poly[1:, 1]

    # Vector from edge endpoint to point and edge vector
    dx, dy = x2 - x1, y2 - y1
    px, py = x - x1, y - y1

    # Cross product: 0 means colinear with the edge.
    cross = px * dy - py * dx
    # Dot product: must be in [0, edge_len_sq] to lie within the segment.
    edge_sq = dx * dx + dy * dy
    dot = px * dx + py * dy

    # Allow a tiny numerical tolerance — vertices are typically expressed
    # as small integers/decimals, so 1e-9 is more than enough.
    eps = 1e-9
    on_line = np.abs(cross) <= eps
    within = (dot >= -eps) & (dot <= edge_sq + eps)
    return np.asarray((on_line & within).any(axis=1))


def signed_distance_to_polygon(points: FloatArray, polygon: FloatArray) -> FloatArray:
    """Return the signed Euclidean distance from each point to the polygon edge.

    Positive inside, negative outside, zero on the boundary. Useful as a
    "confidence" measure for points sitting near a class boundary.

    Args:
        points: (M, 2)
        polygon: (N, 2)

    Returns:
        (M,) float64 array. The magnitude is the unsigned distance to
        the nearest edge; the sign is determined by
        :func:`points_in_polygon`.
    """
    inside = points_in_polygon(points, polygon)
    closed = np.vstack([polygon, polygon[:1]])
    seg_a = closed[:-1]  # (N, 2)
    seg_b = closed[1:]  # (N, 2)

    # For each (point, segment) compute the minimum distance.
    p = points[:, np.newaxis, :]  # (M, 1, 2)
    a = seg_a[np.newaxis, :, :]  # (1, N, 2)
    b = seg_b[np.newaxis, :, :]
    ab = b - a
    ap = p - a
    ab_sq = (ab * ab).sum(axis=2)  # (1, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ab_sq != 0, (ap * ab).sum(axis=2) / ab_sq, 0.0)
    t = np.clip(t, 0.0, 1.0)
    nearest = a + t[..., np.newaxis] * ab  # (M, N, 2)
    diff = p - nearest
    dist = np.sqrt((diff * diff).sum(axis=2))  # (M, N)
    min_dist = np.asarray(dist.min(axis=1), dtype=np.float64)
    return np.asarray(np.where(inside, min_dist, -min_dist), dtype=np.float64)
