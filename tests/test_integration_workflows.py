"""Integration tests for the full 0.3 SoC workflow.

These tests stitch together every 0.3 primitive
(``aggregate_depths`` → ``zonal_aggregate`` → ``combine`` plus the
``classify`` + ``classify_all`` surface) on synthetic inputs that
mirror what a production agronomic pipeline plugs in. They double as
executable specifications of how the pieces are meant to chain.

No network is required. The fixtures fabricate SoilGrids- and
WorldCover-shaped data inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pedotri
from pedotri.uncertainty import Quantiles
from pedotri.zonal import aggregate_depths, zonal_aggregate

H, W = 80, 80
WORLDCOVER_CROPLAND = 40
WORLDCOVER_TREE = 10


def _synthetic_layer(
    mean: float, halfwidth: float, *, shape: tuple[int, int] = (H, W), seed: int = 0
) -> dict[str, Any]:
    """Return a per-depth spec ready to pass to aggregate_depths()."""
    rng = np.random.default_rng(seed)
    m = np.full(shape, mean, dtype=np.float64) + rng.normal(scale=0.25, size=shape)
    return {"mean": m, "uncertainty": Quantiles(m - halfwidth, m + halfwidth)}


def _half_cropland_mask() -> np.ndarray:
    mask = np.full((H, W), WORLDCOVER_TREE, dtype=np.int32)
    mask[:, ::2] = WORLDCOVER_CROPLAND
    return mask


# --- Full SOC stock chain -----------------------------------------------


def test_full_soc_stock_workflow_independent_runs_match() -> None:
    """End-to-end determinism: same inputs + seed → identical stock samples."""
    region = np.zeros((H, W), dtype=bool)
    region[H // 2 :, W // 2 :] = True
    mask = _half_cropland_mask()

    weights = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
    soc_depths = {
        "0-5cm": _synthetic_layer(24.0, 4.0, seed=11),
        "5-15cm": _synthetic_layer(22.0, 4.0, seed=12),
        "15-30cm": _synthetic_layer(19.0, 4.0, seed=13),
    }
    bd_depths = {
        "0-5cm": _synthetic_layer(1.30, 0.10, seed=21),
        "5-15cm": _synthetic_layer(1.35, 0.10, seed=22),
        "15-30cm": _synthetic_layer(1.40, 0.10, seed=23),
    }

    def run() -> tuple[float, float, float]:
        soc_mean, soc_q = aggregate_depths(soc_depths, weights)
        bd_mean, bd_q = aggregate_depths(bd_depths, weights)
        agg = zonal_aggregate(
            region=region,
            properties={
                "soc": {"mean": soc_mean, "uncertainty": soc_q},
                "bd": {"mean": bd_mean, "uncertainty": bd_q},
            },
            mask=mask,
            mask_include=[WORLDCOVER_CROPLAND],
            n_samples=400,
            seed=777,
        )
        # 1 ha pixels (synthetic), 0.30 m depth, tonnes SOC.
        pixel_area = 10_000.0
        stock = agg.combine(
            lambda soc, bd, depth=0.30, area=pixel_area * agg.n_pixels_used: (
                soc * 1e-3 * bd * depth * area * 1e-3
            ),
        )
        return stock.mean, stock.q05, stock.q95

    a = run()
    b = run()
    assert a == b


def test_full_soc_stock_workflow_qbounds_bracket_mean() -> None:
    region = np.ones((H, W), dtype=bool)
    soc_mean, soc_q = aggregate_depths(
        {
            "0-5cm": _synthetic_layer(24.0, 5.0, seed=31),
            "5-15cm": _synthetic_layer(22.0, 5.0, seed=32),
            "15-30cm": _synthetic_layer(19.0, 5.0, seed=33),
        },
        {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15},
    )
    bd_mean, bd_q = aggregate_depths(
        {
            "0-5cm": _synthetic_layer(1.30, 0.15, seed=41),
            "5-15cm": _synthetic_layer(1.35, 0.15, seed=42),
            "15-30cm": _synthetic_layer(1.40, 0.15, seed=43),
        },
        {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15},
    )

    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": {"mean": soc_mean, "uncertainty": soc_q},
            "bd": {"mean": bd_mean, "uncertainty": bd_q},
        },
        n_samples=600,
        seed=99,
    )
    pixel_area = 10_000.0
    stock = agg.combine(
        lambda soc, bd, depth=0.30, area=pixel_area * agg.n_pixels_used: (
            soc * 1e-3 * bd * depth * area * 1e-3
        ),
    )
    assert stock.q05 < stock.mean < stock.q95
    # Sanity: stock ≈ soc·bd·0.30·area on the mean inputs.
    expected_mean = (
        agg["soc"].mean * 1e-3 * agg["bd"].mean * 0.30 * pixel_area * agg.n_pixels_used * 1e-3
    )
    assert stock.mean == pytest.approx(expected_mean, rel=0.02)


# --- classify_all from the same SoilGrids-like data -----------------------


def test_classify_all_from_soilgrids_quantiles_chain() -> None:
    """Mock-SoilGrids → classify_all → ClassifyResult under every 2-axis scheme."""
    # Fake SoilGrids response for one pixel at (27, 45) percent.
    sand, clay = 27.0, 45.0
    out = pedotri.classify_all(sand=sand, clay=clay)
    assert "USDA" in out
    assert "FAO" in out
    # Every result should sit inside some class polygon (these values are
    # not at a corner / edge).
    assert all(r.key is not None for r in out.values())


# --- raster classify with uncertainty + downstream queries ----------------


def test_raster_classify_then_class_probability_query() -> None:
    """Classify a synthetic raster with uncertainty, then ask for the
    pixel-wise probability of a specific class."""
    from pedotri.raster import classify_array_with_uncertainty

    sand = np.full((20, 20), 27.0)
    clay = np.full((20, 20), 45.0)
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 4, sand + 4),
        clay_uncertainty=Quantiles(clay - 4, clay + 4),
        classification="USDA",
        method="monte_carlo",
        n_samples=400,
        seed=3,
    )
    p_clay = r.class_probability("clay")
    assert p_clay.shape == sand.shape
    # Most pixels are firmly in "clay" — mean probability should be high.
    assert float(p_clay.mean()) > 0.5
