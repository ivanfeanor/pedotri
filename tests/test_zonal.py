"""Tests for pedotri.zonal — regional aggregation with uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.uncertainty import Quantiles
from pedotri.zonal import AggregateDistribution, zonal_aggregate


def _uniform_property(shape: tuple[int, int], mean: float, halfwidth: float) -> dict[str, Any]:
    """A uniform-raster property spec in the dict form preferred by 0.3.x."""
    m = np.full(shape, mean, dtype=np.float64)
    return {"mean": m, "uncertainty": Quantiles(m - halfwidth, m + halfwidth)}


def _uniform_property_triple(
    shape: tuple[int, int], mean: float, halfwidth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The deprecated 3-tuple form; kept for the back-compat test below."""
    m = np.full(shape, mean, dtype=np.float64)
    return m, m - halfwidth, m + halfwidth


def test_aggregate_recovers_constant_mean() -> None:
    shape = (40, 40)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=1000,
        seed=0,
    )
    # Across 1600 pixels × 1000 samples, the regional mean is essentially
    # the input mean. The CLT also shrinks the regional Q05/Q95 way
    # below the per-pixel Q05/Q95 width.
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.05)
    assert (agg["soc"].q95 - agg["soc"].q05) < 1.0
    assert agg.n_pixels_used == 1600
    assert agg.mask_coverage == 1.0


def test_aggregate_dispatches_two_properties_independently() -> None:
    shape = (20, 20)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": _uniform_property(shape, 25.0, 4.0),
            "bd": _uniform_property(shape, 1.3, 0.1),
        },
        n_samples=500,
        seed=1,
    )
    assert set(agg.keys()) == {"soc", "bd"}
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.2)
    assert agg["bd"].mean == pytest.approx(1.3, abs=0.01)
    assert agg["soc"].n_pixels_used == agg["bd"].n_pixels_used == 400


def test_mask_narrows_pixel_count() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=np.int32)
    mask[:, ::2] = 40  # 50 % of pixels labelled "cropland"
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        mask=mask,
        mask_include=[40],
        n_samples=200,
        seed=2,
    )
    assert agg.n_pixels_used == 50
    assert agg.mask_coverage == pytest.approx(0.5)


def test_boolean_mask_input() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    mask[5:, :] = True  # bottom half
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 30.0, 2.0)},
        mask=mask,
        n_samples=200,
        seed=3,
    )
    assert agg.n_pixels_used == 50


def test_combine_propagates_through_user_formula() -> None:
    shape = (30, 30)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": _uniform_property(shape, 25.0, 4.0),  # g/kg
            "bd": _uniform_property(shape, 1.3, 0.1),  # g/cm³
        },
        n_samples=2000,
        seed=4,
    )

    # Naive SOC stock per pixel — depth + area passed via defaults.
    stock = agg.combine(
        lambda soc, bd, depth=0.30, area=1.0: soc * 0.001 * bd * 1000.0 * depth * area,
        name="soc_stock_per_m2",
    )
    # 25 × 0.001 × 1.3 × 1000 × 0.30 × 1.0 = 9.75 kg/m²
    assert stock.mean == pytest.approx(9.75, abs=0.1)
    # CLT: the regional product has small variance from the inputs'
    # per-pixel σ averaged across 900 pixels.
    assert stock.std < 0.2
    assert stock.name == "soc_stock_per_m2"
    assert stock.n_pixels_used == 900
    # Sample array carries one entry per draw.
    assert stock.samples.shape == (2000,)


def test_combine_ignores_unused_properties() -> None:
    """combine() should only pass the property samples fn actually asks for."""
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": _uniform_property(shape, 25.0, 4.0),
            "bd": _uniform_property(shape, 1.3, 0.1),
            "ph": _uniform_property(shape, 6.8, 0.3),
        },
        n_samples=200,
        seed=4,
    )
    # Function only consumes soc & bd; ph should be silently dropped.
    stock = agg.combine(
        lambda soc, bd, depth=0.30, area=1.0: soc * 0.001 * bd * 1000.0 * depth * area,
    )
    assert stock.samples.shape == (200,)
    assert stock.mean == pytest.approx(9.75, abs=0.1)


def test_combine_with_var_kwargs_receives_everything() -> None:
    shape = (8, 8)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": _uniform_property(shape, 25.0, 4.0),
            "bd": _uniform_property(shape, 1.3, 0.1),
        },
        n_samples=200,
        seed=5,
    )
    received: list[str] = []

    def stock_fn(**props: np.ndarray) -> np.ndarray:
        received.extend(sorted(props.keys()))
        return np.asarray(props["soc"] * props["bd"])

    agg.combine(stock_fn)
    assert received == ["bd", "soc"]


def test_combine_rejects_scalar_returning_formula() -> None:
    shape = (5, 5)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=100,
        seed=5,
    )
    with pytest.raises(InvalidInputError, match="must return an array"):
        agg.combine(lambda soc: float(soc.mean()))


def test_reproducibility_with_seed() -> None:
    shape = (15, 15)
    region = np.ones(shape, dtype=bool)
    agg1 = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=500,
        seed=2025,
    )
    agg2 = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=500,
        seed=2025,
    )
    np.testing.assert_array_equal(agg1["soc"].samples, agg2["soc"].samples)


def test_region_shape_mismatch_rejected() -> None:
    shape = (10, 10)
    bad_region = np.ones((9, 10), dtype=bool)
    with pytest.raises(InvalidInputError, match="region shape"):
        zonal_aggregate(
            region=bad_region,
            properties={"soc": _uniform_property(shape, 25.0, 4.0)},
            n_samples=100,
            seed=6,
        )


def test_mask_shape_mismatch_rejected() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    bad_mask = np.ones((11, 10), dtype=bool)
    with pytest.raises(InvalidInputError, match="mask shape"):
        zonal_aggregate(
            region=region,
            properties={"soc": _uniform_property(shape, 25.0, 4.0)},
            mask=bad_mask,
            n_samples=100,
            seed=7,
        )


def test_property_spec_dict_with_quantiles() -> None:
    """Dict spec with explicit Quantiles is the recommended form."""
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": {"mean": m, "uncertainty": Quantiles(m - 4, m + 4)},
        },
        n_samples=200,
        seed=42,
    )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_dict_with_sigma() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    s = np.full(shape, 2.0)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": {"mean": m, "sigma": s}},
        n_samples=200,
        seed=42,
    )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_dict_legacy_q05_q95() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": {"mean": m, "q05": m - 4, "q95": m + 4}},
        n_samples=200,
        seed=42,
    )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_two_tuple_with_quantiles() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": (m, Quantiles(m - 4, m + 4))},
        n_samples=200,
        seed=42,
    )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_two_tuple_with_sigma_array() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": (m, np.full(shape, 2.0))},
        n_samples=200,
        seed=42,
    )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_three_tuple_emits_deprecation() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    triple = _uniform_property_triple(shape, 25.0, 4.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        agg = zonal_aggregate(
            region=region,
            properties={"soc": triple},
            n_samples=200,
            seed=42,
        )
    assert agg["soc"].mean == pytest.approx(25.0, abs=0.5)


def test_property_spec_unsupported_shape_rejected() -> None:
    shape = (5, 5)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    with pytest.raises(InvalidInputError, match="spec shape"):
        zonal_aggregate(
            region=region,
            properties={"soc": [m, m, m]},  # list, not tuple/dict
            n_samples=100,
            seed=8,
        )


def test_property_spec_dict_missing_mean_rejected() -> None:
    shape = (5, 5)
    region = np.ones(shape, dtype=bool)
    m = np.full(shape, 25.0)
    with pytest.raises(InvalidInputError, match="'mean' key"):
        zonal_aggregate(
            region=region,
            properties={"soc": {"q05": m, "q95": m + 1}},
            n_samples=100,
            seed=8,
        )


def test_empty_intersection_returns_zero_pixel_distributions() -> None:
    shape = (5, 5)
    region = np.zeros(shape, dtype=bool)  # nothing inside the AOI
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=100,
        seed=9,
    )
    assert agg.n_pixels_used == 0
    assert agg["soc"].samples.size == 0
    assert agg.region_pixels == 0


def test_quantile_helpers_match_numpy() -> None:
    samples = np.random.default_rng(0).normal(loc=10.0, scale=1.0, size=10_000)
    dist = AggregateDistribution(samples=samples, n_pixels_used=42, name="probe")
    assert dist.quantile(0.10) == pytest.approx(float(np.quantile(samples, 0.10)))
    assert dist.summary()["q50"] == pytest.approx(dist.q50)
    assert dist.summary()["name"] == "probe"


def test_hard_threshold_blocks_without_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pedotri.zonal._AGG_WORK_BLOCK", 100)
    monkeypatch.setattr("pedotri.zonal._AGG_WORK_WARN", 50)
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    with pytest.raises(InvalidInputError, match="hard threshold"):
        zonal_aggregate(
            region=region,
            properties={"soc": _uniform_property(shape, 25.0, 4.0)},
            n_samples=100,
            seed=10,
        )


def test_hard_threshold_overridden_by_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pedotri.zonal._AGG_WORK_BLOCK", 100)
    monkeypatch.setattr("pedotri.zonal._AGG_WORK_WARN", 50)
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=100,
        seed=10,
        confirm=True,
    )
    assert agg["soc"].samples.size == 100


def test_geometry_region_rasterizes_with_profile() -> None:
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    shapely = pytest.importorskip("shapely.geometry")

    shape = (20, 20)
    transform = from_origin(0, shape[0], 1, 1)
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "transform": transform,
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float32",
    }

    # Bottom-left quadrant in pixel coordinates, mapped through the
    # raster transform: rows 10..20 (= y 0..10 in world coords), cols 0..10.
    geom = shapely.box(0, 0, 10, 10)
    agg = zonal_aggregate(
        region=geom,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        profile=profile,
        n_samples=200,
        seed=11,
    )
    assert agg.region_pixels == 100  # 10×10 quadrant


def test_geometry_region_requires_profile() -> None:
    shapely = pytest.importorskip("shapely.geometry")
    shape = (10, 10)
    arrays = _uniform_property(shape, 25.0, 4.0)
    with pytest.raises(InvalidInputError, match="profile"):
        zonal_aggregate(
            region=shapely.box(0, 0, 1, 1),
            properties={"soc": arrays},
            n_samples=100,
            seed=12,
        )


# --- aggregate_depths ---------------------------------------------------


def test_aggregate_depths_scalar_inputs_match_handrolled_formula() -> None:
    from pedotri.zonal import aggregate_depths

    samples = {
        "0-5cm": {"mean": 24.0, "uncertainty": Quantiles(18.0, 30.0)},
        "5-15cm": {"mean": 22.0, "uncertainty": Quantiles(16.0, 28.0)},
        "15-30cm": {"mean": 19.0, "uncertainty": Quantiles(13.0, 25.0)},
    }
    weights = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
    mean, q = aggregate_depths(samples, weights)
    # Hand-rolled formula: 30-cm topsoil mean
    w_total = 5 + 10 + 15
    expected = (5 * 24.0 + 10 * 22.0 + 15 * 19.0) / w_total
    assert mean == pytest.approx(expected)
    # Quantiles inherit the same weighting.
    expected_q05 = (5 * 18.0 + 10 * 16.0 + 15 * 13.0) / w_total
    assert q.q05 == pytest.approx(expected_q05)
    assert q.q95 == pytest.approx((5 * 30.0 + 10 * 28.0 + 15 * 25.0) / w_total)


def test_aggregate_depths_array_inputs() -> None:
    from pedotri.zonal import aggregate_depths

    samples = {
        "a": {
            "mean": np.array([10.0, 20.0]),
            "uncertainty": Quantiles(np.array([5.0, 10.0]), np.array([15.0, 30.0])),
        },
        "b": {
            "mean": np.array([30.0, 40.0]),
            "uncertainty": Quantiles(np.array([25.0, 30.0]), np.array([35.0, 50.0])),
        },
    }
    mean, q = aggregate_depths(samples, {"a": 1, "b": 3})
    # Weighted average across the two pixels.
    np.testing.assert_allclose(mean, np.array([0.25 * 10 + 0.75 * 30, 0.25 * 20 + 0.75 * 40]))
    np.testing.assert_allclose(q.q05, np.array([0.25 * 5 + 0.75 * 25, 0.25 * 10 + 0.75 * 30]))


def test_aggregate_depths_sigma_form_converts_to_quantiles() -> None:
    from pedotri.zonal import aggregate_depths

    samples = {"a": {"mean": 20.0, "sigma": 2.0}}
    _mean, q = aggregate_depths(samples, {"a": 1.0})
    # σ → Q05/Q95 round-trip.
    expected_q05 = 20.0 - 1.6448536269514722 * 2.0
    assert q.q05 == pytest.approx(expected_q05)


def test_aggregate_depths_rejects_label_mismatch() -> None:
    from pedotri.zonal import aggregate_depths

    with pytest.raises(InvalidInputError, match="label mismatch"):
        aggregate_depths(
            samples={"a": {"mean": 1.0, "sigma": 0.1}},
            weights={"b": 1},
        )


def test_aggregate_depths_rejects_zero_total_weight() -> None:
    from pedotri.zonal import aggregate_depths

    with pytest.raises(InvalidInputError, match="positive value"):
        aggregate_depths(
            samples={"a": {"mean": 1.0, "sigma": 0.1}},
            weights={"a": 0.0},
        )


def test_aggregate_depths_three_tuple_deprecated() -> None:
    from pedotri.zonal import aggregate_depths

    with pytest.warns(DeprecationWarning, match="deprecated"):
        mean, q = aggregate_depths(
            samples={"a": (20.0, 15.0, 25.0)},
            weights={"a": 1.0},
        )
    assert mean == pytest.approx(20.0)
    assert q.q05 == pytest.approx(15.0)
    assert q.q95 == pytest.approx(25.0)


def test_aggregate_depths_two_tuple_with_quantiles() -> None:
    from pedotri.zonal import aggregate_depths

    mean, q = aggregate_depths(
        samples={"a": (20.0, Quantiles(15.0, 25.0))},
        weights={"a": 1.0},
    )
    assert mean == pytest.approx(20.0)
    assert q.q05 == pytest.approx(15.0)
