"""Tests for uncertainty-aware raster classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.raster import (
    DEFAULT_TOP_K,
    NODATA_CODE,
    classify_array_with_uncertainty,
    classify_geotiff_with_uncertainty,
    write_confidence_geotiff,
    write_probability_stack_geotiff,
)
from pedotri.uncertainty import Quantiles

if TYPE_CHECKING:
    from pathlib import Path


def _two_class_raster() -> tuple[np.ndarray, np.ndarray]:
    """A small (sand, clay) raster straddling a USDA boundary.

    Top half lives inside ``clay`` (sand=10, clay=80). Bottom half
    lives inside ``clay_loam`` (sand=40, clay=28). The middle row sits
    right on a boundary so confidence should be lowest there.
    """
    sand = np.empty((6, 4), dtype=np.float64)
    clay = np.empty((6, 4), dtype=np.float64)
    sand[:3, :] = 10.0
    clay[:3, :] = 80.0
    sand[3:, :] = 40.0
    clay[3:, :] = 28.0
    return sand, clay


# --- distance method ----------------------------------------------------


def test_distance_method_codes_match_deterministic() -> None:
    """With uncertainty, modal codes should still equal the deterministic ones."""
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 3, sand + 3),
        clay_uncertainty=Quantiles(clay - 3, clay + 3),
        classification="USDA",
        method="distance",
    )
    assert r.method == "distance"
    assert r.codes.shape == sand.shape
    assert r.confidence is not None
    assert r.confidence.shape == sand.shape
    # All pixels are firmly inside a class — no NODATA_CODE.
    assert int((r.codes == NODATA_CODE).sum()) == 0
    # Top half = clay; bottom = clay_loam.
    assert r.keys[r.codes[0, 0]] == "clay"
    assert r.keys[r.codes[5, 0]] == "clay_loam"


def test_distance_method_confidence_in_unit_interval() -> None:
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
    )
    assert r.confidence is not None
    finite = r.confidence[np.isfinite(r.confidence)]
    assert finite.size == r.confidence.size  # all pixels classified
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_distance_method_zero_sigma_yields_nan_confidence() -> None:
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=np.zeros_like(sand),
        clay_uncertainty=np.zeros_like(sand),
        classification="USDA",
        method="distance",
    )
    assert r.confidence is not None
    assert np.all(np.isnan(r.confidence))


# --- monte carlo method -------------------------------------------------


def test_monte_carlo_top_k_shape_and_dtypes() -> None:
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=400,
        seed=0,
    )
    assert r.method == "monte_carlo"
    assert r.top_k_codes is not None
    assert r.top_k_probs is not None
    assert r.top_k_codes.shape == (DEFAULT_TOP_K, *sand.shape)
    assert r.top_k_probs.shape == (DEFAULT_TOP_K, *sand.shape)
    assert r.top_k_codes.dtype == np.int16
    assert r.top_k_probs.dtype == np.float32


def test_monte_carlo_per_pixel_probabilities_sum_to_one() -> None:
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=500,
        seed=1,
    )
    assert r.top_k_probs is not None
    assert r.unclassified_probability is not None
    # Sum across the top-k axis plus unclassified ≤ 1 (equality when
    # top_k covers every class with non-zero probability).
    totals = r.top_k_probs.sum(axis=0) + r.unclassified_probability
    assert float(totals.max()) <= 1.0 + 1e-6
    # In our raster every pixel sits well inside a class polygon → low
    # unclassified rate, total mass essentially 1.
    np.testing.assert_allclose(totals, 1.0, atol=0.05)


def test_monte_carlo_modal_class_matches_codes() -> None:
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=400,
        seed=2,
    )
    assert r.top_k_codes is not None
    # Rank-1 (top_k_codes[0]) is the modal class — should agree with
    # codes wherever codes was assigned.
    classified = r.codes != NODATA_CODE
    np.testing.assert_array_equal(r.top_k_codes[0][classified], r.codes[classified])


def test_monte_carlo_entropy_higher_near_boundary() -> None:
    """A point that straddles two classes should be more entropic than one deep inside."""
    sand = np.array([[10.0, 30.0]])
    clay = np.array([[80.0, 38.0]])
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 4, sand + 4),
        clay_uncertainty=Quantiles(clay - 4, clay + 4),
        classification="USDA",
        method="monte_carlo",
        n_samples=600,
        seed=3,
    )
    assert r.entropy is not None
    assert r.entropy[0, 0] < r.entropy[0, 1]


def test_monte_carlo_hard_threshold_blocks_without_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the thresholds so the guard fires cheaply.
    monkeypatch.setattr("pedotri.raster._MC_WORK_BLOCK", 100)
    monkeypatch.setattr("pedotri.raster._MC_WORK_WARN", 50)
    sand, clay = _two_class_raster()
    with pytest.raises(InvalidInputError, match="hard threshold"):
        classify_array_with_uncertainty(
            sand,
            clay,
            sand_uncertainty=Quantiles(sand - 5, sand + 5),
            clay_uncertainty=Quantiles(clay - 5, clay + 5),
            classification="USDA",
            method="monte_carlo",
            n_samples=100,
        )


def test_monte_carlo_hard_threshold_overridden_by_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """confirm=True must bypass the hard-threshold guard (and skip the soft warning)."""
    monkeypatch.setattr("pedotri.raster._MC_WORK_BLOCK", 100)
    monkeypatch.setattr("pedotri.raster._MC_WORK_WARN", 50)
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=100,
        seed=11,
        confirm=True,
    )
    assert r.top_k_probs is not None


def test_monte_carlo_soft_threshold_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pedotri.raster._MC_WORK_BLOCK", 10_000)
    monkeypatch.setattr("pedotri.raster._MC_WORK_WARN", 50)
    sand, clay = _two_class_raster()
    with pytest.warns(UserWarning, match="Monte Carlo workload is large"):
        classify_array_with_uncertainty(
            sand,
            clay,
            sand_uncertainty=Quantiles(sand - 5, sand + 5),
            clay_uncertainty=Quantiles(clay - 5, clay + 5),
            classification="USDA",
            method="monte_carlo",
            n_samples=100,
            seed=4,
        )


# --- input form validation ---------------------------------------------


def test_sigma_input_form_equivalent_to_quantile_form() -> None:
    """Direct σ raster must give the same confidence as the equivalent quantiles."""
    sand, clay = _two_class_raster()
    width = 5.0
    sigma = width / 1.6448536269514722  # σ corresponding to (mean ± width) at Q05/Q95
    r_q = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - width, sand + width),
        clay_uncertainty=Quantiles(clay - width, clay + width),
        classification="USDA",
        method="distance",
    )
    r_s = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=np.full_like(sand, sigma),
        clay_uncertainty=np.full_like(sand, sigma),
        classification="USDA",
        method="distance",
    )
    np.testing.assert_array_equal(r_q.codes, r_s.codes)
    np.testing.assert_allclose(r_q.confidence, r_s.confidence, rtol=1e-6)


def test_bare_tuple_uncertainty_emits_deprecation_warning() -> None:
    sand, clay = _two_class_raster()
    with pytest.warns(DeprecationWarning, match="Quantiles"):
        r = classify_array_with_uncertainty(
            sand,
            clay,
            sand_uncertainty=(sand - 5, sand + 5),
            clay_uncertainty=(clay - 5, clay + 5),
            classification="USDA",
            method="distance",
        )
    assert r.confidence is not None


def test_quantiles_inverted_arrays_rejected() -> None:
    sand, clay = _two_class_raster()
    # q95 < q05 on every pixel → Quantiles constructor rejects.
    with pytest.raises(InvalidInputError, match="q95 must be >= q05"):
        classify_array_with_uncertainty(
            sand,
            clay,
            sand_uncertainty=Quantiles(sand + 1, sand - 1),
            classification="USDA",
            method="distance",
        )


def test_partial_uncertainty_per_axis_is_allowed() -> None:
    """Uncertainty on only one axis should still produce a confidence map."""
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        classification="USDA",
        method="distance",
    )
    assert r.confidence is not None
    finite = r.confidence[np.isfinite(r.confidence)]
    assert finite.size == r.confidence.size
    assert (finite >= 0.0).all()
    assert (finite <= 1.0).all()


def test_mask_excludes_pixels() -> None:
    sand, clay = _two_class_raster()
    mask = np.ones_like(sand, dtype=bool)
    mask[0, 0] = False
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
        mask=mask,
    )
    assert r.codes[0, 0] == NODATA_CODE
    assert np.isnan(r.confidence[0, 0])


def test_unknown_method_rejected() -> None:
    sand, clay = _two_class_raster()
    with pytest.raises(InvalidInputError, match="Unknown method"):
        classify_array_with_uncertainty(
            sand,
            clay,
            sand_uncertainty=Quantiles(sand - 1, sand + 1),
            classification="USDA",
            method="bogus",
        )


# --- GeoTIFF round-trip --------------------------------------------------


def _make_profile(shape: tuple[int, int]) -> dict:
    from rasterio.transform import from_origin

    return {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "transform": from_origin(0, shape[0], 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float64",
    }


def test_write_confidence_geotiff_round_trip(tmp_path: Path) -> None:
    rio = pytest.importorskip("rasterio")
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
    )
    profile = _make_profile(sand.shape)
    out = tmp_path / "confidence.tif"
    write_confidence_geotiff(out, r.confidence, profile=profile, method="distance")
    with rio.open(out) as src:
        read_back = src.read(1)
        tags = src.tags()
        nodata = src.nodata
    assert src.count == 1
    np.testing.assert_allclose(read_back, r.confidence, rtol=1e-6)
    assert tags["method"] == "distance"
    assert nodata is not None
    assert np.isnan(nodata)


def test_write_probability_stack_round_trip(tmp_path: Path) -> None:
    rio = pytest.importorskip("rasterio")
    sand, clay = _two_class_raster()
    r = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=400,
        seed=7,
        top_k=5,
    )
    profile = _make_profile(sand.shape)
    out = tmp_path / "probs.tif"
    write_probability_stack_geotiff(out, r.top_k_codes, r.top_k_probs, profile=profile, keys=r.keys)
    with rio.open(out) as src:
        assert src.count == 10
        descriptions = src.descriptions
        tags = src.tags()
        bands = src.read()
    assert descriptions[0] == "rank_1_class"
    assert descriptions[5] == "rank_1_prob_x255"
    assert tags["top_k"] == "5"
    assert tags["class_0"] == r.keys[0]
    # Rank-1 codes recovered after the nodata remap.
    rank1_codes = np.where(bands[0] == 255, NODATA_CODE, bands[0].astype(np.int16))
    np.testing.assert_array_equal(rank1_codes, r.top_k_codes[0])
    # Probabilities round-trip to within the uint8 quantization step.
    rank1_probs = bands[5].astype(np.float32) / 255.0
    np.testing.assert_allclose(rank1_probs, r.top_k_probs[0], atol=1.0 / 255.0)


def test_probability_stack_rejects_shape_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    profile = _make_profile((3, 3))
    codes = np.zeros((5, 3, 3), dtype=np.int16)
    probs = np.zeros((4, 3, 3), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="share shape"):
        write_probability_stack_geotiff(
            tmp_path / "x.tif", codes, probs, profile=profile, keys=["a"]
        )


def test_probability_stack_rejects_too_many_classes(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    profile = _make_profile((3, 3))
    codes = np.zeros((1, 3, 3), dtype=np.int16)
    probs = np.zeros((1, 3, 3), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="up to 255"):
        write_probability_stack_geotiff(
            tmp_path / "x.tif",
            codes,
            probs,
            profile=profile,
            keys=[f"c{i}" for i in range(256)],
        )


def test_has_confidence_and_probabilities_flags() -> None:
    sand, clay = _two_class_raster()
    dist = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
    )
    mc = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=200,
        seed=7,
    )
    assert dist.has_confidence
    assert not dist.has_probabilities
    assert mc.has_confidence is False  # confidence array isn't populated
    assert mc.has_probabilities is True
    assert dist.n_classes == mc.n_classes == 12  # USDA


def test_modal_confidence_works_for_both_methods() -> None:
    sand, clay = _two_class_raster()
    dist = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
    )
    mc = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=200,
        seed=7,
    )
    d_conf = dist.modal_confidence()
    m_conf = mc.modal_confidence()
    assert d_conf.shape == sand.shape
    assert m_conf.shape == sand.shape
    finite_d = d_conf[np.isfinite(d_conf)]
    finite_m = m_conf[np.isfinite(m_conf)]
    assert (finite_d >= 0).all()
    assert (finite_d <= 1).all()
    assert (finite_m >= 0).all()
    assert (finite_m <= 1).all()


def test_class_probability_returns_per_pixel_mass() -> None:
    sand, clay = _two_class_raster()
    mc = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=400,
        seed=8,
    )
    p_clay = mc.class_probability("clay")
    p_clay_loam = mc.class_probability("clay_loam")
    assert p_clay.shape == sand.shape
    # Top half of the raster lives in "clay"; bottom half in "clay_loam".
    assert p_clay[0, 0] > 0.5
    assert p_clay_loam[5, 0] > 0.5


def test_class_probability_rejects_unknown_class() -> None:
    sand, clay = _two_class_raster()
    mc = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="monte_carlo",
        n_samples=200,
        seed=9,
    )
    with pytest.raises(InvalidInputError, match="Unknown class key"):
        mc.class_probability("not_a_class")


def test_class_probability_rejects_distance_result() -> None:
    sand, clay = _two_class_raster()
    dist = classify_array_with_uncertainty(
        sand,
        clay,
        sand_uncertainty=Quantiles(sand - 5, sand + 5),
        clay_uncertainty=Quantiles(clay - 5, clay + 5),
        classification="USDA",
        method="distance",
    )
    with pytest.raises(InvalidInputError, match="Monte-Carlo"):
        dist.class_probability("clay")


def test_classify_geotiff_with_uncertainty_round_trip(tmp_path: Path) -> None:
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    sand, clay = _two_class_raster()
    base_profile = {
        "driver": "GTiff",
        "height": sand.shape[0],
        "width": sand.shape[1],
        "transform": from_origin(0, sand.shape[0], 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float32",
    }

    def write_band(name: str, arr: np.ndarray) -> str:
        path = tmp_path / name
        prof = dict(base_profile)
        with rio.open(path, "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)
        return str(path)

    paths = {
        "sand_mean": write_band("sand_mean.tif", sand),
        "sand_q05": write_band("sand_q05.tif", sand - 5),
        "sand_q95": write_band("sand_q95.tif", sand + 5),
        "clay_mean": write_band("clay_mean.tif", clay),
        "clay_q05": write_band("clay_q05.tif", clay - 5),
        "clay_q95": write_band("clay_q95.tif", clay + 5),
    }
    # ``classify_geotiff_with_uncertainty`` accepts a (q05_path, q95_path)
    # 2-tuple per axis — handier than wrapping pre-loaded arrays in
    # ``Quantiles``.
    result = classify_geotiff_with_uncertainty(
        paths["sand_mean"],
        paths["clay_mean"],
        sand_uncertainty=(paths["sand_q05"], paths["sand_q95"]),
        clay_uncertainty=(paths["clay_q05"], paths["clay_q95"]),
        classification="USDA",
        method="distance",
    )
    assert result.profile is not None
    assert result.profile["dtype"] == "int16"
    assert result.profile["nodata"] == NODATA_CODE
    assert result.confidence is not None
    assert result.confidence.shape == sand.shape


def test_classify_geotiff_with_uncertainty_single_sigma_path(tmp_path: Path) -> None:
    """Single GeoTIFF path → σ raster for that axis."""
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    sand, clay = _two_class_raster()
    base_profile = {
        "driver": "GTiff",
        "height": sand.shape[0],
        "width": sand.shape[1],
        "transform": from_origin(0, sand.shape[0], 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float32",
    }

    def write_band(name: str, arr: np.ndarray) -> str:
        path = tmp_path / name
        prof = dict(base_profile)
        with rio.open(path, "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)
        return str(path)

    sand_mean_p = write_band("sand_mean.tif", sand)
    clay_mean_p = write_band("clay_mean.tif", clay)
    sand_sigma_p = write_band("sand_sigma.tif", np.full_like(sand, 3.0))

    result = classify_geotiff_with_uncertainty(
        sand_mean_p,
        clay_mean_p,
        sand_uncertainty=sand_sigma_p,
        classification="USDA",
        method="distance",
    )
    # Clay axis is deterministic (no kwarg); sand carries σ=3 everywhere.
    assert result.confidence is not None
    finite = result.confidence[np.isfinite(result.confidence)]
    assert finite.size == result.confidence.size
    assert (finite >= 0.0).all()
    assert (finite <= 1.0).all()
