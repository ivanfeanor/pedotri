"""Tests for uncertainty-aware classification (point path)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pedotri
from pedotri.errors import InvalidInputError
from pedotri.uncertainty import (
    Quantiles,
    _aggregate_class_probabilities,
    _parse_uncertainty,
    sample_compositional,
    sample_correlated_field,
    sample_truncated_normal,
    shannon_entropy,
    sigma_from_quantiles,
)

# --- distribution helpers ------------------------------------------------


def test_sigma_from_quantiles_matches_normal() -> None:
    sigma = sigma_from_quantiles(0.0, 2.0 * 1.6448536269514722)
    assert math.isclose(float(sigma), 1.0, rel_tol=1e-9)


def test_sigma_from_quantiles_vectorized() -> None:
    sigma = sigma_from_quantiles(np.array([0.0, 10.0]), np.array([3.29, 13.29]))
    assert sigma.shape == (2,)
    assert sigma[0] == pytest.approx(1.0, rel=1e-3)
    assert sigma[1] == pytest.approx(1.0, rel=1e-3)


def test_sigma_from_quantiles_rejects_inverted() -> None:
    with pytest.raises(InvalidInputError):
        sigma_from_quantiles(10.0, 5.0)


def test_parse_uncertainty_quantiles_explicit_form() -> None:
    sigma = _parse_uncertainty(Quantiles(20.0, 34.0))
    assert sigma == pytest.approx(14.0 / (2 * 1.6448536269514722), rel=1e-6)


def test_parse_uncertainty_bare_tuple_deprecated() -> None:
    """The bare 2-tuple form still works but emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="Quantiles"):
        sigma = _parse_uncertainty((20.0, 34.0))
    assert sigma == pytest.approx(14.0 / (2 * 1.6448536269514722), rel=1e-6)


def test_parse_uncertainty_scalar_form() -> None:
    sigma = _parse_uncertainty(3.0)
    assert sigma is not None
    assert float(sigma) == 3.0


def test_parse_uncertainty_none() -> None:
    assert _parse_uncertainty(None) is None


def test_parse_uncertainty_rejects_negative_sigma() -> None:
    with pytest.raises(InvalidInputError):
        _parse_uncertainty(-1.0)


def test_quantiles_exposed_at_package_root() -> None:
    """Quantiles is the foundation type — it lives at pedotri.Quantiles."""
    assert pedotri.Quantiles is Quantiles


def test_quantiles_construct_and_helpers() -> None:
    q = Quantiles(20.0, 34.0)
    assert q.q05 == 20.0
    assert q.q95 == 34.0
    assert q.width == pytest.approx(14.0)
    assert q.sigma == pytest.approx(14.0 / (2 * 1.6448536269514722))


def test_quantiles_rejects_inverted() -> None:
    with pytest.raises(InvalidInputError, match="q95 must be >= q05"):
        Quantiles(34.0, 20.0)


def test_quantiles_rejects_shape_mismatch() -> None:
    with pytest.raises(InvalidInputError, match="shape"):
        Quantiles(np.zeros(3), np.zeros(4))


def test_quantiles_accepts_arrays() -> None:
    q = Quantiles(np.array([10.0, 20.0]), np.array([20.0, 30.0]))
    assert q.width.shape == (2,)
    assert q.sigma.shape == (2,)


def test_sample_truncated_normal_shape_and_clip() -> None:
    rng = np.random.default_rng(42)
    samples = sample_truncated_normal(
        mean=np.array([50.0, 50.0]),
        sigma=np.array([10.0, 10.0]),
        n_samples=500,
        rng=rng,
    )
    assert samples.shape == (2, 500)
    assert samples.min() >= 0.0
    assert samples.max() <= 100.0
    # The mean of clipped draws should be close to the input mean when σ ≪ 50.
    assert samples.mean(axis=1) == pytest.approx(np.array([50.0, 50.0]), abs=2.0)


def test_sample_compositional_stays_on_simplex() -> None:
    rng = np.random.default_rng(0)
    sand, clay = sample_compositional(
        np.array([40.0]),
        np.array([20.0]),
        np.array([40.0]),
        np.array([20.0]),
        n_samples=5000,
        rng=rng,
    )
    silt = 100.0 - sand - clay
    # Renormalization may zero out silt but should never make it negative.
    assert silt.min() >= -1e-9
    # Each (sand, clay) is itself in [0, 100].
    assert sand.min() >= 0.0
    assert clay.min() >= 0.0


def test_aggregate_class_probabilities_sums_to_one_minus_unclassified() -> None:
    codes = np.array([[0, 0, 1, 1, -1], [2, 2, 2, 2, 2]])
    probs, unclass = _aggregate_class_probabilities(codes, n_classes=3)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs[0], [0.4, 0.4, 0.0])
    assert unclass[0] == pytest.approx(0.2)
    np.testing.assert_allclose(probs[1], [0.0, 0.0, 1.0])
    assert unclass[1] == 0.0


def test_shannon_entropy_zero_for_pure_distribution() -> None:
    probs = np.array([[1.0, 0.0, 0.0]])
    assert shannon_entropy(probs)[0] == pytest.approx(0.0)


def test_shannon_entropy_max_for_uniform() -> None:
    n = 4
    probs = np.full((1, n), 1.0 / n)
    assert shannon_entropy(probs)[0] == pytest.approx(math.log(n))


# --- end-to-end classify() integration -----------------------------------


def test_classify_deterministic_unchanged() -> None:
    """No uncertainty kwargs → identical behavior to pre-0.3."""
    plain = pedotri.classify(13, 50, "USDA")
    assert plain == "clay"

    detailed = pedotri.classify(13, 50, "USDA", detailed=True)
    assert detailed.key == "clay"
    assert detailed.probabilities is None
    assert detailed.entropy is None
    assert detailed.confidence is None
    assert detailed.unclassified_probability is None


def test_classify_distance_method_confidence_inside_class() -> None:
    """Point deep inside its class should have confidence near 1.0."""
    r = pedotri.classify(
        sand=10,
        clay=80,
        sand_uncertainty=Quantiles(8, 12),
        clay_uncertainty=Quantiles(78, 82),
        classification="USDA",
        detailed=True,
        method="distance",
    )
    assert r.key == "clay"
    assert r.confidence is not None
    assert r.confidence > 0.95


def test_classify_distance_method_confidence_near_boundary() -> None:
    """Point near a class boundary should have confidence near 0.5."""
    deterministic = pedotri.classify(40, 28, "USDA", detailed=True)
    assert deterministic.key is not None
    assert abs(deterministic.distance) < 2.0

    r = pedotri.classify(
        sand=40,
        clay=28,
        sand_uncertainty=Quantiles(30, 50),
        clay_uncertainty=Quantiles(18, 38),
        classification="USDA",
        detailed=True,
        method="distance",
    )
    assert r.confidence is not None
    assert 0.45 < r.confidence < 0.75


def test_classify_monte_carlo_probabilities_sum_to_one() -> None:
    r = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=1000,
        seed=42,
    )
    assert r.probabilities is not None
    total = sum(r.probabilities.values()) + (r.unclassified_probability or 0.0)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_classify_monte_carlo_modal_class_matches_confidence() -> None:
    """ClassifyResult.key should be the modal class, .confidence its probability."""
    r = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=2000,
        seed=42,
    )
    assert r.probabilities is not None
    assert r.key is not None
    modal = max(r.probabilities.items(), key=lambda kv: kv[1])
    assert modal[0] == r.key
    assert r.confidence == pytest.approx(modal[1])


def test_classify_monte_carlo_entropy_low_when_clear_winner() -> None:
    """A point far from any boundary should have entropy near zero."""
    r = pedotri.classify(
        sand=10,
        clay=80,
        sand_uncertainty=Quantiles(8, 12),
        clay_uncertainty=Quantiles(78, 82),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=1000,
        seed=0,
    )
    assert r.entropy is not None
    assert r.entropy < 0.1


def test_classify_uncertainty_auto_promotes_detailed() -> None:
    """Passing uncertainty without detailed=True still returns ClassifyResult."""
    from pedotri.classifier import ClassifyResult

    r = pedotri.classify(
        13,
        50,
        "USDA",
        sand_uncertainty=Quantiles(10, 16),
        clay_uncertainty=Quantiles(45, 55),
    )
    assert isinstance(r, ClassifyResult)
    assert r.confidence is not None
    # Explicit detailed=False is ignored to keep the return type honest.
    r2 = pedotri.classify(
        13,
        50,
        "USDA",
        sand_uncertainty=Quantiles(10, 16),
        clay_uncertainty=Quantiles(45, 55),
        detailed=False,
    )
    assert isinstance(r2, ClassifyResult)


def test_classify_unknown_uncertainty_axis() -> None:
    with pytest.raises(TypeError, match="unknown uncertainty kwarg"):
        pedotri.classify(
            13,
            50,
            "USDA",
            silt_uncertainty=Quantiles(20, 30),
            detailed=True,
        )


def test_classify_bad_method() -> None:
    with pytest.raises(InvalidInputError, match="Unknown classification method"):
        pedotri.classify(
            13,
            50,
            "USDA",
            sand_uncertainty=Quantiles(10, 16),
            detailed=True,
            method="bogus",
        )


def test_classify_array_input_with_uncertainty() -> None:
    results = pedotri.classify(
        sand=[13, 45, 70],
        clay=[50, 24, 10],
        sand_uncertainty=Quantiles(1, 3),
        clay_uncertainty=Quantiles(1, 3),
        classification="FAO",
        detailed=True,
        method="monte_carlo",
        n_samples=300,
        seed=0,
    )
    assert len(results) == 3
    assert [r.key for r in results] == ["fine", "medium", "coarse"]
    for r in results:
        assert r.probabilities is not None
        assert math.isclose(
            sum(r.probabilities.values()) + (r.unclassified_probability or 0.0),
            1.0,
            abs_tol=1e-9,
        )


def test_classify_1d_with_uncertainty() -> None:
    r = pedotri.classify(
        35,
        classification="KACHINSKY",
        physical_clay_uncertainty=Quantiles(28, 42),
        detailed=True,
        method="monte_carlo",
        n_samples=2000,
        seed=1,
    )
    assert r.probabilities is not None
    assert r.entropy is not None
    assert r.entropy > 0.0
    assert math.isclose(
        sum(r.probabilities.values()) + (r.unclassified_probability or 0.0),
        1.0,
        abs_tol=1e-9,
    )


def test_classify_per_point_uncertainty_array() -> None:
    """Per-point σ arrays must broadcast to the input shape."""
    sand_sigma = np.array([5.0, 1.0, 5.0])
    clay_sigma = np.array([5.0, 1.0, 5.0])
    results = pedotri.classify(
        sand=[13, 45, 70],
        clay=[50, 24, 10],
        sand_uncertainty=sand_sigma,
        clay_uncertainty=clay_sigma,
        classification="FAO",
        detailed=True,
        method="monte_carlo",
        n_samples=500,
        seed=2,
    )
    # The middle point has tiny σ → near-1 modal probability.
    assert results[1].confidence is not None
    assert results[1].confidence > 0.99
    # The outer points have larger σ → may be slightly less confident.
    assert results[0].confidence is not None
    assert results[2].confidence is not None


def test_classify_reproducibility_with_seed() -> None:
    r1 = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=500,
        seed=12345,
    )
    r2 = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=500,
        seed=12345,
    )
    assert r1.probabilities == r2.probabilities
    assert r1.entropy == r2.entropy


def test_classify_partial_uncertainty_axes() -> None:
    """Supplying uncertainty for only some axes is allowed; missing axes are deterministic."""
    r = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=500,
        seed=0,
    )
    assert r.probabilities is not None
    # Clay axis has σ=0, so all samples share clay=38; classes split along
    # the sand axis only — still a real distribution.
    assert len(r.probabilities) >= 1


def test_to_dict_includes_uncertainty_fields() -> None:
    r = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        detailed=True,
        method="monte_carlo",
        n_samples=500,
        seed=42,
    )
    d = r.to_dict()
    assert "probabilities" in d
    assert "entropy" in d
    assert "confidence" in d
    assert "unclassified_probability" in d


def test_to_dict_omits_uncertainty_fields_for_deterministic() -> None:
    r = pedotri.classify(13, 50, "USDA", detailed=True)
    d = r.to_dict()
    assert "probabilities" not in d
    assert "entropy" not in d
    assert "confidence" not in d
    assert "unclassified_probability" not in d


# --- spatially-correlated field ------------------------------------------


def test_correlated_field_shape_and_unit_variance() -> None:
    rng = np.random.default_rng(0)
    mean = np.zeros((40, 40))
    sigma = np.ones((40, 40))
    samples = sample_correlated_field(mean, sigma, correlation_range=4.0, n_samples=400, rng=rng)
    assert samples.shape == (400, 40, 40)
    # Per-pixel variance pulls toward 1.0 across many draws (CLT).
    assert samples.var(axis=0).mean() == pytest.approx(1.0, abs=0.1)
    assert abs(samples.mean()) < 0.1


def test_correlated_field_recovers_exponential_kernel() -> None:
    """Empirical correlation at distances ≤ L should track exp(-d/L)."""
    rng = np.random.default_rng(1)
    H = W = 60
    L = 6.0
    samples = sample_correlated_field(
        np.zeros((H, W)),
        np.ones((H, W)),
        correlation_range=L,
        n_samples=600,
        rng=rng,
    )
    centre = samples[:, H // 2, W // 2]
    for d in (1, 3, 5):
        neighbour = samples[:, H // 2, W // 2 + d]
        corr = float(np.corrcoef(centre, neighbour)[0, 1])
        expected = math.exp(-d / L)
        assert abs(corr - expected) < 0.1, (
            f"At d={d}: empirical={corr:.3f}, exp(-d/L)={expected:.3f}"
        )


def test_correlated_field_scales_with_per_pixel_sigma() -> None:
    rng = np.random.default_rng(2)
    mean = np.full((20, 20), 5.0)
    sigma = np.full((20, 20), 3.0)
    samples = sample_correlated_field(mean, sigma, correlation_range=2.0, n_samples=400, rng=rng)
    # Mean field is recovered.
    assert samples.mean(axis=0).mean() == pytest.approx(5.0, abs=0.3)
    # Variance per pixel is sigma².
    assert samples.var(axis=0).mean() == pytest.approx(9.0, rel=0.1)


def test_correlated_field_gaussian_model() -> None:
    rng = np.random.default_rng(3)
    samples = sample_correlated_field(
        np.zeros((30, 30)),
        np.ones((30, 30)),
        correlation_range=4.0,
        n_samples=200,
        rng=rng,
        model="gaussian",
    )
    assert samples.shape == (200, 30, 30)
    assert samples.var(axis=0).mean() == pytest.approx(1.0, abs=0.2)


def test_correlated_field_spherical_model() -> None:
    rng = np.random.default_rng(4)
    L = 5.0
    samples = sample_correlated_field(
        np.zeros((40, 40)),
        np.ones((40, 40)),
        correlation_range=L,
        n_samples=400,
        rng=rng,
        model="spherical",
    )
    # Spherical model: correlation is exactly zero beyond d=L.
    centre = samples[:, 20, 20]
    far = samples[:, 20, 20 + int(L * 2)]  # well beyond the range
    corr = float(np.corrcoef(centre, far)[0, 1])
    assert abs(corr) < 0.1


def test_correlated_field_inflates_regional_uncertainty() -> None:
    """The whole point: correlated draws → larger σ on the regional mean."""
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(10)
    H = W = 40
    indep = sample_correlated_field(
        np.zeros((H, W)),
        np.ones((H, W)),
        correlation_range=0.5,  # effectively independent (sub-pixel range)
        n_samples=500,
        rng=rng_a,
    )
    corr = sample_correlated_field(
        np.zeros((H, W)),
        np.ones((H, W)),
        correlation_range=10.0,
        n_samples=500,
        rng=rng_b,
    )
    indep_regional = indep.reshape(500, -1).mean(axis=1)
    corr_regional = corr.reshape(500, -1).mean(axis=1)
    # Correlated draws should have regional σ at least ~3× that of the
    # near-independent draws on the same H × W grid.
    assert corr_regional.std() > 3 * indep_regional.std()


def test_correlated_field_reproducibility() -> None:
    a = sample_correlated_field(
        np.zeros((20, 20)),
        np.ones((20, 20)),
        correlation_range=3.0,
        n_samples=50,
        rng=np.random.default_rng(99),
    )
    b = sample_correlated_field(
        np.zeros((20, 20)),
        np.ones((20, 20)),
        correlation_range=3.0,
        n_samples=50,
        rng=np.random.default_rng(99),
    )
    np.testing.assert_array_equal(a, b)


def test_correlated_field_rejects_unknown_model() -> None:
    with pytest.raises(InvalidInputError, match="Unknown correlation"):
        sample_correlated_field(
            np.zeros((10, 10)),
            np.ones((10, 10)),
            correlation_range=2.0,
            n_samples=5,
            model="bogus",
        )


def test_correlated_field_rejects_non_2d_mean() -> None:
    with pytest.raises(InvalidInputError, match="2-D"):
        sample_correlated_field(
            np.zeros(10),
            np.ones(10),
            correlation_range=2.0,
            n_samples=5,
        )


def test_correlated_field_rejects_non_positive_range() -> None:
    with pytest.raises(InvalidInputError, match="correlation_range"):
        sample_correlated_field(
            np.zeros((10, 10)),
            np.ones((10, 10)),
            correlation_range=0.0,
            n_samples=5,
        )
