"""Tests for the public classify() API."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pedotri
from pedotri.errors import InvalidInputError, UnknownClassificationError


def test_scalar_returns_string() -> None:
    result = pedotri.classify(13, 50, "USDA")
    assert isinstance(result, str)
    assert result == "clay"  # sand=13, clay=50, silt=37 → clay in USDA


def test_batch_returns_list() -> None:
    result = pedotri.classify([13, 45, 70], [50, 24, 10], "FAO")
    assert result == ["fine", "medium", "coarse"]


def test_numpy_array_input() -> None:
    sand = np.array([13, 45, 70])
    clay = np.array([50, 24, 10])
    result = pedotri.classify(sand, clay, "FAO")
    assert result == ["fine", "medium", "coarse"]


def test_locale_returns_localized_names() -> None:
    assert pedotri.classify(13, 50, "USDA", locale="fr") == "argile"
    assert pedotri.classify(13, 50, "USDA", locale="de") == "Ton"
    assert pedotri.classify(13, 50, "USDA", locale="ru") == "глина"


def test_locale_falls_back_through_chain() -> None:
    # fr-FR → fr → argile
    assert pedotri.classify(13, 50, "USDA", locale="fr-FR") == "argile"


def test_locale_unknown_falls_back_to_english() -> None:
    # No 'ja' on USDA classes — should fall back through to en ("clay")
    assert pedotri.classify(13, 50, "USDA", locale="ja") == "clay"


def test_detailed_returns_classify_result() -> None:
    result = pedotri.classify(13, 50, "USDA", detailed=True)
    assert isinstance(result, pedotri.ClassifyResult)
    assert result.key == "clay"
    assert result.group == "fine"
    assert result.parent is None
    assert result.distance > 0  # strictly inside


def test_detailed_batch_returns_list_of_results() -> None:
    results = pedotri.classify([13, 45], [50, 24], "USDA", detailed=True)
    assert isinstance(results, list)
    assert all(isinstance(r, pedotri.ClassifyResult) for r in results)
    assert results[0].key == "clay"


def test_unclassified_returns_none() -> None:
    # (sand=90, clay=80) → silt = -70%, impossible mixture but caller-allowed
    # since 90+80 > 100. This falls outside every polygon.
    result = pedotri.classify(90, 80, "USDA")
    assert result is None


def test_detailed_unclassified_returns_nan_distance() -> None:
    result = pedotri.classify(90, 80, "USDA", detailed=True)
    assert result.key is None
    assert result.name == ""
    assert math.isnan(result.distance)


def test_rejects_unknown_classification() -> None:
    with pytest.raises(UnknownClassificationError):
        pedotri.classify(50, 25, "NONEXISTENT")


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(InvalidInputError, match="same shape"):
        pedotri.classify([1, 2, 3], [1, 2], "USDA")


def test_rejects_out_of_range() -> None:
    with pytest.raises(InvalidInputError, match="\\[0, 100\\]"):
        pedotri.classify(150, 10, "USDA")
    with pytest.raises(InvalidInputError, match="\\[0, 100\\]"):
        pedotri.classify(10, -5, "USDA")


def test_rejects_nan() -> None:
    with pytest.raises(InvalidInputError, match="NaN"):
        pedotri.classify([1.0, float("nan")], [1.0, 1.0], "USDA")


def test_case_insensitive_classification_key() -> None:
    assert pedotri.classify(13, 50, "usda") == "clay"
    assert pedotri.classify(13, 50, "Usda") == "clay"


def test_classify_with_explicit_classification_object() -> None:
    c = pedotri.get_classification("USDA")
    assert pedotri.classify(13, 50, c) == "clay"


def test_classify_rejects_wrong_classification_type() -> None:
    with pytest.raises(TypeError, match="must be a str or Classification"):
        pedotri.classify(13, 50, 42)  # type: ignore[call-overload]


def test_unknown_classification_error_without_available_list() -> None:
    """Direct instantiation without an ``available`` list is supported and
    omits the 'Available: ...' suffix from the message."""
    err = UnknownClassificationError("NOPE")
    assert err.key == "NOPE"
    assert err.available == []
    assert "Available" not in str(err)


def test_canonical_usda_reference_points() -> None:
    """Spot-check well-known points against the USDA triangle.

    These are not boundary points; each sits clearly inside one class.
    """
    # Pure sand corner: sand=95, clay=2, silt=3
    assert pedotri.classify(95, 2, "USDA") == "sand"
    # Pure clay region: sand=10, clay=70, silt=20
    assert pedotri.classify(10, 70, "USDA") == "clay"
    # Loam region: sand=40, clay=20, silt=40
    assert pedotri.classify(40, 20, "USDA") == "loam"
    # Sandy loam: sand=70, clay=10, silt=20
    assert pedotri.classify(70, 10, "USDA") == "sandy_loam"
    # Silt loam: sand=20, clay=15, silt=65
    assert pedotri.classify(20, 15, "USDA") == "silt_loam"
