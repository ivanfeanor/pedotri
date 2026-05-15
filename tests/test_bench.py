"""Tests for ``pedotri.bench``.

The bench module is primarily a reproducibility helper for users — it
times pedotri vs. ``soiltexture``. Most of the value of the tests is
making sure the timing / sampling helpers are deterministic and that
the report renders cleanly. The full soiltexture comparison only runs
when the optional ``soiltexture`` dependency is importable.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import pedotri
from pedotri.bench import (
    SOILTEXTURE_TO_PEDOTRI_USDA,
    BenchmarkReport,
    BenchmarkRow,
    generate_inputs,
    run_soiltexture_benchmark,
    time_call,
)


def _can_import_soiltexture() -> bool:
    try:
        import soiltexture  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_SOILTEXTURE = _can_import_soiltexture()


def test_generate_inputs_is_deterministic() -> None:
    s1, c1 = generate_inputs(1000, seed=0)
    s2, c2 = generate_inputs(1000, seed=0)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(c1, c2)


def test_generate_inputs_in_feasible_triangle() -> None:
    sand, clay = generate_inputs(10_000, seed=1)
    assert sand.shape == (10_000,)
    assert (sand >= 0).all()
    assert (sand <= 100).all()
    assert (clay >= 0).all()
    assert (clay <= 100).all()
    # Inverse-CDF triangle sample: every (sand, clay) sums to ≤ 100
    assert (sand + clay <= 100.0 + 1e-9).all()


def test_generate_inputs_different_seeds_diverge() -> None:
    s1, _ = generate_inputs(1000, seed=0)
    s2, _ = generate_inputs(1000, seed=1)
    # Astronomically unlikely to match exactly
    assert not np.array_equal(s1, s2)


def test_time_call_returns_minimum() -> None:
    # A trivial callable: time_call should still return a positive float
    t = time_call(lambda: None, repeat=2)
    assert t >= 0.0


def test_soiltexture_remap_covers_all_usda_classes() -> None:
    """The remap dict needs an entry for every USDA class pedotri ships."""
    usda = pedotri.get_classification("USDA")
    pedotri_keys = {c.key for c in usda.classes}
    remap_values = set(SOILTEXTURE_TO_PEDOTRI_USDA.values())
    assert pedotri_keys == remap_values


def test_run_soiltexture_benchmark_rejects_unsupported_classification() -> None:
    with pytest.raises(ValueError, match="USDA, FAO, ISSS, INTERNATIONAL"):
        run_soiltexture_benchmark(sizes=[10], classification="GEPPA")


def test_run_soiltexture_benchmark_handles_missing_soiltexture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When soiltexture isn't importable, the pedotri rows still come back."""
    # Hide soiltexture from the import system: even if it's installed
    # in this dev environment, the test must verify the fallback path.
    monkeypatch.setitem(sys.modules, "soiltexture", None)
    report = run_soiltexture_benchmark(sizes=[100], repeat=1, check_agreement=False)
    libs = {row.library for row in report.rows}
    assert libs == {"pedotri"}


def test_report_format_table() -> None:
    report = BenchmarkReport(classification="USDA")
    report.rows = [
        BenchmarkRow(1000, "pedotri", 0.001, 1_000_000),
        BenchmarkRow(1000, "soiltexture", 0.01, 100_000),
    ]
    report.agreement = 1.0
    text = report.format()
    assert "pedotri" in text
    assert "soiltexture" in text
    assert "100.00%" in text


def test_report_speedup_table() -> None:
    report = BenchmarkReport(classification="USDA")
    report.rows = [
        BenchmarkRow(1000, "pedotri", 0.001, 1_000_000),
        BenchmarkRow(1000, "soiltexture", 0.01, 100_000),
        BenchmarkRow(10_000, "pedotri", 0.01, 1_000_000),
        BenchmarkRow(10_000, "soiltexture", 0.1, 100_000),
    ]
    table = report.speedup_table()
    assert len(table) == 2
    n_values = [row[0] for row in table]
    assert n_values == [1000, 10_000]
    speedups = [row[1] for row in table]
    for sp in speedups:
        assert sp == pytest.approx(10.0)


@pytest.mark.skipif(not _HAS_SOILTEXTURE, reason="soiltexture not installed")
def test_pedotri_vs_soiltexture_agree_on_usda() -> None:
    """Output agreement is the main correctness guarantee of the bench."""
    report = run_soiltexture_benchmark(
        sizes=[200], repeat=1, classification="USDA", check_agreement=True
    )
    libs = {row.library for row in report.rows}
    assert libs == {"pedotri", "soiltexture"}
    assert report.agreement == 1.0


@pytest.mark.skipif(not _HAS_SOILTEXTURE, reason="soiltexture not installed")
def test_pedotri_is_faster_than_soiltexture() -> None:
    """The whole point of the bench: pedotri should be measurably faster."""
    report = run_soiltexture_benchmark(
        sizes=[5_000], repeat=3, classification="USDA", check_agreement=False
    )
    speedup = next(s for n, s in report.speedup_table() if n == 5_000)
    # On all hardware we've tried, this is >5x. Use a forgiving 2x
    # threshold so noisy CI machines don't false-flag.
    assert speedup > 2.0, f"expected pedotri to be >2x faster, got {speedup:.2f}x"
