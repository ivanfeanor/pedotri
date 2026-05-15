"""Tests for ``pedotri.grid`` — target-grid reprojection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.grid import (
    RESAMPLING_METHODS,
    TargetGrid,
    align_to_grid,
    reproject_to_grid,
)


def _wgs84_profile(
    width: int, height: int, *, res: float, dtype: str = "float32"
) -> dict[str, Any]:
    from rasterio.transform import from_origin

    return {
        "driver": "GTiff",
        "crs": "EPSG:4326",
        "transform": from_origin(0.0, height * res, res, res),
        "width": width,
        "height": height,
        "dtype": dtype,
        "nodata": -9999.0 if dtype.startswith("float") else 255,
    }


def _ramp(shape: tuple[int, int], dtype: str = "float32") -> np.ndarray:
    return np.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)


# --- TargetGrid -----------------------------------------------------------


def test_targetgrid_from_profile_round_trips() -> None:
    prof = _wgs84_profile(20, 20, res=1.0)
    grid = TargetGrid.from_profile(prof)
    assert grid.width == 20
    assert grid.height == 20
    assert grid.matches(prof)
    assert grid.shape == (20, 20)
    assert grid.pixel_size == (1.0, 1.0)


def test_targetgrid_from_bounds() -> None:
    grid = TargetGrid.from_bounds((0.0, 0.0, 2.0, 1.0), resolution=0.25)
    assert grid.width == 8
    assert grid.height == 4
    left, bottom, right, top = grid.bounds
    assert (left, bottom, right, top) == (0.0, 0.0, 2.0, 1.0)


def test_targetgrid_rejects_bad_size() -> None:
    from rasterio.transform import from_origin

    with pytest.raises(InvalidInputError, match="positive"):
        TargetGrid(crs="EPSG:4326", transform=from_origin(0, 1, 1, 1), width=0, height=1)


def test_targetgrid_from_profile_rejects_missing_keys() -> None:
    with pytest.raises(InvalidInputError, match="missing"):
        TargetGrid.from_profile({"crs": "EPSG:4326"})


def test_targetgrid_from_bounds_rejects_inverted_bounds() -> None:
    with pytest.raises(InvalidInputError, match="left<right"):
        TargetGrid.from_bounds((2.0, 0.0, 0.0, 1.0), resolution=0.5)


def test_targetgrid_from_bounds_rejects_non_positive_resolution() -> None:
    with pytest.raises(InvalidInputError, match="resolution"):
        TargetGrid.from_bounds((0.0, 0.0, 2.0, 1.0), resolution=0.0)


# --- reproject_to_grid ----------------------------------------------------


def test_reproject_same_grid_is_noop() -> None:
    pytest.importorskip("rasterio")
    src = _ramp((20, 20))
    prof = _wgs84_profile(20, 20, res=1.0)
    grid = TargetGrid.from_profile(prof)
    out, out_prof = reproject_to_grid(src, prof, grid)
    np.testing.assert_array_equal(out, src)
    assert out_prof["width"] == 20
    assert out_prof["height"] == 20


def test_reproject_downsample_uses_average_for_continuous() -> None:
    """Downsampling a smooth float ramp should preserve the mean per cell."""
    pytest.importorskip("rasterio")
    src = _ramp((20, 20))
    prof = _wgs84_profile(20, 20, res=1.0)
    # Target = same grid downsampled 2× (10×10, 2-degree pixels).
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    out, _ = reproject_to_grid(src, prof, target)
    assert out.shape == (10, 10)
    # The default kernel for float downsample is `average`. Each
    # destination pixel covers a 2×2 source window; the mean of a
    # 2×2 ramp window should land at the analytical midpoint.
    # Verify: top-left destination pixel covers source rows 0:2, cols 0:2,
    # values [0, 1, 20, 21] → mean = 10.5.
    assert out[0, 0] == pytest.approx(10.5)


def test_reproject_upsample_uses_bilinear_for_continuous() -> None:
    pytest.importorskip("rasterio")
    src = _ramp((4, 4))
    prof = _wgs84_profile(4, 4, res=2.0)
    # Upsample to 2× resolution: 8×8.
    target = TargetGrid.from_bounds((0.0, 0.0, 8.0, 8.0), resolution=1.0)
    out, _ = reproject_to_grid(src, prof, target)
    assert out.shape == (8, 8)
    # Bilinear preserves the corner extrema (no nearest-neighbour banding).
    assert out.min() == pytest.approx(src.min(), abs=1.0)
    assert out.max() == pytest.approx(src.max(), abs=1.0)


def test_reproject_categorical_uses_nearest() -> None:
    """Integer inputs must round-trip with no class-code averaging."""
    pytest.importorskip("rasterio")
    src = np.zeros((20, 20), dtype=np.uint8)
    src[5:15, 5:15] = 40  # WorldCover-cropland-shaped block
    prof = _wgs84_profile(20, 20, res=1.0, dtype="uint8")
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    out, _ = reproject_to_grid(src, prof, target)
    # Only the two original class codes appear — no averaging artefacts.
    assert set(np.unique(out).tolist()) == {0, 40}


def test_reproject_resampling_override() -> None:
    """Explicit resampling= overrides the dtype-based default."""
    pytest.importorskip("rasterio")
    src = _ramp((20, 20))
    prof = _wgs84_profile(20, 20, res=1.0)
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    out_avg, _ = reproject_to_grid(src, prof, target, resampling="average")
    out_nn, _ = reproject_to_grid(src, prof, target, resampling="nearest")
    # Nearest picks a single source pixel; average pulls toward the
    # local mean → results differ.
    assert not np.array_equal(out_avg, out_nn)


def test_reproject_rejects_unknown_resampling() -> None:
    pytest.importorskip("rasterio")
    prof = _wgs84_profile(20, 20, res=1.0)
    # Use a *different* target so we don't hit the same-grid short circuit.
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    with pytest.raises(InvalidInputError, match="Unknown resampling"):
        reproject_to_grid(_ramp((20, 20)), prof, target, resampling="bogus")


def test_reproject_rejects_non_2d_array() -> None:
    pytest.importorskip("rasterio")
    prof = _wgs84_profile(20, 20, res=1.0)
    target = TargetGrid.from_profile(prof)
    with pytest.raises(InvalidInputError, match="2-D"):
        reproject_to_grid(np.zeros((2, 20, 20)), prof, target)


def test_reproject_rejects_shape_mismatch() -> None:
    pytest.importorskip("rasterio")
    prof = _wgs84_profile(20, 20, res=1.0)
    target = TargetGrid.from_profile(prof)
    with pytest.raises(InvalidInputError, match="shape"):
        reproject_to_grid(np.zeros((10, 10), dtype=np.float32), prof, target)


def test_reproject_inconsistent_nodata_silently_dropped() -> None:
    """Profile that mismatches array dtype shouldn't make rasterio reject."""
    pytest.importorskip("rasterio")
    # Float profile (nodata=-9999) reused with a uint8 raster — defensive
    # path: the impossible-for-uint8 nodata is ignored.
    bad_profile = _wgs84_profile(20, 20, res=1.0, dtype="float32")
    bad_profile["dtype"] = "uint8"  # keep float nodata, change dtype
    arr = np.zeros((20, 20), dtype=np.uint8)
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    out, _ = reproject_to_grid(arr, bad_profile, target)
    assert out.shape == (10, 10)


# --- align_to_grid -------------------------------------------------------


def test_align_to_grid_picks_first_source_as_default_target() -> None:
    pytest.importorskip("rasterio")
    a = _ramp((20, 20))
    b = _ramp((10, 10))
    prof_a = _wgs84_profile(20, 20, res=1.0)
    prof_b = _wgs84_profile(10, 10, res=2.0)
    aligned, target = align_to_grid([(a, prof_a), (b, prof_b)])
    assert target.matches(prof_a)
    for arr, prof in aligned:
        assert arr.shape == (20, 20)
        assert TargetGrid.from_profile(prof).matches(prof_a)


def test_align_to_grid_explicit_target_overrides_first_source() -> None:
    pytest.importorskip("rasterio")
    a = _ramp((20, 20))
    b = _ramp((10, 10))
    prof_a = _wgs84_profile(20, 20, res=1.0)
    prof_b = _wgs84_profile(10, 10, res=2.0)
    target = TargetGrid.from_profile(prof_b)
    aligned, returned = align_to_grid([(a, prof_a), (b, prof_b)], target=target)
    assert returned is target
    for arr, _ in aligned:
        assert arr.shape == (10, 10)


def test_align_to_grid_per_source_resampling() -> None:
    pytest.importorskip("rasterio")
    cont = _ramp((20, 20))
    cat = np.zeros((20, 20), dtype=np.uint8)
    cat[5:15, 5:15] = 40
    prof_cont = _wgs84_profile(20, 20, res=1.0)
    prof_cat = _wgs84_profile(20, 20, res=1.0, dtype="uint8")
    target = TargetGrid.from_bounds((0.0, 0.0, 20.0, 20.0), resolution=2.0)
    # Force bilinear on cont, nearest on cat (cat would auto-pick nearest
    # anyway; this exercises the override pathway).
    aligned, _ = align_to_grid(
        [(cont, prof_cont), (cat, prof_cat)],
        target=target,
        resampling=["bilinear", "nearest"],
    )
    cont_out, cat_out = aligned[0][0], aligned[1][0]
    assert cont_out.dtype == np.float32
    assert cat_out.dtype == np.uint8
    assert set(np.unique(cat_out).tolist()) == {0, 40}


def test_align_to_grid_rejects_mismatched_resampling_length() -> None:
    pytest.importorskip("rasterio")
    prof = _wgs84_profile(20, 20, res=1.0)
    src = _ramp((20, 20))
    with pytest.raises(InvalidInputError, match="length"):
        align_to_grid([(src, prof), (src, prof)], resampling=["bilinear"])


def test_align_to_grid_rejects_empty_sources() -> None:
    with pytest.raises(InvalidInputError, match="no sources"):
        align_to_grid([])


def test_resampling_methods_constant_includes_expected() -> None:
    for required in ("nearest", "bilinear", "average"):
        assert required in RESAMPLING_METHODS
