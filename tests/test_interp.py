"""Tests for ``pedotri.interp`` — kriging of sparse soil samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pedotri.errors import InvalidInputError, PedotriError


def _import_interp() -> Any:
    pytest.importorskip("pykrige")
    pytest.importorskip("scipy")
    from pedotri import interp

    return interp


def test_krige_samples_reproduces_near_constant_field() -> None:
    """Samples clustered tightly around a value should krige to ~that value.

    pykrige's variogram fit rejects exactly-constant fields (zero
    variation breaks the bounded least-squares step), so we use a
    tiny perturbation to mimic real measurement noise.
    """
    interp = _import_interp()
    rng = np.random.default_rng(0)
    xs = np.array([0.0, 1, 0, 1])
    ys = np.array([0.0, 0, 1, 1])
    vals = 42.0 + rng.normal(0, 0.01, size=4)
    grid, _var, profile = interp.krige_samples(
        xs,
        ys,
        vals,
        bbox=(0, 0, 1, 1),
        resolution=0.25,
        variogram="linear",
    )
    np.testing.assert_allclose(grid, 42.0, atol=0.5)
    assert profile["dtype"] == "float64"
    assert profile["count"] == 1


def test_krige_samples_returns_kriging_variance() -> None:
    """Variance grid: zero at sample locations, positive away from them."""
    interp = _import_interp()
    xs = np.array([0.0, 1, 0, 1])
    ys = np.array([0.0, 0, 1, 1])
    vals = np.array([1.0, 2, 3, 4])
    _grid, variance, _ = interp.krige_samples(
        xs,
        ys,
        vals,
        bbox=(0, 0, 1, 1),
        resolution=0.5,
        variogram="linear",
    )
    # Variance must be non-negative everywhere
    assert (variance >= -1e-9).all()


def test_krige_samples_rejects_mismatched_lengths() -> None:
    interp = _import_interp()
    with pytest.raises(InvalidInputError, match="equal length"):
        interp.krige_samples(
            [0, 1, 2],
            [0, 1],
            [0, 1, 2],
            bbox=(0, 0, 1, 1),
            resolution=0.5,
        )


def test_krige_samples_rejects_too_few_samples() -> None:
    interp = _import_interp()
    with pytest.raises(InvalidInputError, match="at least 3"):
        interp.krige_samples(
            [0, 1],
            [0, 1],
            [0, 1],
            bbox=(0, 0, 1, 1),
            resolution=0.5,
        )


def test_krige_samples_rejects_zero_area_bbox() -> None:
    interp = _import_interp()
    with pytest.raises(InvalidInputError, match="bbox"):
        interp.krige_samples(
            [0, 1, 0],
            [0, 0, 1],
            [1.0, 2, 3],
            bbox=(0, 0, 0, 1),
            resolution=0.5,
        )


def test_krige_sand_clay_dict_input() -> None:
    interp = _import_interp()
    rng = np.random.default_rng(0)
    samples = [
        {
            "x": float(rng.uniform(0, 100)),
            "y": float(rng.uniform(0, 100)),
            "sand": float(rng.uniform(20, 70)),
            "clay": float(rng.uniform(10, 40)),
        }
        for _ in range(20)
    ]
    sand_grid, clay_grid, profile = interp.krige_sand_clay(
        samples,
        bbox=(0, 0, 100, 100),
        resolution=10.0,
    )
    assert sand_grid.shape == clay_grid.shape == (10, 10)
    assert profile["transform"] is not None


def test_krige_sand_clay_array_input() -> None:
    interp = _import_interp()
    arr = np.array(
        [
            [0.0, 0.0, 40.0, 20.0],
            [100.0, 0.0, 60.0, 10.0],
            [0.0, 100.0, 30.0, 30.0],
            [100.0, 100.0, 70.0, 5.0],
        ]
    )
    sand_grid, clay_grid, _profile = interp.krige_sand_clay(
        arr,
        bbox=(0, 0, 100, 100),
        resolution=20.0,
    )
    assert sand_grid.shape == clay_grid.shape == (5, 5)


def test_krige_sand_clay_array_wrong_shape_raises() -> None:
    interp = _import_interp()
    arr = np.array([[0.0, 0.0, 40.0]])  # 3 cols, not 4
    with pytest.raises(InvalidInputError, match="N, 4"):
        interp.krige_sand_clay(arr, bbox=(0, 0, 1, 1), resolution=0.5)


def test_krige_sand_clay_dict_missing_keys_raises() -> None:
    interp = _import_interp()
    with pytest.raises(InvalidInputError, match="keys"):
        interp.krige_sand_clay(
            [{"x": 0, "y": 0, "sand": 40}],  # missing 'clay'
            bbox=(0, 0, 1, 1),
            resolution=0.5,
        )


def test_krige_sand_clay_polygon_masks_outside_cells() -> None:
    """Cells outside the mask polygon should be nan in both rasters."""
    interp = _import_interp()
    pytest.importorskip("shapely")
    pytest.importorskip("rasterio")
    from shapely.geometry import Polygon

    # 16 samples on a 10x10 grid corner; mask to a small triangle in the SW
    rng = np.random.default_rng(1)
    samples = [
        {
            "x": float(rng.uniform(0, 100)),
            "y": float(rng.uniform(0, 100)),
            "sand": 40.0 + float(rng.normal(0, 2)),
            "clay": 20.0 + float(rng.normal(0, 2)),
        }
        for _ in range(16)
    ]
    mask = Polygon([(0, 0), (30, 0), (0, 30)])
    sand_grid, clay_grid, _profile = interp.krige_sand_clay(
        samples,
        bbox=(0, 0, 100, 100),
        resolution=10.0,
        mask_polygon=mask,
    )
    # Cell at (90, 90) — well outside the SW triangle — must be nan
    assert np.isnan(sand_grid[0, -1])
    assert np.isnan(clay_grid[0, -1])
    # Cell at (5, 5) — inside the triangle (row 9, col 0 in raster coords) — finite
    assert np.isfinite(sand_grid[-1, 0])


def test_krige_sand_clay_integrates_with_classify_array() -> None:
    """Kriged rasters classify cleanly with no surprises."""
    interp = _import_interp()
    pytest.importorskip("scipy")
    from pedotri.raster import NODATA_CODE, classify_array

    rng = np.random.default_rng(42)
    samples = [
        {
            "x": float(rng.uniform(0, 100)),
            "y": float(rng.uniform(0, 100)),
            "sand": float(rng.uniform(20, 70)),
            "clay": float(rng.uniform(10, 40)),
        }
        for _ in range(20)
    ]
    sand_grid, clay_grid, _profile = interp.krige_sand_clay(
        samples,
        bbox=(0, 0, 100, 100),
        resolution=10.0,
    )
    codes, keys = classify_array(sand_grid, clay_grid, classification="USDA")
    assert codes.shape == (10, 10)
    valid = codes != NODATA_CODE
    assert valid.any(), "kriged grid should yield at least one classified cell"
    for code in np.unique(codes[valid]):
        assert keys[int(code)] in {
            "clay",
            "silty_clay",
            "silty_clay_loam",
            "sandy_clay",
            "sandy_clay_loam",
            "clay_loam",
            "silt",
            "silt_loam",
            "loam",
            "sand",
            "loamy_sand",
            "sandy_loam",
        }


def test_pykrige_missing_raises_pedotri_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pykrige installed, the error message should point at the extra."""
    import sys

    monkeypatch.setitem(sys.modules, "pykrige", None)
    monkeypatch.setitem(sys.modules, "pykrige.ok", None)
    from pedotri import interp

    with pytest.raises(PedotriError, match=r"pip install 'pedotri\[interp\]'"):
        interp.krige_samples(
            [0, 1, 2],
            [0, 1, 2],
            [1.0, 2, 3],
            bbox=(0, 0, 1, 1),
            resolution=0.5,
        )
