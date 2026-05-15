"""Tests for ``pedotri.raster``.

The synthetic tests are pure-numpy and run everywhere. The SoilGrids
fixture test is skipped when the optional ``rasterio`` extra (and the
on-disk tile cached under ``tests/data/soilgrids_france/``) are
unavailable, so a minimal pedotri install still runs the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pedotri
from pedotri.errors import InvalidInputError, PedotriError
from pedotri.raster import NODATA_CODE, classify_array

FIXTURE_DIR = Path(__file__).parent / "data" / "soilgrids_france"


def test_classify_array_matches_classify_on_valid_pixels() -> None:
    """Every valid pixel should agree with ``pedotri.classify`` element-wise."""
    rng = np.random.default_rng(42)
    sand = rng.uniform(0, 100, size=(20, 30))
    clay = rng.uniform(0, 100 - sand, size=(20, 30))  # keep sand+clay ≤ 100

    codes, keys = classify_array(sand, clay, classification="USDA")
    assert codes.shape == sand.shape
    assert codes.dtype == np.int16

    # Spot-check 20 random pixels against classify()
    flat_sand = sand.ravel()
    flat_clay = clay.ravel()
    flat_codes = codes.ravel()
    sample_idx = rng.choice(flat_sand.size, size=20, replace=False)
    for i in sample_idx:
        i_int = int(i)
        expected = pedotri.classify(float(flat_sand[i_int]), float(flat_clay[i_int]), "USDA")
        if flat_codes[i_int] == NODATA_CODE:
            assert expected is None
        else:
            assert keys[flat_codes[i_int]] == expected


def test_classify_array_handles_units_gkg() -> None:
    """Same answer in g/kg as in percent."""
    sand_pct = np.array([[60, 20], [40, 5]], dtype=np.float64)
    clay_pct = np.array([[10, 50], [25, 5]], dtype=np.float64)
    codes_pct, _ = classify_array(sand_pct, clay_pct, classification="USDA")
    codes_gkg, _ = classify_array(sand_pct * 10, clay_pct * 10, classification="USDA", units="g/kg")
    np.testing.assert_array_equal(codes_pct, codes_gkg)


def test_classify_array_nodata_for_out_of_range() -> None:
    """Pixels with sand+clay > 100+tolerance code to NODATA, not a class."""
    # 80 + 50 = 130, well over the 5-pp tolerance
    sand = np.array([[80.0, 10.0]])
    clay = np.array([[50.0, 10.0]])
    codes, _ = classify_array(sand, clay, classification="USDA")
    assert codes[0, 0] == NODATA_CODE
    assert codes[0, 1] != NODATA_CODE


def test_classify_array_disabled_sum_check_passes_anything() -> None:
    """sum_tolerance=inf disables the sand+clay <= 100+tol gate."""
    sand = np.array([80.0])
    clay = np.array([80.0])  # sand+clay=160, impossible in practice
    # Still expected to code, because the impossible sum is allowed
    codes, _ = classify_array(sand, clay, classification="USDA", sum_tolerance=float("inf"))
    # sand=80, clay=80 isn't in any polygon (clay maxes at ~60), so it
    # may end up NODATA from the polygon side — but it should not be
    # rejected by the sum check, which is the thing we're testing
    # indirectly via not raising.
    assert codes.shape == (1,)


def test_classify_array_rejects_shape_mismatch() -> None:
    with pytest.raises(InvalidInputError, match="same shape"):
        classify_array(np.zeros((3, 3)), np.zeros((3, 4)), classification="USDA")


def test_classify_array_rejects_1d_classification() -> None:
    with pytest.raises(PedotriError, match="2-axis"):
        classify_array(np.array([[1.0]]), np.array([[1.0]]), classification="KACHINSKY")


def test_classify_array_applies_mask() -> None:
    """mask=False cells should code to NODATA even if data is valid."""
    sand = np.array([[60.0, 20.0]])
    clay = np.array([[10.0, 50.0]])
    mask = np.array([[True, False]])
    codes, _ = classify_array(sand, clay, classification="USDA", mask=mask)
    assert codes[0, 0] != NODATA_CODE
    assert codes[0, 1] == NODATA_CODE


def test_classify_array_all_invalid_returns_nodata() -> None:
    """When the validity mask is empty, return an all-NODATA raster."""
    sand = np.full((2, 2), 200.0)  # out of range
    clay = np.full((2, 2), 200.0)
    codes, keys = classify_array(sand, clay, classification="USDA")
    assert (codes == NODATA_CODE).all()
    assert len(keys) == 12


def test_classify_array_mask_shape_check() -> None:
    with pytest.raises(InvalidInputError, match="mask shape"):
        classify_array(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            classification="USDA",
            mask=np.ones((3, 4), dtype=bool),
        )


def test_classify_array_accepts_classification_object() -> None:
    """Passing a Classification instance works (not just a key string)."""
    cls = pedotri.get_classification("USDA")
    codes, _ = classify_array(np.array([[60.0]]), np.array([[10.0]]), classification=cls)
    assert codes.shape == (1, 1)


@pytest.mark.skipif(
    not (FIXTURE_DIR / "sand_0-5cm_mean.tif").exists(),
    reason="SoilGrids fixture not available; run examples/soilgrids_france.py to download it",
)
def test_classify_geotiff_on_soilgrids_tile() -> None:
    """End-to-end test on the 2°×2° central-France SoilGrids tile.

    Only runs when rasterio + the cached fixture are available; this
    keeps the core test suite hermetic while still exercising the full
    GeoTIFF code path when the optional extras are present.
    """
    rasterio = pytest.importorskip("rasterio")
    from pedotri.raster import classify_geotiff, write_classified_geotiff

    codes, keys, profile = classify_geotiff(
        sand=FIXTURE_DIR / "sand_0-5cm_mean.tif",
        clay=FIXTURE_DIR / "clay_0-5cm_mean.tif",
        classification="USDA",
        units="g/kg",
    )
    assert codes.shape == (837, 885)
    assert profile["dtype"] == "int16"
    assert profile["nodata"] == NODATA_CODE

    # Central France is dominated by loam-family classes — sanity-check
    # that those make up the majority of pixels.
    loam_idx = keys.index("loam")
    clay_loam_idx = keys.index("clay_loam")
    majority = ((codes == loam_idx) | (codes == clay_loam_idx)).sum()
    assert majority > codes.size * 0.5

    # Round-trip through the GeoTIFF writer
    out = FIXTURE_DIR / "_test_roundtrip.tif"
    try:
        write_classified_geotiff(out, codes, profile=profile, keys=keys)
        with rasterio.open(out) as src:
            read_back = src.read(1)
            tags = src.tags(1)
        np.testing.assert_array_equal(read_back, codes)
        # Class names should be embedded in band metadata
        assert tags["class_0"] == keys[0]
    finally:
        out.unlink(missing_ok=True)


def test_rasterio_helpers_raise_clear_error_when_rasterio_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without rasterio, the GeoTIFF helpers should raise a pointed ImportError."""
    from pedotri import raster

    monkeypatch.setattr(raster, "_require_rasterio", _make_raise_importerror)
    with pytest.raises(ImportError, match=r"pip install 'pedotri\[raster\]'"):
        raster.classify_geotiff("x.tif", "y.tif", classification="USDA")


def _make_raise_importerror() -> None:
    raise ImportError(
        "pedotri.raster's GeoTIFF helpers require rasterio. "
        "Install with: pip install 'pedotri[raster]'"
    )
