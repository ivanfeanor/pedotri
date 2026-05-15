"""Tests for the smoothing / PNG / vector / GeoTIFF-extras pieces.

Coverage of ``pedotri.raster``'s post-processing helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.raster import (
    NODATA_CODE,
    classified_to_features,
    render_classified_png,
    smooth_codes,
    write_classified_geotiff,
    write_features_shapefile,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_smooth_codes_removes_lone_pixel() -> None:
    """A single stray pixel surrounded by class 0 should smooth to 0."""
    pytest.importorskip("scipy")
    codes = np.zeros((5, 5), dtype=np.int16)
    codes[2, 2] = 1
    out = smooth_codes(codes, window=3)
    assert out[2, 2] == 0


def test_smooth_codes_preserves_majority_region() -> None:
    """A 3x3 block of class 1 inside class 0 should survive smoothing."""
    pytest.importorskip("scipy")
    codes = np.zeros((7, 7), dtype=np.int16)
    codes[2:5, 2:5] = 1
    out = smooth_codes(codes, window=3)
    # Center pixel and its 8 neighbors are class 1; smoothing keeps the
    # core (3 of 9 neighbors of the center are class 1 themselves).
    assert out[3, 3] == 1


def test_smooth_codes_preserves_nodata() -> None:
    pytest.importorskip("scipy")
    codes = np.zeros((5, 5), dtype=np.int16)
    codes[0, 0] = NODATA_CODE
    out = smooth_codes(codes, window=3)
    assert out[0, 0] == NODATA_CODE
    assert (out[1:, 1:] == 0).all()


def test_smooth_codes_iterations_are_idempotent_on_uniform_input() -> None:
    """Re-smoothing an already-uniform raster is a no-op."""
    pytest.importorskip("scipy")
    codes = np.full((10, 10), 3, dtype=np.int16)
    out1 = smooth_codes(codes, window=3, iterations=1)
    out2 = smooth_codes(codes, window=3, iterations=5)
    np.testing.assert_array_equal(out1, out2)


def test_smooth_codes_rejects_even_window() -> None:
    pytest.importorskip("scipy")
    with pytest.raises(InvalidInputError, match="odd"):
        smooth_codes(np.zeros((4, 4), dtype=np.int16), window=4)


def test_smooth_codes_rejects_nonpositive_iterations() -> None:
    pytest.importorskip("scipy")
    with pytest.raises(InvalidInputError, match="iterations"):
        smooth_codes(np.zeros((4, 4), dtype=np.int16), iterations=0)


def test_smooth_codes_all_nodata_short_circuits() -> None:
    """An all-nodata raster returns unchanged without invoking scipy."""
    pytest.importorskip("scipy")
    codes = np.full((4, 4), NODATA_CODE, dtype=np.int16)
    out = smooth_codes(codes, window=3)
    np.testing.assert_array_equal(out, codes)


def test_write_classified_geotiff_round_trip_with_colormap(tmp_path: Path) -> None:
    """GeoTIFF must come back identical, with palette + class tags."""
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    codes = np.array([[0, 1, NODATA_CODE], [2, 0, 1]], dtype=np.int16)
    keys = ["loam", "clay", "sand"]
    profile = {
        "driver": "GTiff",
        "height": codes.shape[0],
        "width": codes.shape[1],
        "transform": from_origin(0, 2, 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
    }
    out = tmp_path / "out.tif"
    write_classified_geotiff(out, codes, profile=profile, keys=keys)
    with rio.open(out) as src:
        read_back = src.read(1)
        tags = src.tags(1)
        cmap = src.colormap(1)
        desc = src.descriptions[0]
        nodata = src.nodata
    assert nodata == 255
    # On-disk values: nodata cell remapped to 255, others unchanged
    expected = np.array([[0, 1, 255], [2, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(read_back, expected)
    assert tags["class_0"] == "loam"
    assert tags["class_2"] == "sand"
    assert "soil texture class index" in (desc or "")
    # Color table entries for each class exist + transparent nodata
    for i in range(3):
        assert i in cmap
        assert len(cmap[i]) == 4  # RGBA
    assert cmap[255] == (0, 0, 0, 0)  # transparent nodata


def test_write_classified_geotiff_rejects_too_many_classes(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "transform": from_origin(0, 1, 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
    }
    with pytest.raises(InvalidInputError, match="255"):
        write_classified_geotiff(
            tmp_path / "x.tif",
            np.zeros((1, 1), dtype=np.int16),
            profile=profile,
            keys=[f"k{i}" for i in range(256)],
        )


def test_render_classified_png_writes_nonempty_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    codes = np.array([[0, 1, 0], [1, 0, 1], [NODATA_CODE, 0, 1]], dtype=np.int16)
    out = tmp_path / "map.png"
    render_classified_png(out, codes, keys=["loam", "clay"], title="t", extent=(0, 3, 0, 3))
    assert out.stat().st_size > 1000  # any non-trivial PNG is at least that


def test_classified_to_features_yields_polygons_per_region(tmp_path: Path) -> None:
    """Each contiguous class region becomes one polygon."""
    pytest.importorskip("rasterio")
    pytest.importorskip("shapely")
    from rasterio.transform import from_origin

    # 4x4 with two clean horizontal stripes
    codes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int16,
    )
    transform = from_origin(0, 4, 1, 1)
    feats = classified_to_features(codes, keys=["a", "b"], transform=transform)
    # Expect at least one polygon per class
    class_ids = {f["properties"]["class_id"] for f in feats}
    assert class_ids == {0, 1}


def test_classified_to_features_excludes_nodata() -> None:
    pytest.importorskip("rasterio")
    pytest.importorskip("shapely")
    from rasterio.transform import from_origin

    codes = np.full((3, 3), NODATA_CODE, dtype=np.int16)
    codes[1, 1] = 0
    feats = classified_to_features(codes, keys=["x"], transform=from_origin(0, 3, 1, 1))
    # Only one feature: the single inner cell. No nodata feature emitted.
    assert len(feats) == 1
    assert feats[0]["properties"]["class_id"] == 0


def test_write_features_shapefile_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    fiona = pytest.importorskip("fiona")
    from shapely.geometry import Polygon

    features = [
        {
            "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "properties": {"class_id": 0, "class_key": "loam"},
        },
        {
            "geometry": Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            "properties": {"class_id": 1, "class_key": "clay"},
        },
    ]
    out = tmp_path / "out.shp"
    write_features_shapefile(features, out, crs="EPSG:4326")
    assert out.exists()

    with fiona.open(out) as src:
        rows = list(src)
        assert len(rows) == 2
        keys = sorted(r["properties"]["class_key"] for r in rows)
        assert keys == ["clay", "loam"]


def test_write_features_shapefile_geopackage_driver(tmp_path: Path) -> None:
    """Path suffix selects the OGR driver."""
    pytest.importorskip("shapely")
    fiona = pytest.importorskip("fiona")
    from shapely.geometry import Polygon

    features = [
        {
            "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "properties": {"class_id": 0, "class_key": "loam"},
        }
    ]
    out = tmp_path / "out.gpkg"
    write_features_shapefile(features, out, crs="EPSG:4326")
    with fiona.open(out) as src:
        assert src.driver == "GPKG"
