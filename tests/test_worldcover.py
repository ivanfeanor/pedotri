"""Tests for the WorldCover AOI fetcher (offline-only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pedotri.errors import InvalidInputError
from pedotri.sources import worldcover

if TYPE_CHECKING:
    from pathlib import Path

# --- tile arithmetic ----------------------------------------------------


def test_tile_id_format() -> None:
    assert worldcover.tile_id(3, 48) == "N48E003"
    assert worldcover.tile_id(-3, -3) == "S03W003"
    assert worldcover.tile_id(0, 0) == "N00E000"
    assert worldcover.tile_id(-12, -33) == "S33W012"


def test_tile_id_rejects_non_aligned_origin() -> None:
    with pytest.raises(InvalidInputError):
        worldcover.tile_id(2, 48)  # 2° is not a multiple of 3°


def test_intersecting_tiles_central_france() -> None:
    """Central France stays inside a single horizontal band of tiles."""
    tiles = worldcover.intersecting_tiles((2.5, 46.5, 4.0, 47.9))
    ids = {t[0] for t in tiles}
    assert ids == {"N45E000", "N45E003"}


def test_intersecting_tiles_aligned_bbox_is_one_tile() -> None:
    tiles = worldcover.intersecting_tiles((3.0, 48.0, 6.0, 51.0))
    assert tiles == [("N48E003", 3, 48)]


def test_intersecting_tiles_spans_equator_and_meridian() -> None:
    tiles = worldcover.intersecting_tiles((-1.0, -1.0, 1.0, 1.0))
    ids = {t[0] for t in tiles}
    assert ids == {"S03W003", "S03E000", "N00W003", "N00E000"}


def test_tile_url_includes_version_and_year() -> None:
    assert "v200" in worldcover.tile_url("N48E003", 2021)
    assert "v100" in worldcover.tile_url("N48E003", 2020)
    assert "N48E003" in worldcover.tile_url("N48E003", 2021)


def test_tile_url_rejects_unknown_year() -> None:
    with pytest.raises(InvalidInputError):
        worldcover.tile_url("N48E003", 2025)


# --- bbox + input validation -------------------------------------------


def test_fetch_aoi_rejects_inverted_bbox() -> None:
    with pytest.raises(InvalidInputError, match="west < east"):
        worldcover.fetch_aoi((4.0, 47.0, 2.0, 48.0))


def test_fetch_aoi_rejects_out_of_range() -> None:
    with pytest.raises(InvalidInputError, match="-180"):
        worldcover.fetch_aoi((181.0, 0.0, 182.0, 1.0))


def test_fetch_aoi_rejects_too_large_aoi() -> None:
    # 10° × 10° at ~10 m resolution would materialize ≈ 1.2 Gpix.
    with pytest.raises(InvalidInputError, match="cap"):
        worldcover.fetch_aoi((0.0, 0.0, 10.0, 10.0))


def test_fetch_aoi_rejects_unsupported_year() -> None:
    with pytest.raises(InvalidInputError, match="year"):
        worldcover.fetch_aoi((2.5, 47.0, 2.6, 47.1), year=2099)


def test_bounds_from_geojson_dict() -> None:
    bounds = worldcover._bounds_from_geo_interface(
        {
            "type": "Polygon",
            "coordinates": [[[2.5, 47.0], [2.6, 47.0], [2.6, 47.1], [2.5, 47.1], [2.5, 47.0]]],
        }
    )
    assert bounds == (2.5, 47.0, 2.6, 47.1)


# --- mocked end-to-end fetch -------------------------------------------


class _SyntheticTile:
    """A fake remote WorldCover tile with a constant fill class."""

    def __init__(self, lon0: int, lat0: int, fill: int) -> None:
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.fill = fill


def _fake_read_tile_window(synthetic_tiles: dict[str, _SyntheticTile]):
    """Build a stub for ``worldcover._read_tile_window``."""

    def reader(
        url: str,
        *,
        bbox: tuple[float, float, float, float],
        rasterio: Any,
        window_from_bounds: Any,
    ):
        # The URL ends in '_<TILE>.tif'; recover the tile id from it.
        stem = url.rsplit(".", 1)[0]
        tile = stem.rsplit("_", 1)[-1]
        tile_meta = synthetic_tiles.get(tile)
        if tile_meta is None:
            return None
        tile_bbox = (
            tile_meta.lon0,
            tile_meta.lat0,
            tile_meta.lon0 + worldcover.TILE_DEGREES,
            tile_meta.lat0 + worldcover.TILE_DEGREES,
        )
        west = max(bbox[0], tile_bbox[0])
        south = max(bbox[1], tile_bbox[1])
        east = min(bbox[2], tile_bbox[2])
        north = min(bbox[3], tile_bbox[3])
        if east <= west or north <= south:
            return None
        # Native ≈ 10 m → degrees; mirror the calling function's choice.
        pixel_size = 10.0 / 111_320.0
        cols = max(round((east - west) / pixel_size), 1)
        rows = max(round((north - south) / pixel_size), 1)
        arr = np.full((rows, cols), tile_meta.fill, dtype=np.uint8)
        return arr, (west, south, east, north)

    return reader


def test_fetch_aoi_mocked_single_tile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    fake_tiles = {"N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND)}
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window(fake_tiles))
    # Tiny AOI ~100 m × 100 m inside the synthetic tile.
    aoi = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert aoi.array.dtype == np.uint8
    assert aoi.array.shape[0] > 0
    assert aoi.array.shape[1] > 0
    # Whole AOI is inside the synthetic cropland fill.
    assert int(aoi.array.min()) == worldcover.CROPLAND
    assert int(aoi.array.max()) == worldcover.CROPLAND
    assert aoi.cached is False
    assert aoi.profile["nodata"] == 255
    assert aoi.profile["crs"] == "EPSG:4326"


def test_fetch_aoi_mocked_multi_tile_mosaic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("rasterio")
    # Two horizontally-adjacent synthetic tiles with different classes.
    fake_tiles = {
        "N45E000": _SyntheticTile(0, 45, fill=worldcover.GRASSLAND),
        "N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND),
    }
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window(fake_tiles))
    # AOI straddles the 3°E tile seam.
    aoi = worldcover.fetch_aoi((2.999, 47.0, 3.001, 47.001), year=2021, cache_dir=tmp_path)
    classes = {int(c) for c in np.unique(aoi.array) if c != 255}
    assert classes == {worldcover.GRASSLAND, worldcover.CROPLAND}


def test_fetch_aoi_caches_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    calls: list[str] = []
    fake_tiles = {"N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND)}
    base = _fake_read_tile_window(fake_tiles)

    def counted_reader(url, **kwargs):
        calls.append(url)
        return base(url, **kwargs)

    monkeypatch.setattr(worldcover, "_read_tile_window", counted_reader)

    first = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    second = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert first.cached is False
    assert second.cached is True
    # Only one remote read across both calls.
    assert len(calls) == 1
    np.testing.assert_array_equal(first.array, second.array)


def test_class_fraction_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    fake_tiles = {"N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND)}
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window(fake_tiles))
    aoi = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert aoi.class_fraction(worldcover.CROPLAND) == pytest.approx(1.0)
    assert aoi.class_fraction([worldcover.TREE_COVER]) == pytest.approx(0.0)


def test_fetch_aoi_from_polygon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shapely = pytest.importorskip("shapely.geometry")
    pytest.importorskip("rasterio")
    fake_tiles = {"N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND)}
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window(fake_tiles))
    geom = shapely.box(3.5, 47.0, 3.501, 47.001)
    aoi = worldcover.fetch_aoi_from_polygon(geom, year=2021, cache_dir=tmp_path)
    assert int(aoi.array.min()) == worldcover.CROPLAND


def test_clear_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    fake_tiles = {"N45E003": _SyntheticTile(3, 45, fill=worldcover.CROPLAND)}
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window(fake_tiles))
    worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert sum(1 for _ in tmp_path.glob("*.tif")) == 1
    assert worldcover.clear_cache(cache_dir=tmp_path) == 1
    assert sum(1 for _ in tmp_path.glob("*.tif")) == 0


def test_missing_tile_returns_nodata_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a tile is missing (ocean / not-published), pixels remain nodata."""
    pytest.importorskip("rasterio")
    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_read_tile_window({}))
    aoi = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert int(aoi.array.min()) == 255
    assert int(aoi.array.max()) == 255
