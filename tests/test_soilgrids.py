"""Tests for the SoilGrids 2.0 point fetcher (offline-only).

Every test stubs ``urllib.request.urlopen`` so the suite never touches
the live ISRIC service. The mock returns a canonical SoilGrids 2.0
response shape so we can verify both unit conversion and the on-disk
cache.
"""

from __future__ import annotations

import io
import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from pedotri.errors import InvalidInputError, PedotriError
from pedotri.sources import soilgrids

if TYPE_CHECKING:
    from pathlib import Path

SAND_CLAY_RESPONSE: dict[str, Any] = {
    "type": "Feature",
    "properties": {
        "layers": [
            {
                "name": "sand",
                "depths": [
                    {
                        "label": "0-5cm",
                        "values": {"mean": 350, "Q0.05": 280, "Q0.95": 420},
                    },
                    {
                        "label": "5-15cm",
                        "values": {"mean": 340, "Q0.05": 270, "Q0.95": 410},
                    },
                ],
            },
            {
                "name": "clay",
                "depths": [
                    {
                        "label": "0-5cm",
                        "values": {"mean": 270, "Q0.05": 220, "Q0.95": 320},
                    }
                ],
            },
        ]
    },
}


class _FakeResponse:
    """Context-manager mock for ``urllib.request.urlopen``."""

    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self._buf = io.BytesIO(json.dumps(body).encode("utf-8"))
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "soilgrids_cache"


@pytest.fixture
def fake_urlopen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``urlopen`` with a recorder + canned response."""
    calls: list[str] = []

    def _urlopen(req: Any, timeout: float = 30.0) -> _FakeResponse:
        calls.append(req.full_url if hasattr(req, "full_url") else str(req))
        return _FakeResponse(SAND_CLAY_RESPONSE)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return calls


def test_fetch_point_parses_and_converts_units(cache_dir: Path, fake_urlopen: list[str]) -> None:
    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    assert pt.lon == 2.5
    assert pt.lat == 47.0
    assert pt.cached is False
    # g/kg → percent for sand and clay.
    assert pt.values["sand"]["0-5cm"]["mean"] == pytest.approx(35.0)
    assert pt.values["sand"]["0-5cm"]["Q0.05"] == pytest.approx(28.0)
    assert pt.values["sand"]["0-5cm"]["Q0.95"] == pytest.approx(42.0)
    assert pt.values["clay"]["0-5cm"]["mean"] == pytest.approx(27.0)


def test_fetch_point_sand_clay_helper(cache_dir: Path, fake_urlopen: list[str]) -> None:
    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    sand_mean, _sand_q05, sand_q95, _clay_mean, clay_q05, _clay_q95 = pt.sand_clay()
    assert sand_mean == pytest.approx(35.0)
    assert sand_q95 == pytest.approx(42.0)
    assert clay_q05 == pytest.approx(22.0)


def test_fetch_point_caches_result(cache_dir: Path, fake_urlopen: list[str]) -> None:
    first = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    second = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    assert first.cached is False
    assert second.cached is True
    # Exactly one HTTP call across both fetches.
    assert len(fake_urlopen) == 1
    # Cache file lives under cache_dir.
    assert any(cache_dir.glob("*.json"))


def test_fetch_point_cache_key_separates_distinct_requests(
    cache_dir: Path, fake_urlopen: list[str]
) -> None:
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    soilgrids.fetch_point(2.6, 47.0, cache_dir=cache_dir)  # different lon → cache miss
    soilgrids.fetch_point(
        2.5,
        47.0,
        cache_dir=cache_dir,
        depths=("5-15cm",),  # different depth → cache miss
    )
    assert len(fake_urlopen) == 3


def test_fetch_point_ttl_triggers_refetch(cache_dir: Path, fake_urlopen: list[str]) -> None:
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    # Backdate the cache file beyond a 0.5-day TTL.
    for entry in cache_dir.glob("*.json"):
        old = time.time() - 86400.0
        import os

        os.utime(entry, (old, old))
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir, cache_ttl_days=0.5)
    assert len(fake_urlopen) == 2


def test_fetch_point_infinite_ttl_uses_cache(cache_dir: Path, fake_urlopen: list[str]) -> None:
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    for entry in cache_dir.glob("*.json"):
        import os

        very_old = time.time() - 10 * 365 * 86400.0
        os.utime(entry, (very_old, very_old))
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir, cache_ttl_days=float("inf"))
    assert len(fake_urlopen) == 1


def test_fetch_point_rejects_out_of_range_coords(cache_dir: Path, fake_urlopen: list[str]) -> None:
    with pytest.raises(InvalidInputError, match="lon"):
        soilgrids.fetch_point(200.0, 0.0, cache_dir=cache_dir)
    with pytest.raises(InvalidInputError, match="lat"):
        soilgrids.fetch_point(0.0, 95.0, cache_dir=cache_dir)
    assert len(fake_urlopen) == 0


def test_fetch_point_rejects_unknown_property(cache_dir: Path, fake_urlopen: list[str]) -> None:
    with pytest.raises(InvalidInputError, match="property"):
        soilgrids.fetch_point(2.5, 47.0, properties=("does_not_exist",), cache_dir=cache_dir)
    assert len(fake_urlopen) == 0


def test_fetch_point_rejects_unknown_depth(cache_dir: Path, fake_urlopen: list[str]) -> None:
    with pytest.raises(InvalidInputError, match="depth"):
        soilgrids.fetch_point(2.5, 47.0, depths=("0-1cm",), cache_dir=cache_dir)


def test_fetch_point_raises_on_http_failure(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _failing(*args: Any, **kwargs: Any) -> Any:
        import urllib.error

        raise urllib.error.URLError("nope")

    monkeypatch.setattr("urllib.request.urlopen", _failing)
    with pytest.raises(PedotriError, match="SoilGrids request failed"):
        soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)


def test_fetch_point_raises_on_malformed_response(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _bad(req: Any, timeout: float = 30.0) -> _FakeResponse:
        return _FakeResponse({"oops": True})

    monkeypatch.setattr("urllib.request.urlopen", _bad)
    with pytest.raises(PedotriError, match="missing properties.layers"):
        soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)


def test_clear_cache_removes_entries(cache_dir: Path, fake_urlopen: list[str]) -> None:
    soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    soilgrids.fetch_point(2.6, 47.0, cache_dir=cache_dir)
    assert sum(1 for _ in cache_dir.glob("*.json")) == 2
    removed = soilgrids.clear_cache(cache_dir=cache_dir)
    assert removed == 2
    assert not any(cache_dir.glob("*.json"))


def test_clear_cache_on_missing_dir() -> None:
    assert soilgrids.clear_cache(cache_dir="/tmp/pedotri_does_not_exist_xyz") == 0


def test_typed_accessors(cache_dir: Path, fake_urlopen: list[str]) -> None:
    # Request both depths so the parser keeps them both.
    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir, depths=("0-5cm", "5-15cm"))

    assert pt.properties() == ["clay", "sand"]
    assert pt.depths_for("sand") == ["0-5cm", "5-15cm"]
    # Clay only has 0-5cm in the canned response.
    assert pt.depths_for("clay") == ["0-5cm"]
    assert pt.value("sand", "0-5cm", "mean") == pytest.approx(35.0)
    # value() defaults make the common case terse.
    assert pt.value("sand") == pytest.approx(35.0)


def test_value_helper_pinpoints_missing_level(cache_dir: Path, fake_urlopen: list[str]) -> None:
    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    with pytest.raises(PedotriError, match="property 'soc'"):
        pt.value("soc")
    with pytest.raises(PedotriError, match="depth"):
        pt.value("sand", depth="bogus")
    with pytest.raises(PedotriError, match="value"):
        pt.value("sand", depth="0-5cm", value="median")


def test_as_quantiles_returns_pedotri_quantiles(cache_dir: Path, fake_urlopen: list[str]) -> None:
    from pedotri.uncertainty import Quantiles

    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=cache_dir)
    q = pt.as_quantiles("sand")
    assert isinstance(q, Quantiles)
    assert q.q05 == pytest.approx(28.0)
    assert q.q95 == pytest.approx(42.0)


def test_sand_clay_helper_complains_when_data_missing(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _only_sand(req: Any, timeout: float = 30.0) -> _FakeResponse:
        body = {
            "properties": {
                "layers": [
                    {
                        "name": "sand",
                        "depths": [
                            {
                                "label": "0-5cm",
                                "values": {"mean": 350, "Q0.05": 280, "Q0.95": 420},
                            }
                        ],
                    }
                ]
            }
        }
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", _only_sand)
    pt = soilgrids.fetch_point(2.5, 47.0, properties=("sand",), cache_dir=cache_dir)
    with pytest.raises(PedotriError, match="missing"):
        pt.sand_clay()
