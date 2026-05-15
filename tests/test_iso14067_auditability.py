"""ISO 14067 auditability tests (inheriting 14040 / 14044 LCI machinery).

These checks codify what an external product-CF auditor would verify
before accepting a pedotri-derived value as an LCI input feeding a
14067 carbon footprint of products:

1. **Reproducibility** — same inputs + same seed produce byte-identical
   results across independent runs (14044 §4.4.5 / §4.5.3 quality
   requirement, inherited by 14067).
2. **Provenance completeness** — every result carries a Provenance
   record naming the operation, parameters, software version, and
   upstream data sources.
3. **JSON round-trip** — the audit trail can be exported and re-read
   without information loss.
4. **Sensitivity** — perturbing an input parameter and re-running
   shows up as a different audit record (no silent caching of stale
   numbers).

The full chain is exercised on synthetic inputs so the tests stay
offline and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pedotri
from pedotri.audit import AuditTrail, DataSource, Provenance, make_source
from pedotri.uncertainty import Quantiles
from pedotri.zonal import zonal_aggregate


def _uniform_property(shape: tuple[int, int], mean: float, halfwidth: float) -> dict[str, Any]:
    m = np.full(shape, mean, dtype=np.float64)
    return {"mean": m, "uncertainty": Quantiles(m - halfwidth, m + halfwidth)}


# --- 14044 §4.4.5 / §4.5.3 reproducibility (inherited by 14067) ----------


def test_zonal_aggregate_is_reproducible_with_seed() -> None:
    """Same inputs + same seed → byte-identical sample arrays."""
    shape = (20, 20)
    region = np.ones(shape, dtype=bool)
    spec = {"soc": _uniform_property(shape, 25.0, 4.0)}
    a = zonal_aggregate(region=region, properties=spec, n_samples=400, seed=2024)
    b = zonal_aggregate(region=region, properties=spec, n_samples=400, seed=2024)
    np.testing.assert_array_equal(a["soc"].samples, b["soc"].samples)


def test_correlated_zonal_aggregate_is_reproducible_with_seed() -> None:
    """Correlated mode is also deterministic given the seed."""
    shape = (30, 30)
    region = np.ones(shape, dtype=bool)
    spec = {"soc": _uniform_property(shape, 25.0, 6.0)}
    a = zonal_aggregate(
        region=region,
        properties=spec,
        correlation_range=5.0,
        n_samples=200,
        seed=2024,
    )
    b = zonal_aggregate(
        region=region,
        properties=spec,
        correlation_range=5.0,
        n_samples=200,
        seed=2024,
    )
    np.testing.assert_array_equal(a["soc"].samples, b["soc"].samples)


def test_classification_is_reproducible_with_seed() -> None:
    r1 = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        method="monte_carlo",
        n_samples=500,
        seed=314,
    )
    r2 = pedotri.classify(
        sand=27,
        clay=38,
        sand_uncertainty=Quantiles(22, 32),
        clay_uncertainty=Quantiles(33, 43),
        classification="USDA",
        method="monte_carlo",
        n_samples=500,
        seed=314,
    )
    assert r1.probabilities == r2.probabilities
    assert r1.confidence == r2.confidence


# --- Provenance completeness --------------------------------------------


def test_zonal_aggregate_records_provenance() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={"soc": _uniform_property(shape, 25.0, 4.0)},
        n_samples=200,
        seed=2024,
    )
    prov = agg.provenance
    assert prov is not None
    assert prov.operation == "pedotri.zonal.zonal_aggregate"
    assert prov.parameters["n_samples"] == 200
    assert prov.parameters["properties"] == ["soc"]
    assert prov.seed == 2024
    # Software / timestamp fields are populated.
    assert prov.software["package"] == "pedotri"
    assert prov.software["version"]  # non-empty
    assert prov.timestamp_utc.endswith("+00:00")
    # AggregateDistribution inherits the same provenance.
    assert agg["soc"].provenance is prov


def test_combine_records_upstream_chain() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": _uniform_property(shape, 25.0, 4.0),
            "bd": _uniform_property(shape, 1.3, 0.1),
        },
        n_samples=200,
        seed=2024,
    )
    stock = agg.combine(
        lambda soc, bd, depth=0.30: soc * 1e-3 * bd * 1e3 * depth,
        name="soc_stock_kg_per_m2",
    )
    assert stock.provenance is not None
    assert stock.provenance.operation.endswith("combine")
    # Upstream chain points back at the aggregation step.
    assert stock.provenance.upstream
    assert stock.provenance.upstream[0].operation == "pedotri.zonal.zonal_aggregate"


def test_soilgrids_point_records_data_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the SoilGrids HTTP endpoint and verify the Provenance shape."""

    from pedotri.sources import soilgrids

    canned = {
        "properties": {
            "layers": [
                {
                    "name": "sand",
                    "depths": [
                        {"label": "0-5cm", "values": {"mean": 350, "Q0.05": 280, "Q0.95": 420}}
                    ],
                },
                {
                    "name": "clay",
                    "depths": [
                        {"label": "0-5cm", "values": {"mean": 270, "Q0.05": 220, "Q0.95": 320}}
                    ],
                },
            ]
        }
    }

    class _Resp:
        status = 200

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps(canned).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: _Resp())
    pt = soilgrids.fetch_point(2.5, 47.0, cache_dir=tmp_path)
    assert pt.provenance is not None
    assert pt.provenance.operation == "pedotri.sources.soilgrids.fetch_point"
    assert pt.provenance.parameters["lon"] == 2.5
    assert pt.provenance.parameters["lat"] == 47.0
    assert pt.provenance.sources
    src = pt.provenance.sources[0]
    assert isinstance(src, DataSource)
    assert "SoilGrids" in src.name
    assert src.version == "v2.0"
    assert src.url
    assert src.url.startswith("https://rest.isric.org")


def test_worldcover_aoi_records_data_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the WorldCover tile reader and verify the Provenance shape."""
    pytest.importorskip("rasterio")
    from pedotri.sources import worldcover

    def _fake_reader(url: str, *, bbox, rasterio, window_from_bounds):  # type: ignore[no-untyped-def]
        w, s, e, n = bbox
        return np.full((20, 20), worldcover.CROPLAND, dtype=np.uint8), (w, s, e, n)

    monkeypatch.setattr(worldcover, "_read_tile_window", _fake_reader)
    aoi = worldcover.fetch_aoi((3.5, 47.0, 3.501, 47.001), year=2021, cache_dir=tmp_path)
    assert aoi.provenance is not None
    assert aoi.provenance.operation == "pedotri.sources.worldcover.fetch_aoi"
    src = aoi.provenance.sources[0]
    assert src.name == "ESA WorldCover"
    assert src.version == "v200"


# --- JSON round-trip -----------------------------------------------------


def test_provenance_json_round_trip() -> None:
    prov = Provenance(
        operation="pedotri.test.sample",
        parameters={"n": 10, "seed": 42, "path": Path("/tmp/x.tif")},
        sources=[
            make_source(
                "SoilGrids 2.0",
                version="v2.0",
                url="https://rest.isric.org/soilgrids/v2.0/properties/query",
                accessed_utc="2026-05-15T12:00:00+00:00",
                content_hash="sha256:deadbeef",
            )
        ],
        seed=42,
        notes="reproducibility check",
    )
    encoded = prov.to_json()
    decoded = Provenance.from_dict(json.loads(encoded))
    assert decoded.operation == prov.operation
    assert decoded.parameters["n"] == 10
    assert decoded.sources[0].url == prov.sources[0].url
    assert decoded.seed == 42
    assert decoded.notes == "reproducibility check"


def test_audit_trail_json_round_trip(tmp_path: Path) -> None:
    trail = AuditTrail()
    trail.metadata = {
        "lca_study": "wheat-rotation-2026",
        "auditor": "ACME Verification GmbH",
    }
    trail.append(
        Provenance(
            operation="pedotri.sources.soilgrids.fetch_point",
            parameters={"lon": 2.5, "lat": 47.0},
            sources=[make_source("SoilGrids 2.0", version="v2.0")],
        )
    )
    trail.append(
        Provenance(
            operation="pedotri.zonal.zonal_aggregate",
            parameters={"n_samples": 1000},
            seed=42,
        )
    )
    out_path = tmp_path / "audit.json"
    trail.to_json(out_path)
    rehydrated = AuditTrail.from_json(out_path)
    assert rehydrated.metadata["lca_study"] == "wheat-rotation-2026"
    assert len(rehydrated.records) == 2
    assert rehydrated.records[0].operation.endswith("fetch_point")
    assert rehydrated.records[1].seed == 42


def test_full_workflow_audit_trail_replay_identity() -> None:
    """End-to-end: build a chain, serialise, deserialise, re-execute.

    The seeded result must match the audit record's claimed parameters
    *and* re-running with the same seed produces identical samples —
    that's the practical definition of "reviewer-reproducible" for an
    ISO 14067 LCI step (anchored on 14044 §4.4.5).
    """
    shape = (12, 12)
    region = np.ones(shape, dtype=bool)
    spec = {"soc": _uniform_property(shape, 25.0, 4.0)}

    first = zonal_aggregate(
        region=region,
        properties=spec,
        n_samples=200,
        seed=12345,
    )
    assert first.provenance is not None
    trail = AuditTrail()
    trail.append(first.provenance)
    encoded = trail.to_json()

    reloaded = (
        AuditTrail.from_json_string(encoded)
        if hasattr(AuditTrail, "from_json_string")
        else AuditTrail(records=[Provenance.from_dict(r) for r in json.loads(encoded)["records"]])
    )
    declared_seed = reloaded.records[0].seed
    assert declared_seed == 12345

    replay = zonal_aggregate(
        region=region,
        properties=spec,
        n_samples=200,
        seed=declared_seed,
    )
    np.testing.assert_array_equal(first["soc"].samples, replay["soc"].samples)


# --- Sensitivity / divergence guard --------------------------------------


def test_different_seed_yields_different_audit_record() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    spec = {"soc": _uniform_property(shape, 25.0, 4.0)}
    a = zonal_aggregate(region=region, properties=spec, n_samples=100, seed=1)
    b = zonal_aggregate(region=region, properties=spec, n_samples=100, seed=2)
    assert a.provenance is not None
    assert b.provenance is not None
    assert a.provenance.seed != b.provenance.seed
    # Samples are also different.
    assert not np.array_equal(a["soc"].samples, b["soc"].samples)


def test_changing_n_samples_visible_in_provenance() -> None:
    shape = (10, 10)
    region = np.ones(shape, dtype=bool)
    spec = {"soc": _uniform_property(shape, 25.0, 4.0)}
    a = zonal_aggregate(region=region, properties=spec, n_samples=100, seed=1)
    b = zonal_aggregate(region=region, properties=spec, n_samples=200, seed=1)
    assert a.provenance is not None
    assert b.provenance is not None
    assert a.provenance.parameters["n_samples"] == 100
    assert b.provenance.parameters["n_samples"] == 200


# --- DataSource hash field is preserved ----------------------------------


def test_data_source_round_trips_with_hash() -> None:
    src = make_source(
        "SoilGrids 2.0",
        version="v2.0",
        url="https://example/soc.tif",
        accessed_utc="2026-05-15T12:00:00+00:00",
        content_hash="sha256:abcd1234",
    )
    payload = src.to_dict()
    decoded = DataSource.from_dict(payload)
    assert decoded.content_hash == "sha256:abcd1234"
    assert decoded.url == src.url
