# ISO 14040 auditability

If your pedotri output is going to feed a Life Cycle Assessment, an
external auditor will eventually ask three questions about it: where
did the data come from, what parameters did you use, and can you
re-run the calculation and get the same numbers? `pedotri.audit`
exists to answer all three.

It does **not** try to be a full LCA framework — pedotri is a soil
toolkit, not openLCA. What it does is make the soil-data step of an
LCA *defensible*: every result carries a complete provenance record,
and the audit trail you export is the documentation a reviewer can
hand back to you saying "yes, this is reproducible."

## When you need this

- You're producing data that goes into an ISO 14040 / 14044 LCA
  (carbon accounting, environmental product declarations, regulated
  sustainability reporting).
- You need to prove to a reviewer that the same inputs always give
  the same outputs — *not* approximately, exactly.
- You want to document a sensitivity analysis where the same workflow
  was run with different parameters, and have those runs distinguishable
  by something more authoritative than a filename.

If none of those apply, you can ignore `pedotri.audit` entirely. The
`.provenance` field on every result defaults to populated-but-unused;
existing code paths don't change.

## Quick example

Run a normal pedotri workflow, then look at what came back with it:

```python
import pedotri
from pedotri.audit import AuditTrail
from pedotri.zonal import zonal_aggregate

agg = zonal_aggregate(
    region=village_polygon,
    properties={"soc": "soc_mean.tif", "bd": "bd_mean.tif"},
    correlation_range=2000.0,
    n_samples=1000,
    seed=2024,
)

# Every result already carries its provenance:
print(agg.provenance.operation)                 # 'pedotri.zonal.zonal_aggregate'
print(agg.provenance.seed)                      # 2024
print(agg.provenance.parameters["n_samples"])   # 1000
print(agg.provenance.software["version"])       # '0.4.0' (whatever you've installed)

# combine() inherits an upstream chain:
stock = agg.combine(lambda soc, bd: soc * 1e-3 * bd * 0.3, name="soc_stock_kg")
print(stock.provenance.upstream[0].operation)   # back-link to the aggregation

# Export the chain as a single audit document:
trail = AuditTrail(metadata={"lca_study": "wheat-rotation-2026"})
trail.append(agg.provenance)
trail.append(stock.provenance)
trail.to_json("audit.json")
```

The JSON file is everything an auditor needs to re-run your numbers,
modulo the original data files (which the trail records by name +
version + URL + optional content hash).

## What the standards actually require

The relevant clauses from ISO 14040:2006 and ISO 14044:2006 — and
which pedotri primitive answers each — are:

| Standard clause | Requirement | Pedotri answer |
|---|---|---|
| 14040 §4.5 / 14044 §4.5.3 | LCI data must include source, vintage / version, geographic and technological coverage, and a quality assessment. | `Provenance.sources` is a list of `DataSource` records: name, version, URL, access timestamp, optional content hash. |
| 14044 §4.4.5 | Calculations must be reproducible — a third-party reviewer must be able to re-run them and obtain the same numbers. | All MC paths accept an explicit `seed=`. `Provenance.seed` records it. Re-execution with the same seed produces byte-identical sample arrays — regression-tested in `tests/test_iso14040_auditability.py`. |
| 14044 §4.5.3 | Data quality requirements include uncertainty information. | `pedotri.Quantiles` carries the published 90 % CI; `AggregateDistribution.samples` carries the full posterior. |
| 14044 §4.4.4 | Allocation procedures must be documented. | Pedotri does no allocation itself. The audit trail's `parameters` dict records every numeric choice the caller made (depth, area, weights, correlation range). |
| 14044 §6.4 | Critical review needs documentation spanning goal & scope, LCI, and LCIA. | `AuditTrail.to_json(path)` exports a single self-contained document — the soil-data slice of the documentation pack. |

## The two data classes

```python
from pedotri.audit import DataSource, Provenance, AuditTrail, make_source
```

### `DataSource` — one external dataset

Fields:

- `name` — stable identifier (e.g. `"ISRIC SoilGrids 2.0"`, `"ESA WorldCover"`).
- `version` — version / revision tag (e.g. `"v2.0"`, `"v200"`).
- `url` — *optional* URL the data came from. Audit reviewers
  appreciate this; some agencies require it.
- `accessed_utc` — *optional* ISO 8601 UTC timestamp of the fetch.
- `content_hash` — *optional* hash (typically `"sha256:..."`) of the
  raw bytes. Lets a reviewer detect silent drift if they re-fetch.
- `details` — free-form dict for anything else: bounding box, depth,
  cache-hit flag.

Use `make_source(...)` for a tidy constructor that puts extra kwargs
in `details`:

```python
src = make_source(
    "ISRIC SoilGrids 2.0",
    version="v2.0",
    url="https://rest.isric.org/soilgrids/v2.0/properties/query",
    accessed_utc="2026-05-15T12:00:00+00:00",
    bbox=[2.5, 47.0, 2.6, 47.1],
    cached=False,
)
```

### `Provenance` — one computational step

Fields:

- `operation` — fully-qualified pedotri operation name
  (`"pedotri.sources.soilgrids.fetch_point"`,
  `"pedotri.zonal.zonal_aggregate"`, …).
- `parameters` — the numerical / categorical knobs the caller turned.
- `sources` — list of `DataSource` records consumed by this step.
- `upstream` — list of `Provenance` records whose outputs fed this
  one. Forms the LCI graph back to the original data fetches.
- `seed` — when randomness is involved, the RNG seed.
- `timestamp_utc` — ISO 8601 UTC timestamp.
- `software` — runtime metadata: pedotri version, numpy version,
  Python version, platform.
- `notes` — free-form annotation, populated by the caller for things
  the structure doesn't already cover.

Each pedotri operation auto-populates this; you rarely construct one
by hand unless you're recording a step that lives outside pedotri (a
custom hydrology call, a downstream LCA calculation).

### `AuditTrail` — a flat log of `Provenance` records

Use when you want a single document per workflow run:

```python
trail = AuditTrail(metadata={"lca_study": "wheat-rotation-2026"})
trail.append(agg.provenance)
trail.append(stock.provenance)
trail.to_json("audit.json")          # write to disk
encoded = trail.to_json()             # or get the string
restored = AuditTrail.from_json("audit.json")
```

`AuditTrail.metadata` is for run-level annotations: study name, auditor
contact, scenario identifier, anything that doesn't belong on an
individual Provenance record.

## The replay contract

The reviewer's contract with pedotri is: given an exported audit
trail, the same pedotri version, and access to the recorded data
sources, they can re-run your computation and get byte-identical
numbers. This is regression-tested explicitly in
`tests/test_iso14040_auditability.py`:

```python
# Original run
first = zonal_aggregate(
    region=region, properties=spec, n_samples=200, seed=12345,
)
trail = AuditTrail()
trail.append(first.provenance)
trail.to_json("audit.json")

# Reviewer's machine, later:
reloaded = AuditTrail.from_json("audit.json")
seed = reloaded.records[0].seed   # 12345

replay = zonal_aggregate(
    region=region, properties=spec, n_samples=200, seed=seed,
)
np.testing.assert_array_equal(first["soc"].samples, replay["soc"].samples)
# ↑ passes.
```

Reproducibility holds across:

- **Independent-pixel MC** (`correlation_range=None`).
- **Spatially-correlated MC** (`correlation_range > 0`).
- **Per-point classification** (`classify(..., method="monte_carlo")`).
- **`combine()`** (deterministic given the upstream sample arrays).

It does **not** hold across:

- Different pedotri versions (we make this visible — `software.version`
  is on every record).
- Different numpy versions (BLAS noise; documented limitation).
- Different platforms (Linux vs macOS BLAS implementations differ at
  the bit level on some operations). Same-platform replay is the
  standard contract.

## What pedotri does not claim

- The JSON export is **not** an ISO 14040 LCI report. It's the
  soil-data step's traceability record, suitable for inclusion in a
  larger LCA documentation package. The goal-and-scope section,
  allocation procedures, system boundaries, and impact-assessment
  methodology are out of scope for a soil-texture library.
- Pedotri does not attest to the quality of the underlying SoilGrids
  / WorldCover data. The trail documents what was used; the reviewer
  decides whether that's adequate for the study.

## See also

- [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)
  — the workflow most ISO 14040 / 14044 consumers run.
- [Spatial correlation](Spatial-correlation) — `correlation_range` is a
  modelling choice that materially affects regional Q05/Q95; the audit
  trail records it.
- ISO 14040:2006, *Environmental management — Life cycle assessment —
  Principles and framework.*
- ISO 14044:2006, *Environmental management — Life cycle assessment —
  Requirements and guidelines.*
