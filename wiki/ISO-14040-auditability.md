# ISO 14040 / 14044 auditability

[ISO 14040:2006](https://www.iso.org/standard/37456.html) (*LCA —
Principles and framework*) and [ISO 14044:2006](https://www.iso.org/standard/38498.html)
(*LCA — Requirements and guidelines*) impose specific traceability
requirements on every input to a Life Cycle Assessment, including the
soil-related Life Cycle Inventory (LCI) data many pedotri consumers
produce. Pedotri 0.4 makes those requirements operational with a
small audit-trail machinery that lives in `pedotri.audit`.

This page maps the ISO requirements to the pedotri features that
satisfy them, and walks through an end-to-end example.

## What ISO 14040 / 14044 actually require

The relevant clauses (paraphrased — read the standards for the
authoritative wording):

| Clause | Requirement | Pedotri's answer |
|---|---|---|
| 14040 §4.5 / 14044 §4.5.3 | LCI data must include the source, vintage / version, geographic and technological coverage, and a quality assessment. | Every result dataclass carries a `Provenance` with `sources` (a list of `DataSource` records: name, version, URL, access timestamp, optional content hash). |
| 14044 §4.4.5 | Calculations must be **reproducible** — a third-party reviewer must be able to re-run them and obtain the same numbers. | All MC paths take an explicit `seed=`. `Provenance.seed` records it. Re-execution with the same seed produces byte-identical sample arrays. |
| 14044 §4.5.3 | Data quality requirements include uncertainty information. | `pedotri.Quantiles` + `AggregateDistribution.samples` carry the full posterior; downstream `combine()` propagates it. |
| 14044 §4.4.4 | Allocation procedures must be documented. | Pedotri itself does no allocation. The audit trail's `parameters` dict records every numeric choice the caller made (depth, area, weights, ...). |
| 14044 §6.4 | Critical review of the LCA needs a documentation trail covering goal & scope, LCI, and LCIA stages. | `AuditTrail.to_json(path)` exports a single JSON document with every step's provenance — the soil-data slice of the documentation. |

## The two data classes

```python
from pedotri.audit import Provenance, DataSource, AuditTrail, make_source
```

### `DataSource`

One external dataset that fed a computation. Fields:

- `name` — stable identifier (e.g. `"ISRIC SoilGrids 2.0"`, `"ESA WorldCover"`).
- `version` — version / revision tag (e.g. `"v2.0"`, `"v200"` for WorldCover 2021).
- `url` — *optional* URL the data came from. Audit reviewers love
  this; some agencies require it.
- `accessed_utc` — *optional* ISO 8601 UTC timestamp of the fetch.
- `content_hash` — *optional* hash (typically `"sha256:..."`) of the
  raw bytes. Lets a reviewer detect silent drift if they re-fetch.
- `details` — free-form dict for anything else: spatial bounding box,
  depth, cache-hit flag.

### `Provenance`

One computational step. Fields:

- `operation` — fully-qualified name of the operation
  (`"pedotri.sources.soilgrids.fetch_point"`,
  `"pedotri.zonal.zonal_aggregate"`, …).
- `parameters` — the numerical / categorical knobs the caller turned.
- `sources` — list of `DataSource` records consumed.
- `upstream` — list of `Provenance` records whose outputs fed this
  operation. Forms the LCI graph back to the original fetches.
- `seed` — when randomness is involved, the RNG seed. `None` for
  deterministic operations.
- `timestamp_utc` — ISO 8601 UTC timestamp.
- `software` — runtime metadata: pedotri version, numpy version,
  Python version, platform.
- `notes` — free-form annotation slot, populated by the caller for
  things the structure doesn't already cover (justification of an
  unusual parameter, link to a study protocol, ...).

## End-to-end example

```python
import pedotri
from pedotri.audit import AuditTrail
from pedotri.uncertainty import Quantiles
from pedotri.zonal import zonal_aggregate

# Suppose we ran the synthetic SOC-stock workflow:
agg = zonal_aggregate(
    region=region_mask,
    properties={
        "soc": {"mean": soc_mean, "uncertainty": Quantiles(soc_q05, soc_q95)},
        "bd":  {"mean": bd_mean,  "uncertainty": Quantiles(bd_q05,  bd_q95)},
    },
    correlation_range=2000.0,
    correlation_model="exponential",
    n_samples=1000,
    seed=2024,
)
stock = agg.combine(
    lambda soc, bd, depth=0.30, area=AREA_M2:
        soc * 1e-3 * bd * depth * area,
    name="soc_stock_kg",
)

# Inspect provenance on any result:
print(agg.provenance.operation)           # 'pedotri.zonal.zonal_aggregate'
print(agg.provenance.seed)                # 2024
print(agg.provenance.parameters["correlation_range"])  # 2000.0

print(stock.provenance.operation)         # '...combine'
print(stock.provenance.upstream[0].operation)  # back-link to the aggregation

# Export the full audit log:
trail = AuditTrail(metadata={"lca_study": "wheat-rotation-2026"})
trail.append(agg.provenance)
trail.append(stock.provenance)
trail.to_json("audit.json")
```

The exported JSON is a self-contained document — no other file is
required to reconstruct what pedotri did, modulo the original data
files (which the `DataSource.url` and `content_hash` fields document).

## Replay protocol for critical review

A reviewer with the JSON file and access to pedotri at the recorded
version can replay the computation:

```python
import json
from pedotri.audit import AuditTrail

trail = AuditTrail.from_json("audit.json")
for record in trail.records:
    print(f"{record.operation} → seed={record.seed} → "
          f"sources={[s.name for s in record.sources]}")
# Pick the seed off the relevant step:
seed = trail.records[0].seed
# Re-run the same pedotri call with the same seed → byte-identical samples.
```

`tests/test_iso14040_auditability.py` codifies this contract:

- `test_zonal_aggregate_is_reproducible_with_seed` — same inputs +
  same seed across two independent calls produce identical sample
  arrays.
- `test_correlated_zonal_aggregate_is_reproducible_with_seed` —
  spatially-correlated mode is also deterministic.
- `test_provenance_json_round_trip` — `Provenance.to_dict()` →
  `json.loads(json.dumps(...))` → `Provenance.from_dict(...)`
  preserves every field.
- `test_audit_trail_json_round_trip` — same for `AuditTrail`.
- `test_full_workflow_audit_trail_replay_identity` — full chain:
  produce → serialise → reload → re-run with the recorded seed →
  byte-identical samples.

## What pedotri does *not* claim

- The JSON export is **not** an ISO 14040 LCI report. It's the
  soil-data step's traceability record, suitable for inclusion in a
  larger LCA documentation package. The goal-and-scope section,
  allocation procedures, system boundaries, and impact-assessment
  methodology are out of scope for a soil-texture library.
- Pedotri does not attest to the quality of the underlying SoilGrids
  / WorldCover data. It documents what was used; the reviewer
  decides whether that's adequate for their study.
- Reproducibility is across the same pedotri version + same numpy /
  scipy versions + same platform. Cross-version reproducibility is a
  documented goal but not a guarantee; the `software` dict on every
  Provenance record makes the version drift visible.

## See also

- [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)
  — the workflow that most ISO 14040 / 14044 consumers run.
- [Spatial correlation](Spatial-correlation) — the math behind
  `correlation_range=`, which materially changes regional uncertainty
  estimates and so should be a documented parameter choice.
- ISO 14040:2006, *Environmental management — Life cycle assessment —
  Principles and framework.*
- ISO 14044:2006, *Environmental management — Life cycle assessment —
  Requirements and guidelines.*
