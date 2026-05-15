"""Provenance + audit trail support for ISO 14067 product carbon footprints.

ISO 14067:2018 (*Carbon footprint of products*) inherits its life cycle
inventory machinery from ISO 14040:2006 (§4.5 — life cycle inventory
analysis) and ISO 14044:2006 (§4.2 / §4.3 — data quality, and §4.5.3 —
inventory data requirements). All three demand a documented audit
trail: every value used in the assessment must be traceable to its
source, its version, the parameters that produced it, and — where
randomness is involved — to a seed that lets a critical reviewer
reproduce the result byte-for-byte. Pedotri's soil-derived outputs
typically feed the agricultural / land-use portion of a product CF
(14067 §6.4.5 — soil carbon stock change, §6.4.9 — land use change),
so the audit machinery here is the contract that makes those numbers
defensible to a 14067 reviewer.

This module gives pedotri a small machinery for that:

- :class:`Provenance` — a frozen record describing one computational
  step. Carries the operation name, its input parameters, the data
  sources it consumed, the RNG seed (when any randomness is involved),
  a UTC timestamp, the software version, and the chain of upstream
  :class:`Provenance` records that fed it. Round-trips through JSON.
- :class:`AuditTrail` — a list-like collector that an LCA practitioner
  can pass around to accumulate every record produced during a run.
  ``trail.to_json(path)`` exports a self-contained audit log that
  serves as the documentation the reviewer needs.
- The result dataclasses of ``pedotri.sources`` / ``pedotri.zonal`` /
  ``pedotri.raster`` all gain an optional ``provenance: Provenance |
  None`` field. ``None`` (default) keeps the 0.3 behaviour untouched.
  When populated, the chain of ``.provenance.upstream`` references
  encodes the full LCI graph back to the original data fetches.

What pedotri *does not* claim: that a JSON export of an audit trail
is itself an ISO 14067 product CF report. That report needs a
goal-and-scope section, a functional unit and reference flow, GWP
characterization, allocation procedures, and time-bounded stock-change
accounting (e.g. ΔSOC over an inventory period). Those live in the
downstream LCA / carbon-accounting tool that consumes pedotri's
outputs; the ``Provenance.upstream`` chain is the handoff socket the
downstream tool extends. What the trail does is satisfy the data
traceability and reproducibility requirements (14044 §4.5.3 / §4.4.5,
inherited by 14067) within the soil-data step of a larger product-CF
pipeline.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _utc_iso_now() -> str:
    """ISO 8601 UTC timestamp with second precision — audit-log friendly."""
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def _software_info() -> dict[str, str]:
    """Pedotri + numpy + python runtime versions for the audit trail."""
    import platform
    import sys

    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version

        try:
            pedotri_version = _pkg_version("pedotri")
        except PackageNotFoundError:
            pedotri_version = "unknown"
    except ImportError:  # pragma: no cover
        pedotri_version = "unknown"
    try:
        import numpy as _np

        numpy_version = _np.__version__
    except ImportError:  # pragma: no cover
        numpy_version = "unknown"
    return {
        "package": "pedotri",
        "version": pedotri_version,
        "numpy": numpy_version,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({sys.platform})",
    }


@dataclass(frozen=True, slots=True)
class DataSource:
    """One external dataset that fed a computational step.

    Mirrors the LCI-data fields ISO 14044 §4.5.3 asks for: a stable
    identifier (``name``), a version / revision tag (``version``), and
    optionally where the bytes came from (``url``) and a content hash
    so a reviewer can verify the cached copy hasn't drifted from the
    original.
    """

    name: str
    version: str
    url: str | None = None
    accessed_utc: str | None = None
    content_hash: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.url is not None:
            out["url"] = self.url
        if self.accessed_utc is not None:
            out["accessed_utc"] = self.accessed_utc
        if self.content_hash is not None:
            out["content_hash"] = self.content_hash
        if self.details:
            out["details"] = dict(self.details)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataSource:
        return cls(
            name=payload["name"],
            version=payload["version"],
            url=payload.get("url"),
            accessed_utc=payload.get("accessed_utc"),
            content_hash=payload.get("content_hash"),
            details=dict(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """ISO 14067-style record of one computational step.

    Pedotri populates one of these per result dataclass. The chain
    formed by walking ``.upstream`` recursively is the LCI audit
    trail for that result: every value the operation produced is
    explained by a parent record back to the original data fetches.
    """

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    sources: list[DataSource] = field(default_factory=list)
    upstream: list[Provenance] = field(default_factory=list)
    seed: int | None = None
    timestamp_utc: str = field(default_factory=_utc_iso_now)
    software: dict[str, str] = field(default_factory=_software_info)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "operation": self.operation,
            "timestamp_utc": self.timestamp_utc,
            "software": dict(self.software),
            "parameters": _coerce_jsonable(self.parameters),
        }
        if self.sources:
            out["sources"] = [s.to_dict() for s in self.sources]
        if self.upstream:
            out["upstream"] = [p.to_dict() for p in self.upstream]
        if self.seed is not None:
            out["seed"] = self.seed
        if self.notes is not None:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Provenance:
        return cls(
            operation=payload["operation"],
            parameters=dict(payload.get("parameters", {})),
            sources=[DataSource.from_dict(s) for s in payload.get("sources", [])],
            upstream=[cls.from_dict(p) for p in payload.get("upstream", [])],
            seed=payload.get("seed"),
            timestamp_utc=payload.get("timestamp_utc", _utc_iso_now()),
            software=dict(payload.get("software", _software_info())),
            notes=payload.get("notes"),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(slots=True)
class AuditTrail:
    """Append-only collector for :class:`Provenance` records.

    Two intended usage patterns:

    - Pass the same trail through a chain of pedotri calls, each of
      which appends its own record. At the end, ``trail.to_json()``
      gives you a complete audit log keyed by operation.
    - Skip the trail entirely and reach into ``result.provenance`` —
      every result already carries its own Provenance with its
      upstream graph populated, so the trail is mostly useful for
      flat logs / summary reports.

    The collector is not thread-safe; create one per worker if you
    parallelise.
    """

    records: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, record: Provenance) -> Provenance:
        self.records.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "software": _software_info(),
            "generated_utc": _utc_iso_now(),
            "metadata": dict(self.metadata),
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, path: Path | str | None = None, *, indent: int | None = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload)
        return payload

    @classmethod
    def from_json(cls, path: Path | str) -> AuditTrail:
        raw = json.loads(Path(path).read_text())
        trail = cls(
            records=[Provenance.from_dict(r) for r in raw.get("records", [])],
            metadata=dict(raw.get("metadata", {})),
        )
        return trail


def make_source(
    name: str,
    *,
    version: str = "",
    url: str | None = None,
    accessed_utc: str | None = None,
    content_hash: str | None = None,
    **details: Any,
) -> DataSource:
    """Convenience constructor — flat kwargs land in ``DataSource.details``."""
    return DataSource(
        name=name,
        version=version,
        url=url,
        accessed_utc=accessed_utc,
        content_hash=content_hash,
        details=dict(details),
    )


# --- internals -----------------------------------------------------------


def _coerce_jsonable(obj: Any) -> Any:
    """Best-effort JSON coercion for parameter dicts.

    Audit logs end up as JSON; the parameter dict can contain numpy
    scalars, ``pathlib.Path``, etc. We coerce them to native Python
    types here so downstream ``json.dumps`` doesn't choke.
    """
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return {"_kind": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, (_np.generic,)):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


__all__ = [
    "AuditTrail",
    "DataSource",
    "Provenance",
    "make_source",
]
