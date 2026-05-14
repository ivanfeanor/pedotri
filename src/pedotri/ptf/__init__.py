"""Pedotransfer functions (PTFs) for soil hydraulic properties.

This subpackage estimates water-retention and hydraulic-conductivity
properties from texture (and optional organic-matter, bulk-density,
and gravel content) — the inputs that texture-classification users
typically have on hand.

Available PTFs:

- :func:`saxton_rawls` (2006) — water retention and saturated
  hydraulic conductivity from sand, clay, and organic matter.
- :func:`wosten` (1999) — Mualem-van Genuchten parameters for water
  retention and hydraulic conductivity from texture, bulk density,
  organic matter, and topsoil indicator (the European HYPRES PTF).

All public functions accept either scalar or array-like inputs and
return either a single result object or a list of them, matching the
ergonomics of :func:`pedotri.classify`.
"""

from __future__ import annotations

from pedotri.ptf.saxton_rawls import SaxtonRawlsResult, saxton_rawls
from pedotri.ptf.wosten import WostenResult, wosten

__all__ = [
    "SaxtonRawlsResult",
    "WostenResult",
    "saxton_rawls",
    "wosten",
]
