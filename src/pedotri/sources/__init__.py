"""Data-source fetchers — keep one external dataset per submodule.

Sub-modules are deliberately optional and shallow: each one wraps a
single upstream API (SoilGrids, WorldCover, …) and exposes a small
function set so it can grow independently. Network code lives behind
on-disk caches keyed by the request parameters so an LLM-driven loop
that re-issues the same query hundreds of times doesn't hammer the
upstream service.
"""

from __future__ import annotations
