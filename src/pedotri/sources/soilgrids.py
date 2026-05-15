"""SoilGrids 2.0 point fetcher with an on-disk cache.

The ISRIC SoilGrids 2.0 REST endpoint at
``https://rest.isric.org/soilgrids/v2.0/properties/query`` returns
per-coordinate predictions for soil properties (sand, clay, SOC, bulk
density, …) at six standard depths, with the mean prediction plus
five quantiles. Pedotri uses this to make ``pedotri.classify`` and
``pedotri.zonal.zonal_aggregate`` "just work" from a coordinate alone.

Two concerns are baked in from day one:

- **Caching.** SoilGrids 2.0 is a static dataset that's republished
  rarely. Hitting the same coordinate twice should be free — both for
  the user and the service. Every successful response is persisted
  under ``$XDG_CACHE_HOME/pedotri/soilgrids/`` (typically
  ``~/.cache/pedotri/soilgrids/`` on Linux / macOS), keyed by a sha-256
  of the canonical request payload. A configurable TTL guards against
  cached data ageing past a release.
- **Unit awareness.** SoilGrids ships sand / silt / clay / soc / bd
  in integer g/kg or g/cm³×10 to keep tile sizes small; pedotri's
  classifiers want fractions in percent and BD in g/cm³. The point
  helpers translate units explicitly so the values you get out are
  consumable by ``pedotri.classify`` without further fiddling.

This module deliberately exposes only the point endpoint. The bbox /
WCS path (for tile fetching at scale) is a larger surface and will
land in 0.4 alongside the reprojection + resolution work.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pedotri.errors import InvalidInputError, PedotriError

#: Base REST endpoint for the SoilGrids 2.0 point query.
SOILGRIDS_POINT_URL: str = "https://rest.isric.org/soilgrids/v2.0/properties/query"

#: Property keys SoilGrids 2.0 exposes via the point query.
SUPPORTED_PROPERTIES: tuple[str, ...] = (
    "bdod",
    "cec",
    "cfvo",
    "clay",
    "nitrogen",
    "ocd",
    "ocs",
    "phh2o",
    "sand",
    "silt",
    "soc",
    "wv0010",
    "wv0033",
    "wv1500",
)

#: Standard SoilGrids 2.0 depth intervals.
SUPPORTED_DEPTHS: tuple[str, ...] = (
    "0-5cm",
    "5-15cm",
    "15-30cm",
    "30-60cm",
    "60-100cm",
    "100-200cm",
)

#: Quantile / mean labels accepted by the SoilGrids query string.
SUPPORTED_VALUES: tuple[str, ...] = ("mean", "Q0.05", "Q0.5", "Q0.95", "uncertainty")

#: Native d-factors (per the SoilGrids documentation): integer storage
#: divided by this factor recovers the conventional unit.
_DIVISORS_TO_PERCENT: dict[str, float] = {
    "sand": 10.0,  # g/kg → %
    "silt": 10.0,
    "clay": 10.0,
    "cfvo": 10.0,  # cm³/dm³ → %
    "soc": 10.0,  # dg/kg → g/kg
}
#: Properties whose canonical unit is g/cm³ — SoilGrids stores them
#: as integers ×100.
_DIVISORS_TO_BDOD: dict[str, float] = {"bdod": 100.0}


@dataclass(frozen=True, slots=True)
class SoilGridsPoint:
    """Parsed SoilGrids response for a single (lon, lat) query.

    Properties are returned in their **converted** units:

    - sand / silt / clay / cfvo: percent of fine earth.
    - soc: g / kg.
    - bdod: g / cm³.
    - other properties: SoilGrids native units (see the upstream docs).
    """

    lon: float
    lat: float
    values: dict[str, dict[str, dict[str, float]]]
    raw: dict[str, Any]
    cached: bool

    def value(self, property: str, depth: str = "0-5cm", value: str = "mean") -> float:
        """Return a single value with a clear error if any leg is missing.

        Prefer this over the triple-nested :attr:`values` dict — it
        raises :class:`pedotri.errors.PedotriError` with a self-describing
        message that names the level (``property`` / ``depth`` / ``value``)
        that wasn't fetched, so MCP-driven tool loops can self-correct.
        """
        prop_layer = self.values.get(property)
        if prop_layer is None:
            raise PedotriError(
                f"SoilGrids point at ({self.lon}, {self.lat}) was not fetched "
                f"with property {property!r}. Available properties: "
                f"{sorted(self.values)!r}."
            )
        depth_record = prop_layer.get(depth)
        if depth_record is None:
            raise PedotriError(
                f"SoilGrids point at ({self.lon}, {self.lat}): property "
                f"{property!r} has no depth {depth!r}. Available depths: "
                f"{sorted(prop_layer)!r}."
            )
        if value not in depth_record:
            raise PedotriError(
                f"SoilGrids point at ({self.lon}, {self.lat}): property "
                f"{property!r} depth {depth!r} has no value {value!r}. "
                f"Available values: {sorted(depth_record)!r}."
            )
        return depth_record[value]

    def get(self, property: str, depth: str = "0-5cm", value: str = "mean") -> float:
        """Alias for :meth:`value` — keeps the original 0.3.x spelling working."""
        return self.value(property, depth, value)

    def properties(self) -> list[str]:
        """Sorted list of properties present in this response."""
        return sorted(self.values)

    def depths_for(self, property: str) -> list[str]:
        """Sorted list of depth labels available for ``property``."""
        layer = self.values.get(property)
        if layer is None:
            raise PedotriError(
                f"SoilGrids point at ({self.lon}, {self.lat}) was not fetched "
                f"with property {property!r}. Available: {self.properties()!r}."
            )
        return sorted(layer)

    def as_quantiles(self, property: str, depth: str = "0-5cm") -> Any:
        """Wrap (Q0.05, Q0.95) for ``property`` at ``depth`` as a :class:`Quantiles`.

        The result drops directly into :func:`pedotri.classify`'s
        ``*_uncertainty`` kwargs::

            pt = soilgrids.fetch_point(lon, lat)
            pedotri.classify(
                sand=pt.value("sand"),
                clay=pt.value("clay"),
                sand_uncertainty=pt.as_quantiles("sand"),
                clay_uncertainty=pt.as_quantiles("clay"),
                classification="USDA",
            )
        """
        from pedotri.uncertainty import Quantiles

        return Quantiles(
            self.value(property, depth, "Q0.05"),
            self.value(property, depth, "Q0.95"),
        )

    def sand_clay(self, depth: str = "0-5cm") -> tuple[float, float, float, float, float, float]:
        """Return ``(sand_mean, sand_q05, sand_q95, clay_mean, clay_q05, clay_q95)``.

        Convenience for the most common pedotri use case — feeding
        :func:`pedotri.classify` from a coordinate without manual JSON
        navigation. Values are in percent.
        """
        try:
            return (
                self.value("sand", depth, "mean"),
                self.value("sand", depth, "Q0.05"),
                self.value("sand", depth, "Q0.95"),
                self.value("clay", depth, "mean"),
                self.value("clay", depth, "Q0.05"),
                self.value("clay", depth, "Q0.95"),
            )
        except PedotriError as exc:
            raise PedotriError(
                f"SoilGrids response at ({self.lon}, {self.lat}) is missing "
                f"the required sand/clay mean+Q05+Q95 at depth {depth!r}. "
                "Was the fetch issued with properties=('sand','clay') "
                "and values=('mean','Q0.05','Q0.95')?"
            ) from exc


def fetch_point(
    lon: float,
    lat: float,
    *,
    properties: tuple[str, ...] | list[str] = ("sand", "clay"),
    depths: tuple[str, ...] | list[str] = ("0-5cm",),
    values: tuple[str, ...] | list[str] = ("mean", "Q0.05", "Q0.95"),
    cache_dir: Path | str | None = None,
    cache_ttl_days: float = 30.0,
    timeout: float = 30.0,
) -> SoilGridsPoint:
    """Query SoilGrids 2.0 for one coordinate, with on-disk caching.

    Args:
        lon: Longitude in EPSG:4326 (degrees east, ``-180..180``).
        lat: Latitude in EPSG:4326 (degrees north, ``-90..90``).
        properties: SoilGrids property keys to fetch. Default is
            ``("sand", "clay")`` — enough to feed
            :func:`pedotri.classify`. See :data:`SUPPORTED_PROPERTIES`.
        depths: Standard SoilGrids depth intervals. Default
            ``("0-5cm",)``. See :data:`SUPPORTED_DEPTHS`.
        values: Mean / quantile labels. Default
            ``("mean", "Q0.05", "Q0.95")``. See :data:`SUPPORTED_VALUES`.
        cache_dir: Override the default cache directory
            (``$XDG_CACHE_HOME/pedotri/soilgrids/`` or
            ``~/.cache/pedotri/soilgrids/``). Pass ``Path``/str.
        cache_ttl_days: Entries older than this many days are refetched.
            SoilGrids 2.0 ships infrequently; the 30-day default is
            comfortable. Pass ``float("inf")`` to disable expiry.
        timeout: HTTP timeout in seconds.

    Returns:
        :class:`SoilGridsPoint` with the parsed values translated into
        pedotri-native units (percent for fractions, g/cm³ for BD).
        ``.cached == True`` when the result came from disk.

    Raises:
        InvalidInputError: For coordinates outside the WGS84 extent or
            for unsupported property / depth / value names.
        PedotriError: If the upstream service returns a malformed
            response or a non-200 status.
    """
    if not (-180.0 <= lon <= 180.0):
        raise InvalidInputError(f"lon must be in [-180, 180]; got {lon}.")
    if not (-90.0 <= lat <= 90.0):
        raise InvalidInputError(f"lat must be in [-90, 90]; got {lat}.")

    props = tuple(sorted(set(properties)))
    depths_t = tuple(depths)
    values_t = tuple(values)
    _validate_labels("property", props, SUPPORTED_PROPERTIES)
    _validate_labels("depth", depths_t, SUPPORTED_DEPTHS)
    _validate_labels("value", values_t, SUPPORTED_VALUES)

    payload = _canonical_payload(lon, lat, props, depths_t, values_t)
    cache_path = _cache_path(payload, cache_dir)
    raw = _load_cached(cache_path, cache_ttl_days)
    cached = raw is not None
    if raw is None:
        raw = _http_get(payload, timeout=timeout)
        _save_cached(cache_path, raw)

    converted = _parse_response(raw, props, depths_t, values_t)
    return SoilGridsPoint(lon=lon, lat=lat, values=converted, raw=raw, cached=cached)


def clear_cache(*, cache_dir: Path | str | None = None) -> int:
    """Delete every cached SoilGrids entry in ``cache_dir`` (or the default).

    Returns the number of files removed. Useful when SoilGrids
    publishes a new revision and you want to invalidate the local
    snapshot in one go.
    """
    cache_root = _cache_root(cache_dir)
    if not cache_root.exists():
        return 0
    count = 0
    for entry in cache_root.glob("*.json"):
        entry.unlink()
        count += 1
    return count


# --- internals ---------------------------------------------------------


def _validate_labels(kind: str, labels: tuple[str, ...], allowed: tuple[str, ...]) -> None:
    bad = [lab for lab in labels if lab not in allowed]
    if bad:
        raise InvalidInputError(
            f"Unsupported SoilGrids {kind}(s): {bad!r}. Supported {kind}s: {list(allowed)!r}."
        )


def _canonical_payload(
    lon: float,
    lat: float,
    properties: tuple[str, ...],
    depths: tuple[str, ...],
    values: tuple[str, ...],
) -> dict[str, Any]:
    """Build the canonical request payload used both for HTTP + caching."""
    return {
        "url": SOILGRIDS_POINT_URL,
        "lon": round(lon, 6),
        "lat": round(lat, 6),
        "property": list(properties),
        "depth": list(depths),
        "value": list(values),
    }


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pedotri" / "soilgrids"


def _cache_path(payload: dict[str, Any], cache_dir: Path | str | None) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return _cache_root(cache_dir) / f"{digest}.json"


def _load_cached(path: Path, ttl_days: float) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if ttl_days != float("inf"):
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > ttl_days * 86400.0:
            return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_cached(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(path)


def _http_get(payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    query: list[tuple[str, str]] = [
        ("lon", str(payload["lon"])),
        ("lat", str(payload["lat"])),
    ]
    query.extend(("property", p) for p in payload["property"])
    query.extend(("depth", d) for d in payload["depth"])
    query.extend(("value", v) for v in payload["value"])
    url = f"{payload['url']}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise PedotriError(f"SoilGrids returned HTTP {resp.status} for {url}.")
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise PedotriError(f"SoilGrids request failed: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PedotriError(f"SoilGrids returned non-JSON payload from {url}: {exc}") from exc


def _parse_response(
    raw: dict[str, Any],
    properties: tuple[str, ...],
    depths: tuple[str, ...],
    values: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """Walk the SoilGrids response and translate units."""
    layers = raw.get("properties", {}).get("layers") if isinstance(raw, dict) else None
    if not layers:
        raise PedotriError(
            "SoilGrids response is missing properties.layers; "
            "the upstream payload shape may have changed."
        )
    out: dict[str, dict[str, dict[str, float]]] = {}
    by_name = {layer.get("name"): layer for layer in layers}
    for prop in properties:
        layer = by_name.get(prop)
        if layer is None:
            continue
        divisor = _DIVISORS_TO_PERCENT.get(prop) or _DIVISORS_TO_BDOD.get(prop) or 1.0
        depth_records = layer.get("depths", []) or []
        depth_map: dict[str, dict[str, float]] = {}
        by_label = {d.get("label"): d for d in depth_records}
        for depth in depths:
            depth_entry = by_label.get(depth)
            if depth_entry is None:
                continue
            depth_values = depth_entry.get("values", {}) or {}
            converted: dict[str, float] = {}
            for value in values:
                raw_val = depth_values.get(value)
                if raw_val is None:
                    continue
                try:
                    converted[value] = float(raw_val) / divisor
                except (TypeError, ValueError):
                    continue
            if converted:
                depth_map[depth] = converted
        if depth_map:
            out[prop] = depth_map
    return out


__all__ = [
    "SOILGRIDS_POINT_URL",
    "SUPPORTED_DEPTHS",
    "SUPPORTED_PROPERTIES",
    "SUPPORTED_VALUES",
    "SoilGridsPoint",
    "clear_cache",
    "fetch_point",
]
