"""Load classification definitions from TOML.

A classification TOML file looks like::

    # usda.toml
    [meta]
    key = "USDA"
    axes = ["sand", "clay"]
    default_locale = "en"
    reference = "Soil Survey Division Staff (1993). Soil Survey Manual..."
    url = "https://..."

    [meta.names]
    en = "USDA Soil Texture Triangle"
    fr = "Triangle des textures USDA"

    [[class]]
    key = "clay"
    group = "fine"
    vertices = [[0, 100], [0, 60], [20, 40], [45, 40], [45, 55]]
    [class.names]
    en = "clay"
    fr = "argile"

    [[class]]
    key = "silty_clay"
    ...
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pedotri.errors import ClassificationError
from pedotri.schema import Classification, TextureClass, make_vertices


def load_classification(source: str | Path | bytes | dict[str, Any]) -> Classification:
    """Load a :class:`Classification` from TOML.

    Args:
        source: Either a path to a ``.toml`` file, raw TOML bytes, or a
            pre-parsed dict (useful for tests and entry-point providers).

    Returns:
        A validated :class:`Classification` instance.

    Raises:
        ClassificationError: If the document is missing required fields or
            contains invalid data.
        FileNotFoundError: If a path is given that does not exist.
    """
    data = _to_dict(source)

    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise ClassificationError("Missing or invalid [meta] table.")

    key = _require_str(meta, "key", path="[meta]")
    axes_raw = meta.get("axes")
    if (
        not isinstance(axes_raw, list)
        or not 1 <= len(axes_raw) <= 2
        or not all(isinstance(a, str) for a in axes_raw)
    ):
        raise ClassificationError(
            f"[meta].axes must be a list of 1 or 2 axis-name strings, got {axes_raw!r}."
        )
    axes = tuple(axes_raw)

    names = _coerce_names(meta.get("names", {}), where=f"[meta.names] in {key!r}")
    default_locale = meta.get("default_locale", "en")
    if not isinstance(default_locale, str):
        raise ClassificationError(
            f"[meta].default_locale must be a string, got {default_locale!r}."
        )

    classes_raw = data.get("class")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ClassificationError(
            f"Classification {key!r}: at least one [[class]] entry is required."
        )

    classes = tuple(_parse_class(c, classification_key=key) for c in classes_raw)

    return Classification(
        key=key,
        axes=axes,
        classes=classes,
        names=names,
        default_locale=default_locale,
        reference=_optional_str(meta, "reference"),
        url=_optional_str(meta, "url"),
        description=_optional_str(meta, "description"),
    )


def _to_dict(source: str | Path | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, bytes):
        return tomllib.loads(source.decode("utf-8"))
    path = Path(source)
    with path.open("rb") as f:
        return tomllib.load(f)


def _parse_class(raw: Any, *, classification_key: str) -> TextureClass:
    if not isinstance(raw, dict):
        raise ClassificationError(
            f"Classification {classification_key!r}: each [[class]] entry must be a table."
        )
    key = _require_str(raw, "key", path=f"[[class]] in {classification_key!r}")
    verts_raw = raw.get("vertices")
    interval_raw = raw.get("interval")

    if (verts_raw is None) == (interval_raw is None):
        raise ClassificationError(
            f"Class {key!r}: exactly one of 'vertices' (2-D polygon) or "
            f"'interval' (1-D range) must be set."
        )

    vertices = None
    interval: tuple[float, float] | None = None

    if verts_raw is not None:
        if not isinstance(verts_raw, list):
            raise ClassificationError(f"Class {key!r}: 'vertices' must be a list of [x, y] pairs.")
        try:
            vertices = make_vertices(verts_raw)
        except ClassificationError as exc:
            raise ClassificationError(f"Class {key!r}: {exc}") from exc
    else:
        if (
            not isinstance(interval_raw, list)
            or len(interval_raw) != 2
            or not all(isinstance(x, (int, float)) for x in interval_raw)
        ):
            raise ClassificationError(
                f"Class {key!r}: 'interval' must be a 2-element list of numbers "
                f"[low, high], got {interval_raw!r}."
            )
        interval = (float(interval_raw[0]), float(interval_raw[1]))

    names = _coerce_names(raw.get("names", {}), where=f"[class.names] in {key!r}")
    group = _optional_str(raw, "group")
    parent = _optional_str(raw, "parent")
    return TextureClass(
        key=key,
        vertices=vertices,
        interval=interval,
        names=names,
        group=group,
        parent=parent,
    )


def _require_str(d: dict[str, Any], key: str, *, path: str) -> str:
    val = d.get(key)
    if not isinstance(val, str) or not val:
        raise ClassificationError(
            f"{path}: required field {key!r} must be a non-empty string, got {val!r}."
        )
    return val


def _optional_str(d: dict[str, Any], key: str) -> str | None:
    val = d.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ClassificationError(f"Field {key!r} must be a string when present, got {val!r}.")
    return val


def _coerce_names(raw: Any, *, where: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ClassificationError(f"{where} must be a table of locale → name.")
    out: dict[str, str] = {}
    for locale, name in raw.items():
        if not isinstance(locale, str) or not isinstance(name, str):
            raise ClassificationError(
                f"{where}: entry {locale!r} = {name!r} is not string → string."
            )
        out[locale] = name
    return out
