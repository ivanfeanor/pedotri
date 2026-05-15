"""Tests for the schema dataclasses and TOML loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib

from pedotri.errors import ClassificationError
from pedotri.loader import load_classification
from pedotri.schema import Classification, TextureClass, _resolve_locale, make_vertices

# --- TextureClass --------------------------------------------------------


def test_texture_class_resolves_locale_with_fallback() -> None:
    cls = TextureClass(
        key="clay",
        vertices=make_vertices([(0, 100), (0, 60), (45, 55)]),
        names={"en": "clay", "fr": "argile"},
    )
    assert cls.name("en") == "clay"
    assert cls.name("fr") == "argile"
    assert cls.name("fr-FR") == "argile"  # regional fallback
    assert cls.name("ru") == "clay"  # locale-missing fallback → en
    assert cls.name("zh") == "clay"


def test_texture_class_is_polygon_flag() -> None:
    poly = TextureClass(key="a", vertices=make_vertices([(0, 0), (1, 0), (0, 1)]))
    interval = TextureClass(key="b", interval=(0.0, 10.0))
    assert poly.is_polygon is True
    assert interval.is_polygon is False


def test_resolve_locale_iterates_multi_subtag_chain() -> None:
    # fr-CA-region → fr-CA (miss) → fr (hit). Exercises the loop-back branch.
    assert _resolve_locale({"fr": "argile"}, "fr-CA-region", fallback="X") == "argile"


def test_texture_class_falls_back_to_key_when_no_names() -> None:
    cls = TextureClass(
        key="sandy_loam",
        vertices=make_vertices([(0, 0), (10, 0), (5, 10)]),
        names={},
    )
    assert cls.name() == "sandy_loam"


# --- Classification validation -------------------------------------------


def _ok_class(key: str = "clay") -> TextureClass:
    return TextureClass(key=key, vertices=make_vertices([(0, 0), (10, 0), (5, 10)]))


def test_classification_rejects_empty_classes() -> None:
    with pytest.raises(ClassificationError, match="no classes"):
        Classification(key="X", axes=("sand", "clay"), classes=())


def test_classification_rejects_duplicate_class_keys() -> None:
    with pytest.raises(ClassificationError, match="Duplicate"):
        Classification(key="X", axes=("sand", "clay"), classes=(_ok_class(), _ok_class()))


def test_classification_rejects_identical_axes() -> None:
    with pytest.raises(ClassificationError, match="distinct"):
        Classification(key="X", axes=("sand", "sand"), classes=(_ok_class(),))


def test_classification_rejects_unknown_axis() -> None:
    with pytest.raises(ClassificationError, match="axis"):
        Classification(key="X", axes=("sand", "rocks"), classes=(_ok_class(),))


def test_classification_rejects_unknown_parent() -> None:
    bad_child = TextureClass(
        key="child",
        vertices=make_vertices([(0, 0), (10, 0), (5, 10)]),
        parent="nonexistent",
    )
    with pytest.raises(ClassificationError, match="parent"):
        Classification(key="X", axes=("sand", "clay"), classes=(bad_child,))


def test_classification_accepts_valid_parent() -> None:
    parent = TextureClass(key="loam", vertices=make_vertices([(0, 0), (10, 0), (5, 10)]))
    child = TextureClass(
        key="sandy_loam",
        vertices=make_vertices([(0, 0), (5, 0), (3, 5)]),
        parent="loam",
    )
    c = Classification(key="X", axes=("sand", "clay"), classes=(parent, child))
    assert c.class_by_key("sandy_loam").parent == "loam"


def test_classification_locales_union() -> None:
    c = Classification(
        key="X",
        axes=("sand", "clay"),
        classes=(
            TextureClass(
                key="a",
                vertices=make_vertices([(0, 0), (10, 0), (5, 10)]),
                names={"en": "a", "fr": "ah"},
            ),
            TextureClass(
                key="b",
                vertices=make_vertices([(0, 0), (10, 0), (5, 10)]),
                names={"en": "b", "ru": "бэ"},
            ),
        ),
        names={"en": "X", "de": "X (DE)"},
    )
    assert c.locales() == frozenset({"en", "fr", "ru", "de"})


# --- _resolve_locale -----------------------------------------------------


def test_resolve_locale_strips_regional_subtags() -> None:
    names = {"fr": "argile"}
    assert _resolve_locale(names, "fr-CA", fallback="X") == "argile"
    assert _resolve_locale(names, "fr-FR", fallback="X") == "argile"


def test_resolve_locale_uses_en_then_fallback() -> None:
    assert _resolve_locale({"en": "clay"}, "ru", fallback="X") == "clay"
    assert _resolve_locale({}, "ru", fallback="X") == "X"


# --- Loader --------------------------------------------------------------


_MINIMAL_TOML = b"""
[meta]
key = "TEST"
axes = ["sand", "clay"]
default_locale = "en"
reference = "Test reference"

[meta.names]
en = "Test Classification"
fr = "Classification de test"

[[class]]
key = "clay"
group = "fine"
vertices = [[0, 100], [0, 60], [20, 40], [45, 40], [45, 55]]
[class.names]
en = "clay"
fr = "argile"

[[class]]
key = "loam"
vertices = [[0, 0], [50, 0], [50, 30], [0, 30]]
[class.names]
en = "loam"
"""


def test_load_classification_from_bytes_returns_validated_object() -> None:
    c = load_classification(_MINIMAL_TOML)
    assert c.key == "TEST"
    assert c.axes == ("sand", "clay")
    assert c.name("fr") == "Classification de test"
    assert c.class_by_key("clay").name("fr") == "argile"
    assert c.class_by_key("clay").group == "fine"
    assert c.reference == "Test reference"
    assert c.class_keys() == ("clay", "loam")


def test_load_classification_rejects_missing_meta() -> None:
    with pytest.raises(ClassificationError, match=r"\[meta\]"):
        load_classification(b"[[class]]\nkey='a'\nvertices=[[0,0],[1,0],[0,1]]")


def test_load_classification_rejects_bad_vertices() -> None:
    bad = b"""
[meta]
key = "X"
axes = ["sand", "clay"]
[[class]]
key = "a"
vertices = "not a list"
"""
    with pytest.raises(ClassificationError, match="vertices"):
        load_classification(bad)


def test_load_classification_from_path(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "test.toml"
    p.write_bytes(_MINIMAL_TOML)
    c = load_classification(p)
    assert c.key == "TEST"


def test_make_vertices_promotes_to_float64() -> None:
    v = make_vertices([(0, 100), (50, 50), (100, 0)])
    assert v.dtype == np.float64
    assert v.shape == (3, 2)


def test_make_vertices_rejects_wrong_shape() -> None:
    with pytest.raises(ClassificationError, match="N, 2"):
        make_vertices([(0, 0, 0), (1, 1, 1), (2, 2, 2)])


# --- TextureClass / Classification validation error paths ----------------


def test_texture_class_rejects_both_vertices_and_interval() -> None:
    with pytest.raises(ClassificationError, match="exactly one"):
        TextureClass(
            key="x",
            vertices=make_vertices([(0, 0), (1, 0), (0, 1)]),
            interval=(0.0, 10.0),
        )


def test_texture_class_rejects_neither_vertices_nor_interval() -> None:
    with pytest.raises(ClassificationError, match="exactly one"):
        TextureClass(key="x")


def test_classification_class_by_key_raises_for_missing_key() -> None:
    c = Classification(key="X", axes=("sand", "clay"), classes=(_ok_class(),))
    with pytest.raises(KeyError, match="No class"):
        c.class_by_key("missing")


def test_classification_rejects_empty_key() -> None:
    with pytest.raises(ClassificationError, match="non-empty"):
        Classification(key="", axes=("sand", "clay"), classes=(_ok_class(),))


def test_classification_rejects_zero_or_three_axes() -> None:
    with pytest.raises(ClassificationError, match="length 1"):
        Classification(key="X", axes=(), classes=(_ok_class(),))
    with pytest.raises(ClassificationError, match="length 1"):
        Classification(key="X", axes=("sand", "clay", "silt"), classes=(_ok_class(),))


def test_classification_rejects_empty_class_key() -> None:
    bad = TextureClass(key="", vertices=make_vertices([(0, 0), (1, 0), (0, 1)]))
    with pytest.raises(ClassificationError, match="Empty class key"):
        Classification(key="X", axes=("sand", "clay"), classes=(bad,))


def test_classification_polygon_class_must_have_vertices() -> None:
    interval_class = TextureClass(key="a", interval=(0.0, 10.0))
    with pytest.raises(ClassificationError, match="'vertices' is required"):
        Classification(key="X", axes=("sand", "clay"), classes=(interval_class,))


def test_classification_polygon_class_rejects_too_few_vertices() -> None:
    # make_vertices accepts 2 rows; the polygon validator rejects.
    two_vertex = TextureClass(key="a", vertices=make_vertices([(0, 0), (1, 0)]))
    with pytest.raises(ClassificationError, match="at least 3 vertices"):
        Classification(key="X", axes=("sand", "clay"), classes=(two_vertex,))


def test_classification_polygon_class_rejects_wrong_vertex_shape() -> None:
    # 1-D array slips past TextureClass.__post_init__ (which only checks
    # vertices/interval are mutually exclusive) and is caught by the
    # Classification-level shape validator.
    cls = TextureClass(key="a", vertices=np.zeros((4,), dtype=np.float64))
    with pytest.raises(ClassificationError, match=r"shape \(N, 2\)"):
        Classification(key="X", axes=("sand", "clay"), classes=(cls,))


def test_classification_interval_class_must_have_interval() -> None:
    poly = TextureClass(key="a", vertices=make_vertices([(0, 0), (1, 0), (0, 1)]))
    with pytest.raises(ClassificationError, match="'interval' is required"):
        Classification(key="X", axes=("physical_clay",), classes=(poly,))


def test_classification_interval_class_rejects_empty_interval() -> None:
    cls = TextureClass(key="a", interval=(10.0, 5.0))
    with pytest.raises(ClassificationError, match="empty"):
        Classification(key="X", axes=("physical_clay",), classes=(cls,))
