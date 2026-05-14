"""Smoke tests for built-in classification TOML files.

Each TOML must load cleanly into a validated :class:`Classification`
and meet basic invariants: non-empty classes, valid polygons,
declared locales for at least English.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import pytest

import pedotri
from pedotri.loader import load_classification

if TYPE_CHECKING:
    from pedotri.schema import Classification

_BUILTIN = [
    "usda",
    "fao",
    "international",
    "isss",
    "hypres",
    "geppa",
    "jamagne",
    "embrapa",
    "ka5",
    "kachinsky",
]


def _load(name: str) -> Classification:
    resource = files("pedotri._data.classifications").joinpath(f"{name}.toml")
    return load_classification(resource.read_bytes())


@pytest.mark.parametrize("name", _BUILTIN)
def test_builtin_loads(name: str) -> None:
    c = _load(name)
    assert c.key == name.upper()
    assert len(c.classes) >= 3
    assert "en" in c.locales()


@pytest.mark.parametrize("name", _BUILTIN)
def test_builtin_classes_have_english_names(name: str) -> None:
    c = _load(name)
    for cls in c.classes:
        assert "en" in cls.names, f"{c.key}.{cls.key} missing en name"


@pytest.mark.parametrize("name", _BUILTIN)
def test_builtin_classes_have_reference(name: str) -> None:
    c = _load(name)
    assert c.reference is not None
    assert len(c.reference) > 20


def test_usda_has_twelve_classes() -> None:
    c = _load("usda")
    assert len(c.classes) == 12
    keys = set(c.class_keys())
    assert keys == {
        "clay",
        "silty_clay",
        "silty_clay_loam",
        "sandy_clay",
        "sandy_clay_loam",
        "clay_loam",
        "silt",
        "silt_loam",
        "loam",
        "sand",
        "loamy_sand",
        "sandy_loam",
    }


def test_fao_has_three_classes() -> None:
    c = _load("fao")
    assert c.class_keys() == ("fine", "medium", "coarse")


def test_usda_french_localization() -> None:
    c = _load("usda")
    assert c.class_by_key("clay").name("fr") == "argile"
    assert c.class_by_key("sand").name("fr") == "sable"


def test_usda_groups_assigned() -> None:
    c = _load("usda")
    groups = {cls.group for cls in c.classes}
    # USDA texture groups span fine → coarse
    assert {"fine", "moderately_fine", "medium", "moderately_coarse", "coarse"} <= groups


def test_hypres_reference_points() -> None:
    assert pedotri.classify(90, 5, "HYPRES") == "coarse"
    assert pedotri.classify(40, 40, "HYPRES") == "fine"
    assert pedotri.classify(5, 70, "HYPRES") == "very_fine"
    assert pedotri.classify(5, 5, "HYPRES") == "medium_fine"
    assert pedotri.classify(40, 20, "HYPRES") == "medium"


def test_geppa_reference_points() -> None:
    assert pedotri.classify(5, 70, "GEPPA") == "AA"
    assert pedotri.classify(20, 40, "GEPPA") == "AL"
    assert pedotri.classify(95, 2, "GEPPA") == "S"
    assert pedotri.classify(70, 5, "GEPPA") == "SL"


def test_embrapa_reference_points() -> None:
    assert pedotri.classify(10, 70, "EMBRAPA") == "muito_argilosa"
    assert pedotri.classify(10, 50, "EMBRAPA") == "argilosa"
    assert pedotri.classify(40, 25, "EMBRAPA") == "media"
    assert pedotri.classify(90, 5, "EMBRAPA") == "arenosa"


def test_ka5_reference_points() -> None:
    # KA5 ships the full 31-class subdivision.
    assert pedotri.classify(95, 2, "KA5") == "Ss"  # pure sand
    assert pedotri.classify(10, 80, "KA5") == "Tt"  # pure clay
    assert pedotri.classify(5, 5, "KA5") == "Uu"  # pure silt
    assert pedotri.classify(60, 15, "KA5") == "Sl4"  # strongly loamy sand
    assert pedotri.classify(15, 50, "KA5") == "Tu2"  # weakly silty clay
    assert pedotri.classify(35, 28, "KA5") == "Lt2"  # weakly clayey loam


def test_ka5_has_31_classes() -> None:
    c = _load("ka5")
    assert len(c.classes) == 31


def test_jamagne_distinct_from_geppa() -> None:
    # Jamagne 1967 has no AA class; clay > 45 % is just A.
    j = _load("jamagne")
    g = _load("geppa")
    j_keys = set(j.class_keys())
    g_keys = set(g.class_keys())
    assert "AA" in g_keys
    assert "AA" not in j_keys
    # Both should classify a heavy-clay point, but to different keys.
    assert pedotri.classify(5, 70, "JAMAGNE") == "A"
    assert pedotri.classify(5, 70, "GEPPA") == "AA"


def test_kachinsky_is_one_dimensional() -> None:
    c = _load("kachinsky")
    assert c.axes == ("physical_clay",)
    assert len(c.classes) == 9


def test_kachinsky_classify_positional() -> None:
    assert pedotri.classify(3, "KACHINSKY") == "loose_sand"
    assert pedotri.classify(15, "KACHINSKY") == "sandy_loam"
    assert pedotri.classify(35, "KACHINSKY") == "medium_loam"
    assert pedotri.classify(70, "KACHINSKY") == "medium_clay"
    assert pedotri.classify(100, "KACHINSKY") == "heavy_clay"


def test_kachinsky_classify_array() -> None:
    assert pedotri.classify([3, 35, 90], "KACHINSKY") == [
        "loose_sand",
        "medium_loam",
        "heavy_clay",
    ]


def test_kachinsky_classify_keyword_axis() -> None:
    assert pedotri.classify(physical_clay=15, classification="KACHINSKY") == "sandy_loam"


def test_kachinsky_default_locale_is_russian() -> None:
    c = _load("kachinsky")
    assert c.default_locale == "ru"
    assert c.class_by_key("sandy_loam").name("ru") == "супесь"


def test_kachinsky_detailed_distance_to_interval_edge() -> None:
    # PC=15 sits at the midpoint of [10, 20); distance to nearest edge = 5.
    result = pedotri.classify(15, "KACHINSKY", detailed=True)
    assert result.key == "sandy_loam"
    assert result.distance == 5.0


def test_geppa_default_locale_is_french() -> None:
    c = _load("geppa")
    assert c.default_locale == "fr"
    # When no locale is passed but detailed=True, French should be used
    assert c.class_by_key("AA").name("fr") == "argile lourde"


def test_embrapa_default_locale_is_portuguese() -> None:
    c = _load("embrapa")
    assert c.default_locale == "pt"
    assert c.class_by_key("muito_argilosa").name("pt") == "muito argilosa"


def test_ka5_default_locale_is_german() -> None:
    c = _load("ka5")
    assert c.default_locale == "de"
    assert c.class_by_key("Ss").name("de") == "Reinsand"
    assert c.class_by_key("Tt").name("de") == "Reinton"
