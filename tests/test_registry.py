"""Tests for the classification registry: built-ins, registration, plugins."""

from __future__ import annotations

import pytest

import pedotri
from pedotri.errors import ClassificationError, UnknownClassificationError

_CUSTOM_TOML = b"""
[meta]
key = "MYCUSTOM"
axes = ["sand", "clay"]
reference = "Test"

[[class]]
key = "stuff"
vertices = [[0, 0], [100, 0], [50, 100]]
[class.names]
en = "stuff"
fr = "trucs"
"""


@pytest.fixture(autouse=True)
def _cleanup_custom() -> object:
    """Ensure custom registrations don't leak between tests."""
    pedotri.unregister_classification("MYCUSTOM")
    yield
    pedotri.unregister_classification("MYCUSTOM")


def test_list_includes_builtins() -> None:
    assert set(pedotri.list_classifications()) >= {
        "USDA",
        "FAO",
        "INTERNATIONAL",
        "ISSS",
    }


def test_register_from_bytes() -> None:
    pedotri.register_classification(_CUSTOM_TOML)
    assert "MYCUSTOM" in pedotri.list_classifications()
    assert pedotri.classify(40, 30, "MYCUSTOM") == "stuff"


def test_register_localized_names_resolve() -> None:
    pedotri.register_classification(_CUSTOM_TOML)
    assert pedotri.classify(40, 30, "MYCUSTOM", locale="fr") == "trucs"


def test_register_rejects_duplicate_without_overwrite() -> None:
    pedotri.register_classification(_CUSTOM_TOML)
    with pytest.raises(ClassificationError, match="already registered"):
        pedotri.register_classification(_CUSTOM_TOML)


def test_register_allows_overwrite() -> None:
    pedotri.register_classification(_CUSTOM_TOML)
    pedotri.register_classification(_CUSTOM_TOML, overwrite=True)


def test_register_accepts_classification_object() -> None:
    c = pedotri.load_classification(_CUSTOM_TOML)
    pedotri.register_classification(c)
    assert pedotri.get_classification("MYCUSTOM").key == "MYCUSTOM"


def test_unregister_is_idempotent() -> None:
    pedotri.unregister_classification("NEVER_REGISTERED")
    pedotri.unregister_classification("NEVER_REGISTERED")


def test_get_unknown_classification_raises() -> None:
    with pytest.raises(UnknownClassificationError):
        pedotri.get_classification("NOPE")


def test_get_classification_is_case_insensitive() -> None:
    assert pedotri.get_classification("usda").key == "USDA"
    assert pedotri.get_classification("Usda").key == "USDA"
