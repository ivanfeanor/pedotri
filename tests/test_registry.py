"""Tests for the classification registry: built-ins, registration, plugins."""

from __future__ import annotations

import pytest

import pedotri
from pedotri import registry as reg
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


# --- Entry-point plugin loading ------------------------------------------


_PLUGIN_DICT = {
    "meta": {"key": "PLUGIN_OK", "axes": ["sand", "clay"]},
    "class": [{"key": "x", "vertices": [[0, 0], [100, 0], [50, 100]]}],
}


class _FakeEntryPoint:
    def __init__(self, name: str, factory: object) -> None:
        self.name = name
        self._factory = factory

    def load(self) -> object:
        return self._factory


@pytest.fixture
def _reset_entry_point_state() -> object:
    """Make sure entry-point discovery re-runs for the test and clean up after."""
    reg._state["entry_points_loaded"] = False
    yield
    reg._registry.pop("PLUGIN_OK", None)
    reg._registry.pop("PLUGIN_OBJECT", None)
    reg._state["entry_points_loaded"] = False


@pytest.mark.usefixtures("_reset_entry_point_state")
def test_entry_point_factory_registers_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda group: [_FakeEntryPoint("PLUGIN_OK", lambda: _PLUGIN_DICT)],
    )
    assert "PLUGIN_OK" in pedotri.list_classifications()
    assert pedotri.classify(40, 30, "PLUGIN_OK") == "x"


@pytest.mark.usefixtures("_reset_entry_point_state")
def test_entry_point_returning_classification_object_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factories may return a pre-built Classification instance directly,
    not just raw TOML/dict. Covers the isinstance branch in the loader."""
    obj = pedotri.load_classification(
        {
            "meta": {"key": "PLUGIN_OBJECT", "axes": ["sand", "clay"]},
            "class": [{"key": "x", "vertices": [[0, 0], [100, 0], [50, 100]]}],
        }
    )
    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda group: [_FakeEntryPoint("PLUGIN_OBJECT", lambda: obj)],
    )
    assert pedotri.get_classification("PLUGIN_OBJECT").key == "PLUGIN_OBJECT"


@pytest.mark.usefixtures("_reset_entry_point_state")
def test_broken_entry_point_does_not_break_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> object:
        raise RuntimeError("misbehaving plugin")

    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda group: [
            _FakeEntryPoint("PLUGIN_BAD", boom),
            _FakeEntryPoint("PLUGIN_OK", lambda: _PLUGIN_DICT),
        ],
    )
    keys = pedotri.list_classifications()
    assert "PLUGIN_BAD" not in keys
    assert "PLUGIN_OK" in keys


@pytest.mark.usefixtures("_reset_entry_point_state")
def test_entry_points_discovery_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``importlib.metadata.entry_points()`` itself raises, the registry
    should swallow the failure and continue serving built-ins."""

    def boom(group: str) -> object:
        raise RuntimeError("metadata machinery exploded")

    monkeypatch.setattr(reg, "entry_points", boom)
    # Built-ins must still resolve.
    assert "USDA" in pedotri.list_classifications()


def test_discover_builtin_keys_handles_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken install where the bundled data package is missing should
    return an empty builtin set rather than crashing on import."""

    def boom(pkg: str) -> object:
        raise FileNotFoundError("data package missing")

    monkeypatch.setattr(reg, "files", boom)
    assert reg._discover_builtin_keys() == frozenset()
