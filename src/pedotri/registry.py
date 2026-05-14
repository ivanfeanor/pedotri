"""Classification registry: built-ins, user-registered TOMLs, and entry-point plugins.

Built-in classifications ship as TOML files under
``pedotri/_data/classifications/`` and are lazily loaded on first access.

Third-party packages can contribute classifications by declaring an entry
point in the ``pedotri.classifications`` group::

    [project.entry-points."pedotri.classifications"]
    MY_KEY = "my_package:get_classification"

The referenced callable must accept no arguments and return either a
:class:`~pedotri.schema.Classification` or a path/bytes/dict consumable by
:func:`~pedotri.loader.load_classification`.

Ad-hoc registration is supported via :func:`register_classification`.
"""

from __future__ import annotations

import threading
from importlib.metadata import entry_points
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from pedotri.errors import ClassificationError, UnknownClassificationError
from pedotri.loader import load_classification
from pedotri.schema import Classification

if TYPE_CHECKING:
    from pathlib import Path

_ENTRYPOINT_GROUP = "pedotri.classifications"
_BUILTIN_PACKAGE = "pedotri._data.classifications"

_lock = threading.RLock()
_registry: dict[str, Classification] = {}
_state: dict[str, bool] = {"entry_points_loaded": False}


def _discover_builtin_keys() -> frozenset[str]:
    """Return the keys of TOMLs shipped under ``pedotri/_data/classifications/``.

    Computed lazily on each call from the package's resource listing, so
    adding a new TOML file requires no code change.
    """
    try:
        resources = files(_BUILTIN_PACKAGE)
    except (ModuleNotFoundError, FileNotFoundError):
        return frozenset()
    keys: set[str] = set()
    for entry in resources.iterdir():
        name = entry.name
        if name.endswith(".toml"):
            keys.add(name.removesuffix(".toml").upper())
    return frozenset(keys)


def get_classification(key: str) -> Classification:
    """Return the registered classification for ``key`` (case-insensitive).

    Built-ins and entry-point classifications are loaded lazily on first
    access. Raises :class:`UnknownClassificationError` if the key is not
    registered.
    """
    normalized = key.upper()
    with _lock:
        _ensure_entry_points_loaded()
        if normalized in _registry:
            return _registry[normalized]
        if normalized in _discover_builtin_keys():
            c = _load_builtin(normalized)
            _registry[normalized] = c
            return c
        raise UnknownClassificationError(key, available=list_classifications())


def list_classifications() -> list[str]:
    """Return the sorted list of all registered classification keys."""
    with _lock:
        _ensure_entry_points_loaded()
        seen = set(_registry) | set(_discover_builtin_keys())
        return sorted(seen)


def register_classification(
    source: str | Path | bytes | dict[str, Any] | Classification,
    *,
    overwrite: bool = False,
) -> Classification:
    """Register a custom classification.

    Args:
        source: A TOML path, raw TOML bytes, a pre-parsed dict, or an
            already-constructed :class:`Classification`.
        overwrite: If ``False`` (default), raise when a classification
            with the same key is already registered. Set to ``True`` to
            replace an existing entry — useful when iterating during
            development.

    Returns:
        The (validated) :class:`Classification` instance.
    """
    c = source if isinstance(source, Classification) else load_classification(source)
    key = c.key.upper()
    with _lock:
        _ensure_entry_points_loaded()
        if not overwrite and key in _registry:
            raise ClassificationError(
                f"Classification {key!r} is already registered. Pass overwrite=True to replace it."
            )
        _registry[key] = c
    return c


def unregister_classification(key: str) -> None:
    """Remove a previously registered classification.

    No-op if the key isn't registered. Useful in tests.
    """
    with _lock:
        _registry.pop(key.upper(), None)


def _load_builtin(key: str) -> Classification:
    resource = files(_BUILTIN_PACKAGE).joinpath(f"{key.lower()}.toml")
    data = resource.read_bytes()
    return load_classification(data)


def _ensure_entry_points_loaded() -> None:
    """Discover and register classifications declared via entry points.

    Called lazily and at most once per process. Failures while loading a
    single plugin are isolated — a broken plugin doesn't prevent others
    from registering.
    """
    if _state["entry_points_loaded"]:
        return
    # Set first so re-entry during plugin load can't infinite-loop.
    _state["entry_points_loaded"] = True

    try:
        eps = entry_points(group=_ENTRYPOINT_GROUP)
    except Exception:
        return

    for ep in eps:
        try:
            factory = ep.load()
            raw = factory() if callable(factory) else factory
            c = raw if isinstance(raw, Classification) else load_classification(raw)
        except Exception:
            # A misbehaving plugin should never break the registry for everyone.
            continue
        _registry.setdefault(c.key.upper(), c)
