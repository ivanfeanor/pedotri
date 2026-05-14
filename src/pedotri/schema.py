"""Data model for soil texture classifications.

A *classification* (e.g. USDA, GEPPA) is a set of named texture classes, each
defined by a polygon in the plane of two particle-size fractions (typically
sand+clay percentages). Class names are localizable, classes can be grouped
into a coarser hierarchy, and the whole structure is loadable from a TOML file
so users can define their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pedotri.errors import ClassificationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pedotri._types import FloatArray, Locale


_VALID_AXES = frozenset({"sand", "silt", "clay", "physical_clay"})


@dataclass(slots=True)
class TextureClass:
    """A single named texture class within a classification.

    A class is either a *polygon* in 2-D space (when the parent
    classification has two axes, e.g. USDA's sand-clay plane) or an
    *interval* on a single axis (when the classification is 1-D, e.g.
    Kachinsky's physical clay fraction). Exactly one of
    :attr:`vertices` and :attr:`interval` must be set, matching the
    parent classification's :attr:`Classification.axes` length.

    Treat instances as immutable after construction.

    Attributes:
        key: Stable machine identifier. Used as the return value of
            :func:`pedotri.classify` and as the lookup key for hierarchy
            and i18n.
        vertices: (N, 2) float array of polygon vertices in the
            (axis_x, axis_y) plane. Coordinates are percentages in
            [0, 100]. Closure is implicit. Set this for 2-D classes;
            leave :attr:`interval` as ``None``.
        interval: ``(low, high)`` tuple defining the half-open range
            ``[low, high)`` on the single axis. Set this for 1-D
            classes; leave :attr:`vertices` as ``None``.
        names: Mapping of locale tag → localized display name. ``"en"``
            is the recommended fallback.
        group: Optional coarse grouping (e.g. ``"fine"``, ``"medium"``,
            ``"coarse"``).
        parent: Optional parent class key for hierarchical classifications.
    """

    key: str
    vertices: FloatArray | None = None
    interval: tuple[float, float] | None = None
    names: dict[Locale, str] = field(default_factory=dict)
    group: str | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        has_vertices = self.vertices is not None
        has_interval = self.interval is not None
        if has_vertices == has_interval:
            raise ClassificationError(
                f"Class {self.key!r}: set exactly one of 'vertices' (for 2-D) "
                f"or 'interval' (for 1-D); got "
                f"vertices={'set' if has_vertices else 'None'}, "
                f"interval={'set' if has_interval else 'None'}."
            )

    @property
    def is_polygon(self) -> bool:
        """True if this class is a 2-D polygon, False if a 1-D interval."""
        return self.vertices is not None

    def name(self, locale: Locale = "en") -> str:
        """Resolve the localized class name with locale fallback.

        Falls back through the locale chain ``fr-FR → fr → en`` and finally
        to ``self.key`` if no localized name is available.
        """
        return _resolve_locale(self.names, locale, fallback=self.key)


@dataclass(slots=True)
class Classification:
    """A complete soil texture classification system.

    Attributes:
        key: Uppercase identifier (e.g. ``"USDA"``, ``"GEPPA"``).
        axes: Pair of particle-size fractions used as the x and y axes,
            e.g. ``("sand", "clay")``. The third fraction (silt) is implicit
            and equal to ``100 - x - y``.
        classes: Tuple of :class:`TextureClass`. Order is significant: the
            first class containing a point wins. This matches the convention
            used by most published soil texture triangles, where polygons are
            listed in canonical order and points on shared edges are
            deterministically attributed to the earlier-listed class.
        names: Localized names of the classification itself (for UI display).
        default_locale: Locale to use when none is explicitly requested.
        reference: Bibliographic reference for the polygon definitions.
        url: Optional canonical URL for the reference.
        description: Optional longer description of scope and intended use.
    """

    key: str
    axes: tuple[str, ...]
    classes: tuple[TextureClass, ...]
    names: dict[Locale, str] = field(default_factory=dict)
    default_locale: Locale = "en"
    reference: str | None = None
    url: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def name(self, locale: Locale | None = None) -> str:
        """Resolve the localized classification name."""
        return _resolve_locale(self.names, locale or self.default_locale, fallback=self.key)

    def class_by_key(self, key: str) -> TextureClass:
        """Look up a class by its stable key.

        Raises:
            KeyError: If ``key`` is not a class in this classification.
        """
        for cls in self.classes:
            if cls.key == key:
                return cls
        raise KeyError(
            f"No class {key!r} in classification {self.key!r}. "
            f"Available: {[c.key for c in self.classes]!r}"
        )

    def class_keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.classes)

    def locales(self) -> frozenset[Locale]:
        """Return the union of all locales declared anywhere in this classification."""
        seen: set[Locale] = set(self.names)
        for cls in self.classes:
            seen.update(cls.names)
        return frozenset(seen)

    def _validate(self) -> None:
        if not self.key:
            raise ClassificationError("Classification key must be non-empty.")
        self._validate_axes()
        if not self.classes:
            raise ClassificationError(f"Classification {self.key!r} has no classes.")
        all_keys = {c.key for c in self.classes}
        seen_keys: set[str] = set()
        for cls in self.classes:
            self._validate_class(cls, seen_keys, all_keys)

    def _validate_axes(self) -> None:
        if not 1 <= len(self.axes) <= 2:
            raise ClassificationError(
                f"axes must have length 1 (interval) or 2 (polygon), got {self.axes!r}."
            )
        if len(self.axes) == 2 and self.axes[0] == self.axes[1]:
            raise ClassificationError(f"axes must be distinct, got {self.axes!r}.")
        for ax in self.axes:
            if ax not in _VALID_AXES:
                raise ClassificationError(f"axis {ax!r} not in {sorted(_VALID_AXES)!r}.")

    def _validate_class(
        self,
        cls: TextureClass,
        seen_keys: set[str],
        all_keys: set[str],
    ) -> None:
        if not cls.key:
            raise ClassificationError(f"Empty class key in classification {self.key!r}.")
        if cls.key in seen_keys:
            raise ClassificationError(
                f"Duplicate class key {cls.key!r} in classification {self.key!r}."
            )
        seen_keys.add(cls.key)
        if len(self.axes) == 2:
            self._validate_polygon_class(cls)
        else:
            self._validate_interval_class(cls)
        if cls.parent is not None and cls.parent not in all_keys:
            raise ClassificationError(
                f"Class {cls.key!r}: parent {cls.parent!r} is not a class in {self.key!r}."
            )

    def _validate_polygon_class(self, cls: TextureClass) -> None:
        if cls.vertices is None:
            raise ClassificationError(
                f"Class {cls.key!r}: classification {self.key!r} has 2 axes; "
                f"'vertices' is required."
            )
        if cls.vertices.ndim != 2 or cls.vertices.shape[1] != 2:
            raise ClassificationError(
                f"Class {cls.key!r}: vertices must have shape (N, 2), got {cls.vertices.shape}."
            )
        if cls.vertices.shape[0] < 3:
            raise ClassificationError(
                f"Class {cls.key!r}: a polygon needs at least 3 vertices, "
                f"got {cls.vertices.shape[0]}."
            )

    def _validate_interval_class(self, cls: TextureClass) -> None:
        if cls.interval is None:
            raise ClassificationError(
                f"Class {cls.key!r}: classification {self.key!r} has 1 axis; "
                f"'interval' is required."
            )
        low, high = cls.interval
        if low >= high:
            raise ClassificationError(
                f"Class {cls.key!r}: interval [{low}, {high}) is empty "
                f"(low must be strictly less than high)."
            )


def _resolve_locale(names: Mapping[Locale, str], locale: Locale, *, fallback: str) -> str:
    """Resolve a localized name using a locale fallback chain.

    ``fr-FR`` → ``fr`` → ``en`` → ``fallback``.
    """
    if locale in names:
        return names[locale]
    # Strip region/script subtags one at a time.
    base = locale
    while "-" in base:
        base = base.rsplit("-", 1)[0]
        if base in names:
            return names[base]
    if "en" in names:
        return names["en"]
    return fallback


def make_vertices(coords: Iterable[Iterable[float]]) -> FloatArray:
    """Coerce an iterable of (x, y) pairs into the canonical float64 array."""
    arr = np.asarray(list(coords), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ClassificationError(
            f"vertices must be an (N, 2) array of (x, y) pairs, got shape {arr.shape}."
        )
    return arr
