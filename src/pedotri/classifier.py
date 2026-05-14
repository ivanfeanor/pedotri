"""Core classifier API.

The public entry point :func:`classify` accepts either scalar or array-like
inputs for each axis of the chosen classification, returns the matching
class key (or the localized name when a ``locale`` is given), and works
against any registered :class:`~pedotri.schema.Classification`.

The number of fraction values required matches the classification's
:attr:`~pedotri.schema.Classification.axes` — two for triangle-based
classifications like USDA, one for axis-based classifications like
Kachinsky's physical-clay grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from pedotri.errors import InvalidInputError
from pedotri.geometry import signed_distance_to_polygon
from pedotri.registry import get_classification
from pedotri.schema import Classification

if TYPE_CHECKING:
    from pedotri._types import ArrayLike, FloatArray, Locale, ScalarOrArrayLike
    from pedotri.schema import TextureClass


@dataclass(slots=True, frozen=True)
class ClassifyResult:
    """Detailed classification of a single point.

    Attributes:
        key: The matched class's stable key, or ``None`` if no class
            contains the point.
        name: The localized class name (or ``""`` when key is ``None``).
        group: Coarse grouping of the class (e.g. ``"fine"``) if defined.
        parent: Hierarchical parent class key if defined.
        distance: Signed Euclidean distance to the matched class boundary
            in the axis space. Positive means strictly inside; the larger
            the magnitude, the deeper the point sits inside its class.
            For 1-D classifications, this is the distance to the nearest
            interval endpoint. ``nan`` if the point is unclassified.
    """

    key: str | None
    name: str
    group: str | None
    parent: str | None
    distance: float


@overload
def classify(
    sand: float | int,
    clay: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
) -> str | None: ...


@overload
def classify(
    sand: ArrayLike,
    clay: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
) -> list[str | None]: ...


@overload
def classify(
    sand: float | int,
    clay: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
) -> ClassifyResult: ...


@overload
def classify(
    sand: ArrayLike,
    clay: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
) -> list[ClassifyResult]: ...


@overload
def classify(
    value: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
) -> str | None: ...


@overload
def classify(
    value: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
) -> list[str | None]: ...


@overload
def classify(
    value: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
) -> ClassifyResult: ...


@overload
def classify(
    value: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
) -> list[ClassifyResult]: ...


@overload
def classify(
    *,
    classification: str | Classification,
    locale: Locale | None = ...,
    detailed: bool = ...,
    **axis_fractions: ScalarOrArrayLike,
) -> Any: ...


def classify(
    *args: Any,
    **kwargs: Any,
) -> str | None | list[str | None] | ClassifyResult | list[ClassifyResult]:
    """Classify one or many soil texture points.

    The number and meaning of fraction arguments depends on the chosen
    classification's :attr:`~pedotri.schema.Classification.axes`. The
    function accepts three calling styles:

    Positional, axes-ordered::

        pedotri.classify(70, 20, "USDA")  # sand=70, clay=20
        pedotri.classify(10, "KACHINSKY")  # physical_clay=10

    Keyword fractions with positional classification::

        pedotri.classify(sand=70, clay=20, classification="USDA")

    Mixed::

        pedotri.classify(70, classification="USDA", clay=20)

    Args:
        *fractions_then_classification: Positional arguments. The last
            argument is the classification key or :class:`Classification`
            instance (unless given as a keyword); preceding arguments are
            fraction percentages in axis order.
        classification: Classification key (e.g. ``"USDA"``) or instance.
            Required (either positional or keyword).
        locale: When provided, return localized class names rather than
            class keys. Falls back through the locale chain (e.g.
            ``fr-FR`` → ``fr`` → ``en``).
        detailed: When ``True``, return :class:`ClassifyResult` instances
            with name, group, parent, and signed distance-to-boundary.
        **axis_fractions: Per-axis fraction percentages by axis name
            (e.g. ``sand=70, clay=20`` or ``physical_clay=10``).

    Returns:
        - Scalar input: a class key string (or localized name, or
          :class:`ClassifyResult`), or ``None`` if the point is outside
          every class.
        - Array input: a list of the above, one per input point.

    Raises:
        InvalidInputError: If fraction arrays have mismatched shapes, or
            contain NaN, or are outside the [0, 100] interval, or the
            wrong number of fractions was provided.
        UnknownClassificationError: If ``classification`` is a string
            that is not registered.
    """
    classification = kwargs.pop("classification", None)
    locale: Locale | None = kwargs.pop("locale", None)
    detailed: bool = kwargs.pop("detailed", False)
    cls_obj, fractions = _parse_arguments(args, classification, kwargs)
    arrs, scalar = _coerce_fraction_arrays(fractions, cls_obj.axes)

    matched, distances = _classify(arrs, cls_obj)

    out: list[str | None] | list[ClassifyResult]
    if detailed:
        loc = locale or cls_obj.default_locale
        out = [_make_result(c, loc, d) for c, d in zip(matched, distances, strict=True)]
    else:
        out = [_format_label(c, locale) for c in matched]
    return out[0] if scalar else out


# --- Argument parsing ----------------------------------------------------


def _parse_arguments(
    positional: tuple[Any, ...],
    classification_kwarg: str | Classification | None,
    axis_fractions: dict[str, ScalarOrArrayLike],
) -> tuple[Classification, list[ScalarOrArrayLike]]:
    """Split positional + kwargs into (classification, ordered fractions).

    The last positional arg is the classification unless it was passed
    by keyword. Remaining positionals fill axes in order; remaining
    kwargs fill axes by name.
    """
    if classification_kwarg is not None:
        cls_arg: str | Classification = classification_kwarg
        positional_fractions = list(positional)
    else:
        if not positional:
            raise TypeError(
                "classify() requires a classification (positional last argument "
                "or 'classification' keyword)."
            )
        cls_arg = positional[-1]
        positional_fractions = list(positional[:-1])

    cls_obj = _resolve_classification(cls_arg)
    axes = cls_obj.axes

    if len(positional_fractions) > len(axes):
        raise TypeError(
            f"Classification {cls_obj.key!r} expects {len(axes)} fraction(s) "
            f"(axes {axes!r}), got {len(positional_fractions)} positional."
        )

    ordered: list[ScalarOrArrayLike] = list(positional_fractions)
    for axis in axes[len(positional_fractions) :]:
        if axis not in axis_fractions:
            missing = [a for a in axes[len(positional_fractions) :] if a not in axis_fractions]
            raise TypeError(
                f"Classification {cls_obj.key!r}: missing fraction(s) for axes "
                f"{missing!r}. Provide as positional args in axis order "
                f"({axes!r}) or as keyword args."
            )
        ordered.append(axis_fractions.pop(axis))

    if axis_fractions:
        unknown = sorted(axis_fractions)
        raise TypeError(
            f"Classification {cls_obj.key!r}: unexpected fraction kwarg(s) "
            f"{unknown!r}. Valid axes are {list(axes)!r}."
        )

    return cls_obj, ordered


def _resolve_classification(
    classification: str | Classification,
) -> Classification:
    if isinstance(classification, str):
        return get_classification(classification)
    if isinstance(classification, Classification):
        return classification
    raise TypeError(
        f"classification must be a str or Classification, got {type(classification).__name__}."
    )


def _coerce_fraction_arrays(
    fractions: list[ScalarOrArrayLike], axes: tuple[str, ...]
) -> tuple[list[FloatArray], bool]:
    """Coerce fraction values to 1-D float arrays of equal length.

    Returns ``(arrays, was_scalar)`` where ``was_scalar`` is True if all
    original inputs were scalars (so the caller can unwrap the result).
    """
    raw_arrs = [np.asarray(f, dtype=np.float64) for f in fractions]
    scalar = all(a.ndim == 0 for a in raw_arrs)
    arrs = [np.atleast_1d(a) for a in raw_arrs]

    shapes = {a.shape for a in arrs}
    if len(shapes) > 1:
        shape_list = [a.shape for a in arrs]
        raise InvalidInputError(
            f"All fraction arrays must have the same shape; "
            f"got shapes {shape_list} for axes {list(axes)!r}."
        )

    for axis, arr in zip(axes, arrs, strict=True):
        if np.isnan(arr).any():
            raise InvalidInputError(f"{axis!r} must not contain NaN.")
        if ((arr < 0) | (arr > 100)).any():
            raise InvalidInputError(f"{axis!r} must be a percentage in [0, 100].")
    return arrs, scalar


# --- Classification core -------------------------------------------------


def _classify(
    arrs: list[FloatArray], c: Classification
) -> tuple[list[TextureClass | None], FloatArray]:
    """Dispatch to the polygon or interval matcher based on ``c.axes``."""
    if len(c.axes) == 2:
        points = np.column_stack(arrs)
        return _classify_polygons(points, c)
    return _classify_intervals(arrs[0], c)


def _classify_polygons(
    points: FloatArray, c: Classification
) -> tuple[list[TextureClass | None], FloatArray]:
    """Match 2-D points against polygon classes (first-match-wins)."""
    n_points = points.shape[0]
    matched: list[TextureClass | None] = [None] * n_points
    distances = np.full(n_points, np.nan, dtype=np.float64)
    remaining = np.ones(n_points, dtype=bool)

    for cls in c.classes:
        if not remaining.any():
            break
        assert cls.vertices is not None  # guarded by Classification._validate
        active_idx = np.flatnonzero(remaining)
        sub_points = points[active_idx]
        sub_distances = signed_distance_to_polygon(sub_points, cls.vertices)
        inside = sub_distances >= 0
        if not inside.any():
            continue
        hit_idx = active_idx[inside]
        for i in hit_idx:
            matched[int(i)] = cls
        distances[hit_idx] = sub_distances[inside]
        remaining[hit_idx] = False
    return matched, distances


def _classify_intervals(
    values: FloatArray, c: Classification
) -> tuple[list[TextureClass | None], FloatArray]:
    """Match 1-D scalars against half-open intervals [low, high).

    The half-open convention deterministically attributes boundary
    values to the earlier-listed class when adjacent intervals abut.
    The distance returned is to the nearest interval endpoint.
    """
    n = values.shape[0]
    matched: list[TextureClass | None] = [None] * n
    distances = np.full(n, np.nan, dtype=np.float64)
    remaining = np.ones(n, dtype=bool)

    for cls in c.classes:
        if not remaining.any():
            break
        assert cls.interval is not None
        low, high = cls.interval
        active = remaining & (values >= low) & (values < high)
        if not active.any():
            continue
        for i in np.flatnonzero(active):
            matched[int(i)] = cls
            v = float(values[i])
            distances[i] = min(v - low, high - v)
        remaining &= ~active
    return matched, distances


# --- Result formatting ---------------------------------------------------


def _format_label(cls: TextureClass | None, locale: Locale | None) -> str | None:
    if cls is None:
        return None
    if locale is None:
        return cls.key
    return cls.name(locale)


def _make_result(
    cls: TextureClass | None,
    locale: Locale,
    distance: float,
) -> ClassifyResult:
    if cls is None:
        return ClassifyResult(key=None, name="", group=None, parent=None, distance=float("nan"))
    return ClassifyResult(
        key=cls.key,
        name=cls.name(locale),
        group=cls.group,
        parent=cls.parent,
        distance=float(distance),
    )
