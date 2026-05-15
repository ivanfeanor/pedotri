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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from pedotri.errors import InvalidInputError
from pedotri.geometry import points_in_polygon, signed_distance_to_polygon
from pedotri.registry import get_classification
from pedotri.schema import Classification
from pedotri.units import _convert_inputs

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
        probabilities: Mapping from class key to posterior probability
            when uncertainty information was supplied. ``None`` for the
            deterministic path. Keys with non-zero probability only; the
            unclassified mass (if any) is stored separately in
            :attr:`unclassified_probability`.
        entropy: Shannon entropy of :attr:`probabilities` in nats. ``None``
            for the deterministic path. Maximum value is ``log(n_classes)``.
        confidence: Single scalar in ``[0, 1]`` summarizing how confident
            the classification is. For the distance method, this is the
            normal CDF of ``distance / σ_effective`` — points sitting
            many σ inside their class score near 1.0; points on a boundary
            score 0.5. For the Monte Carlo method, this is the modal-class
            probability. ``None`` for the deterministic path.
        unclassified_probability: Fraction of Monte Carlo samples that
            fell outside every class polygon. ``None`` for the distance
            and deterministic paths.
    """

    key: str | None
    name: str
    group: str | None
    parent: str | None
    distance: float
    probabilities: dict[str, float] | None = None
    entropy: float | None = None
    confidence: float | None = None
    unclassified_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        ``nan`` is mapped to ``None`` so the result encodes cleanly with
        the standard library ``json`` module. Uncertainty-related fields
        are included only when populated, to keep the deterministic
        output unchanged.
        """
        out: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "group": self.group,
            "parent": self.parent,
            "distance": None if math.isnan(self.distance) else self.distance,
        }
        if self.probabilities is not None:
            out["probabilities"] = dict(self.probabilities)
        if self.entropy is not None:
            out["entropy"] = self.entropy
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.unclassified_probability is not None:
            out["unclassified_probability"] = self.unclassified_probability
        return out


@overload
def classify(
    sand: float | int,
    clay: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
    units: str = ...,
) -> str | None: ...


@overload
def classify(
    sand: ArrayLike,
    clay: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
    units: str = ...,
) -> list[str | None]: ...


@overload
def classify(
    sand: float | int,
    clay: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
    units: str = ...,
) -> ClassifyResult: ...


@overload
def classify(
    sand: ArrayLike,
    clay: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
    units: str = ...,
) -> list[ClassifyResult]: ...


@overload
def classify(
    value: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
    units: str = ...,
) -> str | None: ...


@overload
def classify(
    value: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[False] = ...,
    units: str = ...,
) -> list[str | None]: ...


@overload
def classify(
    value: float | int,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
    units: str = ...,
) -> ClassifyResult: ...


@overload
def classify(
    value: ArrayLike,
    classification: str | Classification,
    *,
    locale: Locale | None = ...,
    detailed: Literal[True],
    units: str = ...,
) -> list[ClassifyResult]: ...


@overload
def classify(
    *,
    classification: str | Classification,
    locale: Locale | None = ...,
    detailed: bool = ...,
    units: str = ...,
    **axis_fractions: ScalarOrArrayLike,
) -> Any: ...


@overload
def classify(
    *args: Any,
    classification: str | Classification | None = ...,
    locale: Locale | None = ...,
    detailed: bool = ...,
    units: str = ...,
    method: str = ...,
    n_samples: int = ...,
    seed: Any = ...,
    **kwargs: Any,
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

    The argument shapes are documented through the overload signatures
    above. In short:

    - **fraction values** — one positional argument per axis of the
      chosen classification, in axis order. Each value is either a
      scalar percentage in [0, 100] or an array-like of percentages.
      Fractions may also be passed by axis name as keyword arguments
      (``sand=...``, ``clay=...``, ``physical_clay=...``).
    - **classification** — last positional argument, or the
      ``classification=`` keyword. May be a classification key string
      (e.g. ``"USDA"``) or a :class:`Classification` instance.
    - **locale** — optional locale tag. When provided, class names
      are localized via the fallback chain (``fr-FR`` → ``fr`` →
      ``en``); when omitted, results use the stable class keys.
    - **detailed** — when ``True``, returns
      :class:`ClassifyResult` objects with name, group, parent, and
      signed distance-to-boundary rather than bare strings.
    - **units** — keyword. ``"%"`` (default), ``"g/kg"``, or ``"g/g"``.
      Applied uniformly to every fraction input. Use ``"g/kg"`` when
      your lab reports sand / clay / silt as g/kg (common in European
      soil chemistry), or ``"g/g"`` for the 0-1 mass-fraction convention.
    - **<axis>_uncertainty** — optional keyword per axis (e.g.
      ``sand_uncertainty``, ``clay_uncertainty``). Accepts a
      :class:`~pedotri.uncertainty.Quantiles` (recommended), a scalar
      or array σ, or — for backwards compatibility — a bare
      ``(Q0.05, Q0.95)`` 2-tuple (emits a ``DeprecationWarning``).
      When any uncertainty kwarg is supplied, ``detailed`` is
      automatically promoted to ``True`` so probabilities and
      confidence can be returned, and the function returns a
      :class:`ClassifyResult` regardless of how ``detailed`` was
      passed.
    - **method** — ``"distance"`` (default) or ``"monte_carlo"``. Only
      relevant when uncertainty kwargs are supplied. The distance
      method compares boundary distance to the input σ for a single
      confidence score; Monte Carlo draws ``n_samples`` realizations
      and returns a full probability distribution.
    - **n_samples** — int, default ``1000``. Number of Monte Carlo
      draws per input point.
    - **seed** — optional int or ``np.random.Generator``. Seeds the
      Monte Carlo sampler for reproducibility.

    Returns the matched class key (or localized name, or
    :class:`ClassifyResult`) for scalar input, or a list of those
    values for array-like input. ``None`` is returned for any point
    that lies outside every class polygon / interval.

    Raises :class:`InvalidInputError` if fraction arrays have
    mismatched shapes or contain NaN / out-of-range values, or if the
    wrong number of fractions was provided.
    Raises :class:`UnknownClassificationError` if ``classification``
    is a string that is not registered.
    """
    classification = kwargs.pop("classification", None)
    locale: Locale | None = kwargs.pop("locale", None)
    detailed: bool = kwargs.pop("detailed", False)
    units: str = kwargs.pop("units", "%")
    method: str = kwargs.pop("method", "distance")
    n_samples: int = kwargs.pop("n_samples", 1000)
    seed: Any = kwargs.pop("seed", None)

    uncertainty_kwargs: dict[str, Any] = {}
    for key in list(kwargs):
        if key.endswith("_uncertainty"):
            uncertainty_kwargs[key[: -len("_uncertainty")]] = kwargs.pop(key)

    cls_obj, fractions = _parse_arguments(args, classification, kwargs)

    if uncertainty_kwargs:
        unknown = [a for a in uncertainty_kwargs if a not in cls_obj.axes]
        if unknown:
            raise TypeError(
                f"Classification {cls_obj.key!r}: unknown uncertainty kwarg(s) "
                f"{[a + '_uncertainty' for a in unknown]!r}. Valid axes are "
                f"{list(cls_obj.axes)!r}."
            )
        # Uncertainty data is only meaningful when probabilities /
        # confidence / entropy can be reported, so we auto-promote
        # ``detailed=True`` here. The return type accordingly switches
        # from ``str`` to :class:`ClassifyResult` — the input shape
        # signals this clearly. Without auto-promote, callers had to
        # write ``detailed=True`` on every uncertainty call, which was
        # pure boilerplate that got forgotten and surfaced a confusing
        # error.
        detailed = True
        if method not in ("distance", "monte_carlo"):
            raise InvalidInputError(
                f"Unknown classification method {method!r}. "
                "Use 'distance' (default) or 'monte_carlo'."
            )

    if units != "%":
        fractions = [_convert_inputs(f, units) for f in fractions]
    arrs, scalar = _coerce_fraction_arrays(fractions, cls_obj.axes)

    sigmas = _resolve_uncertainty(uncertainty_kwargs, cls_obj.axes, arrs[0].shape)

    extras: _Extras
    if sigmas is None:
        codes, distances = _classify(arrs, cls_obj, with_distance=detailed)
        extras = _Extras()
    elif method == "distance":
        codes, distances = _classify(arrs, cls_obj, with_distance=True)
        extras = _Extras(confidences=_distance_confidences(distances, sigmas))
    else:
        codes, distances = _classify(arrs, cls_obj, with_distance=True)
        extras = _monte_carlo_extras(arrs, sigmas, cls_obj, n_samples, seed)

    classes = cls_obj.classes
    out: list[str | None] | list[ClassifyResult]
    if detailed:
        loc = locale or cls_obj.default_locale
        out = _build_detailed(codes, distances, classes, loc, extras)
    else:
        out = _build_labels(codes, classes, locale)
    return out[0] if scalar else out


def classify_all(
    sand: float | int,
    clay: float | int,
    *,
    locale: Locale | None = None,
    schemes: list[str] | None = None,
    units: str = "%",
) -> dict[str, ClassifyResult]:
    """Classify the same (sand, clay) point under every 2-axis classification.

    This mirrors the common agronomic-API pattern of presenting one
    sample against several regional taxonomies side-by-side (USDA for
    USA, FAO for international, GEPPA for France, KA5 for Germany, …)
    instead of forcing the caller to issue N separate
    :func:`classify` calls.

    Args:
        sand: Sand fraction (scalar). See ``units``.
        clay: Clay fraction (scalar). See ``units``.
        locale: Optional locale tag forwarded to each classification.
            ``None`` keeps the stable class key as ``.name``.
        schemes: Restrict to a subset of classification keys
            (e.g. ``["USDA", "FAO", "GEPPA"]``). ``None`` runs every
            registered 2-axis classification.
        units: Same as :func:`classify` (``"%"``, ``"g/kg"``, ``"g/g"``).

    Returns:
        Dict mapping classification key to :class:`ClassifyResult`.
        1-axis classifications such as ``KACHINSKY`` are skipped (their
        axis differs from sand/clay), so the result includes only
        classifications that consume the sand-clay simplex.

    Example::

        >>> import pedotri
        >>> all_classes = pedotri.classify_all(sand=27, clay=45)
        >>> {k: v.key for k, v in all_classes.items()}
        {'USDA': 'clay', 'FAO': 'fine', 'GEPPA': '...', ...}
    """
    from pedotri.registry import get_classification, list_classifications

    if schemes is None:
        schemes = list_classifications()

    out: dict[str, ClassifyResult] = {}
    for key in schemes:
        cls_obj = get_classification(key)
        if len(cls_obj.axes) != 2:
            continue
        result = classify(
            sand=float(sand),
            clay=float(clay),
            classification=cls_obj,
            detailed=True,
            locale=locale,
            units=units,
        )
        out[key] = result
    return out


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
    arrs: list[FloatArray], c: Classification, *, with_distance: bool = True
) -> tuple[np.ndarray, FloatArray]:
    """Dispatch to the polygon or interval matcher based on ``c.axes``.

    Returns ``(codes, distances)``. ``codes`` is an ``int16`` array
    where each element is either the index of the matched
    :class:`TextureClass` in ``c.classes`` or ``-1`` for unclassified
    points. Using int codes — rather than a Python ``list`` of
    ``TextureClass`` references — keeps the scatter inside the class
    loop fully vectorized and lets the result formatter index a small
    pre-built label table once per call.

    ``with_distance`` is honoured by the 2-D path: skipping the
    per-segment distance computation roughly triples throughput on the
    common ``classify(...)`` call where the caller doesn't ask for a
    :class:`ClassifyResult`.
    """
    if len(c.axes) == 2:
        points = np.column_stack(arrs)
        return _classify_polygons(points, c, with_distance=with_distance)
    return _classify_intervals(arrs[0], c)


def _classify_polygons(
    points: FloatArray, c: Classification, *, with_distance: bool
) -> tuple[np.ndarray, FloatArray]:
    """Match 2-D points against polygon classes (first-match-wins).

    When ``with_distance`` is false, only the boolean point-in-polygon
    test runs; ``distances`` is returned as an all-NaN array. The
    detailed (``ClassifyResult``) path uses the signed-distance test
    so it can populate :attr:`ClassifyResult.distance`.

    A cheap axis-aligned bounding-box pre-filter wraps every polygon
    test: only points falling within the polygon's bbox proceed to the
    O(M·N) ray-cast / signed-distance computation. For typical
    classifications where individual class polygons cover only a
    fraction of the (sand %, clay %) plane, this skips the heavy work
    for the majority of points on the majority of classes.
    """
    n_points = points.shape[0]
    codes = np.full(n_points, -1, dtype=np.int16)
    distances = np.full(n_points, np.nan, dtype=np.float64)
    remaining = np.ones(n_points, dtype=bool)

    xs = points[:, 0]
    ys = points[:, 1]

    for idx, cls in enumerate(c.classes):
        if not remaining.any():
            break
        assert cls.vertices is not None  # guarded by Classification._validate
        v = cls.vertices
        x_min, x_max = v[:, 0].min(), v[:, 0].max()
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        candidates = remaining & (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        if not candidates.any():
            continue
        active_idx = np.flatnonzero(candidates)
        sub_points = points[active_idx]
        if with_distance:
            sub_distances = signed_distance_to_polygon(sub_points, v)
            inside = sub_distances >= 0
        else:
            inside = points_in_polygon(sub_points, v)
            sub_distances = None
        if not inside.any():
            continue
        hit_idx = active_idx[inside]
        codes[hit_idx] = idx
        if sub_distances is not None:
            distances[hit_idx] = sub_distances[inside]
        remaining[hit_idx] = False
    return codes, distances


def _classify_intervals(values: FloatArray, c: Classification) -> tuple[np.ndarray, FloatArray]:
    """Match 1-D scalars against half-open intervals [low, high).

    The half-open convention deterministically attributes boundary
    values to the earlier-listed class when adjacent intervals abut.
    The distance returned is to the nearest interval endpoint.
    """
    n = values.shape[0]
    codes = np.full(n, -1, dtype=np.int16)
    distances = np.full(n, np.nan, dtype=np.float64)
    remaining = np.ones(n, dtype=bool)

    for idx, cls in enumerate(c.classes):
        if not remaining.any():
            break
        assert cls.interval is not None
        low, high = cls.interval
        active = remaining & (values >= low) & (values < high)
        if not active.any():
            continue
        codes[active] = idx
        sub = values[active]
        distances[active] = np.minimum(sub - low, high - sub)
        remaining &= ~active
    return codes, distances


# --- Result formatting ---------------------------------------------------


def _build_labels(
    codes: np.ndarray,
    classes: tuple[TextureClass, ...],
    locale: Locale | None,
) -> list[str | None]:
    """Map a vector of class indices to a list of label strings.

    The label table is computed once per call (one entry per class) and
    a single Python-level pass over ``codes.tolist()`` produces the
    output. This avoids the per-point Python attribute access that
    dominated the non-detailed path's profile.
    """
    labels = (
        [cls.key for cls in classes] if locale is None else [cls.name(locale) for cls in classes]
    )
    return [None if c < 0 else labels[c] for c in codes.tolist()]


def _build_detailed(
    codes: np.ndarray,
    distances: FloatArray,
    classes: tuple[TextureClass, ...],
    locale: Locale,
    extras: _Extras,
) -> list[ClassifyResult]:
    """Map class indices + distances (+ optional uncertainty extras) to results."""
    names = [cls.name(locale) for cls in classes]
    out: list[ClassifyResult] = []
    code_list = codes.tolist()
    dist_list = distances.tolist()
    has_extras = extras.has_any()
    for i, (c, d) in enumerate(zip(code_list, dist_list, strict=True)):
        extras_i = extras.at(i) if has_extras else _ExtrasView()
        if c < 0:
            out.append(
                ClassifyResult(
                    key=None,
                    name="",
                    group=None,
                    parent=None,
                    distance=float("nan"),
                    probabilities=extras_i.probabilities,
                    entropy=extras_i.entropy,
                    confidence=extras_i.confidence,
                    unclassified_probability=extras_i.unclassified_probability,
                )
            )
            continue
        cls = classes[c]
        out.append(
            ClassifyResult(
                key=cls.key,
                name=names[c],
                group=cls.group,
                parent=cls.parent,
                distance=d,
                probabilities=extras_i.probabilities,
                entropy=extras_i.entropy,
                confidence=extras_i.confidence,
                unclassified_probability=extras_i.unclassified_probability,
            )
        )
    return out


# --- Uncertainty plumbing ------------------------------------------------


@dataclass(slots=True)
class _ExtrasView:
    """Per-point slice of the optional uncertainty extras."""

    probabilities: dict[str, float] | None = None
    entropy: float | None = None
    confidence: float | None = None
    unclassified_probability: float | None = None


@dataclass(slots=True)
class _Extras:
    """Vectorized container for the optional per-point uncertainty outputs.

    The three array fields are populated only by paths that produce
    them. ``probabilities_index`` is the resolved class key per column
    of ``probabilities``; it's stored once rather than re-created per
    point. Distance method fills ``confidences`` only; Monte Carlo
    fills all four.
    """

    confidences: FloatArray | None = None
    probabilities: FloatArray | None = None  # shape (n_points, n_classes)
    probabilities_index: list[str] | None = None  # class keys
    entropies: FloatArray | None = None
    unclassified_probs: FloatArray | None = None

    def has_any(self) -> bool:
        return (
            self.confidences is not None
            or self.probabilities is not None
            or self.entropies is not None
            or self.unclassified_probs is not None
        )

    def at(self, i: int) -> _ExtrasView:
        view = _ExtrasView()
        if self.confidences is not None:
            v = float(self.confidences[i])
            view.confidence = None if math.isnan(v) else v
        if self.probabilities is not None and self.probabilities_index is not None:
            row = self.probabilities[i]
            view.probabilities = {
                self.probabilities_index[j]: float(row[j])
                for j in range(row.shape[0])
                if row[j] > 0
            }
            # For Monte Carlo, modal-class probability also serves as
            # the confidence — overwrite whatever the caller had so the
            # field stays single-source-of-truth for the active method.
            if row.size:
                view.confidence = float(row.max())
        if self.entropies is not None:
            v = float(self.entropies[i])
            view.entropy = None if math.isnan(v) else v
        if self.unclassified_probs is not None:
            view.unclassified_probability = float(self.unclassified_probs[i])
        return view


def _resolve_uncertainty(
    uncertainty_kwargs: dict[str, Any],
    axes: tuple[str, ...],
    shape: tuple[int, ...],
) -> list[FloatArray] | None:
    """Build per-axis σ arrays in axis order, or return ``None`` if no kwargs.

    Axes without an uncertainty entry get an all-zero σ — they behave
    deterministically while letting the other axes carry uncertainty.
    The output broadcasts each user value to the input point shape so
    downstream code can index uniformly.
    """
    from pedotri.uncertainty import _parse_uncertainty

    if not uncertainty_kwargs:
        return None
    sigmas: list[FloatArray] = []
    for axis in axes:
        raw = uncertainty_kwargs.get(axis)
        if raw is None:
            sigmas.append(np.zeros(shape, dtype=np.float64))
            continue
        parsed = _parse_uncertainty(raw)
        if parsed is None:
            sigmas.append(np.zeros(shape, dtype=np.float64))
            continue
        try:
            broadcast = np.broadcast_to(parsed, shape).astype(np.float64, copy=True)
        except ValueError as exc:
            raise InvalidInputError(
                f"{axis}_uncertainty shape {parsed.shape} cannot broadcast to "
                f"fraction shape {shape}."
            ) from exc
        sigmas.append(broadcast)
    return sigmas


def _distance_confidences(distances: FloatArray, sigmas: list[FloatArray]) -> FloatArray:
    """Per-point confidence via Φ(distance / σ_effective).

    σ_effective is the quadratic mean across axes — an isotropic proxy
    when the boundary orientation is unknown. NaN distances (unclassified
    points) and zero σ produce NaN confidences.
    """
    stacked = np.stack(sigmas, axis=0)
    sigma_eff = np.sqrt(np.mean(stacked * stacked, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = distances / sigma_eff
    conf = 0.5 * (1.0 + _erf_array(z / np.sqrt(2.0)))
    conf = np.where(sigma_eff > 0, conf, np.nan)
    conf = np.where(np.isnan(distances), np.nan, conf)
    return conf


def _erf_array(x: FloatArray) -> FloatArray:
    """Vectorized ``erf`` — delegated to :mod:`pedotri.uncertainty`."""
    from pedotri.uncertainty import _erf_array as _erf_impl

    return _erf_impl(x)


def _monte_carlo_extras(
    arrs: list[FloatArray],
    sigmas: list[FloatArray],
    c: Classification,
    n_samples: int,
    seed: Any,
) -> _Extras:
    """Sample ``n_samples`` realizations per point, classify, tally probs.

    For 2-axis classifications, sampling respects the compositional
    constraint ``sand + clay ≤ 100`` via the helper in
    :mod:`pedotri.uncertainty`. For 1-axis classifications, the single
    axis is sampled as a clipped truncated normal.
    """
    from pedotri.uncertainty import (
        _aggregate_class_probabilities,
        sample_compositional,
        sample_truncated_normal,
        shannon_entropy,
    )

    if n_samples < 1:
        raise InvalidInputError(f"n_samples must be >= 1, got {n_samples}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if len(c.axes) == 2:
        sand_samples, clay_samples = sample_compositional(
            arrs[0], sigmas[0], arrs[1], sigmas[1], n_samples, rng=rng
        )
        flat = [sand_samples.reshape(-1), clay_samples.reshape(-1)]
    else:
        samples = sample_truncated_normal(arrs[0], sigmas[0], n_samples, rng=rng)
        flat = [samples.reshape(-1)]

    flat_codes, _ = _classify(flat, c, with_distance=False)
    n_points = arrs[0].size
    codes_grid = flat_codes.reshape(n_points, n_samples)

    n_classes = len(c.classes)
    probs, unclass_rate = _aggregate_class_probabilities(codes_grid, n_classes)
    entropies = shannon_entropy(probs)

    class_keys = [cls.key for cls in c.classes]
    return _Extras(
        probabilities=probs,
        probabilities_index=class_keys,
        entropies=entropies,
        unclassified_probs=unclass_rate,
    )
