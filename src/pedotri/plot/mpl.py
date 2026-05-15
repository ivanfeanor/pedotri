"""Matplotlib backend for texture diagrams.

Available when ``matplotlib`` is installed (``pip install pedotri[matplotlib]``).
Renders the same equilateral sand-silt-clay triangle as the SVG renderer,
but as a :class:`matplotlib.figure.Figure` so users can compose with
their existing plotting code, save to PDF/PNG/etc., or further annotate.

For 1-D classifications a horizontal banded axis is drawn instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pedotri.errors import PedotriError
from pedotri.registry import get_classification

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from pedotri.schema import Classification, TextureClass


_GROUP_COLORS: dict[str, str] = {
    "coarse": "#4a90d9",
    "moderately_coarse": "#7eb4e0",
    "medium": "#d4d8a8",
    "medium_fine": "#e8c987",
    "moderately_fine": "#d98e5a",
    "fine": "#a93f3f",
}
_DEFAULT_COLOR = "#cccccc"


def render_mpl(
    classification: str | Classification,
    *,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[str] | None = None,
    locale: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
    show_grid: bool = True,
    figsize: tuple[float, float] = (8, 7),
) -> Figure:
    """Render a texture diagram as a :class:`matplotlib.figure.Figure`.

    Args:
        classification: Classification key or instance.
        points: Optional (sand, clay) overlay points for 2-D
            classifications, or 1-tuples / scalars for 1-D.
        point_labels: Optional labels printed beside each marker.
        locale: Locale for class names in tooltips and legend.
        title: Optional figure title.
        show_legend: Show class colour legend.
        show_grid: For 2-D, draw 10 % interior gridlines.
        figsize: (width, height) inches passed to ``plt.figure``.

    Returns:
        Configured ``Figure`` ready for ``.savefig(...)`` or display.

    Raises:
        ModuleNotFoundError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401 — pulled in via patch import
        from matplotlib.patches import Polygon
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "render_mpl() requires matplotlib. Install with `pip install pedotri[matplotlib]`."
        ) from exc

    c = get_classification(classification) if isinstance(classification, str) else classification
    loc = locale or c.default_locale

    if len(c.axes) == 2:
        return _render_triangle_mpl(
            c,
            loc=loc,
            points=points,
            point_labels=point_labels,
            title=title,
            show_legend=show_legend,
            show_grid=show_grid,
            figsize=figsize,
            polygon_cls=Polygon,
        )
    if len(c.axes) == 1:
        return _render_axis_mpl(
            c,
            loc=loc,
            points=points,
            point_labels=point_labels,
            title=title,
            show_legend=show_legend,
            figsize=figsize,
        )
    raise PedotriError(f"Cannot render classification with {len(c.axes)} axes.")


def render_png(
    classification: str | Classification,
    *,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[str] | None = None,
    locale: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
    show_grid: bool = True,
    figsize: tuple[float, float] = (8, 7),
    dpi: int = 200,
) -> bytes:
    """Render a texture diagram to PNG bytes.

    Thin wrapper over :func:`render_mpl` that saves to PNG via
    matplotlib. The default 200 dpi yields ~1600 x 1400 px from the
    default 8 x 7" figure — crisp on retina displays at typical
    chat-embed widths, and the PNG stays comfortably small
    (~200 KB).

    Args:
        classification: Classification key or instance.
        points / point_labels / locale / title / show_legend /
            show_grid / figsize: forwarded to :func:`render_mpl`.
        dpi: Output resolution. 200 is a good "retina-ready" default
            for chat embeds; bump to 300+ for print-quality.

    Returns:
        Raw PNG bytes ready to write to a file or base64-encode.

    Raises:
        ModuleNotFoundError: If matplotlib is not installed.
    """
    import io

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "render_png() requires matplotlib. Install with `pip install pedotri[matplotlib]`."
        ) from exc

    # Headless backend so it works in any process (MCP server, CI, ...).
    matplotlib.use("Agg", force=True)

    fig = render_mpl(
        classification,
        points=points,
        point_labels=point_labels,
        locale=locale,
        title=title,
        show_legend=show_legend,
        show_grid=show_grid,
        figsize=figsize,
    )
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def _triangle_xy(sand: float, clay: float) -> tuple[float, float]:
    """Barycentric (sand %, clay %) → equilateral (x, y) coordinates."""
    s = sand / 100.0
    cy = clay / 100.0
    x = s + 0.5 * cy
    y = math.sin(math.radians(60)) * cy
    return x, y


def _render_triangle_mpl(
    c: Classification,
    *,
    loc: str,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    title: str | None,
    show_legend: bool,
    show_grid: bool,
    figsize: tuple[float, float],
    polygon_cls: Any,
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw triangle frame.
    ax.plot(
        [0, 1, 0.5, 0],
        [0, 0, math.sin(math.radians(60)), 0],
        color="#333",
        linewidth=1.5,
    )

    if show_grid:
        _draw_grid_mpl(ax)

    # Polygons.
    seen_groups: set[str] = set()
    for cls in c.classes:
        assert cls.vertices is not None
        verts_eq = [_triangle_xy(s, k) for s, k in cls.vertices]
        color = _color_for(cls)
        label_for_legend = None
        if show_legend and cls.group and cls.group not in seen_groups:
            label_for_legend = cls.group.replace("_", " ")
            seen_groups.add(cls.group)
        poly = polygon_cls(
            verts_eq,
            closed=True,
            facecolor=color,
            alpha=0.55,
            edgecolor="#777",
            linewidth=0.5,
            label=label_for_legend,
        )
        ax.add_patch(poly)
        # Class key as a faint annotation at the polygon centroid.
        cx = sum(x for x, _ in verts_eq) / len(verts_eq)
        cy = sum(y for _, y in verts_eq) / len(verts_eq)
        ax.text(
            cx,
            cy,
            cls.key,
            ha="center",
            va="center",
            fontsize=7,
            color="#222",
            alpha=0.8,
        )

    # Overlay points.
    if points is not None:
        for i, pt in enumerate(points):
            if len(pt) != 2:
                raise PedotriError(f"2-D diagram expects (sand, clay) points; got {pt!r}.")
            x, y = _triangle_xy(float(pt[0]), float(pt[1]))
            ax.plot(
                x,
                y,
                "o",
                color="#0066cc",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            if point_labels and i < len(point_labels):
                ax.annotate(
                    point_labels[i],
                    xy=(x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=9,
                )

    # Axis labels at the three corners.
    ax.text(0.5, -0.05, "% sand", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        0.78,
        math.sin(math.radians(60)) / 2 + 0.05,
        "% clay",
        ha="center",
        rotation=-60,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.22,
        math.sin(math.radians(60)) / 2 + 0.05,
        "% silt",
        ha="center",
        rotation=60,
        fontsize=11,
        fontweight="bold",
    )

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if show_legend and seen_groups:
        # Deduplicate and place legend to the right.
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
            title=c.name(loc),
            title_fontsize=10,
        )
    fig.tight_layout()
    return fig


def _draw_grid_mpl(ax: Any) -> None:
    """Draw 10 % gridlines parallel to each side."""
    for p in range(10, 100, 10):
        # Constant clay (parallel to bottom).
        x1, y1 = _triangle_xy(0, p)
        x2, y2 = _triangle_xy(100 - p, p)
        ax.plot([x1, x2], [y1, y2], color="#e0e0e0", linewidth=0.5, zorder=0)
        # Constant sand (parallel to right side).
        x1, y1 = _triangle_xy(p, 0)
        x2, y2 = _triangle_xy(p, 100 - p)
        ax.plot([x1, x2], [y1, y2], color="#e0e0e0", linewidth=0.5, zorder=0)
        # Constant silt (parallel to left side).
        x1, y1 = _triangle_xy(100 - p, 0)
        x2, y2 = _triangle_xy(0, 100 - p)
        ax.plot([x1, x2], [y1, y2], color="#e0e0e0", linewidth=0.5, zorder=0)


def _render_axis_mpl(
    c: Classification,
    *,
    loc: str,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    title: str | None,
    show_legend: bool,
    figsize: tuple[float, float],
) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    intervals = [cls.interval for cls in c.classes if cls.interval is not None]
    lo = min(iv[0] for iv in intervals)
    hi = max(iv[1] for iv in intervals)

    for cls in c.classes:
        assert cls.interval is not None
        x1, x2 = cls.interval
        color = _color_for(cls)
        ax.axvspan(
            x1,
            x2,
            facecolor=color,
            alpha=0.7,
            edgecolor="#777",
            linewidth=0.5,
            label=f"{cls.key} — {cls.name(loc)}",
        )

    if points is not None:
        for i, pt in enumerate(points):
            v = float(pt[0]) if hasattr(pt, "__getitem__") else float(pt)  # type: ignore[arg-type]
            ax.plot(
                v,
                0.5,
                "o",
                color="#0066cc",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            if point_labels and i < len(point_labels):
                ax.annotate(
                    point_labels[i],
                    xy=(v, 0.5),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"% {c.axes[0]}", fontsize=11, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(c.classes)),
            frameon=False,
            fontsize=9,
            title=c.name(loc),
            title_fontsize=10,
        )
    fig.tight_layout()
    return fig


def _color_for(cls: TextureClass) -> str:
    return _GROUP_COLORS.get(cls.group or "", _DEFAULT_COLOR)
