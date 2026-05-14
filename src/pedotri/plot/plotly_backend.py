"""Plotly backend for texture diagrams.

Available when ``plotly`` is installed (``pip install pedotri[plotly]``).
For 2-D classifications this uses Plotly's native ``Scatterternary`` so
the diagram inherits all of Plotly's interactivity: hover tooltips,
zoom, and the ability to embed as standalone HTML.

For 1-D classifications, a banded bar chart is produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pedotri.errors import PedotriError
from pedotri.registry import get_classification

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def render_plotly(
    classification: str | Classification,
    *,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[str] | None = None,
    locale: str | None = None,
    title: str | None = None,
    width: int = 720,
    height: int = 640,
) -> Any:
    """Render a texture diagram as a :class:`plotly.graph_objects.Figure`.

    Args:
        classification: Classification key or instance.
        points: Optional (sand, clay) overlay points for 2-D, or
            1-tuples / scalars for 1-D.
        point_labels: Optional labels shown alongside markers.
        locale: Locale for class names in tooltips and legend.
        title: Optional figure title.
        width: Pixel width.
        height: Pixel height.

    Returns:
        A ``plotly.graph_objects.Figure`` ready for ``.show()`` or
        ``.write_html(...)``.

    Raises:
        ModuleNotFoundError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "render_plotly() requires plotly. Install with `pip install pedotri[plotly]`."
        ) from exc

    c = get_classification(classification) if isinstance(classification, str) else classification
    loc = locale or c.default_locale

    if len(c.axes) == 2:
        return _render_ternary_plotly(
            c,
            loc=loc,
            points=points,
            point_labels=point_labels,
            title=title,
            width=width,
            height=height,
            go=go,
        )
    if len(c.axes) == 1:
        return _render_axis_plotly(
            c,
            loc=loc,
            points=points,
            point_labels=point_labels,
            title=title,
            width=width,
            height=height,
            go=go,
        )
    raise PedotriError(f"Cannot render classification with {len(c.axes)} axes.")


def _render_ternary_plotly(
    c: Classification,
    *,
    loc: str,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    title: str | None,
    width: int,
    height: int,
    go: Any,
) -> Any:
    """Build a ternary diagram with one filled polygon per class.

    Plotly's ``Scatterternary`` accepts (a, b, c) sand/silt/clay triplets;
    its built-in ternary axes already handle the equilateral layout.
    """
    fig = go.Figure()

    seen_groups: set[str] = set()
    for cls in c.classes:
        assert cls.vertices is not None
        # Close the polygon so the fill renders correctly.
        verts = [*list(cls.vertices), cls.vertices[0]]
        sand = [float(v[0]) for v in verts]
        clay = [float(v[1]) for v in verts]
        silt = [100.0 - s - cl for s, cl in zip(sand, clay, strict=True)]
        color = _color_for(cls)

        show_in_legend = cls.group is not None and cls.group not in seen_groups
        if cls.group:
            seen_groups.add(cls.group)

        fig.add_trace(
            go.Scatterternary(
                a=sand,
                b=clay,
                c=silt,
                mode="lines",
                fill="toself",
                fillcolor=color,
                opacity=0.55,
                line={"color": "#777", "width": 0.5},
                name=cls.group.replace("_", " ") if cls.group else cls.key,
                legendgroup=cls.group or cls.key,
                showlegend=show_in_legend,
                hovertemplate=(
                    f"<b>{cls.key}</b> — {cls.name(loc)}<br>"
                    "sand=%{a:.0f}%, clay=%{b:.0f}%, silt=%{c:.0f}%<extra></extra>"
                ),
            )
        )

    if points is not None:
        sand_pts: list[float] = []
        clay_pts: list[float] = []
        silt_pts: list[float] = []
        labels: list[str] = []
        for i, pt in enumerate(points):
            if len(pt) != 2:
                raise PedotriError(f"2-D diagram expects (sand, clay) points; got {pt!r}.")
            s_, cl_ = float(pt[0]), float(pt[1])
            sand_pts.append(s_)
            clay_pts.append(cl_)
            silt_pts.append(100.0 - s_ - cl_)
            labels.append((point_labels[i] if point_labels and i < len(point_labels) else "") or "")
        fig.add_trace(
            go.Scatterternary(
                a=sand_pts,
                b=clay_pts,
                c=silt_pts,
                mode="markers+text" if any(labels) else "markers",
                marker={"size": 10, "color": "#0066cc", "line": {"color": "white", "width": 1.5}},
                text=labels,
                textposition="top right",
                name="samples",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        ternary={
            "sum": 100,
            "aaxis": {"title": "% sand", "min": 0, "linewidth": 1.5},
            "baxis": {"title": "% clay", "min": 0, "linewidth": 1.5},
            "caxis": {"title": "% silt", "min": 0, "linewidth": 1.5},
            "bgcolor": "#fafafa",
        },
        margin={"l": 60, "r": 60, "t": 60 if title else 30, "b": 40},
    )
    return fig


def _render_axis_plotly(
    c: Classification,
    *,
    loc: str,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    title: str | None,
    width: int,
    height: int,
    go: Any,
) -> Any:
    fig = go.Figure()

    for cls in c.classes:
        assert cls.interval is not None
        x1, x2 = cls.interval
        color = _color_for(cls)
        fig.add_shape(
            type="rect",
            x0=x1,
            x1=x2,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.7,
            line={"color": "#777", "width": 0.5},
        )
        # Invisible trace to populate the legend.
        fig.add_trace(
            go.Scatter(
                x=[(x1 + x2) / 2],
                y=[0.5],
                mode="markers",
                marker={"size": 0, "color": color},
                name=f"{cls.key} — {cls.name(loc)} [{x1:g}, {x2:g})",
                hoverinfo="skip",
            )
        )

    if points is not None:
        xs: list[float] = []
        labels: list[str] = []
        for i, pt in enumerate(points):
            v = float(pt[0]) if hasattr(pt, "__getitem__") else float(pt)  # type: ignore[arg-type]
            xs.append(v)
            labels.append((point_labels[i] if point_labels and i < len(point_labels) else "") or "")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[0.5] * len(xs),
                mode="markers+text" if any(labels) else "markers",
                marker={"size": 12, "color": "#0066cc", "line": {"color": "white", "width": 1.5}},
                text=labels,
                textposition="top center",
                name="samples",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis={"title": f"% {c.axes[0]}", "range": [0, 100]},
        yaxis={"visible": False, "range": [0, 1]},
        plot_bgcolor="#fafafa",
        margin={"l": 60, "r": 60, "t": 60 if title else 30, "b": 60},
    )
    return fig


def _color_for(cls: TextureClass) -> str:
    return _GROUP_COLORS.get(cls.group or "", _DEFAULT_COLOR)
