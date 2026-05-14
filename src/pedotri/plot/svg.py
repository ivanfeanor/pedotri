"""Pure-Python SVG renderer for soil texture diagrams.

The renderer has no plotting dependencies — it builds the SVG document
as a string. For 2-D classifications it draws the canonical equilateral
sand-silt-clay triangle with class polygons coloured by textural group
(when available) and an optional legend. For 1-D classifications it
draws a horizontal bar split into the named intervals.

The :class:`TextureDiagram` wrapper exposes a Jupyter ``_repr_svg_``
method so a single line of code displays the diagram inline in a
notebook cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from pedotri.errors import PedotriError
from pedotri.registry import get_classification

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pedotri.schema import Classification, TextureClass


# A curated group → fill colour palette, going from coarse (blue) to
# fine (red) along the agronomic gradient. Falls back to a neutral grey
# for classes without a group tag.
_GROUP_COLORS: dict[str, str] = {
    "coarse": "#4a90d9",
    "moderately_coarse": "#7eb4e0",
    "medium": "#d4d8a8",
    "medium_fine": "#e8c987",
    "moderately_fine": "#d98e5a",
    "fine": "#a93f3f",
}
_DEFAULT_COLOR = "#cccccc"
_BG_COLOR = "#fafafa"
_FRAME_COLOR = "#333333"
_LABEL_COLOR = "#222222"
_POINT_COLOR = "#0066cc"


@dataclass(slots=True)
class _Layout:
    """Pixel geometry for the diagram canvas."""

    width: float
    height: float
    tri_x: float  # left edge of triangle
    tri_y: float  # top edge of triangle
    tri_w: float  # side length
    tri_h: float  # height = tri_w * sin60


def render_svg(
    classification: str | Classification,
    *,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[str] | None = None,
    locale: str | None = None,
    title: str | None = None,
    width: int = 720,
    show_legend: bool = True,
    show_grid: bool = True,
) -> str:
    """Render a soil texture diagram as a standalone SVG string.

    Args:
        classification: Classification key (e.g. ``"USDA"``) or instance.
        points: Optional list of (sand, clay) pairs for 2-D
            classifications, or (value,) / scalars for 1-D. Drawn as
            overlay markers.
        point_labels: Optional labels for each marker.
        locale: Locale tag for class names in the legend; falls back to
            the classification's default locale.
        title: Optional title rendered above the diagram.
        width: SVG viewport width in pixels.
        show_legend: Include a class legend to the right of the diagram.
        show_grid: For 2-D, draw the 10 % interior grid lines.

    Returns:
        SVG document as a string.
    """
    c = get_classification(classification) if isinstance(classification, str) else classification
    if len(c.axes) == 2:
        return _render_triangle(
            c,
            points=points,
            point_labels=point_labels,
            locale=locale,
            title=title,
            width=width,
            show_legend=show_legend,
            show_grid=show_grid,
        )
    if len(c.axes) == 1:
        return _render_axis(
            c,
            points=points,
            point_labels=point_labels,
            locale=locale,
            title=title,
            width=width,
            show_legend=show_legend,
        )
    raise PedotriError(f"Cannot render classification with {len(c.axes)} axes.")


class TextureDiagram:
    """Renderable wrapper that displays inline in Jupyter notebooks.

    Construct with the same arguments as :func:`render_svg`; the
    notebook will call :meth:`_repr_svg_` to display the diagram.
    """

    def __init__(
        self,
        classification: str | Classification,
        *,
        points: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[str] | None = None,
        locale: str | None = None,
        title: str | None = None,
        width: int = 720,
        show_legend: bool = True,
        show_grid: bool = True,
    ) -> None:
        self._kwargs = {
            "classification": classification,
            "points": points,
            "point_labels": point_labels,
            "locale": locale,
            "title": title,
            "width": width,
            "show_legend": show_legend,
            "show_grid": show_grid,
        }

    def render(self) -> str:
        """Return the SVG string."""
        return render_svg(**self._kwargs)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> None:
        """Write the SVG document to ``path``."""
        Path(path).write_text(self.render(), encoding="utf-8")

    def _repr_svg_(self) -> str:
        return self.render()


# --- 2-D triangle --------------------------------------------------------


def _render_triangle(
    c: Classification,
    *,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    locale: str | None,
    title: str | None,
    width: int,
    show_legend: bool,
    show_grid: bool,
) -> str:
    loc = locale or c.default_locale
    margin = 60
    legend_w = 220 if show_legend else 0
    title_h = 40 if title else 12
    tri_w = float(width - 2 * margin - legend_w)
    tri_h = tri_w * math.sin(math.radians(60))
    total_h = int(tri_h + 2 * margin + title_h + 40)

    layout = _Layout(
        width=float(width),
        height=float(total_h),
        tri_x=float(margin),
        tri_y=float(margin + title_h),
        tri_w=tri_w,
        tri_h=tri_h,
    )

    parts: list[str] = []
    parts.append(_svg_header(layout))
    if title:
        parts.append(_svg_title(title, layout))
    if show_grid:
        parts.append(_svg_grid_lines(layout))
    parts.extend(_svg_class_polygons(c, layout, loc))
    parts.append(_svg_triangle_frame(layout))
    parts.extend(_svg_axis_labels(c, layout, loc))
    if points is not None:
        parts.append(_svg_points(points, point_labels, layout))
    if show_legend:
        parts.append(_svg_legend(c, layout, loc, x=margin + tri_w + 20))
    parts.append("</svg>")
    return "\n".join(parts)


def _triangle_xy(sand: float, clay: float, layout: _Layout) -> tuple[float, float]:
    """Barycentric (sand %, clay %) → SVG pixel (x, y) on equilateral triangle.

    Vertices:
      - sand=100, clay=0  → lower-right
      - sand=0,   clay=100 → top
      - sand=0,   clay=0   → lower-left (i.e. silt=100)
    """
    s = sand / 100.0
    c_ = clay / 100.0
    # Position in unit equilateral with left at (0,0), right at (1,0), top at (0.5, sin60).
    x_norm = s + 0.5 * c_
    y_norm = math.sin(math.radians(60)) * c_
    x = layout.tri_x + x_norm * layout.tri_w
    # SVG y grows downwards; flip relative to the triangle's bottom edge.
    y = layout.tri_y + layout.tri_h - y_norm * layout.tri_w
    return x, y


def _svg_header(layout: _Layout) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {layout.width:.0f} {layout.height:.0f}" '
        f'width="{layout.width:.0f}" height="{layout.height:.0f}" '
        f'font-family="-apple-system, system-ui, sans-serif">'
        f'<rect width="100%" height="100%" fill="{_BG_COLOR}"/>'
    )


def _svg_title(title: str, layout: _Layout) -> str:
    return (
        f'<text x="{layout.width / 2:.0f}" y="32" '
        f'text-anchor="middle" font-size="18" font-weight="600" '
        f'fill="{_LABEL_COLOR}">{escape(title)}</text>'
    )


def _svg_grid_lines(layout: _Layout) -> str:
    """Draw 10 % interior gridlines parallel to each side."""
    lines: list[str] = []
    for p in range(10, 100, 10):
        # Lines of constant clay (parallel to bottom edge).
        x1, y1 = _triangle_xy(0, p, layout)
        x2, y2 = _triangle_xy(100 - p, p, layout)
        lines.append(_line(x1, y1, x2, y2, stroke="#e0e0e0"))
        # Lines of constant sand (parallel to right edge).
        x1, y1 = _triangle_xy(p, 0, layout)
        x2, y2 = _triangle_xy(p, 100 - p, layout)
        lines.append(_line(x1, y1, x2, y2, stroke="#e0e0e0"))
        # Lines of constant silt (parallel to left edge).
        x1, y1 = _triangle_xy(100 - p, 0, layout)
        x2, y2 = _triangle_xy(0, 100 - p, layout)
        lines.append(_line(x1, y1, x2, y2, stroke="#e0e0e0"))
    return "<g>" + "".join(lines) + "</g>"


def _svg_class_polygons(c: Classification, layout: _Layout, loc: str) -> Iterable[str]:
    """Yield filled polygons for each class."""
    for cls in c.classes:
        assert cls.vertices is not None
        pts = [_triangle_xy(s, k, layout) for s, k in cls.vertices]
        path_d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        fill = _color_for(cls)
        title_text = f"{cls.key} — {cls.name(loc)}"
        yield (
            f'<polygon points="{path_d}" fill="{fill}" '
            f'fill-opacity="0.55" stroke="#777" stroke-width="0.5">'
            f"<title>{escape(title_text)}</title>"
            f"</polygon>"
        )


def _svg_triangle_frame(layout: _Layout) -> str:
    x_left, y_left = _triangle_xy(0, 0, layout)
    x_right, y_right = _triangle_xy(100, 0, layout)
    x_top, y_top = _triangle_xy(0, 100, layout)
    return (
        f'<polygon points="{x_left:.1f},{y_left:.1f} '
        f'{x_right:.1f},{y_right:.1f} {x_top:.1f},{y_top:.1f}" '
        f'fill="none" stroke="{_FRAME_COLOR}" stroke-width="1.5"/>'
    )


def _svg_axis_labels(c: Classification, layout: _Layout, loc: str) -> Iterable[str]:
    """Axis names and 0/100 markers at each corner."""
    del c, loc  # unused; could later be used for axis-specific labels
    x_left, y_left = _triangle_xy(0, 0, layout)
    x_right, y_right = _triangle_xy(100, 0, layout)
    x_top, y_top = _triangle_xy(0, 100, layout)
    yield (
        f'<text x="{(x_left + x_right) / 2:.0f}" y="{y_left + 28:.0f}" '
        f'text-anchor="middle" font-size="13" font-weight="600" '
        f'fill="{_LABEL_COLOR}">% sand</text>'
    )
    yield (
        f'<text x="{(x_right + x_top) / 2 + 30:.0f}" '
        f'y="{(y_right + y_top) / 2:.0f}" font-size="13" '
        f'font-weight="600" fill="{_LABEL_COLOR}" '
        f'transform="rotate(60 {(x_right + x_top) / 2 + 30:.0f} '
        f'{(y_right + y_top) / 2:.0f})">% clay</text>'
    )
    yield (
        f'<text x="{(x_left + x_top) / 2 - 30:.0f}" '
        f'y="{(y_left + y_top) / 2:.0f}" font-size="13" '
        f'font-weight="600" fill="{_LABEL_COLOR}" '
        f'text-anchor="end" '
        f'transform="rotate(-60 {(x_left + x_top) / 2 - 30:.0f} '
        f'{(y_left + y_top) / 2:.0f})">% silt</text>'
    )


def _svg_points(
    points: Sequence[Sequence[float]],
    labels: Sequence[str] | None,
    layout: _Layout,
) -> str:
    parts: list[str] = ['<g class="points">']
    for i, pt in enumerate(points):
        if len(pt) != 2:
            raise PedotriError(f"2-D diagram expects (sand, clay) points; got {pt!r}.")
        sand, clay = float(pt[0]), float(pt[1])
        x, y = _triangle_xy(sand, clay, layout)
        label = (labels[i] if labels else "") or ""
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
            f'fill="{_POINT_COLOR}" stroke="white" stroke-width="1.5">'
            f"<title>sand={sand:g}, clay={clay:g}"
            f"{(' — ' + label) if label else ''}</title>"
            f"</circle>"
        )
        if label:
            parts.append(
                f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="11" '
                f'fill="{_LABEL_COLOR}">{escape(label)}</text>'
            )
    parts.append("</g>")
    return "".join(parts)


def _svg_legend(c: Classification, layout: _Layout, loc: str, *, x: float) -> str:
    y = layout.tri_y
    parts: list[str] = []
    parts.append(
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="13" font-weight="600" '
        f'fill="{_LABEL_COLOR}">{escape(c.name(loc))}</text>'
    )
    row_h = 18
    line_y = y + 18
    for cls in c.classes:
        color = _color_for(cls)
        parts.append(
            f'<rect x="{x:.0f}" y="{line_y - 11:.0f}" width="13" '
            f'height="13" fill="{color}" fill-opacity="0.7" '
            f'stroke="#777" stroke-width="0.5"/>'
        )
        text = f"{cls.key} — {cls.name(loc)}"
        parts.append(
            f'<text x="{x + 20:.0f}" y="{line_y:.0f}" font-size="11" '
            f'fill="{_LABEL_COLOR}">{escape(text)}</text>'
        )
        line_y += row_h
    return "<g>" + "".join(parts) + "</g>"


# --- 1-D axis bar --------------------------------------------------------


def _render_axis(
    c: Classification,
    *,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[str] | None,
    locale: str | None,
    title: str | None,
    width: int,
    show_legend: bool,
) -> str:
    loc = locale or c.default_locale
    margin = 60
    title_h = 40 if title else 12
    bar_w = width - 2 * margin
    bar_h = 50
    legend_h = 22 * len(c.classes) if show_legend else 0
    total_h = title_h + bar_h + 80 + legend_h + margin

    layout = _Layout(
        width=float(width),
        height=float(total_h),
        tri_x=float(margin),
        tri_y=float(margin + title_h),
        tri_w=float(bar_w),
        tri_h=float(bar_h),
    )

    parts: list[str] = [_svg_header(layout)]
    if title:
        parts.append(_svg_title(title, layout))

    # Determine domain from class intervals.
    intervals = [cls.interval for cls in c.classes if cls.interval is not None]
    lo = min(iv[0] for iv in intervals)
    hi = max(iv[1] for iv in intervals)

    def x_of(v: float) -> float:
        return layout.tri_x + (v - lo) / (hi - lo) * layout.tri_w

    bar_top = layout.tri_y
    bar_bot = layout.tri_y + layout.tri_h

    for cls in c.classes:
        assert cls.interval is not None
        x1 = x_of(cls.interval[0])
        x2 = x_of(cls.interval[1])
        color = _color_for(cls)
        parts.append(
            f'<rect x="{x1:.1f}" y="{bar_top:.1f}" '
            f'width="{x2 - x1:.1f}" height="{layout.tri_h:.1f}" '
            f'fill="{color}" fill-opacity="0.7" stroke="#777" '
            f'stroke-width="0.5">'
            f"<title>{escape(cls.key)} — {escape(cls.name(loc))}: "
            f"[{cls.interval[0]:g}, {cls.interval[1]:g})</title>"
            f"</rect>"
        )
        # Tick at the lower edge of each interval.
        parts.append(
            f'<line x1="{x1:.1f}" y1="{bar_bot:.1f}" '
            f'x2="{x1:.1f}" y2="{bar_bot + 5:.1f}" '
            f'stroke="{_FRAME_COLOR}" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{x1:.1f}" y="{bar_bot + 18:.1f}" '
            f'font-size="10" text-anchor="middle" '
            f'fill="{_LABEL_COLOR}">{cls.interval[0]:g}</text>'
        )

    # Final tick at the right edge.
    parts.append(
        f'<line x1="{x_of(hi):.1f}" y1="{bar_bot:.1f}" '
        f'x2="{x_of(hi):.1f}" y2="{bar_bot + 5:.1f}" '
        f'stroke="{_FRAME_COLOR}" stroke-width="0.8"/>'
    )
    parts.append(
        f'<text x="{x_of(hi):.1f}" y="{bar_bot + 18:.1f}" '
        f'font-size="10" text-anchor="middle" '
        f'fill="{_LABEL_COLOR}">{hi:g}</text>'
    )

    # Axis label.
    parts.append(
        f'<text x="{layout.width / 2:.0f}" y="{bar_bot + 38:.0f}" '
        f'text-anchor="middle" font-size="13" font-weight="600" '
        f'fill="{_LABEL_COLOR}">% {c.axes[0]}</text>'
    )

    if points is not None:
        for i, pt in enumerate(points):
            v = float(pt[0]) if len(pt) >= 1 else float(pt)  # type: ignore[arg-type]
            x = x_of(v)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{(bar_top + bar_bot) / 2:.1f}" '
                f'r="5" fill="{_POINT_COLOR}" stroke="white" '
                f'stroke-width="1.5"/>'
            )
            label = (point_labels[i] if point_labels else "") or ""
            if label:
                parts.append(
                    f'<text x="{x:.1f}" y="{bar_top - 6:.1f}" '
                    f'font-size="11" text-anchor="middle" '
                    f'fill="{_LABEL_COLOR}">{escape(label)}</text>'
                )

    if show_legend:
        legend_y = bar_bot + 60
        parts.append(
            f'<text x="{layout.tri_x:.0f}" y="{legend_y:.0f}" '
            f'font-size="13" font-weight="600" '
            f'fill="{_LABEL_COLOR}">{escape(c.name(loc))}</text>'
        )
        line_y = legend_y + 20
        for cls in c.classes:
            color = _color_for(cls)
            parts.append(
                f'<rect x="{layout.tri_x:.0f}" y="{line_y - 11:.0f}" '
                f'width="13" height="13" fill="{color}" '
                f'fill-opacity="0.7" stroke="#777" stroke-width="0.5"/>'
            )
            assert cls.interval is not None
            text = f"{cls.key} — {cls.name(loc)} [{cls.interval[0]:g}, {cls.interval[1]:g})"
            parts.append(
                f'<text x="{layout.tri_x + 20:.0f}" y="{line_y:.0f}" '
                f'font-size="11" fill="{_LABEL_COLOR}">{escape(text)}</text>'
            )
            line_y += 22

    parts.append("</svg>")
    return "\n".join(parts)


# --- Helpers -------------------------------------------------------------


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="0.5"/>'
    )


def _color_for(cls: TextureClass) -> str:
    return _GROUP_COLORS.get(cls.group or "", _DEFAULT_COLOR)
