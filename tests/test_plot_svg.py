"""Tests for the pure-SVG renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pedotri
from pedotri.plot import TextureDiagram, render_svg

if TYPE_CHECKING:
    import pathlib


def _is_well_formed_svg(s: str) -> bool:
    return s.strip().startswith("<svg") and s.strip().endswith("</svg>")


def test_render_usda_returns_svg_string() -> None:
    svg = render_svg("USDA")
    assert _is_well_formed_svg(svg)
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg


def test_render_includes_all_class_polygons() -> None:
    svg = render_svg("USDA")
    # Each class should appear in a <title> tooltip on its polygon.
    for cls in pedotri.get_classification("USDA").classes:
        assert cls.key in svg


def test_render_localized_legend_names() -> None:
    svg = render_svg("USDA", locale="fr")
    assert "argile" in svg
    assert "sable" in svg


def test_render_with_overlay_points() -> None:
    svg = render_svg(
        "USDA",
        points=[(13, 50), (60, 15)],
        point_labels=["A", "B"],
    )
    assert "<circle" in svg
    assert ">A<" in svg
    assert ">B<" in svg


def test_render_title_is_embedded() -> None:
    svg = render_svg("USDA", title="My Diagram")
    assert "My Diagram" in svg


def test_render_ka5_includes_31_polygons() -> None:
    svg = render_svg("KA5", show_legend=False, show_grid=False)
    # Each class produces a <polygon> + a frame polygon; count must be ≥ 31.
    assert svg.count("<polygon") >= 31


def test_render_kachinsky_uses_axis_bar() -> None:
    """1-D classifications render as a horizontal bar, not a triangle."""
    svg = render_svg("KACHINSKY", title="Kachinsky")
    assert _is_well_formed_svg(svg)
    # Bar segments are rendered as <rect> (not <polygon>) for 1-D.
    # The Kachinsky classification has 9 classes plus the background rect.
    assert svg.count("<rect") >= 10
    # And there is no triangle frame for 1-D.
    assert "<polygon" not in svg


def test_render_kachinsky_with_overlay_point() -> None:
    svg = render_svg("KACHINSKY", points=[(35,)], point_labels=["sample"])
    assert "sample" in svg


def test_texture_diagram_repr_svg() -> None:
    d = TextureDiagram("USDA", title="t")
    out = d._repr_svg_()
    assert _is_well_formed_svg(out)


def test_texture_diagram_save_writes_file(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "out.svg"
    TextureDiagram("USDA").save(p)
    content = p.read_text()
    assert _is_well_formed_svg(content)


def test_render_show_grid_false_omits_grid() -> None:
    with_grid = render_svg("USDA", show_grid=True)
    without_grid = render_svg("USDA", show_grid=False)
    assert with_grid.count("<line") > without_grid.count("<line")


def test_render_show_legend_false_omits_legend() -> None:
    with_legend = render_svg("USDA", show_legend=True)
    without_legend = render_svg("USDA", show_legend=False)
    # Legend includes a label row per class
    usda = pedotri.get_classification("USDA")
    sample_class_name = usda.classes[0].name("en")
    # Class name appears as polygon tooltip text either way, but the
    # legend adds extra occurrences; legend-disabled version should have
    # strictly fewer occurrences.
    assert with_legend.count(sample_class_name) > without_legend.count(sample_class_name)


def test_render_unknown_classification_raises() -> None:
    with pytest.raises(pedotri.UnknownClassificationError):
        render_svg("NOT_A_REAL_KEY")
