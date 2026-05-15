"""Tests for the optional matplotlib and plotly backends.

These tests only run if the relevant extra is installed; the dev group
in ``pyproject.toml`` pulls both in for CI.
"""

from __future__ import annotations

import pytest

import pedotri

pytest.importorskip("matplotlib")
pytest.importorskip("plotly.graph_objects")

import matplotlib
import plotly.graph_objects as go
from matplotlib.figure import Figure

import pedotri.plot
from pedotri.errors import PedotriError
from pedotri.plot import render_mpl, render_plotly

_ = matplotlib  # keep the import alive for matplotlib's side effects


# --- matplotlib backend --------------------------------------------------


def test_render_mpl_returns_figure() -> None:
    fig = render_mpl("USDA")
    assert isinstance(fig, Figure)


def test_render_mpl_with_overlay_points() -> None:
    fig = render_mpl("USDA", points=[(40, 25), (60, 10)], point_labels=["A", "B"])
    ax = fig.axes[0]
    annotations = [child.get_text() for child in ax.texts]
    assert "A" in annotations
    assert "B" in annotations


def test_render_mpl_kachinsky_one_dimensional() -> None:
    fig = render_mpl("KACHINSKY", title="K test", points=[(35,)])
    ax = fig.axes[0]
    assert ax.get_ylim() == (0.0, 1.0)


def test_render_mpl_localized_legend() -> None:
    fig = render_mpl("USDA", locale="fr")
    legend = fig.axes[0].get_legend()
    assert legend is not None


def test_render_mpl_unknown_classification_raises() -> None:
    with pytest.raises(pedotri.UnknownClassificationError):
        render_mpl("NOT_REAL")


# --- plotly backend ------------------------------------------------------


def test_render_plotly_returns_figure() -> None:
    fig = render_plotly("USDA")
    assert isinstance(fig, go.Figure)


def test_render_plotly_uses_ternary_layout() -> None:
    fig = render_plotly("USDA", title="t")
    assert fig.layout.ternary is not None


def test_render_plotly_includes_overlay_points() -> None:
    fig = render_plotly("USDA", points=[(40, 25)], point_labels=["sample"])
    traces = list(fig.data)
    assert len(traces) > 0
    assert traces[-1].name == "samples"


def test_render_plotly_kachinsky_one_dimensional() -> None:
    fig = render_plotly("KACHINSKY", title="K test")
    assert fig.layout.xaxis.title.text == "% physical_clay"


def test_render_plotly_unknown_classification_raises() -> None:
    with pytest.raises(pedotri.UnknownClassificationError):
        render_plotly("NOT_REAL")


def test_render_plotly_kachinsky_with_points() -> None:
    """1-D classifications accept points as either scalars or (x,) pairs and
    render them as overlay markers on the horizontal banded axis."""
    fig = render_plotly("KACHINSKY", points=[(35,), (12,)], point_labels=["A", "B"])
    traces = list(fig.data)
    assert traces[-1].name == "samples"
    assert list(traces[-1].x) == [35.0, 12.0]


def test_render_plotly_rejects_2d_point_with_wrong_arity() -> None:
    with pytest.raises(PedotriError, match="2-D diagram"):
        render_plotly("USDA", points=[(40, 25, 35)])


def test_plot_module_getattr_rejects_unknown_attribute() -> None:
    """``pedotri.plot.__getattr__`` lazily imports recognised backends and
    raises AttributeError for anything else."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = pedotri.plot.does_not_exist
