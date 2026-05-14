"""Texture-diagram rendering.

The built-in :func:`render_svg` returns a standalone SVG string with no
external dependencies — suitable for saving to disk, embedding in HTML,
or displaying inline in Jupyter notebooks via the
:class:`TextureDiagram` wrapper.

Optional backends are exposed under the same subpackage when the
relevant extras are installed:

- ``pedotri.plot.mpl`` (``pip install pedotri[matplotlib]``)
- ``pedotri.plot.plotly`` (``pip install pedotri[plotly]``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pedotri.plot.svg import TextureDiagram, render_svg

if TYPE_CHECKING:
    # Re-export the lazy-loaded backends for type checkers without
    # forcing the optional dependency at import time.
    from pedotri.plot.mpl import render_mpl
    from pedotri.plot.plotly_backend import render_plotly


def __getattr__(name: str) -> Any:
    """Lazy import for optional backends.

    Accessing :func:`render_mpl` or :func:`render_plotly` triggers the
    backend module load (and the underlying optional dependency); the
    error message points at the right extras-install command when the
    dependency is missing.
    """
    if name == "render_mpl":
        from pedotri.plot.mpl import render_mpl

        return render_mpl
    if name == "render_plotly":
        from pedotri.plot.plotly_backend import render_plotly

        return render_plotly
    raise AttributeError(f"module 'pedotri.plot' has no attribute {name!r}")


__all__ = ["TextureDiagram", "render_mpl", "render_plotly", "render_svg"]
