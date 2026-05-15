"""pedotri — modern soil texture classification and pedotransfer functions.

Quick reference::

    import pedotri

    pedotri.classify(13, 50, "USDA")
    # 'clay'

    pedotri.classify([13, 45], [50, 24], "FAO")
    # ['fine', 'medium']

    pedotri.classify(13, 50, "USDA", locale="fr")
    # 'argile'

    pedotri.classify(13, 50, "USDA", detailed=True)
    # ClassifyResult(key='clay', name='clay', group='fine', ...)
"""

from __future__ import annotations

import contextlib as _contextlib

from pedotri import audit, dem, grid, sources, uncertainty, zonal
from pedotri.classifier import ClassifyResult, classify, classify_all
from pedotri.errors import (
    ClassificationError,
    InvalidInputError,
    PedotriError,
    UnknownClassificationError,
)
from pedotri.loader import load_classification
from pedotri.plot import TextureDiagram, render_svg
from pedotri.registry import (
    get_classification,
    list_classifications,
    register_classification,
    unregister_classification,
)
from pedotri.schema import Classification, TextureClass
from pedotri.uncertainty import Quantiles

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("pedotri")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0+unknown"
del _pkg_version, PackageNotFoundError


# Optional DataFrame accessors are registered as a side effect of
# importing pedotri whenever pandas / polars are available, so users
# can write ``df.soil.classify(...)`` without an explicit import.
with _contextlib.suppress(ImportError):
    from pedotri import _pandas_accessor  # noqa: F401

with _contextlib.suppress(ImportError):
    from pedotri import _polars_namespace  # noqa: F401

__all__ = [
    "Classification",
    "ClassificationError",
    "ClassifyResult",
    "InvalidInputError",
    "PedotriError",
    "Quantiles",
    "TextureClass",
    "TextureDiagram",
    "UnknownClassificationError",
    "__version__",
    "audit",
    "classify",
    "classify_all",
    "dem",
    "get_classification",
    "grid",
    "list_classifications",
    "load_classification",
    "register_classification",
    "render_svg",
    "sources",
    "uncertainty",
    "unregister_classification",
    "zonal",
]
