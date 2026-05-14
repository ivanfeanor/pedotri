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

from pedotri.classifier import ClassifyResult, classify
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

__version__ = "0.1.0"


# Optional DataFrame accessors are registered as a side effect of
# importing pedotri whenever pandas / polars are available, so users
# can write ``df.soil.classify(...)`` without an explicit import.
import contextlib as _contextlib

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
    "TextureClass",
    "TextureDiagram",
    "UnknownClassificationError",
    "__version__",
    "classify",
    "get_classification",
    "list_classifications",
    "load_classification",
    "register_classification",
    "render_svg",
    "unregister_classification",
]
