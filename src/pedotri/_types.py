"""Internal type aliases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.float64]

# IETF-style locale tag, e.g. "en", "fr", "fr-FR", "ru-RU"
Locale: TypeAlias = str

# Array-like inputs *excluding* bare scalars (those are handled by separate
# overloads so the return type can be narrowed to a single string).
ArrayLike: TypeAlias = Union[
    Sequence[float],
    Sequence[int],
    "NDArray[np.floating]",
    "NDArray[np.integer]",
]

# Either a scalar percentage or an array-like of percentages.
ScalarOrArrayLike: TypeAlias = float | int | ArrayLike
