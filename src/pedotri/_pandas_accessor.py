"""pandas DataFrame accessor: ``df.soil.classify(...)``.

Auto-registered when ``import pedotri`` is called *and* pandas is
importable. With ``pip install pedotri[pandas]`` this gives users a
fluent way to classify a column of soil samples without having to pull
arrays out by hand.

Usage::

    import pandas as pd
    import pedotri  # noqa: F401 — side-effect import registers the accessor

    df = pd.DataFrame({"sand": [60, 20], "clay": [10, 50]})
    df["texture"] = df.soil.classify("USDA")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

import pedotri
from pedotri.registry import get_classification

if TYPE_CHECKING:
    from pedotri.schema import Classification


@pd.api.extensions.register_dataframe_accessor("soil")
class SoilAccessor:
    """``df.soil`` namespace.

    The accessor resolves each axis of the chosen classification to a
    column in the DataFrame, defaulting to the axis name (``"sand"``,
    ``"clay"``, ``"physical_clay"``, ...) and accepting an explicit
    override via the ``columns`` argument.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def classify(
        self,
        classification: str | Classification,
        *,
        columns: dict[str, str] | None = None,
        locale: str | None = None,
        detailed: bool = False,
    ) -> pd.Series:
        """Classify every row of the DataFrame.

        Args:
            classification: Classification key or instance.
            columns: Optional mapping from axis name → DataFrame column
                name. By default each axis is read from a column with
                the same name.
            locale: Locale tag for class names; ``None`` returns keys.
            detailed: When ``True``, returns a Series of
                :class:`~pedotri.ClassifyResult` instances.

        Returns:
            Pandas Series indexed like the source DataFrame.
        """
        c = (
            get_classification(classification)
            if isinstance(classification, str)
            else classification
        )
        col_map = columns or {axis: axis for axis in c.axes}
        axis_arrays: dict[str, Any] = {}
        for axis in c.axes:
            col_name = col_map.get(axis, axis)
            if col_name not in self._df.columns:
                raise KeyError(
                    f"DataFrame is missing column {col_name!r} "
                    f"(axis {axis!r}). Provide it or pass "
                    f"columns={{{axis!r}: <your_column>}}."
                )
            axis_arrays[axis] = self._df[col_name].to_numpy()

        result = pedotri.classify(classification=c, locale=locale, detailed=detailed, **axis_arrays)
        return pd.Series(result, index=self._df.index, name="texture")
