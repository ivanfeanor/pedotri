"""polars DataFrame namespace: ``df.soil.classify(...)``.

Auto-registered when ``import pedotri`` is called *and* polars is
importable.

Usage::

    import polars as pl
    import pedotri  # noqa: F401

    df = pl.DataFrame({"sand": [60, 20], "clay": [10, 50]})
    df.with_columns(texture=df.soil.classify("USDA"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

import pedotri
from pedotri.registry import get_classification

if TYPE_CHECKING:
    from pedotri.schema import Classification


@pl.api.register_dataframe_namespace("soil")
class SoilNamespace:
    """``df.soil`` namespace for polars DataFrames."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def classify(
        self,
        classification: str | Classification,
        *,
        columns: dict[str, str] | None = None,
        locale: str | None = None,
        detailed: bool = False,
    ) -> pl.Series:
        """Classify every row of the DataFrame.

        Returns a polars :class:`Series` of class keys (or localized
        names, or :class:`ClassifyResult` instances) aligned to the
        source DataFrame.
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
        return pl.Series("texture", result)
