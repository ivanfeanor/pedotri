"""Tests for the pandas and polars DataFrame accessors."""

from __future__ import annotations

import pytest

import pedotri

pytest.importorskip("pandas")
pytest.importorskip("polars")

import pandas as pd
import polars as pl

# Reference textures used across tests (verified against pedotri.classify):
#   (60, 10) → sandy_loam       (silt = 30)
#   (60, 20) → sandy_clay_loam  (boundary; lookup-order winner)
#   (20, 50) → clay
#   (70, 5)  → sandy_loam


# --- pandas accessor -----------------------------------------------------


def test_pandas_accessor_classifies_2d() -> None:
    df = pd.DataFrame({"sand": [60, 20, 70], "clay": [10, 50, 5]})
    series = df.soil.classify("USDA")
    assert list(series) == ["sandy_loam", "clay", "sandy_loam"]
    assert list(series.index) == list(df.index)


def test_pandas_accessor_classifies_1d() -> None:
    df = pd.DataFrame({"physical_clay": [3, 35, 80]})
    series = df.soil.classify("KACHINSKY")
    assert list(series) == ["loose_sand", "medium_loam", "heavy_clay"]


def test_pandas_accessor_localized_names() -> None:
    df = pd.DataFrame({"sand": [60, 20], "clay": [10, 50]})
    series = df.soil.classify("USDA", locale="fr")
    assert list(series) == ["limon sableux", "argile"]


def test_pandas_accessor_column_remap() -> None:
    df = pd.DataFrame({"S": [60, 20], "C": [10, 50]})
    series = df.soil.classify("USDA", columns={"sand": "S", "clay": "C"})
    assert list(series) == ["sandy_loam", "clay"]


def test_pandas_accessor_missing_column_raises() -> None:
    df = pd.DataFrame({"sand": [60], "x": [10]})  # missing 'clay'
    with pytest.raises(KeyError, match="clay"):
        df.soil.classify("USDA")


def test_pandas_accessor_detailed_returns_results() -> None:
    df = pd.DataFrame({"sand": [60], "clay": [10]})
    series = df.soil.classify("USDA", detailed=True)
    result = series.iloc[0]
    assert isinstance(result, pedotri.ClassifyResult)
    assert result.key == "sandy_loam"


# --- polars accessor -----------------------------------------------------


def test_polars_accessor_classifies_2d() -> None:
    df = pl.DataFrame({"sand": [60, 20, 70], "clay": [10, 50, 5]})
    series = df.soil.classify("USDA")  # type: ignore[attr-defined]
    assert series.to_list() == ["sandy_loam", "clay", "sandy_loam"]


def test_polars_accessor_classifies_1d() -> None:
    df = pl.DataFrame({"physical_clay": [3, 35, 80]})
    series = df.soil.classify("KACHINSKY")  # type: ignore[attr-defined]
    assert series.to_list() == ["loose_sand", "medium_loam", "heavy_clay"]


def test_polars_accessor_localized() -> None:
    df = pl.DataFrame({"sand": [60, 20], "clay": [10, 50]})
    series = df.soil.classify("GEPPA", locale="fr")  # type: ignore[attr-defined]
    assert series.to_list() == ["limon sableux", "argile"]


def test_polars_accessor_column_remap() -> None:
    df = pl.DataFrame({"S": [60], "C": [10]})
    series = df.soil.classify(  # type: ignore[attr-defined]
        "USDA", columns={"sand": "S", "clay": "C"}
    )
    assert series.to_list() == ["sandy_loam"]


def test_polars_accessor_missing_column_raises() -> None:
    df = pl.DataFrame({"sand": [60], "x": [10]})
    with pytest.raises(KeyError, match="clay"):
        df.soil.classify("USDA")  # type: ignore[attr-defined]
