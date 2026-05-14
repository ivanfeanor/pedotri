"""Smoke tests — verify the package imports cleanly."""

from __future__ import annotations

import pedotri


def test_version() -> None:
    assert isinstance(pedotri.__version__, str)
    assert pedotri.__version__.count(".") >= 2
