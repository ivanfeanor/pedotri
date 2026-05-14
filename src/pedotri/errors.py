"""Exception hierarchy for pedotri."""

from __future__ import annotations


class PedotriError(Exception):
    """Base class for all pedotri exceptions."""


class ClassificationError(PedotriError):
    """Raised when a classification definition is invalid."""


class UnknownClassificationError(PedotriError, KeyError):
    """Raised when a requested classification key is not registered."""

    def __init__(self, key: str, available: list[str] | None = None) -> None:
        msg = f"Unknown classification: {key!r}."
        if available:
            msg += f" Available: {sorted(available)!r}."
        super().__init__(msg)
        self.key = key
        self.available = available or []


class InvalidInputError(PedotriError, ValueError):
    """Raised when input arrays are malformed (shape mismatch, NaNs, out of range, etc.)."""
