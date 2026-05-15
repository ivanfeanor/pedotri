"""Error-path tests for the TOML loader.

The happy-path loader tests live in ``test_schema.py``; this module
exercises the validation branches that reject malformed input.
"""

from __future__ import annotations

import pytest

from pedotri.errors import ClassificationError
from pedotri.loader import load_classification


def test_loader_accepts_pre_parsed_dict() -> None:
    """Covers the dict short-circuit in _to_dict."""
    data = {
        "meta": {"key": "TEST", "axes": ["sand", "clay"]},
        "class": [{"key": "a", "vertices": [[0, 0], [1, 0], [0, 1]]}],
    }
    c = load_classification(data)
    assert c.key == "TEST"


@pytest.mark.parametrize(
    ("toml_bytes", "match"),
    [
        # [meta].axes: not a list / wrong length / wrong element type
        (b'[meta]\nkey="X"\naxes="sand"\n[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]', "axes"),
        (b'[meta]\nkey="X"\naxes=[]\n[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]', "axes"),
        (b'[meta]\nkey="X"\naxes=[1,2]\n[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]', "axes"),
        # [meta].default_locale not a string
        (
            b'[meta]\nkey="X"\naxes=["sand","clay"]\ndefault_locale=42\n'
            b'[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]',
            "default_locale",
        ),
        # No [[class]] entries at all
        (b'[meta]\nkey="X"\naxes=["sand","clay"]', "at least one"),
        # [[class]] interval shape wrong
        (
            b'[meta]\nkey="X"\naxes=["physical_clay"]\n[[class]]\nkey="a"\ninterval=[1,2,3]',
            "interval",
        ),
        (
            b'[meta]\nkey="X"\naxes=["physical_clay"]\n[[class]]\nkey="a"\ninterval=["lo","hi"]',
            "interval",
        ),
        # Class has both vertices and interval
        (
            b'[meta]\nkey="X"\naxes=["sand","clay"]\n[[class]]\nkey="a"\n'
            b"vertices=[[0,0],[1,0],[0,1]]\ninterval=[0,10]",
            "exactly one",
        ),
        # Class has neither vertices nor interval
        (b'[meta]\nkey="X"\naxes=["sand","clay"]\n[[class]]\nkey="a"', "exactly one"),
        # vertices that pass the list check but fail make_vertices (1-D)
        (
            b'[meta]\nkey="X"\naxes=["sand","clay"]\n[[class]]\nkey="a"\nvertices=[0,0,1,1]',
            r"\(N, 2\)",
        ),
        # _require_str: missing / non-string [meta].key
        (b'[meta]\naxes=["sand","clay"]\n[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]', "key"),
        (
            b'[meta]\nkey=42\naxes=["sand","clay"]\n[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]',
            "key",
        ),
        # _optional_str: non-string optional field (group is optional but typed)
        (
            b'[meta]\nkey="X"\naxes=["sand","clay"]\n[[class]]\nkey="a"\n'
            b"group=1\nvertices=[[0,0],[1,0],[0,1]]",
            "string when present",
        ),
        # _coerce_names: not a table
        (
            b'[meta]\nkey="X"\naxes=["sand","clay"]\nnames="not-a-table"\n'
            b'[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]',
            "locale",
        ),
    ],
)
def test_loader_rejects_malformed_inputs(toml_bytes: bytes, match: str) -> None:
    with pytest.raises(ClassificationError, match=match):
        load_classification(toml_bytes)


def test_loader_rejects_non_string_name_value() -> None:
    """_coerce_names rejects entries where the value isn't a string.

    TOML can't directly encode a non-string locale *key*, so the
    string→string check fires on the value side.
    """
    toml = (
        b'[meta]\nkey="X"\naxes=["sand","clay"]\n'
        b"[meta.names]\nen=42\n"
        b'[[class]]\nkey="a"\nvertices=[[0,0],[1,0],[0,1]]'
    )
    with pytest.raises(ClassificationError, match="string"):
        load_classification(toml)


def test_loader_rejects_non_dict_class_entry() -> None:
    """[[class]] entries must be tables. TOML enforces this at parse time
    for the array-of-tables syntax, but the loader's own check guards
    against dict-source input that hand-builds the structure.
    """
    data = {
        "meta": {"key": "X", "axes": ["sand", "clay"]},
        "class": ["not-a-table"],
    }
    with pytest.raises(ClassificationError, match="table"):
        load_classification(data)
