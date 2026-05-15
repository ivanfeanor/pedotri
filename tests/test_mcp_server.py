"""Tests for the MCP server.

These exercise the registered handlers without spinning up a real
stdio transport. The handlers themselves delegate to pedotri.ai, so
the heavy logic is covered by tests/test_ai.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

pytest.importorskip("mcp")

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ImageContent,
    ListToolsRequest,
    TextContent,
)

from pedotri.mcp_server import _wrap_render_result, build_server

if TYPE_CHECKING:
    from collections.abc import Awaitable

T = TypeVar("T")


def _await(awaitable: Awaitable[T]) -> T:
    """asyncio.run() wrapper that satisfies mypy's strict signature."""

    async def runner() -> T:
        return await awaitable

    return asyncio.run(runner())


def test_build_server_registers_pedotri_handlers() -> None:
    server = build_server()
    assert server.name == "pedotri"
    assert "soil texture" in (server.instructions or "")


def test_list_tools_returns_full_schema_set() -> None:
    """The MCP list_tools handler should report every pedotri.ai tool."""
    server = build_server()
    handler = server.request_handlers[ListToolsRequest]
    result: Any = _await(handler(ListToolsRequest(method="tools/list")))
    tool_names = {t.name for t in result.root.tools}
    assert tool_names == {
        "classify_soil",
        "classify_soil_1d",
        "list_classifications",
        "classification_info",
        "saxton_rawls",
        "wosten",
        "convert_particle_size",
        "render_diagram",
    }


def test_call_tool_classify_soil_returns_json_text() -> None:
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="classify_soil",
            arguments={"sand": 60, "clay": 20, "classification": "USDA"},
        ),
    )
    result: Any = _await(handler(request))
    contents = result.root.content
    assert len(contents) == 1
    payload = json.loads(contents[0].text)
    assert payload["key"] == "sandy_clay_loam"


def test_call_tool_schema_validation_returns_is_error() -> None:
    """Inputs that violate the JSON Schema are caught by the MCP layer
    and surfaced as isError=True with a plain-text message — before
    our handler is even called. This is exactly the layered validation
    we want for LLM tool-use loops.

    A negative value violates ``minimum: 0`` on every fraction field.
    (We no longer cap at 100 because the ``units`` parameter accepts
    g/kg inputs up to 1000.)
    """
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="classify_soil",
            arguments={"sand": 60, "clay": -5, "classification": "USDA"},
        ),
    )
    result: Any = _await(handler(request))
    assert result.root.isError is True
    assert "less than" in result.root.content[0].text or "minimum" in result.root.content[0].text


def test_call_tool_render_diagram_returns_image_content() -> None:
    """render_diagram emits ImageContent so MCP clients render the
    diagram inline rather than printing markup. PNG when matplotlib is
    installed (bundled in the [mcp] extra), SVG as a graceful fallback
    when it isn't.
    """
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="render_diagram",
            arguments={"classification": "USDA"},
        ),
    )
    result: Any = _await(handler(request))
    contents = result.root.content
    # First block is the image, second is a minimal text companion.
    assert len(contents) == 2
    assert isinstance(contents[0], ImageContent)
    mime = contents[0].mimeType
    assert mime in {"image/png", "image/svg+xml"}
    decoded = base64.standard_b64decode(contents[0].data)
    if mime == "image/png":
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    else:
        assert decoded.startswith(b"<svg")
        assert decoded.rstrip().endswith(b"</svg>")
    # The text companion describes what was rendered but does not
    # re-include the image bytes.
    assert isinstance(contents[1], TextContent)
    companion = json.loads(contents[1].text)
    assert companion["format"] in {"png", "svg+xml"}
    assert companion["rendered_inline"] is True
    assert companion["classification"] == "USDA"


def test_call_tool_render_diagram_prefers_png_when_matplotlib_available() -> None:
    """When matplotlib is installed (it is in the dev group + [mcp] extra),
    the MCP server should emit PNG, which renders inline reliably in
    every MCP client we know of."""
    pytest.importorskip("matplotlib")
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="render_diagram",
            arguments={"classification": "USDA"},
        ),
    )
    result: Any = _await(handler(request))
    assert result.root.content[0].mimeType == "image/png"


def test_call_tool_render_diagram_honours_explicit_svg_format() -> None:
    """Callers can opt into SVG with format='svg' for clients that
    render vector content (text editors, custom apps, future builds)."""
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="render_diagram",
            arguments={"classification": "USDA", "format": "svg"},
        ),
    )
    result: Any = _await(handler(request))
    assert result.root.content[0].mimeType == "image/svg+xml"
    decoded = base64.standard_b64decode(result.root.content[0].data)
    assert decoded.startswith(b"<svg")


def test_call_tool_render_diagram_error_still_returns_json_envelope() -> None:
    """When render_diagram errors (unknown classification), the response
    is the standard JSON envelope, not an empty/broken ImageContent."""
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="render_diagram",
            arguments={"classification": "DOES_NOT_EXIST"},
        ),
    )
    result: Any = _await(handler(request))
    contents = result.root.content
    assert len(contents) == 1
    assert isinstance(contents[0], TextContent)
    payload = json.loads(contents[0].text)
    assert payload["error"] == "UnknownClassificationError"


def test_wrap_render_result_returns_none_for_non_string_content() -> None:
    assert _wrap_render_result({"format": "png", "content": 123}) is None


def test_wrap_render_result_returns_none_for_unknown_format() -> None:
    assert _wrap_render_result({"format": "jpeg", "content": "abc"}) is None


def test_call_tool_handler_error_returns_json_envelope() -> None:
    """Errors raised inside our handler (after schema validation) come
    back as a JSON envelope embedded in the text content."""
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    # Unknown classification is *not* a schema violation (we don't
    # enumerate every key in the schema), so it reaches our handler
    # and surfaces via pedotri.ai.run's envelope.
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="classify_soil",
            arguments={"sand": 60, "clay": 20, "classification": "DOES_NOT_EXIST"},
        ),
    )
    result: Any = _await(handler(request))
    payload = json.loads(result.root.content[0].text)
    assert payload["error"] == "UnknownClassificationError"
    assert "USDA" in payload["available"]
