"""Tests for the MCP server.

These exercise the registered handlers without spinning up a real
stdio transport. The handlers themselves delegate to pedotri.ai, so
the heavy logic is covered by tests/test_ai.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

pytest.importorskip("mcp")

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListToolsRequest,
)

from pedotri.mcp_server import build_server

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
    we want for LLM tool-use loops."""
    server = build_server()
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="classify_soil",
            arguments={"sand": 60, "clay": 200, "classification": "USDA"},
        ),
    )
    result: Any = _await(handler(request))
    assert result.root.isError is True
    assert "100" in result.root.content[0].text


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
