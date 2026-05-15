"""Model Context Protocol server for pedotri.

Run via ``pedotri-mcp`` (registered as a console script when pedotri
is installed with the ``[mcp]`` extra) to expose pedotri to MCP-aware
clients such as Claude Desktop, Cursor, and any Anthropic-SDK
application that speaks MCP.

The tool surface mirrors :mod:`pedotri.ai`: classification (2-D + 1-D),
discovery, pedotransfer functions, particle-size conversion, and SVG
rendering. Errors are returned as structured JSON-text content so the
calling model can self-correct.

To install for Claude Desktop, add to your config (usually
``~/Library/Application Support/Claude/claude_desktop_config.json`` on
macOS)::

    {"mcpServers": {"pedotri": {"command": "pedotri-mcp"}}}

Claude will then see eight pedotri tools at the start of every
conversation and can call them in tool-use loops.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pedotri.ai

if TYPE_CHECKING:
    from mcp.server import Server
    from mcp.types import TextContent, Tool


_INSTRUCTIONS = (
    "pedotri is a soil texture classification and pedotransfer toolkit. "
    "Useful for any question that starts from sand / silt / clay "
    "percentages (or physical-clay percentage for Kachinsky): you can "
    "classify the soil under any of ~14 worldwide systems, estimate "
    "water-retention properties from texture, convert between "
    "particle-size standards, and render texture diagrams as SVG. Call "
    "list_classifications and classification_info to discover what is "
    "available before classifying."
)


def build_server() -> Server:
    """Construct the low-level MCP server with pedotri tools registered.

    Importing :mod:`mcp` is deferred to this function so a bare
    ``pip install pedotri`` doesn't pull in the SDK; the import lives
    behind the ``[mcp]`` extra.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ModuleNotFoundError as exc:  # pragma: no cover - tested elsewhere
        raise ModuleNotFoundError(
            "pedotri.mcp_server requires the 'mcp' SDK. Install with "
            "`pip install pedotri[mcp]` or `pip install mcp`."
        ) from exc

    server: Server = Server("pedotri", instructions=_INSTRUCTIONS)
    schemas = {s["name"]: s for s in pedotri.ai.tool_schemas()}

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=schema["description"],
                inputSchema=schema["input_schema"],
            )
            for name, schema in schemas.items()
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = pedotri.ai.run(name, arguments)
        text = json.dumps(result, ensure_ascii=False)
        return [TextContent(type="text", text=text)]

    return server


async def _run_stdio() -> None:
    """Run the server over stdio (the default Claude-Desktop transport)."""
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for the ``pedotri-mcp`` console script."""
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
