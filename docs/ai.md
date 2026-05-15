# LLM tools and MCP server

pedotri ships first-class tool wrappers and a Model Context Protocol server so language models can call its functions directly.

## What's available

Eight tools cover the user-facing surface:

| Tool                    | What it does                                        |
|-------------------------|-----------------------------------------------------|
| `classify_soil`         | 2-D sand-clay classification (USDA, FAO, …)         |
| `classify_soil_1d`      | Single-axis classification (Kachinsky)              |
| `list_classifications`  | Discovery of every registered classification        |
| `classification_info`   | Axes, classes, reference, locales for one classifier |
| `saxton_rawls`          | Water retention + saturated conductivity (PTF)      |
| `wosten`                | Mualem-van Genuchten parameters (HYPRES PTF)        |
| `convert_particle_size` | USDA / ISSS / KA5 cutoff conversion                 |
| `render_diagram`        | Standalone SVG texture diagram                      |

All tools accept JSON inputs and return JSON results. Errors come back as structured envelopes (`{"error": "InvalidInputError", "message": "..."}`) — never as raised exceptions — so an agent loop can recover without parsing English error strings.

## Anthropic / OpenAI tool use

`pedotri.ai.tool_schemas()` returns Anthropic-shaped schemas that drop directly into the Messages API:

```python
import pedotri.ai
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=pedotri.ai.tool_schemas(),
    messages=[
        {
            "role": "user",
            "content": "What's the USDA texture of 60% sand, 20% clay?",
        }
    ],
)

for block in response.content:
    if block.type == "tool_use":
        result = pedotri.ai.run(block.name, block.input)
        # → {"key": "sandy_clay_loam", "name": "sandy clay loam",
        #    "group": "moderately_fine", "parent": null, "distance": 0.0}
```

The same schemas work with the OpenAI Responses / Chat Completions APIs after the format conversion their SDK does for you. Tool names and input schemas are stable across the two formats.

## Claude Desktop (MCP)

Install the MCP extra and add pedotri to your Claude Desktop config:

```bash
pip install "pedotri[mcp]"
```

On macOS, edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pedotri": {
      "command": "pedotri-mcp"
    }
  }
}
```

(On Windows, the file is at `%APPDATA%\Claude\claude_desktop_config.json`.)

Restart Claude Desktop. The eight pedotri tools appear in every conversation and Claude can call them while reasoning about soil samples.

## Layered validation

The MCP server validates inputs **twice**:

1. **At the protocol layer**, against the tool's JSON Schema. Out-of-range inputs (negative percentages, missing required fields) are rejected before pedotri code runs. The client sees an `isError=true` content block with a plain-text reason.
2. **Inside pedotri**, against the function's own preconditions (unknown classification key, NaN, polygon-coverage gap). These surface as JSON envelopes in the tool's text content.

Both paths give the model enough information to fix its inputs and retry without human intervention.

## Direct Python dispatcher

For non-MCP integrations (a custom agent loop, an internal HTTP service, a Jupyter notebook driving an LLM), the dispatcher is directly callable:

```python
import pedotri.ai

# Schemas
schemas = pedotri.ai.tool_schemas()  # list[dict]

# Dispatch
result = pedotri.ai.run(
    "saxton_rawls",
    {"sand": 40, "clay": 20, "organic_matter": 2.0},
)
# → {"wilting_point": 0.122, "field_capacity": 0.253, ...}

# Error envelope
err = pedotri.ai.run(
    "classify_soil",
    {"sand": 60, "clay": 20, "classification": "NOPE"},
)
# → {"error": "UnknownClassificationError", "message": "...",
#    "available": ["AVERY", "CHINA", ...]}
```

The dispatcher never raises for user-facing errors — caller code can treat every response uniformly.

## Adding your own tools

To expose a custom classification through the same MCP server, register it on import and the `list_classifications` / `classify_soil` tools pick it up automatically:

```python
import pedotri

pedotri.register_classification("path/to/my_scheme.toml")
# MCP clients now see MY_SCHEME in list_classifications results.
```

For classifications shipped as a third-party package, declare an entry point in the `pedotri.classifications` group of your `pyproject.toml` — see the [Custom classifications guide](classifications/custom.md) for details. pedotri-mcp picks them up automatically on next start.
