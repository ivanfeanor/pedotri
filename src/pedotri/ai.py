"""Tool-shaped wrappers for LLM / agent use.

This module exposes the most useful pedotri operations as JSON-in /
JSON-out functions with Anthropic-style tool schemas attached. The
schemas drop directly into the ``tools=`` argument of the Anthropic
Messages API (or, after a thin format conversion, the OpenAI
Responses / Chat Completions API).

The same surface is what :mod:`pedotri.mcp_server` exposes to MCP
clients (Claude Desktop, Cursor, ...).

Quick usage::

    import pedotri.ai

    pedotri.ai.tool_schemas()
    # [{"name": "classify_soil", "description": "...", "input_schema": {...}},
    #  {"name": "list_classifications", ...}, ...]

    pedotri.ai.run("classify_soil", {"sand": 60, "clay": 20, "classification": "USDA"})
    # {"key": "sandy_clay_loam", "name": "sandy clay loam", ...}

    # Errors come back as structured envelopes, not raised:
    pedotri.ai.run("classify_soil", {"sand": 60, "clay": 200, "classification": "USDA"})
    # {"error": "InvalidInputError", "message": "'clay' must be a percentage in [0, 100]."}
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pedotri
from pedotri.errors import (
    ClassificationError,
    InvalidInputError,
    PedotriError,
    UnknownClassificationError,
)
from pedotri.plot import render_svg
from pedotri.psd import convert as _psd_convert
from pedotri.ptf import saxton_rawls, wosten

# --- Tool schemas --------------------------------------------------------


def tool_schemas() -> list[dict[str, Any]]:
    """Return the full list of pedotri tools as Anthropic-style schemas.

    Each entry has ``name``, ``description``, and ``input_schema``
    (a JSON Schema describing the tool's arguments). The schemas can
    be passed directly to ``anthropic.Anthropic().messages.create(...,
    tools=pedotri.ai.tool_schemas())``.
    """
    return [
        _classify_soil_schema(),
        _classify_soil_1d_schema(),
        _list_classifications_schema(),
        _classification_info_schema(),
        _saxton_rawls_schema(),
        _wosten_schema(),
        _convert_particle_size_schema(),
        _render_diagram_schema(),
    ]


def _classify_soil_schema() -> dict[str, Any]:
    return {
        "name": "classify_soil",
        "description": (
            "Classify a soil sample on the sand-silt-clay simplex. "
            "Use for any 2-D classification (USDA, FAO, GEPPA, KA5, etc.). "
            "For 1-D classifications like Kachinsky, use classify_soil_1d. "
            "Inputs are percentages; silt is implicit (silt = 100 - sand - clay)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Percent sand, in [0, 100].",
                },
                "clay": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Percent clay, in [0, 100].",
                },
                "classification": {
                    "type": "string",
                    "description": (
                        "Classification key, e.g. 'USDA', 'FAO', 'GEPPA', "
                        "'KA5', 'NORTHCOTE'. Call list_classifications "
                        "for the full set."
                    ),
                },
                "locale": {
                    "type": "string",
                    "description": (
                        "Optional locale tag for class names (e.g. 'fr', "
                        "'de', 'ru'). When omitted, the stable class key "
                        "is returned. Always returns localized name in "
                        "the 'name' field regardless."
                    ),
                },
            },
            "required": ["sand", "clay", "classification"],
        },
    }


def _classify_soil_1d_schema() -> dict[str, Any]:
    return {
        "name": "classify_soil_1d",
        "description": (
            "Classify a soil sample on a single axis. Used by 1-D "
            "classifications like KACHINSKY (physical clay <0.01 mm). "
            "Call classification_info first to learn which axis the "
            "classification expects."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Percent for the classification's axis (e.g. physical clay).",
                },
                "classification": {
                    "type": "string",
                    "description": (
                        "1-D classification key. Currently 'KACHINSKY' is "
                        "the only built-in 1-D classification."
                    ),
                },
                "locale": {"type": "string", "description": "Optional locale tag."},
            },
            "required": ["value", "classification"],
        },
    }


def _list_classifications_schema() -> dict[str, Any]:
    return {
        "name": "list_classifications",
        "description": (
            "List every registered classification key. Useful for "
            "discovery before calling classify_soil or "
            "classification_info."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


def _classification_info_schema() -> dict[str, Any]:
    return {
        "name": "classification_info",
        "description": (
            "Describe one classification: its axes, classes, reference, "
            "default locale, and per-class metadata. Use this before "
            "classify_soil if you don't know whether a system is 1-D or "
            "2-D, or to discover the localized names for a locale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "description": "Classification key (e.g. 'USDA').",
                },
                "locale": {
                    "type": "string",
                    "description": "Optional locale tag for class names.",
                },
            },
            "required": ["classification"],
        },
    }


def _saxton_rawls_schema() -> dict[str, Any]:
    return {
        "name": "saxton_rawls",
        "description": (
            "Estimate soil hydraulic properties (wilting point, field "
            "capacity, saturation, plant-available water, saturated "
            "hydraulic conductivity, bulk density, air-entry tension) "
            "from texture and organic matter via the Saxton & Rawls "
            "(2006) pedotransfer function. Use for any agronomy water-"
            "balance question that starts from sand/clay/OM."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {"type": "number", "minimum": 0, "maximum": 100},
                "clay": {"type": "number", "minimum": 0, "maximum": 100},
                "organic_matter": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Percent organic matter (mass). Default 1.0 % if omitted.",
                },
                "density_factor": {
                    "type": "number",
                    "minimum": 0.9,
                    "maximum": 1.3,
                    "description": (
                        "Optional compaction factor. 1.0 = normal regression bulk density "
                        "(default). 1.1 = 10 % more compacted."
                    ),
                },
            },
            "required": ["sand", "clay"],
        },
    }


def _wosten_schema() -> dict[str, Any]:
    return {
        "name": "wosten",
        "description": (
            "Estimate Mualem-van Genuchten parameters (theta_s, alpha, n, "
            "K_s, L) via the European HYPRES PTF (Wösten et al. 1999). "
            "Returns parameters of the *continuous* water-retention "
            "curve; use saxton_rawls() if you need direct field-capacity "
            "/ wilting-point numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {"type": "number", "minimum": 0, "maximum": 100},
                "silt": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                "clay": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                "organic_matter": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 100,
                    "description": "Percent organic matter, strictly positive. Default 1.0.",
                },
                "bulk_density": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Dry bulk density (g/cm³). Default 1.4.",
                },
                "topsoil": {
                    "type": "boolean",
                    "description": "True for topsoil, False for subsoil. Default true.",
                },
            },
            "required": ["sand", "silt", "clay"],
        },
    }


def _convert_particle_size_schema() -> dict[str, Any]:
    return {
        "name": "convert_particle_size",
        "description": (
            "Convert sand/silt/clay percentages between particle-size "
            "standards (USDA, FAO, ISSS, INTERNATIONAL, KA5) which use "
            "different sand-silt cutoffs. Required when classifying a "
            "USDA-measured sample with an ISSS-based scheme. The clay "
            "fraction (<0.002 mm) is unchanged across all standards. "
            "Note: KACHINSKY is *not* convertible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {"type": "number", "minimum": 0, "maximum": 100},
                "silt": {"type": "number", "minimum": 0, "maximum": 100},
                "clay": {"type": "number", "minimum": 0, "maximum": 100},
                "source": {
                    "type": "string",
                    "enum": ["USDA", "FAO", "ISSS", "INTERNATIONAL", "KA5"],
                },
                "target": {
                    "type": "string",
                    "enum": ["USDA", "FAO", "ISSS", "INTERNATIONAL", "KA5"],
                },
            },
            "required": ["sand", "silt", "clay", "source", "target"],
        },
    }


def _render_diagram_schema() -> dict[str, Any]:
    return {
        "name": "render_diagram",
        "description": (
            "Render a soil texture diagram as a standalone SVG string. "
            "Optionally overlay sample points. The SVG is suitable for "
            "embedding in HTML, displaying in a chat, or saving to disk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "description": "Classification key (e.g. 'USDA').",
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "description": (
                        "Optional samples. (sand, clay) pairs for 2-D, or "
                        "single-element [value] arrays for 1-D."
                    ),
                },
                "point_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "locale": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["classification"],
        },
    }


# --- Dispatcher ----------------------------------------------------------


def run(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name and return a JSON-serializable result.

    Errors are *returned* as structured envelopes — not raised — so
    LLM tool-use loops can recover without parsing English exception
    strings.

    Args:
        tool: Tool name as listed in :func:`tool_schemas`.
        arguments: Tool-specific keyword arguments.

    Returns:
        On success, a dict with the tool's output. On error, a dict
        with at least ``"error"`` (exception class name) and
        ``"message"`` keys.
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {
            "error": "UnknownTool",
            "message": f"Unknown tool {tool!r}. Available: {sorted(_HANDLERS)!r}.",
        }
    try:
        return handler(arguments)
    except UnknownClassificationError as exc:
        return {
            "error": "UnknownClassificationError",
            "message": str(exc),
            "available": exc.available,
        }
    except (InvalidInputError, ClassificationError, PedotriError) as exc:
        return {"error": type(exc).__name__, "message": str(exc)}
    except (TypeError, ValueError, KeyError) as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def _h_classify_soil(args: dict[str, Any]) -> dict[str, Any]:
    sand = _require(args, "sand", (int, float))
    clay = _require(args, "clay", (int, float))
    classification = _require(args, "classification", str)
    locale = args.get("locale")
    result = pedotri.classify(
        float(sand), float(clay), classification, detailed=True, locale=locale
    )
    return result.to_dict()


def _h_classify_soil_1d(args: dict[str, Any]) -> dict[str, Any]:
    value = _require(args, "value", (int, float))
    classification = _require(args, "classification", str)
    locale = args.get("locale")
    result = pedotri.classify(float(value), classification, detailed=True, locale=locale)
    return result.to_dict()


def _h_list_classifications(args: dict[str, Any]) -> dict[str, Any]:
    del args
    return {"classifications": pedotri.list_classifications()}


def _h_classification_info(args: dict[str, Any]) -> dict[str, Any]:
    classification = _require(args, "classification", str)
    locale = args.get("locale")
    c = pedotri.get_classification(classification)
    loc = locale or c.default_locale
    return {
        "key": c.key,
        "name": c.name(loc),
        "axes": list(c.axes),
        "default_locale": c.default_locale,
        "available_locales": sorted(c.locales()),
        "reference": c.reference,
        "url": c.url,
        "description": c.description,
        "classes": [
            {
                "key": cls.key,
                "name": cls.name(loc),
                "group": cls.group,
                "parent": cls.parent,
                "region": (
                    {"type": "interval", "low": cls.interval[0], "high": cls.interval[1]}
                    if cls.interval is not None
                    else {
                        "type": "polygon",
                        "vertices": cls.vertices.tolist() if cls.vertices is not None else [],
                    }
                ),
            }
            for cls in c.classes
        ],
    }


def _h_saxton_rawls(args: dict[str, Any]) -> dict[str, Any]:
    sand = float(_require(args, "sand", (int, float)))
    clay = float(_require(args, "clay", (int, float)))
    om = float(args.get("organic_matter", 1.0))
    df = float(args.get("density_factor", 1.0))
    return saxton_rawls(sand, clay, om, density_factor=df).to_dict()


def _h_wosten(args: dict[str, Any]) -> dict[str, Any]:
    sand = float(_require(args, "sand", (int, float)))
    silt = float(_require(args, "silt", (int, float)))
    clay = float(_require(args, "clay", (int, float)))
    om = float(args.get("organic_matter", 1.0))
    bd = float(args.get("bulk_density", 1.4))
    topsoil = bool(args.get("topsoil", True))
    return wosten(
        sand, silt, clay, organic_matter=om, bulk_density=bd, topsoil=topsoil
    ).to_dict()


def _h_convert_particle_size(args: dict[str, Any]) -> dict[str, Any]:
    sand = _require(args, "sand", (int, float))
    silt = _require(args, "silt", (int, float))
    clay = _require(args, "clay", (int, float))
    source = _require(args, "source", str)
    target = _require(args, "target", str)
    s, si, c = _psd_convert(sand, silt, clay, source=source, target=target)
    return {
        "sand": float(s[0]),
        "silt": float(si[0]),
        "clay": float(c[0]),
        "source": source,
        "target": target,
    }


def _h_render_diagram(args: dict[str, Any]) -> dict[str, Any]:
    classification = _require(args, "classification", str)
    points = args.get("points")
    point_labels = args.get("point_labels")
    locale = args.get("locale")
    title = args.get("title")
    svg = render_svg(
        classification,
        points=points,
        point_labels=point_labels,
        locale=locale,
        title=title,
    )
    return {"format": "svg", "content": svg}


def _require(args: dict[str, Any], key: str, types: type | tuple[type, ...]) -> Any:
    if key not in args:
        raise TypeError(f"Missing required argument {key!r}.")
    value = args[key]
    if not isinstance(value, types):
        type_names = (
            types.__name__
            if isinstance(types, type)
            else " or ".join(t.__name__ for t in types)
        )
        raise TypeError(
            f"Argument {key!r} must be {type_names}, got {type(value).__name__}."
        )
    return value


_Handler = Callable[[dict[str, Any]], dict[str, Any]]

_HANDLERS: dict[str, _Handler] = {
    "classify_soil": _h_classify_soil,
    "classify_soil_1d": _h_classify_soil_1d,
    "list_classifications": _h_list_classifications,
    "classification_info": _h_classification_info,
    "saxton_rawls": _h_saxton_rawls,
    "wosten": _h_wosten,
    "convert_particle_size": _h_convert_particle_size,
    "render_diagram": _h_render_diagram,
}


__all__ = ["run", "tool_schemas"]
