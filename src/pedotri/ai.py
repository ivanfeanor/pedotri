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
            "Default input units are percent (0-100); silt is implicit "
            "(silt = 100 - sand - clay). Pass units='g/kg' or units='g/g' "
            "when the lab report uses those instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "Sand fraction. Default units: percent in [0, 100]. "
                        "See the 'units' parameter."
                    ),
                },
                "clay": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Clay fraction. Default units: percent.",
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
                "units": _units_schema(),
            },
            "required": ["sand", "clay", "classification"],
        },
    }


def _units_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": ["%", "g/kg", "g/g"],
        "default": "%",
        "description": (
            "Units of the fraction inputs. '%' (default) for percent in "
            "[0, 100]; 'g/kg' for grams per kilogram (common in European "
            "soil lab reports for organic matter); 'g/g' for the 0-1 mass "
            "fraction convention."
        ),
    }


def _classify_soil_1d_schema() -> dict[str, Any]:
    return {
        "name": "classify_soil_1d",
        "description": (
            "Classify a soil sample on a single axis. Used by 1-D "
            "classifications like KACHINSKY (physical clay <0.01 mm). "
            "Call classification_info first to learn which axis the "
            "classification expects. Default input units are percent; "
            "pass units='g/kg' or 'g/g' to override."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "Fraction for the classification's axis (e.g. "
                        "physical clay). Default units: percent."
                    ),
                },
                "classification": {
                    "type": "string",
                    "description": (
                        "1-D classification key. Currently 'KACHINSKY' is "
                        "the only built-in 1-D classification."
                    ),
                },
                "locale": {"type": "string", "description": "Optional locale tag."},
                "units": _units_schema(),
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
            "balance question that starts from sand/clay/OM. "
            "Important: organic_matter is organic MATTER, not organic "
            "carbon. If the lab reports organic carbon (very common), "
            "multiply by 1.724 (Van Bemmelen factor) before passing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Sand fraction. Default units: percent.",
                },
                "clay": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Clay fraction. Default units: percent.",
                },
                "organic_matter": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "Organic-matter fraction (NOT organic carbon). "
                        "Default units: percent. Default value 1.0 %."
                    ),
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
                "units": _units_schema(),
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
            "/ wilting-point numbers. organic_matter is organic MATTER, "
            "not organic carbon (multiply OC by 1.724 if needed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Sand fraction. Default units: percent.",
                },
                "silt": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Silt fraction, strictly positive (the PTF "
                        "contains 1/silt and ln(silt) terms). "
                        "Default units: percent."
                    ),
                },
                "clay": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Clay fraction, strictly positive. Default units: percent.",
                },
                "organic_matter": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Organic-matter fraction, strictly positive. "
                        "Default units: percent. Default value 1.0."
                    ),
                },
                "bulk_density": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Dry bulk density in g/cm³ (always; not affected "
                        "by the units parameter). Default 1.4."
                    ),
                },
                "topsoil": {
                    "type": "boolean",
                    "description": "True for topsoil, False for subsoil. Default true.",
                },
                "units": _units_schema(),
            },
            "required": ["sand", "silt", "clay"],
        },
    }


def _convert_particle_size_schema() -> dict[str, Any]:
    return {
        "name": "convert_particle_size",
        "description": (
            "Convert sand/silt/clay fractions between particle-size "
            "standards (USDA, FAO, ISSS, INTERNATIONAL, KA5) which use "
            "different sand-silt cutoffs. Required when classifying a "
            "USDA-measured sample with an ISSS-based scheme. The clay "
            "fraction (<0.002 mm) is unchanged across all standards. "
            "Output is always in percent. Note: KACHINSKY is *not* "
            "convertible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sand": {"type": "number", "minimum": 0},
                "silt": {"type": "number", "minimum": 0},
                "clay": {"type": "number", "minimum": 0},
                "source": {
                    "type": "string",
                    "enum": ["USDA", "FAO", "ISSS", "INTERNATIONAL", "KA5"],
                },
                "target": {
                    "type": "string",
                    "enum": ["USDA", "FAO", "ISSS", "INTERNATIONAL", "KA5"],
                },
                "units": _units_schema(),
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
    units = args.get("units", "%")
    result = pedotri.classify(
        float(sand),
        float(clay),
        classification,
        detailed=True,
        locale=locale,
        units=units,
    )
    return result.to_dict()


def _h_classify_soil_1d(args: dict[str, Any]) -> dict[str, Any]:
    value = _require(args, "value", (int, float))
    classification = _require(args, "classification", str)
    locale = args.get("locale")
    units = args.get("units", "%")
    result = pedotri.classify(
        float(value), classification, detailed=True, locale=locale, units=units
    )
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
    units = args.get("units", "%")
    return saxton_rawls(
        sand, clay, om, density_factor=df, units=units
    ).to_dict()


def _h_wosten(args: dict[str, Any]) -> dict[str, Any]:
    sand = float(_require(args, "sand", (int, float)))
    silt = float(_require(args, "silt", (int, float)))
    clay = float(_require(args, "clay", (int, float)))
    om = float(args.get("organic_matter", 1.0))
    bd = float(args.get("bulk_density", 1.4))
    topsoil = bool(args.get("topsoil", True))
    units = args.get("units", "%")
    return wosten(
        sand,
        silt,
        clay,
        organic_matter=om,
        bulk_density=bd,
        topsoil=topsoil,
        units=units,
    ).to_dict()


def _h_convert_particle_size(args: dict[str, Any]) -> dict[str, Any]:
    sand = _require(args, "sand", (int, float))
    silt = _require(args, "silt", (int, float))
    clay = _require(args, "clay", (int, float))
    source = _require(args, "source", str)
    target = _require(args, "target", str)
    units = args.get("units", "%")
    s, si, c = _psd_convert(
        sand, silt, clay, source=source, target=target, units=units
    )
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
