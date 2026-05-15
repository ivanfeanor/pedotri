"""Tests for pedotri.ai (JSON tool dispatcher)."""

from __future__ import annotations

import base64
import json
import math

import pytest

import pedotri
from pedotri import ai
from pedotri.ptf import saxton_rawls, wosten


def test_tool_schemas_complete() -> None:
    schemas = ai.tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        "classify_soil",
        "classify_soil_1d",
        "classify_point",
        "list_classifications",
        "classification_info",
        "saxton_rawls",
        "wosten",
        "convert_particle_size",
        "render_diagram",
    }
    # Every schema must be Anthropic tool-shaped.
    for s in schemas:
        assert s["name"]
        assert s["description"]
        assert s["input_schema"]["type"] == "object"
        # And must JSON-serialize cleanly.
        json.dumps(s)


def test_classify_soil_round_trip() -> None:
    result = ai.run(
        "classify_soil",
        {"sand": 60, "clay": 20, "classification": "USDA"},
    )
    assert result["key"] == "sandy_clay_loam"
    assert result["group"] == "moderately_fine"
    # Distance is a finite float (point is well inside the polygon).
    assert isinstance(result["distance"], float)
    assert math.isfinite(result["distance"])


def test_classify_soil_with_locale() -> None:
    result = ai.run(
        "classify_soil",
        {"sand": 13, "clay": 50, "classification": "USDA", "locale": "fr"},
    )
    assert result["key"] == "clay"
    assert result["name"] == "argile"


def test_classify_soil_1d() -> None:
    result = ai.run(
        "classify_soil_1d",
        {"value": 35, "classification": "KACHINSKY", "locale": "ru"},
    )
    assert result["key"] == "medium_loam"
    assert result["name"] == "суглинок средний"


def test_list_classifications() -> None:
    result = ai.run("list_classifications", {})
    assert "USDA" in result["classifications"]
    assert "KACHINSKY" in result["classifications"]
    assert len(result["classifications"]) >= 10


def test_classification_info_includes_classes() -> None:
    result = ai.run("classification_info", {"classification": "FAO"})
    assert result["key"] == "FAO"
    assert result["axes"] == ["sand", "clay"]
    keys = {c["key"] for c in result["classes"]}
    assert keys == {"fine", "medium", "coarse"}
    # Each class has a region description, JSON-serializable.
    for cls in result["classes"]:
        assert cls["region"]["type"] in {"polygon", "interval"}
    json.dumps(result)


def test_classification_info_kachinsky_uses_interval_region() -> None:
    result = ai.run("classification_info", {"classification": "KACHINSKY"})
    assert result["axes"] == ["physical_clay"]
    for cls in result["classes"]:
        assert cls["region"]["type"] == "interval"
        assert "low" in cls["region"]
        assert "high" in cls["region"]


def test_saxton_rawls_returns_floats() -> None:
    result = ai.run(
        "saxton_rawls",
        {"sand": 40, "clay": 20, "organic_matter": 2.0},
    )
    for k in (
        "wilting_point",
        "field_capacity",
        "saturation",
        "available_water",
        "saturated_conductivity",
        "bulk_density",
        "air_entry_tension",
    ):
        assert isinstance(result[k], float)
    assert 0 < result["wilting_point"] < result["field_capacity"] < result["saturation"] < 1


def test_saxton_rawls_density_factor() -> None:
    normal = ai.run("saxton_rawls", {"sand": 60, "clay": 10})
    compacted = ai.run(
        "saxton_rawls",
        {"sand": 60, "clay": 10, "density_factor": 1.10},
    )
    assert compacted["bulk_density"] > normal["bulk_density"]


def test_wosten_returns_van_genuchten_params() -> None:
    result = ai.run(
        "wosten",
        {
            "sand": 60,
            "silt": 30,
            "clay": 10,
            "organic_matter": 2.5,
            "bulk_density": 1.4,
        },
    )
    assert result["theta_r"] == 0.01
    assert 0 < result["theta_s"] < 1
    assert result["n"] > 1
    assert result["alpha"] > 0


def test_convert_particle_size() -> None:
    result = ai.run(
        "convert_particle_size",
        {
            "sand": 60,
            "silt": 30,
            "clay": 10,
            "source": "USDA",
            "target": "ISSS",
        },
    )
    # USDA → ISSS moves silt → sand (smaller cutoff)
    assert result["sand"] > 60
    assert result["silt"] < 30
    assert result["clay"] == 10.0


def test_render_diagram_defaults_to_png() -> None:
    """When matplotlib is available (it is in the dev environment),
    render_diagram returns PNG by default — the format every MCP client
    can render inline reliably."""
    result = ai.run("render_diagram", {"classification": "USDA"})
    assert result["format"] == "png"
    assert result["encoding"] == "base64"
    # PNG signature is 89 50 4E 47 0D 0A 1A 0A.
    decoded = base64.standard_b64decode(result["content"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_diagram_returns_svg_when_requested() -> None:
    """Callers (or MCP clients that handle vector content) can opt
    into SVG explicitly via the format keyword."""
    result = ai.run("render_diagram", {"classification": "USDA", "format": "svg"})
    assert result["format"] == "svg"
    assert result["encoding"] == "text"
    assert result["content"].startswith("<svg")
    assert result["content"].rstrip().endswith("</svg>")


def test_render_diagram_unknown_format_rejected() -> None:
    result = ai.run("render_diagram", {"classification": "USDA", "format": "jpeg"})
    assert result["error"] == "InvalidInputError"
    assert "format" in result["message"]


# --- Error envelopes -----------------------------------------------------


def test_unknown_tool_returns_error_envelope() -> None:
    result = ai.run("does_not_exist", {})
    assert result["error"] == "UnknownTool"
    assert "Unknown tool" in result["message"]


def test_unknown_classification_returns_envelope_with_available() -> None:
    result = ai.run(
        "classify_soil",
        {"sand": 60, "clay": 20, "classification": "NOPE"},
    )
    assert result["error"] == "UnknownClassificationError"
    assert "USDA" in result["available"]


def test_out_of_range_input_returns_envelope() -> None:
    result = ai.run(
        "classify_soil",
        {"sand": 60, "clay": 200, "classification": "USDA"},
    )
    assert result["error"] == "InvalidInputError"
    assert "100" in result["message"]


def test_missing_argument_returns_envelope() -> None:
    result = ai.run("classify_soil", {"sand": 60})
    assert result["error"] == "TypeError"
    assert "clay" in result["message"]


def test_wrong_argument_type_returns_envelope() -> None:
    result = ai.run(
        "classify_soil",
        {"sand": "sixty", "clay": 20, "classification": "USDA"},
    )
    assert result["error"] == "TypeError"
    assert "sand" in result["message"]


def test_classify_point_explicit_no_uncertainty() -> None:
    """No quantiles → behaves like classify_soil; uncertainty fields stay absent."""
    result = ai.run(
        "classify_point",
        {"sand": 13, "clay": 50, "classification": "USDA"},
    )
    assert result["key"] == "clay"
    assert "confidence" not in result
    assert "probabilities" not in result


def test_classify_point_explicit_with_uncertainty_distance() -> None:
    result = ai.run(
        "classify_point",
        {
            "sand": 27,
            "clay": 38,
            "sand_q05": 22,
            "sand_q95": 32,
            "clay_q05": 33,
            "clay_q95": 43,
            "classification": "USDA",
            "method": "distance",
        },
    )
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_point_explicit_with_uncertainty_monte_carlo() -> None:
    result = ai.run(
        "classify_point",
        {
            "sand": 27,
            "clay": 38,
            "sand_q05": 22,
            "sand_q95": 32,
            "clay_q05": 33,
            "clay_q95": 43,
            "classification": "USDA",
            "method": "monte_carlo",
            "n_samples": 500,
            "seed": 42,
        },
    )
    probs = result["probabilities"]
    assert isinstance(probs, dict)
    assert probs
    total = sum(probs.values()) + result.get("unclassified_probability", 0.0)
    assert abs(total - 1.0) < 1e-6


def test_classify_point_requires_one_mode() -> None:
    assert ai.run("classify_point", {})["error"] == "InvalidInputError"


def test_classify_point_rejects_mixed_modes() -> None:
    err = ai.run(
        "classify_point",
        {"sand": 13, "clay": 50, "lon": 2.5, "lat": 47.0},
    )
    assert err["error"] == "InvalidInputError"
    assert "(sand, clay) or (lon, lat)" in err["message"]


def test_classify_point_rejects_partial_quantiles() -> None:
    err = ai.run(
        "classify_point",
        {"sand": 27, "clay": 38, "sand_q05": 22, "classification": "USDA"},
    )
    assert err["error"] == "InvalidInputError"
    assert "sand_q05" in err["message"]


def test_classify_point_from_soilgrids_with_mock(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Stub SoilGrids HTTP and verify the lon/lat mode pipes through to classify()."""

    from pedotri.sources import soilgrids

    canned = {
        "properties": {
            "layers": [
                {
                    "name": "sand",
                    "depths": [
                        {
                            "label": "0-5cm",
                            "values": {"mean": 350, "Q0.05": 280, "Q0.95": 420},
                        }
                    ],
                },
                {
                    "name": "clay",
                    "depths": [
                        {
                            "label": "0-5cm",
                            "values": {"mean": 270, "Q0.05": 220, "Q0.95": 320},
                        }
                    ],
                },
            ]
        }
    }

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(canned).encode("utf-8")

    monkeypatch.setattr(soilgrids.urllib.request, "urlopen", lambda req, timeout=30.0: _Resp())
    # Redirect cache into tmp_path so a real cache dir isn't polluted.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    result = ai.run(
        "classify_point",
        {
            "lon": 2.5,
            "lat": 47.0,
            "classification": "USDA",
            "method": "monte_carlo",
            "n_samples": 500,
            "seed": 42,
        },
    )
    assert result["key"] is not None
    assert "probabilities" in result
    assert result["source"]["name"] == "soilgrids"
    assert result["source"]["cached"] is False
    assert result["source"]["values"]["sand"]["mean"] == pytest.approx(35.0)


def test_result_dataclasses_have_to_dict() -> None:
    """The to_dict() methods underpin every JSON return path."""
    cr = pedotri.classify(60, 20, "USDA", detailed=True)
    assert isinstance(cr.to_dict(), dict)

    sr = saxton_rawls(40, 20)
    assert isinstance(sr.to_dict(), dict)
    assert sr.to_dict()["wilting_point"] == pytest.approx(sr.wilting_point)

    w = wosten(60, 30, 10, organic_matter=2.5, bulk_density=1.4)
    assert isinstance(w.to_dict(), dict)
    assert w.to_dict()["theta_r"] == 0.01
