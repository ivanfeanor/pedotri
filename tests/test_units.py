"""Tests for the pedotri.units module and the `units=` keyword."""

from __future__ import annotations

import numpy as np
import pytest

import pedotri
import pedotri.ai
from pedotri import units as pedotri_units
from pedotri.errors import InvalidInputError
from pedotri.psd import convert as psd_convert
from pedotri.ptf import saxton_rawls, wosten

# --- Conversion helpers --------------------------------------------------


def test_g_per_kg_to_percent() -> None:
    assert float(pedotri_units.g_per_kg_to_percent(250)[0]) == pytest.approx(25.0)
    assert float(pedotri_units.g_per_kg_to_percent(0)[0]) == 0.0
    assert float(pedotri_units.g_per_kg_to_percent(1000)[0]) == 100.0


def test_percent_to_g_per_kg_round_trip() -> None:
    pct = 37.5
    g = pedotri_units.percent_to_g_per_kg(pct)
    assert float(pedotri_units.g_per_kg_to_percent(g)[0]) == pytest.approx(pct)


def test_g_per_g_to_percent() -> None:
    assert float(pedotri_units.g_per_g_to_percent(0.25)[0]) == pytest.approx(25.0)


def test_percent_to_g_per_g() -> None:
    assert float(pedotri_units.percent_to_g_per_g(25.0)[0]) == pytest.approx(0.25)


def test_organic_carbon_to_organic_matter_van_bemmelen() -> None:
    typical = float(pedotri_units.organic_carbon_to_organic_matter(1.16)[0])
    assert typical == pytest.approx(2.0, abs=0.005)
    # The Van Bemmelen factor itself
    one_pct = float(pedotri_units.organic_carbon_to_organic_matter(1.0)[0])
    assert one_pct == pytest.approx(1.724)


def test_organic_matter_to_organic_carbon_inverse() -> None:
    om = 2.5
    oc = pedotri_units.organic_matter_to_organic_carbon(om)
    back = pedotri_units.organic_carbon_to_organic_matter(oc)
    assert float(back[0]) == pytest.approx(om)


def test_oc_to_om_custom_factor() -> None:
    # Pribyl 2010 suggests 1.9 for less-humified material
    assert float(pedotri_units.organic_carbon_to_organic_matter(1.0, factor=1.9)[0]) == 1.9


def test_oc_to_om_rejects_nonpositive_factor() -> None:
    with pytest.raises(InvalidInputError, match="strictly positive"):
        pedotri_units.organic_carbon_to_organic_matter(1.0, factor=0)
    with pytest.raises(InvalidInputError, match="strictly positive"):
        pedotri_units.organic_matter_to_organic_carbon(1.0, factor=-1.0)


def test_to_percent_passes_arrays() -> None:
    out = pedotri_units.to_percent(np.array([100, 200, 300]), "g/kg")
    assert out.tolist() == [10.0, 20.0, 30.0]


def test_unknown_units_raise() -> None:
    with pytest.raises(InvalidInputError, match="Unknown units"):
        pedotri_units.to_percent(50, "kg/kg")  # type: ignore[arg-type]


# --- classify(units=...) ------------------------------------------------


def test_classify_default_units_is_percent() -> None:
    assert pedotri.classify(60, 20, "USDA") == "sandy_clay_loam"
    # Same call with explicit units="%" is identical.
    assert pedotri.classify(60, 20, "USDA", units="%") == "sandy_clay_loam"


def test_classify_g_per_kg_matches_percent() -> None:
    in_pct = pedotri.classify(60, 20, "USDA", detailed=True)
    in_gkg = pedotri.classify(600, 200, "USDA", detailed=True, units="g/kg")
    assert in_pct.key == in_gkg.key
    assert in_pct.distance == pytest.approx(in_gkg.distance)


def test_classify_g_per_g_matches_percent() -> None:
    in_pct = pedotri.classify(60, 20, "USDA")
    in_gg = pedotri.classify(0.6, 0.2, "USDA", units="g/g")
    assert in_pct == in_gg


def test_classify_1d_units() -> None:
    in_pct = pedotri.classify(35, "KACHINSKY")
    in_gkg = pedotri.classify(350, "KACHINSKY", units="g/kg")
    assert in_pct == in_gkg == "medium_loam"


def test_classify_unknown_units_raises() -> None:
    with pytest.raises(InvalidInputError, match="Unknown units"):
        pedotri.classify(60, 20, "USDA", units="ppm")


# --- PTFs ----------------------------------------------------------------


def test_saxton_rawls_units_round_trip() -> None:
    pct = saxton_rawls(40, 20, 2.0)
    gkg = saxton_rawls(400, 200, 20, units="g/kg")
    gg = saxton_rawls(0.40, 0.20, 0.020, units="g/g")
    assert pct.wilting_point == pytest.approx(gkg.wilting_point)
    assert pct.wilting_point == pytest.approx(gg.wilting_point)
    assert pct.bulk_density == pytest.approx(gkg.bulk_density)


def test_wosten_units_round_trip() -> None:
    pct = wosten(60, 30, 10, organic_matter=2.5, bulk_density=1.4)
    gkg = wosten(600, 300, 100, organic_matter=25, bulk_density=1.4, units="g/kg")
    assert pct.theta_s == pytest.approx(gkg.theta_s)
    assert pct.alpha == pytest.approx(gkg.alpha)


def test_wosten_bulk_density_not_converted() -> None:
    """bulk_density is always g/cm³ — the units keyword must not touch it."""
    same_bd = wosten(600, 300, 100, organic_matter=25, bulk_density=1.4, units="g/kg")
    # If bulk_density were also converted, the value would be wildly off
    # and theta_s would drift far from the percent baseline.
    pct = wosten(60, 30, 10, organic_matter=2.5, bulk_density=1.4)
    assert same_bd.theta_s == pytest.approx(pct.theta_s)


# --- psd.convert --------------------------------------------------------


def test_psd_convert_units_keyword() -> None:
    s_pct, si_pct, c_pct = psd_convert(60, 30, 10, source="USDA", target="ISSS")
    s_gkg, si_gkg, c_gkg = psd_convert(
        600, 300, 100, source="USDA", target="ISSS", units="g/kg"
    )
    # psd.convert always returns percent regardless of input units.
    assert s_pct[0] == pytest.approx(s_gkg[0])
    assert si_pct[0] == pytest.approx(si_gkg[0])
    assert c_pct[0] == pytest.approx(c_gkg[0])


# --- AI tool schemas include units --------------------------------------


def test_ai_schemas_advertise_units_param() -> None:
    schemas = pedotri.ai.tool_schemas()
    has_units = {
        s["name"]
        for s in schemas
        if "units" in s["input_schema"].get("properties", {})
    }
    # Every tool that takes fraction values exposes `units`.
    assert has_units >= {
        "classify_soil",
        "classify_soil_1d",
        "saxton_rawls",
        "wosten",
        "convert_particle_size",
    }


def test_ai_run_accepts_units() -> None:
    result = pedotri.ai.run(
        "classify_soil",
        {"sand": 600, "clay": 200, "classification": "USDA", "units": "g/kg"},
    )
    assert result["key"] == "sandy_clay_loam"
