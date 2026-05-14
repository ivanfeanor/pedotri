"""Tests for the pedotri CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pedotri.cli import main

if TYPE_CHECKING:
    import pathlib


def test_list_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "USDA" in out
    assert "KACHINSKY" in out


def test_info_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["info", "USDA"])
    assert code == 0
    out = capsys.readouterr().out
    assert "USDA" in out
    assert "sand, clay" in out
    assert "Soil Survey" in out
    assert "clay" in out  # listed as a class


def test_info_unknown_classification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["info", "NOPE"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Unknown" in err or "error" in err


def test_classify_2d_positional(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["classify", "-c", "USDA", "--sand", "60", "--clay", "20"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "sandy_clay_loam"


def test_classify_1d(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["classify", "-c", "KACHINSKY", "--physical-clay", "35"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "medium_loam"


def test_classify_localized(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "classify",
            "-c",
            "USDA",
            "--sand",
            "60",
            "--clay",
            "20",
            "--locale",
            "fr",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "limon argilo-sableux"


def test_classify_missing_axis(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["classify", "-c", "USDA", "--sand", "60"])
    assert code == 1
    err = capsys.readouterr().err
    assert "clay" in err


def test_classify_csv(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inp = tmp_path / "in.csv"
    inp.write_text("sand,clay\n60,20\n10,70\n80,5\n")
    out = tmp_path / "out.csv"
    code = main(
        [
            "classify",
            "-c",
            "USDA",
            "--csv",
            str(inp),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    rows = out.read_text().splitlines()
    assert rows[0] == "sand,clay,texture"
    assert rows[1].endswith("sandy_clay_loam")
    assert rows[2].endswith("clay")
    assert rows[3].endswith("loamy_sand")


def test_classify_csv_missing_column(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inp = tmp_path / "in.csv"
    inp.write_text("sand,X\n60,20\n")
    code = main(["classify", "-c", "USDA", "--csv", str(inp)])
    assert code == 1
    err = capsys.readouterr().err
    assert "clay" in err


def test_render_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["render", "-c", "USDA"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("<svg")
    assert out.strip().endswith("</svg>")


def test_render_to_file(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "usda.svg"
    code = main(
        [
            "render",
            "-c",
            "USDA",
            "-o",
            str(target),
            "--title",
            "USDA test",
            "--locale",
            "fr",
        ]
    )
    assert code == 0
    content = target.read_text()
    assert content.startswith("<svg")
    assert "USDA test" in content
    assert "argile" in content


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "pedotri" in capsys.readouterr().out
