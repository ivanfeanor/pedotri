"""Command-line interface for pedotri.

Subcommands:

- ``pedotri list`` — list registered classifications.
- ``pedotri info <key>`` — show details for a single classification.
- ``pedotri classify`` — classify a sample (positional fractions or
  ``--csv`` for batch).
- ``pedotri render`` — write an SVG diagram for a classification.

The CLI is stdlib-only (argparse) so installing pedotri without any
extras gives a working ``pedotri`` command.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TextIO

import pedotri
from pedotri.errors import PedotriError
from pedotri.plot import render_svg


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns the process exit code (0 on success, 1 on user-facing
    errors, 2 on argparse usage errors).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PedotriError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pedotri",
        description="Soil texture classification and pedotransfer functions.",
    )
    parser.add_argument("--version", action="version", version=f"pedotri {pedotri.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="List registered classifications.")
    p_list.set_defaults(handler=_cmd_list)

    p_info = subparsers.add_parser("info", help="Show details for a classification.")
    p_info.add_argument("key", help="Classification key (e.g. USDA).")
    p_info.add_argument("--locale", help="Locale tag for class names.")
    p_info.set_defaults(handler=_cmd_info)

    p_classify = subparsers.add_parser("classify", help="Classify one or many samples.")
    p_classify.add_argument("-c", "--classification", required=True, help="Classification key.")
    p_classify.add_argument("--sand", type=float, help="Percent sand (single sample).")
    p_classify.add_argument("--clay", type=float, help="Percent clay (single sample).")
    p_classify.add_argument(
        "--physical-clay",
        type=float,
        dest="physical_clay",
        help="Percent physical clay (for 1-D classifications).",
    )
    p_classify.add_argument(
        "--csv",
        type=Path,
        help="CSV input file with axis-named columns (overrides single sample).",
    )
    p_classify.add_argument("--output", type=Path, help="Write CSV results to this file.")
    p_classify.add_argument("--locale", help="Return localized names instead of class keys.")
    p_classify.set_defaults(handler=_cmd_classify)

    p_render = subparsers.add_parser("render", help="Render a texture diagram as SVG.")
    p_render.add_argument("-c", "--classification", required=True, help="Classification key.")
    p_render.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write SVG to file (stdout if omitted).",
    )
    p_render.add_argument("--locale", help="Locale for class names in the legend.")
    p_render.add_argument("--title", help="Title rendered above the diagram.")
    p_render.add_argument("--width", type=int, default=720, help="SVG width in pixels.")
    p_render.add_argument("--no-legend", action="store_true", help="Hide the legend.")
    p_render.add_argument("--no-grid", action="store_true", help="Hide the 10%% gridlines.")
    p_render.set_defaults(handler=_cmd_render)

    return parser


# --- Command handlers ----------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    del args
    for key in pedotri.list_classifications():
        print(key)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    c = pedotri.get_classification(args.key)
    loc = args.locale or c.default_locale
    print(f"{c.key} — {c.name(loc)}")
    print(f"  axes:    {', '.join(c.axes)}")
    print(f"  classes: {len(c.classes)}")
    print(f"  locale:  {loc} (default: {c.default_locale})")
    if c.reference:
        print(f"  ref:     {c.reference}")
    if c.url:
        print(f"  url:     {c.url}")
    if c.description:
        print()
        print(c.description)
    print()
    print("Classes:")
    for cls in c.classes:
        region = (
            f"[{cls.interval[0]:g}, {cls.interval[1]:g})"
            if cls.interval is not None
            else f"polygon ({len(cls.vertices) if cls.vertices is not None else 0} vertices)"
        )
        group = f"  [{cls.group}]" if cls.group else ""
        print(f"  {cls.key:<20} {cls.name(loc):<28} {region}{group}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    if args.csv is not None:
        return _classify_csv(args)
    return _classify_single(args)


def _classify_single(args: argparse.Namespace) -> int:
    c = pedotri.get_classification(args.classification)
    fractions: dict[str, float] = {}
    for axis in c.axes:
        val = getattr(args, axis.replace("-", "_"), None)
        if val is None:
            print(
                f"error: missing --{axis.replace('_', '-')} for classification {c.key!r}.",
                file=sys.stderr,
            )
            return 1
        fractions[axis] = val
    result = pedotri.classify(  # type: ignore[call-overload]
        classification=c, locale=args.locale, **fractions
    )
    print(result)
    return 0


def _classify_csv(args: argparse.Namespace) -> int:
    c = pedotri.get_classification(args.classification)
    with args.csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("error: empty CSV.", file=sys.stderr)
            return 1
        for axis in c.axes:
            if axis not in reader.fieldnames:
                print(
                    f"error: CSV is missing column {axis!r} (axes: {list(c.axes)}).",
                    file=sys.stderr,
                )
                return 1
        rows = list(reader)

    columns = {axis: [float(r[axis]) for r in rows] for axis in c.axes}
    results = pedotri.classify(  # type: ignore[call-overload]
        classification=c, locale=args.locale, **columns
    )

    out: TextIO = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=[*list(reader.fieldnames), "texture"])
        writer.writeheader()
        for row, texture in zip(rows, results, strict=True):
            row["texture"] = texture if texture is not None else ""
            writer.writerow(row)
    finally:
        if args.output:
            out.close()
    if args.output:
        print(f"Wrote {len(rows)} rows to {args.output}.", file=sys.stderr)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    svg = render_svg(
        args.classification,
        locale=args.locale,
        title=args.title,
        width=args.width,
        show_legend=not args.no_legend,
        show_grid=not args.no_grid,
    )
    if args.output:
        args.output.write_text(svg, encoding="utf-8")
        print(f"Wrote {args.output}.", file=sys.stderr)
    else:
        sys.stdout.write(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
