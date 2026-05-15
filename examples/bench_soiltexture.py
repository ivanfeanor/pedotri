"""Benchmark pedotri.classify against the `soiltexture` Python package.

Both libraries implement point-in-polygon over the USDA / FAO / ISSS /
INTERNATIONAL triangles, but pedotri uses a vectorized numpy crossing-
number test while ``soiltexture`` calls ``matplotlib.path.Path.contains_point``
inside a Python ``for`` loop. The agreement check confirms they produce
the same class labels (after normalizing soiltexture's "sandy clay loam"
to pedotri's stable "sandy_clay_loam" key).

Run with::

    pip install soiltexture pedotri
    python examples/bench_soiltexture.py
"""

from __future__ import annotations

from pedotri.bench import run_soiltexture_benchmark


def main() -> None:
    report = run_soiltexture_benchmark(
        sizes=[1_000, 10_000, 100_000, 740_745],
        classification="USDA",
        repeat=3,
    )
    print(report.format())
    print()
    for n, speedup in report.speedup_table():
        print(f"  pedotri is {speedup:5.1f}x faster than soiltexture at n={n:,}")


if __name__ == "__main__":
    main()
