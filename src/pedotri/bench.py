"""Reproducible benchmarks against alternative texture-classification libraries.

The flagship comparison is against the `soiltexture
<https://pypi.org/project/soiltexture/>`_ Python package
(`sagitta1618/soiltexture <https://github.com/sagitta1618/soiltexture>`_),
which implements the same point-in-polygon idea via
``matplotlib.path.Path.contains_point`` in a Python ``for`` loop. The
classifications it supports (``USDA``, ``FAO``, ``ISSS``,
``INTERNATIONAL``) all have direct pedotri equivalents, so the inputs
are byte-identical and the result keys round-trip with a small
remapping.

The benchmark is structured so you can drop it into a notebook,
include the numbers in a paper / blog post, or re-run it against
new pedotri releases to track regressions::

    from pedotri.bench import run_soiltexture_benchmark

    report = run_soiltexture_benchmark(sizes=[1_000, 10_000, 100_000])
    print(report.format())

The runner is deliberately self-contained: input arrays are seeded
deterministically and the timing loop uses :class:`time.perf_counter`
plus ``repeat`` cycles, so two consecutive runs on the same machine
should agree within a few percent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import pedotri

if TYPE_CHECKING:
    from collections.abc import Callable


#: Pedotri uses ``"sandy_clay_loam"`` while soiltexture uses
#: ``"sandy clay loam"``; the rest of the USDA / FAO / ISSS /
#: INTERNATIONAL classes differ in the same predictable way (underscore
#: vs. space). This map normalizes the soiltexture output so we can
#: directly compare against pedotri's stable keys without manual
#: remapping per benchmark.
SOILTEXTURE_TO_PEDOTRI_USDA: dict[str, str] = {
    "clay": "clay",
    "silty clay": "silty_clay",
    "silty clay loam": "silty_clay_loam",
    "sandy clay": "sandy_clay",
    "sandy clay loam": "sandy_clay_loam",
    "clay loam": "clay_loam",
    "silt": "silt",
    "silt loam": "silt_loam",
    "loam": "loam",
    "sand": "sand",
    "loamy sand": "loamy_sand",
    "sandy loam": "sandy_loam",
}


@dataclass(slots=True, frozen=True)
class BenchmarkRow:
    """Timing result for one (n_points, library) cell."""

    n_points: int
    library: str
    seconds: float
    points_per_second: float


@dataclass(slots=True)
class BenchmarkReport:
    """Collected timings + a 100 %-agreement check on outputs."""

    classification: str
    rows: list[BenchmarkRow] = field(default_factory=list)
    agreement: float | None = None

    def format(self) -> str:
        """Pretty-print the report as a fixed-width table for stdout / READMEs."""
        widths = (10, 14, 14, 16)
        header = (
            f"{'n_points':>{widths[0]}}  "
            f"{'library':>{widths[1]}}  "
            f"{'seconds':>{widths[2]}}  "
            f"{'points/sec':>{widths[3]}}"
        )
        lines = [
            f"benchmark vs. soiltexture (classification={self.classification!r})",
            "-" * len(header),
            header,
            "-" * len(header),
        ]
        lines.extend(
            f"{r.n_points:>{widths[0]},}  "
            f"{r.library:>{widths[1]}}  "
            f"{r.seconds:>{widths[2]}.4f}  "
            f"{r.points_per_second:>{widths[3]},.0f}"
            for r in self.rows
        )
        if self.agreement is not None:
            lines += ["", f"output agreement: {self.agreement * 100:.2f}%"]
        return "\n".join(lines)

    def speedup_table(self) -> list[tuple[int, float]]:
        """Return [(n_points, pedotri_speed / soiltexture_speed), ...]."""
        by_size: dict[int, dict[str, float]] = {}
        for r in self.rows:
            by_size.setdefault(r.n_points, {})[r.library] = r.points_per_second
        return [
            (n, pair["pedotri"] / pair["soiltexture"])
            for n, pair in sorted(by_size.items())
            if "pedotri" in pair and "soiltexture" in pair
        ]


def generate_inputs(
    n: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` valid (sand, clay) samples in [0, 100] with sand+clay ≤ 100.

    The distribution is uniform over the triangular feasible region
    ``{(s, c) : s ≥ 0, c ≥ 0, s + c ≤ 100}``. This guarantees no
    nonsense inputs (impossible silt) and reasonable class coverage.
    """
    rng = np.random.default_rng(seed)
    # Inverse-CDF sampling for the triangle (s, c) such that s+c ≤ 100:
    # draw two uniforms and flip if they sum > 100.
    a = rng.uniform(0.0, 100.0, size=n)
    b = rng.uniform(0.0, 100.0, size=n)
    over = (a + b) > 100.0
    a[over] = 100.0 - a[over]
    b[over] = 100.0 - b[over]
    return a.astype(np.float64), b.astype(np.float64)


def time_call(fn: Callable[[], object], *, repeat: int = 3) -> float:
    """Return the minimum wall time (s) across ``repeat`` runs of ``fn``."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def run_soiltexture_benchmark(
    sizes: list[int] | None = None,
    *,
    classification: str = "USDA",
    seed: int = 0,
    repeat: int = 3,
    check_agreement: bool = True,
) -> BenchmarkReport:
    """Time pedotri vs. ``soiltexture`` on equivalent inputs.

    Args:
        sizes: Sample counts to benchmark, e.g. ``[1_000, 10_000, 100_000]``.
            Defaults to ``[1_000, 10_000, 100_000]``.
        classification: Pedotri classification key. Must be one of
            ``USDA / FAO / ISSS / INTERNATIONAL`` (the set both
            libraries share).
        seed: Random seed for input generation.
        repeat: Each timing is the *minimum* of ``repeat`` runs.
        check_agreement: When ``True``, also run both libraries at the
            smallest size, normalize soiltexture's label format
            (``"sandy clay loam"`` → ``"sandy_clay_loam"``), and record
            the fraction of points on which they agree.

    Returns:
        :class:`BenchmarkReport`. Empty rows for ``soiltexture`` if the
        package is not importable — the pedotri timings are still
        reported so a CI run without the optional dependency degrades
        gracefully.
    """
    if sizes is None:
        sizes = [1_000, 10_000, 100_000]
    if classification not in {"USDA", "FAO", "ISSS", "INTERNATIONAL"}:
        raise ValueError(
            f"soiltexture only ships {{USDA, FAO, ISSS, INTERNATIONAL}}; "
            f"got classification={classification!r}."
        )

    report = BenchmarkReport(classification=classification)

    try:
        import soiltexture

        soiltexture_mod: Any | None = soiltexture
    except ImportError:
        soiltexture_mod = None

    for n in sizes:
        sand, clay = generate_inputs(n, seed=seed)

        def _run_pedotri(s: np.ndarray = sand, c: np.ndarray = clay) -> object:
            return pedotri.classify(s, c, classification)

        t_p = time_call(_run_pedotri, repeat=repeat)
        report.rows.append(BenchmarkRow(n, "pedotri", t_p, n / t_p))

        if soiltexture_mod is not None:

            def _run_soiltexture(
                s: np.ndarray = sand,
                c: np.ndarray = clay,
                mod: Any = soiltexture_mod,
            ) -> object:
                return mod.getTextures(s.tolist(), c.tolist(), classification=classification)

            t_s = time_call(
                _run_soiltexture,
                repeat=repeat,
            )
            report.rows.append(BenchmarkRow(n, "soiltexture", t_s, n / t_s))

    if check_agreement and soiltexture_mod is not None and classification == "USDA":
        sand, clay = generate_inputs(min(sizes), seed=seed)
        pedotri_out = pedotri.classify(sand, clay, classification)
        soiltexture_out = soiltexture_mod.getTextures(
            sand.tolist(), clay.tolist(), classification=classification
        )
        normalized = [SOILTEXTURE_TO_PEDOTRI_USDA.get(x, x) for x in soiltexture_out]
        agree = sum(1 for a, b in zip(pedotri_out, normalized, strict=True) if a == b)
        report.agreement = agree / len(pedotri_out)

    return report


__all__ = [
    "SOILTEXTURE_TO_PEDOTRI_USDA",
    "BenchmarkReport",
    "BenchmarkRow",
    "generate_inputs",
    "run_soiltexture_benchmark",
    "time_call",
]
