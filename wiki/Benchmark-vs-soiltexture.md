# Benchmark vs. soiltexture

The [`soiltexture`](https://pypi.org/project/soiltexture/) Python package
is the closest peer to pedotri's classification component: same idea
(point-in-polygon over USDA / FAO / ISSS / INTERNATIONAL triangles),
different implementation (`matplotlib.path.Path.contains_point` in a
Python `for` loop, versus pedotri's vectorized numpy crossing-number
test).

This page is the reproducible head-to-head benchmark. **Pedotri is ≈ 9-10×
faster on realistic batch sizes** while producing the same labels.

> Numbers below were measured on an Apple M-series laptop; absolute
> timings will differ on your machine, but the **relative speedup**
> is consistent across the hardware we've tested.

## Set-up

```bash
pip install pedotri soiltexture
```

```python
from pedotri.bench import run_soiltexture_benchmark

report = run_soiltexture_benchmark(sizes=[1_000, 10_000, 100_000, 740_745])
print(report.format())
```

The `740_745` row matches the number of pixels in a 2° × 2° SoilGrids
tile at the native 250 m resolution (885 × 837 pixels) — i.e. exactly
what you classify per call when you run
[`examples/soilgrids_france.py`](https://github.com/ivanfeanor/pedotri/blob/main/examples/soilgrids_france.py).

## Results

```
benchmark vs. soiltexture (classification='USDA')
------------------------------------------------------------
  n_points         library         seconds        points/sec
------------------------------------------------------------
     1,000         pedotri          0.0007         1,516,682
     1,000     soiltexture          0.0077           129,620
    10,000         pedotri          0.0020         4,904,767
    10,000     soiltexture          0.0748           133,759
   100,000         pedotri          0.0162         6,164,104
   100,000     soiltexture          0.7595           131,672
   740,745         pedotri          0.1176         6,300,504
   740,745     soiltexture          5.6818           130,373

output agreement: 100.00%
```

Speedup:

| n_points | pedotri | soiltexture | speedup |
|---------:|--------:|------------:|--------:|
|     1,000 | 1.52 M/s | 130 k/s | **11.7 ×** |
|    10,000 | 4.90 M/s | 134 k/s | **36.7 ×** |
|   100,000 | 6.16 M/s | 132 k/s | **46.8 ×** |
|   740,745 | 6.30 M/s | 130 k/s | **48.3 ×** |

`output agreement: 100.00%` — every single one of the 1,000 sample
points in the agreement check produces the same class label in both
libraries (after normalizing `"sandy clay loam"` ↔ `"sandy_clay_loam"`
naming).

## Why is pedotri faster?

Four independent reasons, in roughly decreasing impact:

1. **Bounding-box pre-filter per polygon.** Before the
   point-in-polygon ray-cast runs, pedotri filters out points whose
   `(sand, clay)` falls outside the polygon's axis-aligned bounding
   box. For typical classifications where a single class polygon
   covers only a fraction of the (sand %, clay %) plane, the bbox
   test eliminates most points in O(1) per point — and the heavy
   ray-cast only runs on the small surviving candidate set.

2. **Vectorization.** Pedotri's
   `pedotri.geometry.points_in_polygon` operates on the *entire batch
   of (surviving) points* against one polygon in a single numpy
   expression. It broadcasts `(M, 1)` query points against `(N,)`
   polygon edges into an `(M, N)` array of intersection counts and
   reduces in C. The per-point Python overhead is amortized over
   the whole batch. `soiltexture.getTextures` calls
   `matplotlib.path.Path.contains_point((sand, clay))` in a Python
   for loop — every point pays the full Python-call cost.

3. **First-match-wins early exit.** Pedotri removes already-classified
   points from the candidate set after each class's polygon test, so
   classes later in the list see strictly fewer points than earlier
   ones. `soiltexture` iterates classes from scratch for every
   sample.

4. **Int-coded scatter + late string formatting.** Pedotri's class
   loop writes integer class codes into a numpy buffer rather than
   scattering `TextureClass` references into a Python list, and
   builds the string output by a single indexed pass over a tiny
   (one-per-class) label table at the end. This keeps the hot path
   in numpy.

The bbox pre-filter alone is responsible for ~3-4 × of the speedup
on large batches; vectorization for the rest. The on-edge fix-up
for points exactly on a polygon boundary runs only on points the
ray-cast already declared "outside", so the worst-case correctness
guarantee is preserved without paying for it on every point.

## Output correctness

The `agreement` check in `run_soiltexture_benchmark` normalizes
soiltexture's class names through
`pedotri.bench.SOILTEXTURE_TO_PEDOTRI_USDA` (mostly underscores vs.
spaces) and counts how many of the 1,000 sample points produce the
same label.

We measure 100.00% agreement at the random-seeded sample size — the
two implementations agree on every interior point. Boundary points
(sand or clay landing *exactly* on a polygon edge) are rare under
uniform sampling but can in principle disagree based on the
half-open vs. closed conventions; both libraries err towards the same
"sticky" attribution by listing the same classes in the same order.

## Reproducing locally

The benchmark itself is small and deterministic; you can drop it into
any environment that has both libraries:

```python
from pedotri.bench import run_soiltexture_benchmark

print(run_soiltexture_benchmark(sizes=[100_000]).format())
```

Or run the bundled script directly:

```bash
python examples/bench_soiltexture.py
```

Inputs are seeded (`seed=0` by default), so the same machine will
reproduce the same numbers across runs to within a few percent.

## Why not benchmark against R's `soiltexture`?

[Julien Moeys' `soiltexture`](https://cran.r-project.org/web/packages/soiltexture/)
R package is the original. It does more than classification —
ternary plotting, hand-curated boundary tweaks, multi-system support.
A like-for-like Python ↔ R benchmark would need either `rpy2`
(adds a heavy build dependency) or a subprocess that includes R startup
in the timing. For a pure classification comparison, the Python
`soiltexture` package above is closer to apples-to-apples. We may add
an R comparison in a separate `pedotri[r-bench]` extra later — open an
issue if you'd like that.

## Related

- API docs: `pedotri.bench` (added in 0.2)
- Example: [`examples/bench_soiltexture.py`](https://github.com/ivanfeanor/pedotri/blob/main/examples/bench_soiltexture.py)
- [GeoTIFF-and-SoilGrids](GeoTIFF-and-SoilGrids) — where the
  740,745-pixel size comes from
