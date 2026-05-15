# Spatial correlation in regional uncertainty

When you average values from neighbouring pixels, the regional mean's
uncertainty depends on how much those pixels share. If they're truly
independent, the regional uncertainty shrinks by √N (basic central
limit theorem). If they share most of their predictive error — as
SoilGrids pixels do, because neighbouring pixels usually share the
same covariates and training samples — the regional uncertainty is
much wider than the √N shrinkage suggests.

Pedotri 0.3 documented this gap in big bold print: its independent-
pixel sampler produced regional Q05/Q95 that were a known **lower
bound** on the truth. 0.4 closes the gap with an opt-in correlated
sampler.

## When you need this

If you're propagating uncertainty through any **regional** estimate —
SOC stock per village, mean clay over a commune, water-holding
capacity on a farm — and reporting Q05/Q95 to a downstream consumer
who's going to make a decision with that interval, you need the
correlated mode. The independent-pixel default is fine when:

- You don't care about the interval width (just the mean).
- Your AOI is so small that there are only a handful of pixels in it
  (the spatial-correlation effect is dominated by sample noise anyway).
- You're producing a per-pixel map, not a regional aggregate.

Otherwise: pass `correlation_range=`.

## Quick example

```python
from pedotri.zonal import zonal_aggregate

agg = zonal_aggregate(
    region=village_polygon,
    properties={"soc": "soc_mean.tif"},
    correlation_range=2000.0,        # 2 km, in CRS units (UTM metres here)
    correlation_model="exponential",  # default
    n_samples=1000,
    seed=0,
)

agg["soc"].mean   # same as the independent-pixel mode
agg["soc"].q05    # noticeably wider than the independent-pixel Q05
```

The mean is unchanged — adding correlation doesn't bias the regional
average. The interval, on the other hand, can widen by an integer
factor. Whether it should is a function of how spatially correlated
your data actually is.

## Independent vs correlated — concrete numbers

`examples/aoi_soc_stock.py` runs the same SOC stock calculation in
both modes on a synthetic 100 × 100 grid:

| Mode | SOC mean (g/kg) | SOC Q05 / Q95 (g/kg) | Stock σ (t) |
|---|---|---|---|
| Independent (`correlation_range=None`) | 20.82 | 20.79 / 20.85 | 0.74 |
| Correlated (`correlation_range=4`) | 20.82 | 20.45 / 21.21 | 5.10 |

Same numerical setup, same RNG seed — only the correlation flag flipped.
The Q05/Q95 width on the regional mean grows from 0.06 g/kg to 0.76 g/kg
(13× wider); the SOC stock σ grows from 0.74 t to 5.10 t (7× wider).
Which one to report depends on what spatial correlation length is
defensible for your data — see *Picking the right correlation range*
below.

## Three correlation kernels

Pedotri ships three classical isotropic correlation functions. All
satisfy `ρ(0) = 1` (perfectly correlated at zero distance) and `ρ(d) →
0` for large `d`. `L` is the correlation range — a length scale, in
pixels after pedotri's conversion from your `correlation_range=`
argument in CRS units.

**`"exponential"` (default)** — `ρ(d) = exp(−d / L)`.

The model implicitly assumed by most geostatistical soil-mapping
work (Hengl et al. 2017). Heavy-tailed: pixels at `d = L` still
correlate at 37 %, at `d = 3L` at 5 %. Use when soil property
variation is *not* mean-square differentiable, which is the empirical
regime at metre / kilometre scales. Mathematically the Matérn ν = ½
special case.

**`"gaussian"`** — `ρ(d) = exp(−(d / L)²)`.

Drops off much faster (Matérn ν → ∞). Smooth, infinitely
differentiable — mathematically convenient but rarely a faithful
description of soil data. Use only if you've fit a Gaussian variogram
externally and want to match it.

**`"spherical"`** — piecewise polynomial with compact support.

```
ρ(d) = 1 − 1.5·(d/L) + 0.5·(d/L)³     for d ≤ L
ρ(d) = 0                              for d > L
```

The classic Matheron (1963) variogram model — exactly zero beyond the
range, no heavy tail. Use when the physical interpretation calls for
a hard cutoff (patch sizes that you know don't extend beyond `L`).

## Picking the right correlation range

The honest answer: it's a parameter you choose, defend, and
sensitivity-test. The starting points:

1. **For SoilGrids workflows**: read the empirical range off the
   covariance appendix in Hengl et al. 2017. Sand / clay across most
   of Europe sits at 5–15 km.
2. **Don't push `L` larger than half the AOI dimension.** Beyond that
   the regional draws degenerate into a single-mode distribution —
   every pixel becomes effectively the same draw, and the regional
   mean has no information from spatial averaging.
3. **Sensitivity-sweep over `L ∈ [L/2, L, 2L]`** and report all three.
   That's what an LCA reviewer expects — a single Q05/Q95 pair is
   not a defensible answer when the uncertainty depends materially on
   a modelling choice.

## How the sampler works

Drawing a Gaussian random field with a specified correlation function
on an `H × W` grid naively means Cholesky-factoring an `HW × HW`
covariance matrix — `O((HW)³)` and out of memory at any realistic
size. The standard fast alternative is **circulant embedding** (Davies
1987, Wood & Chan 1994):

1. Embed the target grid in a torus that's at least twice as large.
   The resulting covariance matrix is *circulant* — its
   eigenvectors are the discrete Fourier basis, and its eigenvalues
   are the FFT of its first row. `np.fft.fft2` gives you those in
   `O(N log N)`.
2. For each realisation: generate complex standard normals `z` on the
   padded grid, multiply by `√λ` (the eigenvalue weights), inverse-FFT,
   take the real part, crop back to the original grid.
3. Multiply by the per-pixel σ raster, add the per-pixel mean.

The full implementation lives in
`pedotri.uncertainty.sample_correlated_field`. It accepts a 2-D mean
+ σ raster, a correlation range in pixels, a sample count and seed.
You can call it directly when you want correlated draws outside the
zonal-aggregation flow.

### Variance-preservation proof

With `z_k ~ CN(0, 1)` (complex standard normal, `E[|z_k|²] = 2`) and
`∑_k λ_k = N · ρ(0) = N` (inverse DFT of `λ` at origin recovers `ρ(0)
= 1`), the sampled field

```
x_n = √N · IFFT(√λ · z)_n = (1/√N) · ∑_k √λ_k · z_k · exp(+2πi k·n / N)
```

has expected squared modulus

```
E[|x_n|²] = (1/N) · ∑_k λ_k · E[|z_k|²] = (1/N) · 2 · N = 2.
```

Taking the real part halves the variance: `Var[Re x_n] = 1`. So the
sampled field is unit-variance at every pixel before the per-pixel σ
kicks in — exactly what the independent sampler produces, just with
the prescribed correlation structure layered on top.

`tests/test_uncertainty.py` regression-tests this numerically on
realistic-sized grids:

- `test_correlated_field_shape_and_unit_variance` — per-pixel variance
  hits 1.0 within 10 % across 400 samples on a 40 × 40 grid.
- `test_correlated_field_recovers_exponential_kernel` — empirical
  correlation tracks `exp(−d / L)` within 0.1 absolute on a 60 × 60
  grid with 600 samples.
- `test_correlated_field_inflates_regional_uncertainty` — regional σ
  is at least 3× wider than the independent-pixel case under
  `L = 10` pixels.

## CRS units vs pixel units

When you pass `correlation_range=2000.0` to `zonal_aggregate`, it's
in the **CRS units of the target grid**. UTM metres → 2 km. WGS84
degrees → an enormous correlation length you almost certainly didn't
mean. The conversion to pixels uses the target grid's average pixel
side; the sampler operates entirely in pixel space internally.

Pure-ndarray inputs (no profile, no target grid) have no CRS reference,
so `correlation_range=` is taken as-is in pixels — useful for tests,
not for production.

## References

- Davies, R. B. (1987). Algorithm AS 219: Tridiagonal approximation
  for sampling from a stationary Gaussian process. *Applied Statistics*
  36: 247–251.
- Hengl, T. et al. (2017). SoilGrids250m: Global gridded soil
  information based on machine learning. *PLOS ONE* 12(2): e0169748.
- Matheron, G. (1963). Principles of geostatistics. *Economic Geology*
  58: 1246–1266.
- Wood, A. T. A. & Chan, G. (1994). Simulation of stationary Gaussian
  processes in [0, 1]^d. *Journal of Computational and Graphical
  Statistics* 3(4): 409–432.

## See also

- [Regional aggregation & SOC stock](Regional-aggregation-and-SOC-stock)
  — `correlation_range=` in context.
- [Uncertainty-aware classification](Uncertainty-aware-classification)
  — the per-pixel uncertainty story that this regional layer builds on.
- [ISO 14067 auditability](ISO-14067-auditability) — when you report a
  regional Q05/Q95, the audit trail must record the correlation range
  you used.
