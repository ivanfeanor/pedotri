# Spatially-correlated uncertainty sampling

0.3's `zonal_aggregate` documented a known weakness in big bold print:
it treats every pixel inside the AOI as an independent draw from its
per-pixel posterior. That gives a regional Q05/Q95 that's a **lower
bound** on the truth, because the central-limit shrinkage `σ_regional
= σ_per_pixel / √n_pixels` assumes independence — and SoilGrids
residuals are anything but. Two adjacent 250 m pixels share most of
their predictive uncertainty (same covariates, same neighbouring
training samples), so the regional posterior is much wider than the
independent-pixel calculation suggests.

0.4 fixes this with an optional spatial-correlation mode in
`zonal_aggregate`:

```python
agg = zonal_aggregate(
    region=village_polygon,
    properties={"soc": "soc_mean.tif"},
    correlation_range=2000.0,        # 2 km, in CRS units
    correlation_model="exponential",  # default
    n_samples=1000,
    seed=0,
)
```

Under the hood we draw whole 2-D Gaussian fields with a stationary
isotropic correlation function and average over the effective AOI per
draw. The result has the same per-pixel mean and variance the
independent sampler produced, but the regional mean carries a much
wider distribution.

## The three correlation kernels

Pedotri ships three classical isotropic correlation functions. All
satisfy `ρ(0) = 1` (unit variance at zero lag) and `ρ(d) → 0` as `d
→ ∞`. Let `L` be the correlation range (a length scale, in pixels
after conversion from CRS units).

### Exponential — `ρ(d) = exp(−d / L)`

The default, and the model implicitly assumed by most geostatistical
soil-mapping workflows (Hengl et al. 2017, *Random forest in
spatial prediction*). Heavy-tailed: pixels at `d = L` still have ~37 %
correlation, pixels at `d = 3L` still have ~5 %. Useful when the
underlying random process is *not* mean-square differentiable —
which is the empirical regime for soil properties at metre/kilometre
scales.

Mathematically equivalent to the Matérn covariance with smoothness
parameter `ν = 1/2`.

### Gaussian — `ρ(d) = exp(−(d / L)²)`

Drops off much faster: pixels at `d = L` have correlation `exp(−1) ≈
0.37`, pixels at `d = 2L` already at `exp(−4) ≈ 0.018`. The
underlying random field is *infinitely* mean-square differentiable —
mathematically convenient but rarely a faithful description of soil
data. Use it when you've fit a Gaussian variogram externally and
want pedotri to match it.

### Spherical — piecewise polynomial, compact support

```
ρ(d) = 1 − 1.5·(d/L) + 0.5·(d/L)³     for d ≤ L
ρ(d) = 0                              for d > L
```

The Matheron (1963) variogram model — exactly zero beyond the range,
no heavy tail. Useful when the physical interpretation calls for a
hard cutoff (e.g. patch sizes you know exceed any plausible
correlation length).

## How the sampler works (circulant embedding)

Given an isotropic correlation function `ρ` on a grid of size `H × W`,
we want to draw realisations of a Gaussian field whose covariance
between pixels `(i, j)` and `(i', j')` is `ρ(d)` where `d` is the
Euclidean distance between them. The naive approach — Cholesky-factor
the `HW × HW` covariance matrix — is `O((HW)³)` and runs out of memory
fast.

Davies (1987) and Wood & Chan (1994) noticed that if we embed the
target grid in a torus that's at least twice as large, the resulting
covariance matrix becomes *circulant*, and its eigendecomposition is
free: the eigenvalues are the discrete Fourier transform of its first
row, which `np.fft.fft2` gives us in `O(N log N)`.

The full algorithm (`pedotri.uncertainty.sample_correlated_field`):

1. **Pad** the grid to `(pad_factor · H, pad_factor · W)`. Default
   `pad_factor=2` is enough to suppress wrap-around aliasing when
   `L < H / 2`.
2. **Build** the first row of the circulant covariance: for each
   padded pixel, distance to the origin (with periodic minimum), then
   `ρ(d)`.
3. **Spectrum**: `λ = real(fft2(ρ))`. Tiny negative values from
   floating-point round-off are clipped to zero.
4. **Sample**: for each requested realisation, draw complex standard
   normals `z` of shape `(pad_factor·H, pad_factor·W)`, multiply
   element-wise by `√λ`, take `√N · real(ifft2(...))` (where
   `N = pad_factor² · H · W`), and crop to `(H, W)`.
5. **Scale and shift**: multiply by the per-pixel σ raster, add the
   per-pixel mean raster.

### Variance-preservation proof

With `z_k ~ CN(0, 1)` having `E[|z_k|²] = 2` (real and imaginary parts
each unit-variance N(0,1)), and `∑_k λ_k = N · ρ(0) = N` (the inverse
DFT of `λ` evaluated at the origin recovers `ρ(0)`, and `ρ(0) = 1`),
the field

```
x_n = √N · ifft2(√λ · z)_n = (1/√N) · ∑_k √λ_k · z_k · exp(+2πi k·n / N)
```

has expected squared modulus

```
E[|x_n|²] = (1/N) · ∑_k λ_k · E[|z_k|²] = (1/N) · 2 · N = 2.
```

Taking the real part halves the variance: `Var[Re x_n] = 1`. So the
field is unit-variance at every pixel before the per-pixel σ kicks in
— exactly what the independent sampler delivers, but with the
prescribed correlation. The regression test
`test_correlated_field_shape_and_unit_variance` checks this
numerically on a 40×40 grid and 400 samples.

## What "correlation_range" means in `zonal_aggregate`

When you pass `correlation_range=2000.0` to `zonal_aggregate`, the
number is interpreted in the **CRS units of the target grid**. If
your target grid is a UTM projection in metres, that's 2 km. If it's
EPSG:4326 in degrees, that's a giant correlation length you almost
certainly didn't mean. Pedotri does the conversion to pixels using
the target grid's average pixel side; the sampler operates entirely
in pixel space internally.

When neither a target grid nor a profile is in play (pure-ndarray
inputs), `correlation_range` is taken as-is in pixels. The unit
tests use this short path.

## Empirical verification

The implementation passes three regression checks that lock the
math down:

- **Unit variance per pixel** holds within 10 % across 400 draws on a
  40 × 40 grid.
- **Empirical correlation at d ≤ L** tracks `exp(−d / L)` within 0.1
  absolute on a 60 × 60 grid with 600 draws.
- **Regional Q05/Q95 inflation** vs. independent draws on the same
  grid is at least 3× when `L = 10` pixels — i.e. the whole point of
  the feature, demonstrated in
  `tests/test_uncertainty.py::test_correlated_field_inflates_regional_uncertainty`.

## Picking the right correlation range

For a SoilGrids-derived workflow, the practical recipe is:

1. Read off the empirical range from the SoilGrids covariance
   appendix (Hengl et al. 2017 publish a variogram per property). For
   sand and clay across most of Europe, 5–15 km gets you within a
   factor of two of the published value.
2. Don't push `correlation_range` larger than half the AOI
   dimension. Beyond that the regional draws degenerate into a
   single-mode distribution because every pixel becomes effectively
   the same draw.
3. Sensitivity-sweep over `correlation_range ∈ [L/2, L, 2L]` and
   report all three — that's an LCA reviewer's expected output, not
   a single Q05/Q95 pair.

## References

- Beven, K. J., & Kirkby, M. J. (1979). A physically based,
  variable contributing area model of basin hydrology. *Hydrological
  Sciences Bulletin* 24(1): 43–69. (TWI, not correlation — included
  here only because we use the same `sample_correlated_field`
  primitive in the TWI integration tests.)
- Davies, R. B. (1987). Algorithm AS 219: Tridiagonal approximation
  for sampling from a stationary Gaussian process. *Applied
  Statistics* 36: 247–251.
- Wood, A. T. A., & Chan, G. (1994). Simulation of stationary
  Gaussian processes in [0, 1]^d. *Journal of Computational and
  Graphical Statistics* 3(4): 409–432.
- Hengl, T. et al. (2017). SoilGrids250m: Global gridded soil
  information based on machine learning. *PLOS ONE* 12(2): e0169748.
- Matheron, G. (1963). Principles of geostatistics. *Economic
  Geology* 58: 1246–1266.
