# Pedotransfer functions

A **pedotransfer function** (PTF) is a regression that estimates a
*hard-to-measure* soil property — typically a hydraulic-property
curve point or a Van Genuchten parameter — from *easy-to-measure*
inputs (sand, silt, clay, organic matter, bulk density). PTFs are
empirical: they only generalize within the range of soils they were
fit against.

Pedotri ships two of the most-used PTFs:

| Function | Predicts | Trained on | Best for |
|---|---|---|---|
| `pedotri.ptf.saxton_rawls` | θ at wilting (1500 kPa), field capacity (33 kPa), saturation; K&#x209B;, bulk density, air-entry tension | USDA-style mineral soils, US data, primarily agricultural | Quick water-balance modelling, agronomic decision support |
| `pedotri.ptf.wosten` (HYPRES) | Mualem-Van Genuchten parameters (θᵣ, θₛ, α, n, K&#x209B;, l) | HYPRES European soil dataset, continental EU | Driving full Richards-equation simulators (Hydrus, SWAP) |

Both are vectorized: pass arrays, get arrays.

## Saxton-Rawls (2006)

Saxton & Rawls 2006 is the de-facto default for water-balance models
that need only field capacity and wilting point. The inputs are
sand %, clay %, and organic matter %. Silt is implicit.

```python
from pedotri.ptf import saxton_rawls

r = saxton_rawls(sand=40, clay=20, organic_matter=2.0)
r.wilting_point             # θ at -1500 kPa, m³/m³
r.field_capacity            # θ at -33 kPa,   m³/m³
r.saturation                # θ at saturation, m³/m³
r.available_water           # FC − WP (the "TAW" number)
r.saturated_conductivity    # K_s, mm/h
r.bulk_density              # g/cm³
r.air_entry_tension         # ψ_e, kPa
```

### When does Saxton-Rawls fit your soil?

It was trained on agricultural soils with **organic matter ≤ 8 %** and
**clay ≤ 60 %**. Outside that envelope:

- **Histosols / peats**: don't use Saxton-Rawls — pedotri doesn't
  refuse, but the predictions are unreliable. Use a dedicated peat PTF
  or stick to measured retention curves.
- **Vertisols (>60 % clay, smectite-rich)**: regression
  underestimates available water; the swelling/shrinking behaviour
  isn't captured.
- **Sodic soils**: structure is collapsed, K&#x209B; predictions are
  optimistic by an order of magnitude.

### What does `density_factor` do?

Saxton-Rawls 2006 includes a compaction adjustment (Eq. 6-8). The
`density_factor` keyword applies it:

```python
# Compaction-corrected: roots see denser soil after heavy machinery
r_compact = saxton_rawls(40, 20, 2.0, density_factor=1.10)
print(r_compact.field_capacity, r_compact.bulk_density)
```

`density_factor=1.0` (default) gives the baseline regression. A
factor > 1 raises bulk density (so saturation θ drops and K&#x209B; falls);
a factor < 1 represents tilled / loosened soil. The valid range is
[0.9, 1.2] in the original paper — pedotri does *not* clip outside
that band, but you should not exceed it without justification.

### A complete one-liner with lab-report units

European labs often report organic *carbon* in g/kg, not organic
*matter* in %. Combine the `units=` keyword with the OC→OM helper:

```python
from pedotri.ptf import saxton_rawls
from pedotri.units import organic_carbon_to_organic_matter

oc_g_per_kg = 11.6  # straight from the lab
om_pct = float(organic_carbon_to_organic_matter(oc_g_per_kg / 10.0)[0])  # → ~2.0 %
saxton_rawls(sand=400, clay=200, organic_matter=om_pct * 10, units="g/kg")
```

(Or, more idiomatic: convert everything to percent at the top and call
`saxton_rawls` without `units=`. Either works.)

## Wösten / HYPRES (1999)

Wösten 1999 is the standard for **continuous Van Genuchten parameters**
across European soils. The inputs include silt and bulk density on
top of Saxton-Rawls', plus a topsoil/subsoil flag.

```python
from pedotri.ptf import wosten

w = wosten(
    sand=40, silt=40, clay=20,
    organic_matter=2.0, bulk_density=1.40,
    topsoil=True,
)
w.theta_s                   # θₛ, saturated water content
w.alpha                     # α, 1/cm
w.n                         # shape parameter
w.saturated_conductivity    # K_s, cm/day
w.l                         # Mualem pore-connectivity, dimensionless
w.theta_r                   # θᵣ, residual water content (fixed at 0.01)
```

### When does HYPRES fit your soil?

HYPRES was fit on **5521 European soil horizons** (mostly
W. & N. Europe). Outside that domain:

- **Tropical / lateritic soils**: not represented at all. Don't use.
- **Volcanic ash / andic horizons**: also outside the training set.
- **Heavy clays > 60 %**: extrapolation, predicted *n* tends toward
  pathologically small values.

If you're outside Europe, prefer **Rosetta** (USDA) or a regional PTF
fitted to your soil type. (Pedotri doesn't ship Rosetta; it's
neural-network-based and adds a TF/PyTorch dependency we want to keep
optional.)

### Topsoil vs subsoil

The `topsoil=True` flag selects the topsoil regression coefficients
from Wösten et al. 1999 Table 4. Use it for the A horizon (0-30 cm
typically); `topsoil=False` for B / C horizons.

### Bulk density: don't trust default values

The HYPRES regression is *sensitive* to bulk density — a 0.1 g/cm³
error shifts θₛ by 3-5 %. If you don't have a measured BD, either:

- Use the Saxton-Rawls BD prediction as a placeholder: it's
  texture-derived, so it captures the gross effect but not the
  compaction state.
- Or report your hydraulic predictions with an explicit BD uncertainty
  band by running the PTF at BD ± 0.1 g/cm³ and taking the spread.

## Choosing a PTF

| Need | Recommended |
|---|---|
| FC, WP, available water for an agronomic model | Saxton-Rawls |
| Full Van Genuchten curve to drive Hydrus / SWAP | Wösten |
| European soil | Either; Wösten if you have BD |
| US / global | Saxton-Rawls (or Rosetta externally) |
| Tropical / volcanic | Neither built-in PTF is appropriate — use a regional one or measure |

## What pedotri does *not* compute

These are deliberate exclusions (out of scope for 0.x):

- **Rosetta** (Schaap et al. 2001): requires a packaged neural-network
  model; planned for an optional extra.
- **Pachepsky / SCS-CN curve number**: hydrology-specific and not a
  classical PTF.
- **Pedotransfer for organic horizons**: would need a separate
  training set; not in 0.x.

If you want one of these, open an issue describing your use case and
the dataset you'd like fitted against.

## Related

- [Units-and-organic-matter](Units-and-organic-matter) — OC ↔ OM conversion, g/kg inputs
- [Classifications-and-when-to-use-them](Classifications-and-when-to-use-them) — HYPRES classification → Wösten PTF
- API docs: [`pedotri.ptf.saxton_rawls`](https://ivanfeanor.github.io/pedotri/ptf/#saxton-rawls), [`pedotri.ptf.wosten`](https://ivanfeanor.github.io/pedotri/ptf/#wosten)
