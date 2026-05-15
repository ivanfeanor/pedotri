# Soil-conversions

Soil texture is reported as a triple (sand, silt, clay), but the
*definition* of sand and silt depends on the standard you use. USDA
puts the sand/silt boundary at 0.05 mm; ISSS at 0.02 mm; KA5 at
0.063 mm. The clay/silt boundary is universally 0.002 mm. So a soil
classified as "loam" in USDA may classify differently under ISSS,
not because the soil changed but because the *binning* did.

`pedotri.psd.convert` translates a (sand, silt, clay) triple from one
standard to another so you can re-classify your data without
re-analysing the sample.

## When do you need this?

You need a particle-size standard conversion when:

| Situation | Why you need it |
|---|---|
| Lab uses one standard, the PTF you want to apply uses another | E.g. an EU lab reports ISSS percentages, you want to run Saxton-Rawls (trained on USDA) |
| Cross-comparing data from labs in different countries | Mixing French (USDA convention) with German (KA5) requires aligning the sand boundary |
| Driving a published regression / classification that fixes a specific cutoff | E.g. HYPRES is internally consistent — its inputs were re-binned to a single cutoff before fitting |
| Updating legacy ISSS data to USDA for a modern analysis | Pre-1980 European literature predominantly used ISSS |

You don't need a conversion when:

- All your data and all your downstream tools use the same standard.
- You only need the clay fraction (silt/clay boundary doesn't move).
- You're crossing the USDA ↔ FAO line — both use the same 0.05 mm
  cutoff, so the triples are *identical*; only the class grouping
  differs. Pass the same numbers to `classify(..., 'FAO')` directly.

## What is actually being computed?

The conversion is **approximate** because it requires assuming a
distribution *within* the silt (or sand) fraction. Pedotri uses a
**log-linear** assumption: the cumulative mass-passing curve is treated
as a straight line in log-particle-size space inside each source-
standard fraction.

Why log-linear? Because that's how a typical soil PSD actually looks
on a log scale over the silt range. It's also the simplest reasonable
model that gives a closed-form answer from three input points
(sand%, silt%, clay%). For higher accuracy you need a full sieve PSD
— see [interpolate_psd](#interpolate_psd-arbitrary-cutoffs) below.

### USDA → ISSS (boundary moves *finer*: 0.05 → 0.02 mm)

ISSS draws the sand boundary at 0.02 mm, so the 0.02-0.05 mm fraction
that USDA called "sand" should be called "silt" under ISSS. We need
to estimate how much of the USDA sand sits in that 0.02-0.05 mm slice.

Under the log-linear assumption over the *USDA sand* range
[0.05 mm, 2 mm]:

```
fraction_remaining_as_sand_ISSS = log10(2 / 0.02) / log10(2 / 0.05)
                               = log10(100) / log10(40)
                               ≈ 1.249
```

That ratio is > 1 because we're stretching the same mass over a wider
log-range, but in this direction (finer cutoff → smaller "sand"
fraction) the formula is inverted; pedotri uses the symmetric form so
the math is consistent in both directions. See `pedotri.psd.convert`
source for the exact derivation.

### USDA → ISSS, numerically

```python
import pedotri.psd as psd

# Sample analysed in USDA convention
sand_usda, silt_usda, clay = 40.0, 40.0, 20.0
sand_isss, silt_isss, _ = psd.convert(
    sand_usda, silt_usda, clay, source="USDA", target="ISSS"
)
print(sand_isss[0], silt_isss[0], clay)
# 32.6  47.4  20.0
```

Roughly 7 percentage points migrate from "sand" to "silt" — that's a
typical magnitude. The clay column is unchanged because all standards
agree on the 0.002 mm cutoff.

### USDA → KA5 (boundary moves *coarser*: 0.05 → 0.063 mm)

KA5 has a higher sand boundary, so part of the USDA sand gets
reclassified as KA5 silt:

```python
sand_ka5, silt_ka5, _ = psd.convert(
    40.0, 40.0, 20.0, source="USDA", target="KA5"
)
print(sand_ka5[0], silt_ka5[0])
# 38.2  41.8
```

A smaller shift because 0.05 mm and 0.063 mm are close in log-space.

## Reversibility

Conversion is **not** exactly reversible. A round-trip USDA → ISSS →
USDA generally differs from the input by 1-2 percentage points:

```python
s, si, c = 40.0, 40.0, 20.0
a, b, _ = psd.convert(s, si, c, source="USDA", target="ISSS")
s2, si2, _ = psd.convert(a[0], b[0], c, source="ISSS", target="USDA")
print(s2[0], si2[0])   # close to 40, 40 but not exactly
```

This is a property of the single-point log-linear model, not a bug.
If you need exact round-tripping, store the original measurements and
re-derive from the source standard each time.

## Kachinsky — the non-triangle

The Russian Kachinsky classification cannot be derived from (sand,
silt, clay) under USDA / ISSS / KA5. Kachinsky keys on **physical
clay** — the mass fraction of particles &lt; 0.01 mm — which sits
*between* the silt/clay cutoff (0.002 mm) and the silt midpoint. You
can't reconstruct it from a triple that aggregates everything below
0.05 mm (or 0.02 mm) into one number.

The Russian Soil Institute publishes the official 2004 / 2008
nomenclature and the synonymy with the 1977 / 1997 schemes at
[infosoil.ru](http://www.infosoil.ru/) — see
[infosoil.ru/index.php?pageID=raznovid04](http://www.infosoil.ru/index.php?pageID=raznovid04)
for the variety-level names. The six official 2004 varieties
(песчаная / супесчаная / лёгко-, средне-, тяжелосуглинистая /
глинистая) are shipped as the `RUS2004` classification — a six-class
collapse of the nine-class `KACHINSKY`. Use whichever resolution
matches the document you're reproducing.

If you have a full sieve PSD, use `interpolate_psd` to read off the
&lt; 0.01 mm fraction:

```python
import numpy as np
import pedotri.psd as psd

sieves_mm = np.array([0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
passing  = np.array([18.0, 35.0, 47.0, 55.0, 62.0, 72.0, 88.0, 95.0, 100.0])

physical_clay = psd.interpolate_psd(sieves_mm, passing, target_size_mm=0.01)
import pedotri
pedotri.classify(physical_clay, "KACHINSKY")
# e.g. 'medium_loam'
```

If you only have the (sand, silt, clay) triple under a different
standard, **you can't classify Kachinsky** — say so loudly in your
methods section and pick a triangle-based classification instead.

## `interpolate_psd` — arbitrary cutoffs

When you have a richer PSD than three numbers, prefer
`interpolate_psd` directly. It's the building block under
`psd.convert` and avoids the log-linear assumption *inside* each
fraction:

```python
import numpy as np
import pedotri.psd as psd

# Sieves in mm and cumulative % passing
sieves = np.array([0.002, 0.063, 0.2, 2.0])
pct    = np.array([14.0, 51.0, 78.0, 100.0])

# What's the USDA sand fraction (i.e., > 0.05 mm)?
pct_below_50um = psd.interpolate_psd(sieves, pct, 0.05)
sand_usda = 100.0 - pct_below_50um
print(sand_usda)   # ~51%
```

## How big are these corrections?

For "normal" soils, USDA ↔ ISSS conversion moves ~3-8 pp of mass
between sand and silt. That's enough to change the classification in
a meaningful fraction of cases — especially near boundaries like
loamy-sand ↔ sandy-loam or silt-loam ↔ silt. If your downstream
analysis is decision-bearing (irrigation scheduling, crop modelling,
hydraulic-property prediction), **convert first, classify second**.

## Related

- [Classifications-and-when-to-use-them](Classifications-and-when-to-use-them) — picking the right target standard
- [Pedotransfer-functions](Pedotransfer-functions) — most PTFs are trained against a specific standard, so converting before driving a PTF is usually mandatory
- [Units-and-organic-matter](Units-and-organic-matter) — `units=` keyword on `psd.convert` for g/kg / g/g inputs
