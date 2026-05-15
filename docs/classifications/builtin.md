# Built-in classifications

| Key             | Region        | Axes           | Classes | Reference                                          |
|-----------------|---------------|----------------|---------|----------------------------------------------------|
| `USDA`          | USA (global)  | sand, clay     | 12      | USDA Soil Survey Manual (Handbook 18, 1993)        |
| `FAO`           | International | sand, clay     | 3       | Verheye & Ameryckx 1984 (FAO grouping)             |
| `INTERNATIONAL` | International | sand, clay     | 11      | Leeper & Uren 1993                                 |
| `ISSS`          | International | sand, clay     | 12      | ISSS / Verheye & Ameryckx 1984                     |
| `AVERY`         | UK            | sand, clay     | 12      | Avery 1980 (Soil Survey of England and Wales)      |
| `JAMAGNE`       | France        | sand, clay     | 13      | Jamagne 1967                                       |
| `GEPPA`         | France        | sand, clay     | 14      | Baize & Jamagne 1995 (GEPPA-Aisne refinement)      |
| `HYPRES`        | Europe        | sand, clay     | 5       | Wösten et al. 1999                                 |
| `KA5`           | Germany       | sand, clay     | 31      | Bodenkundliche Kartieranleitung 5, 2005            |
| `PTG`           | Poland        | sand, clay     | 6       | Polskie Towarzystwo Gleboznawcze 2008              |
| `NORTHCOTE`     | Australia     | sand, clay     | 16      | Northcote 1979 (Factual Key)                       |
| `CHINA`         | China         | sand, clay     | 6       | GB/T 17296-2009                                    |
| `EMBRAPA`       | Brazil        | sand, clay     | 5       | Embrapa, SiBCS 5ª ed. 2018                         |
| `KACHINSKY`     | Russia / CIS  | physical_clay  | 9       | Kachinsky 1965 (by particles < 0.01 mm)            |

## Particle-size cutoffs

Most built-in classifications operate on the standard sand–silt–clay simplex, but the *particle-size cutoffs* differ between standards. See [`pedotri.psd`](../api.md) for conversions between them.

| Standard | Sand–silt boundary | Silt–clay boundary |
|----------|-------------------:|-------------------:|
| USDA / FAO                | 0.050 mm | 0.002 mm |
| ISSS / INTERNATIONAL      | 0.020 mm | 0.002 mm |
| KA5                       | 0.063 mm | 0.002 mm |

Convert before classifying when your data is on a different standard from the classification you want to use:

```python
from pedotri.psd import convert

# Lab report in USDA cutoffs; classify with ISSS-based scheme
sand, silt, clay = convert(60, 30, 10, source="USDA", target="ISSS")
```

## Localized class names

All built-in classifications ship with class names in `en`, `fr`, `de`, `es`, `ru`, and `pt` (where the translation is well established). Classifications native to a particular language default to that language:

- `GEPPA` and `JAMAGNE` → `fr`
- `KA5` → `de`
- `KACHINSKY` → `ru`
- `EMBRAPA` → `pt`
- `PTG` → `pl`
- `CHINA` → `zh`
- All others → `en`

`pedotri.classify(..., locale=None)` always returns the stable class key; pass a locale tag to get the localized name.

## Polygon provenance

Built-in polygons fall into two tiers of fidelity:

**Tier 1 — direct transcription from primary literature.** Polygon vertices come from the original published reference and have been spot-checked against published reference points or against authoritative public implementations:

- `USDA` — Soil Survey Manual, Handbook 18 (1993), Table 3.1
- `FAO` / `ISSS` — Verheye & Ameryckx (1984), Pedologie
- `HYPRES` — Wösten et al. (1999), Geoderma 90(3-4)
- `AVERY` — geometric clone of USDA per Avery 1980 (which adopted USDA boundaries)
- `EMBRAPA` — simplified 5-class grouping per SiBCS 5ª ed. (2018)

**Tier 2 — interpreted from textbook descriptions.** Boundaries reconstructed from class definitions stated in the cited reference; geometry is plausible but may need refinement once a domain reviewer cross-checks against authoritative reference points:

- `KA5` — full 31-class subdivision derived by converting the published (clay %, silt %) class ranges to (sand %, clay %) polygons
- `GEPPA` / `JAMAGNE` — French texture triangle boundaries from Jamagne 1967 and Baize & Jamagne 1995; the AA / A split is the documented difference between the historical and modern variants
- `NORTHCOTE` — Australian factual-key boundaries mapped onto the sand-clay plane (the Factual Key is primarily a field-test protocol, not a lab-data triangle)
- `PTG` — six-class simplification of the Polish 2008 standard's finer class set
- `CHINA` — six-class grouping per GB/T 17296-2009
- `KACHINSKY` — 1-D interval definitions per Качинский (1965) (no geometric ambiguity since intervals are explicit)

The property-based test suite asserts that every Tier 1 and Tier 2 classification *tiles* the full sand-silt-clay simplex with no gaps; coverage holes are caught by `tests/test_properties.py::test_tiling_classifications_cover_simplex`.

## Hierarchy

Where the underlying reference provides a coarser grouping, the class `group` field exposes it. Typical groups across the built-ins:

- `fine` — clay-dominant
- `moderately_fine` — clay loams
- `medium_fine` — loams with elevated clay
- `medium` — loams and silts
- `moderately_coarse` — sandy loams
- `coarse` — sands

The plotting backends use these groups to colour-code class polygons along the coarse-to-fine gradient.
