# Classifications and when to use them

Soil texture classifications carve the sand-silt-clay simplex into
different sets of named regions. There is no "best" system — each one
encodes the agronomic / pedological priorities of the community that
designed it. Picking the right one matters more than people usually
admit: a "clay loam" in USDA is *not* the same physical material as a
"clay loam" in Avery (England & Wales), and the soil-water curves you
get from a PTF will reflect whichever system the regression was
trained on.

Pedotri ships 14 built-in classifications. Here is what each one is
actually good for, the regions it was designed against, and the
gotchas you'll hit if you use it outside that intended scope.

## Triangle (2-D) classifications

| Key | Region | Use it when… | Watch out for… |
|---|---|---|---|
| **USDA** | USA + global default | You're publishing internationally, sharing data across labs, or unsure. The lingua franca. | Coarse near the sand-silt boundary; misses fine clay distinctions important in Mediterranean soils. |
| **FAO** | International — coarse | You need a quick **coarse / medium / fine** grouping (3 classes). Common in FAO yield modelling and global crop maps. | Too coarse for field-scale agronomy; collapses ~12 USDA classes into 3. |
| **GEPPA** | France | French soil-survey data, INRA / GIS Sol references. Modern refinement of Jamagne. | 14 classes — finer than most international systems; "AL" / "ALo" boundaries may not round-trip cleanly with USDA. |
| **JAMAGNE** | France (legacy) | Reproducing pre-1995 French studies. Otherwise use GEPPA. | Superseded by GEPPA in current practice. |
| **HYPRES** | Europe (continental) | You're driving the Wösten (1999) PTF or any HYPRES-derived hydraulic-property database. | Only 5 classes; intentionally aligned with the Wösten parameter set, not for descriptive use. |
| **KA5** | Germany | German soil survey data, *Bodenkundliche Kartieranleitung* references. | 31 subclasses — much finer than USDA. Many subclasses share a parent so think about whether you actually need the full split. |
| **ISSS** | International, classical | Older European literature; pre-1980 reference. Uses 0.02 mm sand-silt cutoff. | Cutoff differs from USDA — see [Soil-conversions](Soil-conversions). |
| **INTERNATIONAL** | International (Leeper & Uren) | Australian / textbook references that pre-date Northcote. | Often confused with ISSS — verify which one your source uses. |
| **AVERY** | UK (England & Wales) | UK soil-survey data; current SSEW practice. | Slightly different boundary geometry from USDA in the loam region. |
| **NORTHCOTE** | Australia | Australian agronomy, especially heavy clays. Fine subdivision of the high-clay corner. | Less useful in sandy / loamy soils. |
| **CHINA** | China (GB/T 17296-2009) | Chinese national soil-survey data. | The 6-class split is closer to FAO's grouping than USDA's. |
| **EMBRAPA** | Brazil (SiBCS) | Brazilian agronomy / EMBRAPA datasets. | 5 classes — coarse; mostly used as a grouping after measurement. |
| **PTG** | Poland | Polish soil-survey data (PTG 2008). | Six classes, regional vocabulary. |

## One-axis (1-D) classifications

| Key | Region | Input | Use it when… |
|---|---|---|---|
| **KACHINSKY** | Russia & CIS | `physical_clay` — % particles &lt; 0.01 mm | Reading Russian / Soviet-era soil literature (9-class scheme). Cannot be converted from sand/silt/clay alone — you need a sieve PSD. See [Soil-conversions](Soil-conversions#kachinsky-the-non-triangle). |
| **RUS2004** | Russia (current) | `physical_clay` | Reproducing labels from the 2004 "Классификация и диагностика почв России" or the 2008 *Полевой определитель почв России*. Six classes; a strict coarsening of `KACHINSKY` (loose+cohesive sand → "песчаная"; light/medium/heavy clay → "глинистая"). Use this when you want the official Russian soil-survey "разновидности" label. |

## How to choose

1. **Where is the data from?** Use the local convention. USDA is *not* a safe default for French INRA data — it'll silently drop the GEPPA-specific distinctions your colleagues care about.
2. **What are you going to do with it?** If you're feeding a PTF, use the classification the PTF was *trained* against. HYPRES → Wösten. USDA → Saxton-Rawls.
3. **What is the resolution of your input?** Don't classify into 31 KA5 subclasses if your lab gave you sand/silt/clay rounded to the nearest 5 %. The classification reports false precision.

## Quick decision tree

```
Got geographic data with a specific country/region?
├── Yes → use the regional classification (GEPPA, KA5, AVERY, …)
└── No → 
    ├── Need it for a hydraulic-property PTF?
    │   ├── Saxton-Rawls (USDA-trained) → USDA
    │   ├── Wösten / HYPRES (continental EU) → HYPRES
    │   └── Other → match the PTF's training set
    └── Just need a label?
        ├── Coarse summary (3 classes) → FAO
        └── Detailed (12 classes) → USDA
```

## Related

- [Soil-conversions](Soil-conversions) — moving (sand, silt, clay) between standards with different cutoffs
- [Pedotransfer-functions](Pedotransfer-functions) — which classification each PTF expects
- [Custom-classifications-in-practice](Custom-classifications-in-practice) — when none of the built-ins fit your data
