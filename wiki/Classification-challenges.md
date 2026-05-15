# Classification challenges — and how pedotri helps

Anyone who's tried to *use* soil-texture classifications across
projects, countries, or time periods runs into the same recurring
problems. They're not specific to one country or one decade — they
show up wherever soil scientists have classified soils, which is
everywhere. This page lays out those problems and points to the
pedotri features that address each.

This isn't a "look how good pedotri is" pitch. Most of these
problems aren't *fully* solvable by a library — they're rooted in how
classifications were authored. But several of them are at least
*mitigable*, and that's where pedotri tries to be useful.

## 1. The same name does not mean the same soil

"Clay loam" in **USDA** (clay 27-40 %, sand 20-45 %) is not the same
material as "clay loam" in **Avery** (UK), which is not the same as a
"clay loam" entry in older textbooks following the now-superseded
**International** system.

Even within one system, the *underlying particle-size cutoff* drifts:

| System | Sand boundary | Silt boundary | "Loam" meaning |
|---|---|---|---|
| USDA | 0.05 mm | 0.002 mm | a specific polygon |
| ISSS | 0.02 mm | 0.002 mm | a *different* polygon — looks similar on the triangle but the input numbers don't match |
| KA5 | 0.063 mm | 0.002 mm | another polygon |
| Russian (Качинский, 1965) | — | particles <0.01 mm | a 1-D classification by "physical clay" — not a triangle at all |

**Consequence:** a sample reported as "loam" in one document and used
as "loam" in another may classify into a wholly different polygon
once the analyst tries to plot it on the receiving system's triangle.

**What pedotri does:**

- Ships **15 classifications** as built-ins, so you don't have to
  guess what someone else's "loam" means — you can re-classify the
  sample under the convention you actually want.
- Provides `pedotri.psd.convert(..., source=, target=)` to convert
  the (sand, silt, clay) triple between the four sand-cutoff
  conventions (USDA / FAO / ISSS / KA5). The conversion is a
  log-linear approximation, not exact, and we explain the assumptions
  on the [Soil-conversions](Soil-conversions) page so you know when
  to trust it.

```python
import pedotri
import pedotri.psd as psd

# Sample described under ISSS — re-classify under USDA
sand_usda, silt_usda, clay = psd.convert(
    40.0, 40.0, 20.0, source="ISSS", target="USDA"
)
pedotri.classify(float(sand_usda[0]), clay, "USDA")
```

## 2. Different classifications use different *axes*

Most international systems classify on the **sand-silt-clay
triangle** (a 2-D simplex). The Russian Качинский / *Полевой
определитель почв России* (2008) uses a **single axis**: physical
clay, the mass fraction of particles &lt;0.01 mm. The two cannot be
mapped 1-to-1 from a three-number lab report alone — the 0.002-0.01
mm range that Качинский folds into "physical clay" cuts across the
USDA silt/clay boundary, so you'd need a full sieve PSD to translate
losslessly.

This isn't a bug — it's a deliberate scientific choice. The Russian
community decided that a single, water-retention-correlated cutoff is
more useful for the soils they work with than the triangle.

**What pedotri does:**

- Treats axes as **first-class metadata** on every classification.
  `pedotri.classify` adapts its calling convention to the axes the
  classification declares:

  ```python
  pedotri.classify(60, 20, "USDA")          # two axes: sand, clay
  pedotri.classify(35, "KACHINSKY")          # one axis: physical_clay
  pedotri.classify(35, "RUS2004")            # same axis, six-class official label
  ```

- Provides `pedotri.psd.interpolate_psd(sieves, passing, target)` so
  you can derive physical clay from a richer PSD when you have one —
  the right way to translate to Качинский, instead of pretending the
  triangle is enough.

- Documents on the [Soil-conversions](Soil-conversions) page exactly
  *why* the triangle ↔ physical-clay conversion is lossy and which
  measurements you'd need to do it properly.

## 3. Classifications drift over time

A "GEPPA" classification (France, 1995) is a refinement of "Jamagne"
(1967), which was itself a refinement of older French soil-survey
work. Across the Russian classifications, the **1977** "Классификация
и диагностика почв СССР", the **1997** revision, and the **2004**
"Классификация и диагностика почв России" each renamed taxa, merged
or split classes, and changed diagnostic horizons. The 2004 work even
introduces a wholly new family of taxa for **anthropogenically
transformed soils** and **technogenic surface formations (ТПО)** that
simply did not exist in 1977.

The Russian Soil Institute's [infosoil.ru](http://www.infosoil.ru/)
maintains explicit **synonymy tables** — one for 1977 ↔ 1997, one for
1997 ↔ 2004, and one for the Russian 1988 soil map ↔ WRB/FAO. The
mere fact that those tables need to be authored at all is the
problem: a "клееземы тундровые" of 1988 maps onto multiple distinct
WRB / FAO entries depending on which sub-property you key on.

The French / Russian cases are not exotic. Australian *Northcote* has
gone through similar refinements; the German KA5 expanded to its
current 31 subclasses from a smaller predecessor; ISSS lost ground to
USDA partly *because* it was harder to re-key against on new
measurements.

**What pedotri does:**

- Ships both legacy and modern classifications side-by-side: **GEPPA
  and JAMAGNE** (France), **KACHINSKY 9-class and RUS2004 6-class**
  (Russia), **ISSS and USDA** (international). You can re-classify
  the same sample under each one and see what changes — which is
  often more informative than picking a single one and hoping it's
  the right answer.

- Treats class **keys** as stable identifiers separately from class
  **names**, so renaming a class (or translating it) never breaks
  downstream code. This is the same pattern that lets us add the
  6-class Russian 2004 variety scheme alongside Качинский without
  shadowing or renaming anything.

- Doesn't pretend to ship synonymy tables. Those are domain knowledge
  that belongs in a citable reference (and we link to one for the
  Russian case in [Soil-conversions](Soil-conversions)). Pedotri's
  job is to let you *classify in either system and compare* — not to
  invent a cross-reference.

## 4. Classifications are not always *texture* classifications

Most national soil-survey schemes are **profile-genetic**: they
classify soils by what genetic horizons formed, in what order, under
what climate and parent material. Texture is one *property* used to
diagnose a horizon — not the classification itself.

The Russian 2004 system is explicit about this: the variety
("разновидности") level — the only level on which texture appears —
is *outside* the hierarchical taxonomic system. WRB does the same:
texture appears as a qualifier (Arenic / Loamic / Clayic / Siltic),
not as a primary axis.

**Consequence:** if you have a profile description and want to
classify it under WRB or the Russian 2004 system end-to-end, you need
a horizon analyzer, not a texture classifier.

**What pedotri does *not* try to do:**

- It does not classify by genetic horizons, parent material, water
  regime, or any non-textural property. It is a **texture
  classifier**, full stop. Profile-genetic classification (WRB,
  Soil Taxonomy, the Russian 2004 system at типа/подтипа level) is
  out of scope.

**What pedotri *does* do:**

- Provides texture classification as a clean, composable building
  block. If your pipeline starts with profile analysis and ends with
  "OK, what texture variety is this Ap horizon?", pedotri's
  `classify` is the right answer for the last step. It's *not* the
  right answer for "is this a Чернозём or a Кастанозём?" — that's a
  different (and much harder) classification problem.

- Returns enough metadata via `detailed=True` to be honest about how
  close a sample is to a class boundary:

  ```python
  result = pedotri.classify(45, 15, "USDA", detailed=True)
  result.key       # 'loam'
  result.distance  # 2.3 — modest distance from the boundary
  ```

  A small `distance` value is your signal to *not* over-trust the
  label: the sample is borderline and a different system, or a small
  measurement error, would put it elsewhere.

## 5. There are always disturbed soils that nobody planned for

Every classification was designed against a population of "well-
formed" soils. Urban soils, mine spoils, landfill caps, dredged
sediments, ash deposits — these are real surface materials,
classifiable in the trivial sense (they have a sand / silt / clay
fraction you can measure), but in the rich sense they fall through
every genetic-classification framework. The Russian community made
this explicit: their 2004 scheme adds a wholly separate ствол
("trunk") for **антропогенно-преобразованные почвы** (anthropogenically
transformed soils) and a parallel non-soil system of **ТПО** —
*techogenic surface formations* (квазизёмы, артифабрикаты, литостраты,
урбистраты, …) — that explicitly **cannot** be a genetic soil
classification because they have no genetic horizons.

**What pedotri does:**

- Honors the input you give it. If you measure the texture of an
  urban litho­strat (urban dredge), pedotri will classify the
  *texture* fine — that's a meaningful, defensible measurement. It
  will not tell you "this is не почва" (not a soil) because that's
  not a textural judgment.

- Lets you ship a **custom classification** for your domain. If your
  workflow uses an urban-soils grouping like "compactible / friable /
  contaminated", you can author it as a TOML and load it through
  `pedotri.register_classification`. See [Custom-classifications-in-
  practice](Custom-classifications-in-practice).

## 6. Most classifications were never i18n'd

Russian soil literature uses Russian class names; French uses French;
German uses German — and the receiving English-speaking pipeline
often silently re-labels everything as USDA's English names, losing
the connection to the source convention.

**What pedotri does:**

- Every built-in classification ships **class names in six
  languages** (en / fr / de / es / ru / pt), plus the source-region
  language as default (`KACHINSKY` defaults to `ru`, `GEPPA` to `fr`,
  `EMBRAPA` to `pt`). The class **key** (`"sandy_clay_loam"`) stays
  stable across locales; only the **display name** changes.

  ```python
  pedotri.classify(13, 50, "USDA")                     # 'clay' (key)
  pedotri.classify(13, 50, "USDA", locale="ru")        # 'глина'
  pedotri.classify(13, 50, "GEPPA", locale="ru")       # 'тяжёлая глина'
  ```

- Falls back through the locale chain
  (`fr-FR` → `fr` → `en` → class key) so a partial translation never
  silently drops to an empty string.

## 7. People don't have one classification — they have several

In practice, a single dataset gets classified under several systems
over its lifetime: a French soil sample analyzed at INRA might be
GEPPA-labeled in its original record, USDA-labeled when entered into
an international database, FAO-labeled when summarized for a
continental-scale model, and Качинский-labeled when discussed in
collaboration with Russian colleagues.

Tools that handle one classification at a time impose a translation
cost at every interface.

**What pedotri does:**

- Treats all classifications **uniformly**. The same `classify()`
  call works against every built-in and every user-registered TOML.
  Same DataFrame accessor (`df.soil.classify("USDA")`,
  `df.soil.classify("GEPPA")`), same vectorized performance, same
  error handling.

- Ships a `units=` keyword so the *input* convention (%, g/kg, g/g)
  can also drift between projects without forcing manual conversion
  at every boundary.

## Where pedotri can't help

Some problems aren't ours to solve, and it's better to say so:

- **No reference data conversion.** If you have a sample classified
  as "глеезём арктический" in the Russian 1988 soil map legend, no
  amount of texture classification will give you the WRB equivalent
  — that's a synonymy / type-conversion problem in
  *profile-genetic* classification space.
  [infosoil.ru/index.php?pageID=mapcorr](http://www.infosoil.ru/index.php?pageID=mapcorr)
  publishes the table; pedotri does not duplicate it.
- **No fitted PTFs outside Saxton-Rawls / Wösten.** Russian PTFs
  (Vadyunina-Korchagina, Kachinsky bulk-density relations) are not
  shipped. If you need them, ship them via the same TOML extension
  mechanism a user can extend custom classifications with — or open
  an issue.
- **No genetic / morphological classification.** As above. WRB and
  Soil Taxonomy are out of scope.

## Related

- [Classifications-and-when-to-use-them](Classifications-and-when-to-use-them) — what each built-in is good for
- [Soil-conversions](Soil-conversions) — translating triples between cutoffs
- [Custom-classifications-in-practice](Custom-classifications-in-practice) — when none of the built-ins fit
