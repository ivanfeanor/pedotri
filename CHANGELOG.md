# Changelog

All notable changes to pedotri are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] — 2026-05-15

### Added

#### 0.4 — geospatial plumbing, spatial correlation, DEM, ISO 14040

- **`pedotri.grid`** — `TargetGrid` dataclass (CRS + affine + width + height) with `from_profile` / `from_bounds` constructors. `reproject_to_grid(array, profile, target)` and `align_to_grid(sources, target=)` warp heterogeneous inputs onto a common grid via `rasterio.warp` with per-dtype resampling defaults (nearest for integers, bilinear when upsampling continuous, average when downsampling continuous). Documented affine + GDAL inverse-mapping math.
- **`zonal_aggregate(target_grid=)`** — path-based property bands and the mask raster are auto-aligned to the target grid before the AOI is computed. ndarray inputs stay raw. Fixes the "SoilGrids 250 m + WorldCover 10 m" mismatch that 0.3 left to the user.
- **`pedotri.uncertainty.sample_correlated_field`** — FFT-based circulant-embedding sampler (Davies 1987 / Wood & Chan 1994) for spatially-correlated Gaussian fields with three isotropic correlation kernels: exponential (default, Matérn ν=½), Gaussian, and spherical. Documented variance-preservation proof; regression tests verify empirical correlation matches `exp(−d / L)` within 0.1 absolute on 600-sample grids.
- **`zonal_aggregate(correlation_range=, correlation_model=)`** — opt-in spatially-correlated MC for regional aggregation. Turns the documented "lower bound" on regional Q05/Q95 into a real bound by sampling whole correlated fields instead of independent pixels.
- **`pedotri.dem`** — `slope` / `aspect` (Horn 1981), `profile_curvature` / `plan_curvature` (Zevenbergen & Thorne 1987), and `topographic_wetness_index` (Beven & Kirkby 1979). Pure numpy, with the published formulas inline in the docstrings + a wiki page covering the derivations.
- **`pedotri.audit`** — `Provenance`, `DataSource`, `AuditTrail` for ISO 14040 / 14044 LCI auditability. Every result dataclass (`SoilGridsPoint`, `WorldCoverAOI`, `ZonalAggregate`, `AggregateDistribution`) gains an optional `.provenance` field auto-populated by its producer. `AuditTrail.to_json(path)` exports a self-contained traceability log with operation, parameters, sources, seed, software version, and the upstream chain. Replayable: same seed + same parameters → byte-identical samples (regression-tested).
- **Four new wiki pages**: `Reprojection-and-target-grids`, `Spatial-correlation`, `DEM-derived-indices`, `ISO-14040-auditability`. Each carries the math and references (Davies 1987; Wood & Chan 1994; Horn 1981; Zevenbergen & Thorne 1987; Beven & Kirkby 1979; ISO 14040:2006 / 14044:2006).
- **`examples/aoi_soc_stock.py`** refreshed to use `correlation_range=` and expose the audit trail in the printed summary — the SOC stock Q05/Q95 now visibly widens (from ~2 t to ~16 t on the synthetic 5 000 ha AOI) when the correlated sampler kicks in, matching the documented expectation.

#### API polish pass — ergonomic refactor of the 0.3 surface

After a self-review against the [mef-agroref](https://github.com/) production codebase, the 0.3 surface got one focused cleanup pass before tagging:

- **`Quantiles(q05, q95)` dataclass** in `pedotri.uncertainty` — the new self-documenting form for `*_uncertainty` kwargs. The bare 2-tuple `(q05, q95)` shorthand still works but emits a `DeprecationWarning` recommending `Quantiles(...)`; tuples of two numbers were genuinely ambiguous (could be read as `(low, high)` of an error bar, `(−err, +err)`, …) and silently produced wrong σ.
- **One uncertainty kwarg per axis everywhere.** `classify_array_with_uncertainty` and `classify_geotiff_with_uncertainty` no longer expose `sand_q05` / `sand_q95` / `sand_sigma` (six kwargs for two axes); they now take `sand_uncertainty=` / `clay_uncertainty=` accepting `Quantiles`, scalar/array σ, or — for the GeoTIFF helper — a `(q05_path, q95_path)` 2-tuple of file paths. Matches the point API.
- **`zonal_aggregate(properties=...)` accepts richer spec shapes.** A property's value can now be a dict (`{"mean": …, "uncertainty": Quantiles(q05, q95)}` or `{"mean": …, "sigma": …}` or the legacy `{"mean": …, "q05": …, "q95": …}`) or a 2-tuple `(mean, uncertainty)`. The original 3-tuple `(mean, q05, q95)` keeps working but emits a `DeprecationWarning`.
- **Auto-promote `detailed=True` on uncertainty.** Calling `pedotri.classify(..., sand_uncertainty=Quantiles(...))` now returns a `ClassifyResult` automatically — no more "Uncertainty kwargs require detailed=True" error. The boilerplate the old check enforced added nothing.
- **`ZonalAggregate.combine(fn)` inspects fn's signature.** Lambdas only need to declare the property names they actually consume; extras are silently dropped. Functions using `**kwargs` still receive every property.
- **`RasterClassification` helper API.** Added `has_confidence`, `has_probabilities`, `n_classes` properties, a `modal_confidence()` method that picks the right field for the active method, and `class_probability(class_key)` that walks the top-k stack so you can answer "per-pixel probability that this is *clay*" in one call.
- **`SoilGridsPoint` typed accessors.** `pt.value(property, depth, value)` raises a self-describing `PedotriError` pinpointing which level wasn't fetched. `pt.depths_for(property)`, `pt.properties()`, and `pt.as_quantiles(property, depth)` (returns a `Quantiles` ready to feed `pedotri.classify`) cover the common navigation patterns without dropping into the raw triple-nested dict.
- **Private helpers prefixed.** `_parse_uncertainty`, `_aggregate_class_probabilities`, `_distance_confidence`, `_effective_sigma` are no longer exposed; the public `pedotri.uncertainty` surface is now `Quantiles` + `sigma_from_quantiles` + `sample_truncated_normal` + `sample_compositional` + `shannon_entropy`. (`parse_uncertainty` remains importable as a back-compat alias.)
- **`pedotri.classify_all(sand, clay, *, locale=, schemes=, units=)`** — classify the same point under every 2-axis built-in (USDA + FAO + GEPPA + KA5 + …) in one call, returning `dict[scheme_key, ClassifyResult]`. Mirrors the agroref pattern of side-by-side multi-scheme reports.
- **`pedotri.zonal.aggregate_depths(samples, weights)`** — depth-weighted aggregation of per-depth mean + Q0.05 + Q0.95 bands into a single 0–30 cm layer (the standard `5 : 10 : 15` interval-width recipe), propagating the uncertainty envelope. Replaces the hand-rolled numpy code most multi-depth SoilGrids ingest pipelines carry around.
- **`examples/aoi_soc_stock.py`** — end-to-end SOC-stock-on-cropland workflow chaining `aggregate_depths` → `zonal_aggregate` → `combine` on synthetic inputs (offline). Runs in under a second; swap the `build_synthetic_*` helpers for real loaders to switch to production data.
- **`pedotri.Quantiles`** — re-export at the package root. Less reach-around for callers who write `from pedotri import Quantiles`.

### Performance

- **Vectorized `erf`** via the Abramowitz & Stegun 7.1.26 approximation in pure numpy. Replaces `np.vectorize(math.erf)` everywhere it appeared (distance-method confidence). Measured ~4.7× speedup on 1 Mpix workloads — pure-numpy, no new dependency. See the "Numerical implementation notes" section of [Uncertainty-aware classification](https://github.com/ivanfeanor/pedotri/wiki/Uncertainty-aware-classification) for the rationale.
- **Vectorized `_aggregate_class_probabilities`** via a single global `np.bincount` over flattened `row × n_classes + code` indices. Replaces the per-row Python loop. Measured 2–4× speedup at typical raster MC sizes.

#### Uncertainty-aware classification (`pedotri.uncertainty`)

- **Distribution helpers (`pedotri.uncertainty`)** — `sigma_from_quantiles`, `parse_uncertainty`, `sample_truncated_normal`, `sample_compositional`, `aggregate_class_probabilities`, `shannon_entropy`. SoilGrids and similar maps publish Q0.05 / Q0.50 / Q0.95 layers; these helpers turn that quantile data into distribution parameters and draw composition samples that respect `sand + silt + clay = 100`.
- **`<axis>_uncertainty` kwargs on `pedotri.classify`** — pass either a `(Q0.05, Q0.95)` tuple or a σ scalar / array per axis (`sand_uncertainty=`, `clay_uncertainty=`, `physical_clay_uncertainty=`, …). Requires `detailed=True` so the result carries probabilities and confidence.
- **`method=` keyword on `pedotri.classify`** — `"distance"` (default, cheap) computes `Φ(distance / σ_effective)` and returns a single confidence score for the modal class; `"monte_carlo"` draws `n_samples` realizations, classifies each, and returns a full probability distribution plus Shannon entropy.
- **Extended `ClassifyResult`** — new optional fields `probabilities`, `entropy`, `confidence`, `unclassified_probability`. The deterministic path leaves all four as `None` so existing code is unaffected; `to_dict()` only emits these fields when populated.

#### Uncertainty-aware raster classification (`pedotri.raster`)

- **`pedotri.raster.classify_array_with_uncertainty(sand_mean, clay_mean, *, sand_q05=, sand_q95=, sand_sigma=, …, classification=, method=, n_samples=, seed=, top_k=, chunk_pixels=, confirm=)`** — pixel-wise uncertainty propagation. Accepts either `(Q05, Q95)` rasters or a direct σ raster per axis (mixing forms across axes is allowed). Returns a `RasterClassification` dataclass with modal codes + keys plus method-specific outputs: `confidence` for `"distance"`, or `top_k_codes` / `top_k_probs` / `entropy` / `unclassified_probability` for `"monte_carlo"`. The MC path chunks the raster (default 50 000 valid pixels per chunk) to cap peak memory, and a pre-flight guard warns above 1e8 pixel-samples of work and aborts above 1e9 unless `confirm=True`.
- **`pedotri.raster.classify_geotiff_with_uncertainty(...)`** — convenience wrapper reading SoilGrids-style mean + quantile (or σ) GeoTIFFs from disk, validating alignment, and forwarding to the array path. Returns the same `RasterClassification` with a populated `profile` aligned with the mean raster.
- **`pedotri.raster.write_confidence_geotiff(path, confidence, *, profile, method=)`** — single-band float32 confidence raster, NaN for unclassified / zero-σ pixels, tagged with the producing method.
- **`pedotri.raster.write_probability_stack_geotiff(path, top_k_codes, top_k_probs, *, profile, keys)`** — 10-band uint8 GeoTIFF (5 class-code bands + 5 probability bands scaled to 0–255). Band descriptions identify `rank_k_class` / `rank_k_prob_x255`; dataset tags record the class-key mapping and the active `top_k`.
- **`examples/uncertainty_demo.py`** — reproducible synthetic-raster demo that emits modal-class, distance-confidence, MC-entropy PNGs/TIFFs plus the 10-band probability stack to `docs/images/`.

#### Regional aggregation (`pedotri.zonal`)

- **`pedotri.zonal.zonal_aggregate(*, region=, properties=, mask=, mask_include=, profile=, n_samples=, seed=, confirm=)`** — Monte-Carlo aggregation of property rasters over an AOI. ``region`` accepts a boolean ndarray, a shapely geometry, or any ``__geo_interface__`` object. ``properties`` is a mapping of ``name -> (mean, q05, q95)`` triples (ndarrays or paths). Optional ``mask`` + ``mask_include`` restrict the aggregation to a land-cover class allowlist (works with ESA WorldCover out of the box). Returns a :class:`pedotri.zonal.ZonalAggregate` with one :class:`AggregateDistribution` per property and a regional ``mask_coverage`` summary. Independent-pixel sampling in 0.3.x — the reported regional Q05/Q95 is a documented lower bound on the true posterior width.
- **`AggregateDistribution`** — posterior over a single regional aggregate. Exposes ``mean``, ``std``, ``q05``, ``q50``, ``q95``, ``quantile(q)``, plus the raw ``samples`` array.
- **`ZonalAggregate.combine(fn, *, name=)`** — propagate per-property samples through an arbitrary user formula (typical case: ``stock = soc · bd · depth · area``). Defaults on the lambda let you bake in scalar constants without leaking them through the aggregation API.

#### Data sources (`pedotri.sources`)

- **`pedotri.sources.soilgrids.fetch_point(lon, lat, *, properties=, depths=, values=, cache_dir=, cache_ttl_days=, timeout=)`** — query SoilGrids 2.0 for one coordinate (default sand + clay at 0–5 cm with mean + Q0.05 + Q0.95). Responses are persisted under `$XDG_CACHE_HOME/pedotri/soilgrids/` (override via `cache_dir=`), keyed by a sha-256 of the canonical request payload, with a 30-day TTL by default. Units are translated back into pedotri-native percent / g/cm³ on the way out.
- **`SoilGridsPoint`** dataclass with `.values`, `.cached`, and a `sand_clay(depth="0-5cm")` convenience returning `(mean, Q05, Q95)` for both axes ready to feed `pedotri.classify(..., sand_uncertainty=, clay_uncertainty=)`.
- **`pedotri.sources.soilgrids.clear_cache()`** — invalidate the local snapshot in one call (returns the number of files removed).
- **`pedotri.sources.worldcover.fetch_aoi(bbox, *, year=2021, cache_dir=, cache_ttl_days=365, max_pixels=)`** — fetch ESA WorldCover 10 m land cover for a bounding box. Computes intersecting 3°×3° tiles, opens each remote Cloud-Optimized GeoTIFF via GDAL `/vsicurl/`, reads only the AOI window, and mosaics across tile boundaries — a 1 km² village pulls a few hundred KB instead of a multi-gigabyte tile. Cached on disk by `(bbox, year)` digest with a 365-day TTL. AOI size capped at 25 Mpix by default. Class-code constants (`CROPLAND = 40`, `TREE_COVER = 10`, `BUILT_UP = 50`, …) and a `WorldCoverAOI.class_fraction(codes)` helper for quick summaries.
- **`pedotri.sources.worldcover.fetch_aoi_from_polygon(geom, *, year=, …)`** — convenience that derives the bbox from any shapely / `__geo_interface__` geometry before delegating to `fetch_aoi`. The returned raster plugs directly into `pedotri.zonal.zonal_aggregate(..., mask=aoi.array, mask_include=[CROPLAND])`.

#### MCP tool: `classify_point`

- Single new MCP tool that subsumes both the explicit `classify_soil` flow and the SoilGrids fetch path. Modes are disambiguated by the inputs: explicit `sand`/`clay` (with optional `*_q05`/`*_q95`) or coordinates `lon`/`lat` (auto-fetches SoilGrids + uses its uncertainty). Returns the same `ClassifyResult` shape with probabilities + confidence when uncertainty is available, plus a `source` block describing where the values came from.

## [0.2.0] — 2026-05-15

### Added

#### Classifications

- **`RUS2004` classification** — the six-class granulometric "variety" (разновидности) scheme used in the modern Russian soil-survey documents (*Классификация и диагностика почв России*, 2004; *Полевой определитель почв России*, 2008). Keyed on the same `physical_clay` axis as `KACHINSKY` and a strict coarsening of it (`loose_sand` + `cohesive_sand` → `sandy`; `light_clay` + `medium_clay` + `heavy_clay` → `clay`). Class names ship in six languages, defaulting to Russian.

#### Raster classification (`pedotri.raster`)

- **`pedotri.raster.classify_array(sand, clay, *, classification, units=, mask=, sum_tolerance=)`** — vectorized point-in-polygon classification on numpy rasters of any shape. Returns an `int16` class-code array plus an index→class-key list. Pure numpy — no GDAL dependency on the hot path.
- **`pedotri.raster.classify_geotiff(sand, clay, *, classification, units=, sum_tolerance=)`** — convenience wrapper that reads two GeoTIFFs via the optional `[raster]` extra (rasterio), validates alignment (CRS / transform / shape), and returns codes + keys + a write-ready profile.
- **`pedotri.raster.write_classified_geotiff(path, codes, *, profile, keys, colormap=None)`** — writes a **paletted uint8** GeoTIFF with a color table (auto-built from matplotlib `tab20`, customizable via `colormap=`). QGIS / `gdalinfo` pick up both the palette and the class names automatically; `NODATA_CODE` maps to `255` on disk so GDAL accepts the palette.
- **`pedotri.raster.smooth_codes(codes, *, window=3, iterations=1)`** — majority filter on class-coded rasters, vectorized as one-hot box-filter + argmax via `scipy.ndimage.uniform_filter`. Cleans up salt-and-pepper noise without a Python-callback mode filter; preserves `NODATA_CODE` pixels.
- **`pedotri.raster.render_classified_png(path, codes, *, keys, title=, extent=, xlabel=, ylabel=, …)`** — categorical map with a side colorbar legend (matplotlib).
- **`pedotri.raster.classified_to_features(codes, *, keys, transform, simplify_tolerance=None)`** — polygonize a class-coded raster into shapely features (one feature per contiguous class region; nodata excluded).
- **`pedotri.raster.write_features_shapefile(features, path, *, crs)`** — write features to ESRI Shapefile, GeoPackage, or GeoJSON (driver chosen by file suffix). Uses fiona directly; no geopandas dependency.
- **`pedotri.raster.NODATA_CODE = -1`** sentinel for masked, NaN, out-of-range, or unclassified pixels.

#### Field-scale interpolation (`pedotri.interp`)

- **`pedotri.interp.krige_samples(xs, ys, values, *, bbox, resolution, variogram=, n_lags=)`** — ordinary kriging on a regular grid via pyKrige. Returns `(grid, variance, profile)` so callers can use the per-cell kriging variance as a confidence map.
- **`pedotri.interp.krige_sand_clay(samples, *, bbox, resolution, variogram=, mask_polygon=, crs=)`** — convenience wrapper: kriges sand and clay independently from one sample table, optionally masks to a field polygon, and returns rasters ready for `pedotri.raster.classify_array`.

#### Benchmark utilities (`pedotri.bench`)

- **`pedotri.bench.run_soiltexture_benchmark(sizes, *, classification, seed, repeat, check_agreement)`** — reproducible head-to-head benchmark against the [`soiltexture`](https://pypi.org/project/soiltexture/) Python package, with input normalization (sand/clay percentages in the feasible triangle), `time.perf_counter`-based timing, and a 100 %-agreement check on labels.
- **`pedotri.bench.BenchmarkReport`** with `format()` (fixed-width table) and `speedup_table()` (per-N pedotri/soiltexture speedup).
- **`pedotri.bench.SOILTEXTURE_TO_PEDOTRI_USDA`** — mapping table for `"sandy clay loam"` → `"sandy_clay_loam"` etc., so output strings round-trip cleanly between libraries.

#### Extras

- **`pedotri[raster]`** = `rasterio + scipy` (GeoTIFF I/O + smoothing).
- **`pedotri[vector]`** = `shapely + fiona` (Shapefile / GeoPackage / GeoJSON output).
- **`pedotri[interp]`** = `pykrige + scipy + rasterio` (kriging + raster profile + polygon masking).

#### Example scripts

- **`examples/soilgrids_france.py`** — full end-to-end regional demo: downloads the ISRIC SoilGrids 2.0 sand / silt / clay 2°×2° tile (central France, 2-4°E × 46-48°N, 0-5 cm depth) via WCS, classifies all 740,745 pixels under USDA, smooths 3×3, writes paletted GeoTIFF + PNG.
- **`examples/bench_soiltexture.py`** — runs the soiltexture head-to-head benchmark at 1k / 10k / 100k / 740,745 sample sizes.
- **`examples/field_kriging.py`** — field-scale pipeline: 20 in-situ samples on a hexagonal 100 m × 100 m plot → krige sand + clay → mask → USDA-classify → smooth → write paletted GeoTIFF + PNG + Shapefile.
- **`examples/central_france_field.py`** — minimal-survey variant: a pseudo-random 10 ha field inside the central-France demo box, **3 in-situ samples** placed in distinct quadrants, ordinary kriging with a linear variogram (the only model robust to so few points), classified USDA map with sample markers overlaid.
- **Committed output artifacts** under `docs/images/` — paletted GeoTIFFs + PNGs of all three example workflows. PNG previews embedded in the README and on the [Examples-gallery](https://github.com/ivanfeanor/pedotri/wiki/Examples-gallery) wiki page.

#### Documentation

- New `wiki/` source directory (push-ready for `github.com/ivanfeanor/pedotri.wiki.git`) with long-form pages: *Classifications & when to use them*, *Classification-challenges*, *Soil-conversions*, *Pedotransfer-functions*, *Units-and-organic-matter*, *GeoTIFF-and-SoilGrids*, *Map-products-and-field-sampling*, *Examples-gallery*, *Benchmark-vs-soiltexture*, *Custom-classifications-in-practice*.
- `docs/raster.md` mkdocs page documenting the raster workflow.
- `docs/api.md` extended with reference entries for the new `raster`, `bench`, and `interp` modules.

### Changed

- **`pedotri.classify` is up to ≈ 13× faster than 0.1.3** on large batches (and ≈ 48× faster than `soiltexture` on USDA, up from ≈ 9-10× before any 0.2.0 optimizations). Four independent improvements layered:
  - **Skip the per-edge signed-distance computation on the non-detailed path** (`src/pedotri/classifier.py:_classify_polygons`). It was unconditional in 0.1.x and accounted for ~70 % of runtime. ~485 k → ~1.2 M pts/sec on its own.
  - **Bounding-box pre-filter per polygon** (`src/pedotri/classifier.py:_classify_polygons`). Points outside the polygon's axis-aligned bbox are skipped before the O(M·N) ray-cast. The dominant additional win.
  - **On-edge fix-up only for points the ray-cast already said were outside** (`src/pedotri/geometry.py:points_in_polygon`). The MxN on-edge allocation is skipped for the (often majority) already-inside set.
  - **Int-coded scatter through `_classify_polygons`** + precomputed label table in `_build_labels` (`src/pedotri/classifier.py`). The hot path no longer assigns `TextureClass` objects into a Python list per point.

  Throughput on the 740,745-pixel SoilGrids tile: **485 k → 6.30 M points/sec** on a 2024 M-series Mac. All optimizations preserve byte-identical outputs against 0.1.3; `pedotri.bench.run_soiltexture_benchmark` still reports 100 % agreement with `soiltexture` on USDA inputs.

### Benchmark snapshot

`pedotri` vs. `soiltexture` 1.0.4 on USDA classification, Apple M-series, pedotri 0.2.0:

| n_points | pedotri (pts/sec) | soiltexture (pts/sec) | speedup |
|---------:|------------------:|----------------------:|--------:|
|     1,000 | 1,516,000 |  130,000 | **11.7 ×** |
|    10,000 | 4,905,000 |  134,000 | **36.7 ×** |
|   100,000 | 6,164,000 |  132,000 | **46.8 ×** |
|   740,745 | 6,300,000 |  130,000 | **48.3 ×** |

Output agreement: 100.00 %.

## [0.1.3] — 2026-05-15

### Changed

- **MCP `render_diagram` now returns PNG by default** so the diagram renders inline in every MCP client (Claude Desktop, Cursor, …). PNG is rasterized via matplotlib (added to the `[mcp]` extra so `pip install pedotri[mcp]` gets everything needed for inline rendering). The `image/svg+xml` `ImageContent` path that 0.1.1/0.1.2 used worked in some clients but not in Claude Desktop's current build.
- **`render_diagram` gains a `format` parameter** taking `"png"` (default, fail-safe) or `"svg"` (vector, for clients that support it / HTML embedding / post-processing). The tool description tells the model explicitly when to pick each.
- **`pedotri.ai.run("render_diagram", ...)` response shape**: now `{"format": "png"|"svg", "encoding": "base64"|"text", "content": ...}`. PNG mode falls back to SVG automatically when matplotlib isn't installed, so callers without the `[mcp]` extra still get a usable response.

### Added

- `pedotri.plot.render_png(classification, ...)` helper that renders directly to PNG bytes at 200 dpi (retina-ready for typical chat-embed widths, ~200 KB files). Re-uses the existing matplotlib backend for visual consistency with `render_mpl`.

## [0.1.2] — 2026-05-15

### Fixed

- `pedotri.__version__` now reads from package metadata via `importlib.metadata` instead of a hardcoded string, so it always matches the installed wheel version. The 0.1.1 release shipped with `__version__ == "0.1.0"` because the constant wasn't bumped alongside `pyproject.toml`; making the version a single source of truth prevents that class of drift.

## [0.1.1] — 2026-05-15

### Changed

- **MCP `render_diagram` now returns `ImageContent`** (mimeType `image/svg+xml`) so Claude Desktop / Cursor / other MCP clients render the diagram inline instead of printing the SVG markup as a text blob. Successful renders return an `ImageContent` block plus a minimal `TextContent` companion describing what was rendered. Error responses are unchanged — the standard JSON envelope is still emitted as `TextContent` so the model can self-correct in a tool-use loop.

## [0.1.0] — 2026-05-15

### Added

#### Classification engine

- Vectorized point-in-polygon classifier (`pedotri.classify`) for 2-D sand-clay classifications and interval-based 1-D classifications (e.g. Kachinsky's physical-clay axis).
- TOML-based classification authoring with full validation; user-defined classifications via `register_classification()` or `pedotri.classifications` entry-point group.
- Locale fallback chain (`fr-FR` → `fr` → `en` → class key) for class names.
- Vectorized geometry: ~360k points/sec for the 2-D classifier on a single thread.

#### Built-in classifications

Fourteen classifications shipped, with class names in en / fr / de / es / ru / pt (and pl / pt / zh / ru native where applicable):

- USDA (USA, 12 classes)
- FAO (international, 3 classes)
- INTERNATIONAL (Leeper & Uren 1993, 11 classes)
- ISSS (international, 12 classes)
- JAMAGNE (France, 13 classes — Jamagne 1967)
- GEPPA (France, 14 classes — Baize & Jamagne 1995, modern refinement)
- HYPRES (Europe, 5 classes — Wösten et al. 1999)
- KA5 (Germany, full 31-class subdivision per Bodenkundliche Kartieranleitung 5)
- EMBRAPA (Brazil, 5 classes — SiBCS 5ª ed. 2018)
- KACHINSKY (Russia, 9 1-D classes — Kachinsky 1965)
- NORTHCOTE (Australia, 16 classes — Northcote 1979, fine clay subdivisions)
- PTG (Poland, 6 classes — PTG 2008)
- CHINA (national, 6 classes — GB/T 17296-2009)
- AVERY (UK, 12 classes — Avery 1980)

#### Pedotransfer functions

- `pedotri.ptf.saxton_rawls(sand, clay, organic_matter)` — Saxton & Rawls (2006). Wilting point, field capacity, saturation, plant-available water, saturated hydraulic conductivity, bulk density, air-entry tension. Optional `density_factor` keyword applies Eq. 6-8 compaction correction.
- `pedotri.ptf.wosten(sand, silt, clay, organic_matter, bulk_density, topsoil)` — HYPRES continuous PTF for Mualem-van Genuchten parameters.

#### Plotting

- Built-in pure-Python SVG renderer (`pedotri.plot.render_svg`, `pedotri.plot.TextureDiagram`). No plotting dependency required.
- Optional matplotlib backend (`pedotri[matplotlib]` extra).
- Optional plotly backend (`pedotri[plotly]` extra) using native ternary subplots.

#### Particle-size conversions

- `pedotri.psd.convert(sand, silt, clay, source=, target=)` for the USDA / FAO / ISSS / INTERNATIONAL / KA5 sand-silt cutoff conversions under a log-linear interior model.
- `pedotri.psd.interpolate_psd()` for arbitrary cutoff interpolation from a full sieve PSD.

#### Units

- `pedotri.units` module: g/kg ↔ % and g/g ↔ %, plus organic-carbon ↔ organic-matter conversion via the Van Bemmelen factor (1.724, configurable).
- `units=` keyword on `classify()`, `saxton_rawls()`, `wosten()`, and `psd.convert()` for `"%"` (default), `"g/kg"`, or `"g/g"`.

#### DataFrame accessors

- `df.soil.classify(...)` for both pandas and polars, auto-registered on `import pedotri` when the relevant DataFrame library is installed.

#### LLM integration

- `pedotri.ai` module: Anthropic-style tool schemas for 8 operations (`classify_soil`, `classify_soil_1d`, `list_classifications`, `classification_info`, `saxton_rawls`, `wosten`, `convert_particle_size`, `render_diagram`) plus a JSON-in / JSON-out dispatcher with structured error envelopes.
- `pedotri-mcp` console script (`pedotri[mcp]` extra): Model Context Protocol server speaking stdio, suitable for Claude Desktop, Cursor, and any MCP-aware client.

#### CLI

- `pedotri` console script with `list`, `info`, `classify` (single + CSV batch), and `render` subcommands. Stdlib-only (argparse).

#### Tooling and standards

- uv-managed environment, `pyproject.toml` PEP 621 metadata, src layout.
- ruff (lint + format), mypy strict, pytest + hypothesis property-based tests.
- mkdocs-material documentation site with mkdocstrings API reference.
- GitHub Actions CI matrix: Python 3.11 / 3.12 / 3.13 × Ubuntu / macOS / Windows.
- Tag-triggered release workflow with PyPI trusted publishing.

### Documentation

- README with quickstart, classifications table, plotting, accessors, CLI, AI integration, units, and PTFs.
- mkdocs site: quickstart, built-in / custom-classification / i18n guides, PTFs, plotting, CLI, AI tools, units, and full API reference.
- Polygon provenance notes (Tier 1 direct-from-literature vs. Tier 2 textbook-derived).

[Unreleased]: https://github.com/ivanfeanor/pedotri/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.2.0
[0.1.3]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.1.3
[0.1.2]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.1.2
[0.1.1]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.1.1
[0.1.0]: https://github.com/ivanfeanor/pedotri/releases/tag/v0.1.0
