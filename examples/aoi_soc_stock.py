"""End-to-end: SOC stock with uncertainty for an AOI restricted to cropland.

This script chains the four pieces 0.3 adds to pedotri into a single
workflow that mirrors what a real agronomic application needs:

1. **Land-cover mask** — ``pedotri.sources.worldcover.fetch_aoi`` would
   pull ESA WorldCover for the AOI; here we manufacture an equivalent
   synthetic mask so the script runs without network access.
2. **Multi-depth property rasters** — three SoilGrids-style bands
   (0–5 cm, 5–15 cm, 15–30 cm) for SOC and bulk density, each carrying
   the mean and Q0.05 / Q0.95 envelope. In production these come from
   ``pedotri.sources.soilgrids`` (bbox path landing in 0.4) or any
   in-house COG stack.
3. **Depth aggregation** — ``pedotri.zonal.aggregate_depths`` collapses
   the three per-depth bands into a single 0–30 cm layer with weights
   ``5 : 10 : 15`` (the standard interval-width recipe), propagating
   the uncertainty envelope.
4. **Regional aggregation + SOC stock** — ``pedotri.zonal.zonal_aggregate``
   restricts to cropland pixels, draws ``n_samples`` realisations from
   each pixel's distribution, and reports a posterior over the regional
   mean. ``.combine()`` then propagates SOC × BD × depth × area into a
   distribution over the AOI's total SOC stock.

The shapes / sizes are small enough to run instantly. Replace
``_build_synthetic_*`` with your real data loaders to switch to a
production workflow.
"""

from __future__ import annotations

import numpy as np

from pedotri.uncertainty import Quantiles
from pedotri.zonal import aggregate_depths, zonal_aggregate

H, W = 200, 200
AOI_TRANSFORM_PIXELS_PER_M = 100  # synthetic: each pixel is 100 m
RNG_SEED = 11
N_SAMPLES = 1000

# Cropland class in ESA WorldCover v200. Re-exported as
# ``pedotri.sources.worldcover.CROPLAND``.
WORLDCOVER_CROPLAND = 40


def build_synthetic_aoi():
    """A 200 × 200 grid; the bottom-right quadrant is the AOI."""
    region = np.zeros((H, W), dtype=bool)
    region[H // 2 :, W // 2 :] = True
    return region


def build_synthetic_worldcover():
    """Half cropland (code 40), half tree-cover (code 10), in a stripe pattern.

    Mimics what ``worldcover.fetch_aoi(bbox).array`` returns when an
    AOI straddles the boundary between two land-cover classes.
    """
    mask = np.full((H, W), 10, dtype=np.int32)  # tree cover
    mask[:, ::2] = WORLDCOVER_CROPLAND  # every other column = cropland
    return mask


def build_synthetic_soilgrids_depth(mean_value: float, q_halfwidth: float, noise_seed: int):
    """Return a single-depth (mean, Q05, Q95) raster trio for one property."""
    rng = np.random.default_rng(noise_seed)
    mean_arr = np.full((H, W), mean_value, dtype=np.float64) + rng.normal(scale=0.4, size=(H, W))
    q05_arr = mean_arr - q_halfwidth
    q95_arr = mean_arr + q_halfwidth
    return mean_arr, q05_arr, q95_arr


def main() -> None:
    region = build_synthetic_aoi()
    land_cover = build_synthetic_worldcover()

    # --- Multi-depth SoilGrids stack -----------------------------------
    # SOC and BD at three standard depths. In production these would
    # come from rasterio file reads or pedotri.sources.soilgrids.
    soc_depths = {
        "0-5cm": build_synthetic_soilgrids_depth(24.0, 4.0, noise_seed=1),
        "5-15cm": build_synthetic_soilgrids_depth(22.0, 4.0, noise_seed=2),
        "15-30cm": build_synthetic_soilgrids_depth(19.0, 4.0, noise_seed=3),
    }
    bd_depths = {
        "0-5cm": build_synthetic_soilgrids_depth(1.30, 0.10, noise_seed=4),
        "5-15cm": build_synthetic_soilgrids_depth(1.35, 0.10, noise_seed=5),
        "15-30cm": build_synthetic_soilgrids_depth(1.40, 0.10, noise_seed=6),
    }

    # --- 0–30 cm depth aggregation -------------------------------------
    weights = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}

    soc_topsoil_mean, soc_topsoil_q = aggregate_depths(
        samples={
            depth: {"mean": m, "uncertainty": Quantiles(q05, q95)}
            for depth, (m, q05, q95) in soc_depths.items()
        },
        weights=weights,
    )
    bd_topsoil_mean, bd_topsoil_q = aggregate_depths(
        samples={
            depth: {"mean": m, "uncertainty": Quantiles(q05, q95)}
            for depth, (m, q05, q95) in bd_depths.items()
        },
        weights=weights,
    )

    # --- Regional aggregation, cropland-only ---------------------------
    agg = zonal_aggregate(
        region=region,
        properties={
            "soc": {"mean": soc_topsoil_mean, "uncertainty": soc_topsoil_q},
            "bd": {"mean": bd_topsoil_mean, "uncertainty": bd_topsoil_q},
        },
        mask=land_cover,
        mask_include=[WORLDCOVER_CROPLAND],
        n_samples=N_SAMPLES,
        seed=RNG_SEED,
    )

    # --- Derived quantity: SOC stock (t / AOI) -------------------------
    # SOC density per pixel = SOC [g/kg] × 1e-3 × BD [g/cm³ ≡ t/m³] × depth [m]
    # × pixel area [m²]. Then sum across AOI pixels = SOC density times
    # pixel count × pixel area. With area folded into the constant, the
    # regional mean of "SOC density per pixel" times pixel count × area
    # = AOI total.
    pixel_area_m2 = AOI_TRANSFORM_PIXELS_PER_M**2  # 100 m × 100 m = 10 000 m²
    aoi_area_m2 = float(agg.n_pixels_used) * pixel_area_m2

    stock = agg.combine(
        lambda soc, bd, depth=0.30, area=aoi_area_m2: (
            # tonnes SOC across the whole AOI cropland slice
            soc * 1e-3 * bd * depth * area * 1e-3
        ),
        name="soc_stock_tonnes",
    )

    # --- Report --------------------------------------------------------
    print(f"AOI region pixels:   {agg.region_pixels:,}")
    print(f"Cropland pixels:     {agg.n_pixels_used:,}")
    print(f"Mask coverage:       {agg.mask_coverage:.1%}")
    print(f"Cropland area:       {aoi_area_m2:,.0f} m² = {aoi_area_m2 / 10_000:.1f} ha")
    print()
    print(
        f"SOC mean  (g/kg):    mean={agg['soc'].mean:6.2f}  "
        f"Q05={agg['soc'].q05:6.2f}  Q95={agg['soc'].q95:6.2f}"
    )
    print(
        f"BD  mean  (g/cm³):   mean={agg['bd'].mean:6.3f}  "
        f"Q05={agg['bd'].q05:6.3f}  Q95={agg['bd'].q95:6.3f}"
    )
    print()
    print(
        f"SOC stock (t/AOI):   mean={stock.mean:8.1f}  "
        f"Q05={stock.q05:8.1f}  Q95={stock.q95:8.1f}  "
        f"σ={stock.std:7.2f}"
    )


if __name__ == "__main__":
    main()
