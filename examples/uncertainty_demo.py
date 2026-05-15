"""Uncertainty-aware soil-texture classification on a synthetic raster.

This example builds a small "SoilGrids-style" patch with mean + Q05 + Q95
bands for both sand and clay, then runs the two methods exposed in 0.3:

1. ``method="distance"`` — cheap confidence proxy from
   ``Φ(distance / σ_effective)``.
2. ``method="monte_carlo"`` — full per-pixel probability distribution
   over USDA classes, plus Shannon entropy.

Outputs (under ``docs/images/`` so the wiki can embed them via GitHub
raw URLs):

- ``uncertainty_demo_class.{png,tif}`` — modal-class map.
- ``uncertainty_demo_confidence.{png,tif}`` — distance-method confidence.
- ``uncertainty_demo_entropy.{png,tif}`` — Monte-Carlo Shannon entropy.
- ``uncertainty_demo_probabilities.tif`` — 10-band top-5 stack
  (5 class-code bands + 5 probability bands).

No network access required. The synthetic field is fully reproducible
from the seed in the source — change ``RNG_SEED`` to explore other
realizations.

Run with::

    pip install 'pedotri[raster,matplotlib]'
    python examples/uncertainty_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pedotri.raster import (
    classify_array_with_uncertainty,
    write_classified_geotiff,
    write_confidence_geotiff,
    write_probability_stack_geotiff,
)

H, W = 200, 200
RNG_SEED = 7
N_SAMPLES = 1000

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"


def build_synthetic_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sand_mean, clay_mean, sand_sigma, clay_sigma)`` rasters.

    Sand grades left-to-right from sandy soil to clayey soil. Clay
    counter-grades top-to-bottom. A low-frequency noise field perturbs
    each band so isolines aren't trivially straight. Uncertainty (σ)
    is *spatially varying* — lowest in the center of the map and rising
    toward the edges, so the resulting confidence map has structure
    that's not just a copy of the distance-to-boundary map.
    """
    rng = np.random.default_rng(RNG_SEED)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    x_norm = x / (W - 1)
    y_norm = y / (H - 1)

    sand_mean = 20.0 + 50.0 * x_norm + 5.0 * (y_norm - 0.5)
    clay_mean = 50.0 - 40.0 * x_norm + 5.0 * (0.5 - y_norm)

    # Smooth low-frequency noise via repeated averaging of a coarse field.
    noise = rng.normal(size=(H, W))
    for _ in range(20):
        noise = 0.25 * (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
        )
    noise = (noise - noise.mean()) / noise.std()
    sand_mean = np.clip(sand_mean + 4.0 * noise, 0.0, 100.0)
    clay_mean = np.clip(clay_mean + 3.0 * noise, 0.0, 100.0)
    # Keep silt non-negative.
    overshoot = sand_mean + clay_mean - 100.0
    over = overshoot > 0
    sand_mean[over] -= overshoot[over] * 0.5
    clay_mean[over] -= overshoot[over] * 0.5

    # σ rises from ~2 in the center to ~7 at the corners.
    dx = x_norm - 0.5
    dy = y_norm - 0.5
    radial = np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5)
    sigma_field = 2.0 + 5.0 * radial
    sand_sigma = sigma_field
    clay_sigma = sigma_field * 0.8

    return sand_mean, clay_mean, sand_sigma, clay_sigma


def make_profile(transform_origin: tuple[float, float] = (0.0, H)) -> dict:
    """A minimal rasterio profile aligned with the synthetic grid."""
    from rasterio.transform import from_origin

    return {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "transform": from_origin(transform_origin[0], transform_origin[1], 1, 1),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float32",
    }


def render_scalar_map(
    path: Path,
    arr: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a single-band float array as a heatmap PNG with a colorbar."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_scalar_geotiff(path: Path, arr: np.ndarray, *, profile: dict) -> None:
    """Single-band float32 GeoTIFF (NaN nodata)."""
    import rasterio

    out_profile = dict(profile)
    out_profile.update(dtype="float32", count=1, nodata=float("nan"), compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sand_mean, clay_mean, sand_sigma, clay_sigma = build_synthetic_field()
    profile = make_profile()

    # --- distance method ------------------------------------------------
    distance = classify_array_with_uncertainty(
        sand_mean,
        clay_mean,
        sand_sigma=sand_sigma,
        clay_sigma=clay_sigma,
        classification="USDA",
        method="distance",
    )

    # --- monte carlo ----------------------------------------------------
    mc = classify_array_with_uncertainty(
        sand_mean,
        clay_mean,
        sand_sigma=sand_sigma,
        clay_sigma=clay_sigma,
        classification="USDA",
        method="monte_carlo",
        n_samples=N_SAMPLES,
        seed=RNG_SEED,
    )

    # --- write class GeoTIFF + PNG -------------------------------------
    class_tif = OUT_DIR / "uncertainty_demo_class.tif"
    class_png = OUT_DIR / "uncertainty_demo_class.png"
    class_profile = dict(profile)
    class_profile.update(dtype="int16", nodata=-1, count=1, compress="deflate")
    write_classified_geotiff(class_tif, mc.codes, profile=class_profile, keys=mc.keys)

    from pedotri.raster import render_classified_png

    render_classified_png(
        class_png,
        mc.codes,
        keys=mc.keys,
        title="USDA modal class (synthetic SoilGrids-style patch)",
        figsize=(8.0, 6.5),
    )

    # --- confidence (distance) -----------------------------------------
    conf_tif = OUT_DIR / "uncertainty_demo_confidence.tif"
    conf_png = OUT_DIR / "uncertainty_demo_confidence.png"
    write_confidence_geotiff(conf_tif, distance.confidence, profile=profile, method="distance")
    render_scalar_map(
        conf_png,
        distance.confidence,
        title="Distance-method confidence — Φ(d / σ_eff)",
        colorbar_label="confidence",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )

    # --- entropy (monte carlo) -----------------------------------------
    ent_tif = OUT_DIR / "uncertainty_demo_entropy.tif"
    ent_png = OUT_DIR / "uncertainty_demo_entropy.png"
    write_scalar_geotiff(ent_tif, mc.entropy, profile=profile)
    render_scalar_map(
        ent_png,
        mc.entropy,
        title="Monte-Carlo Shannon entropy (1000 draws / pixel)",
        colorbar_label="entropy (nats)",
        cmap="magma",
    )

    # --- top-5 probability stack ---------------------------------------
    probs_tif = OUT_DIR / "uncertainty_demo_probabilities.tif"
    probs_profile = dict(profile)
    probs_profile["count"] = 10
    write_probability_stack_geotiff(
        probs_tif,
        mc.top_k_codes,
        mc.top_k_probs,
        profile=probs_profile,
        keys=mc.keys,
    )

    # --- summary printout ----------------------------------------------
    print(f"Wrote outputs to {OUT_DIR}/:")
    for path in sorted(OUT_DIR.glob("uncertainty_demo_*")):
        print(f"  {path.name:42s} {path.stat().st_size:>10,} bytes")
    classified = mc.codes != -1
    print(
        f"\nClasses present (modal): {sorted({mc.keys[c] for c in mc.codes[classified].tolist()})}"
    )
    print(
        f"Confidence (distance) — min={float(np.nanmin(distance.confidence)):.3f}, "
        f"mean={float(np.nanmean(distance.confidence)):.3f}, "
        f"max={float(np.nanmax(distance.confidence)):.3f}"
    )
    print(
        f"Entropy   (MC)        — min={float(np.nanmin(mc.entropy)):.3f}, "
        f"mean={float(np.nanmean(mc.entropy)):.3f}, "
        f"max={float(np.nanmax(mc.entropy)):.3f}"
    )


if __name__ == "__main__":
    main()
