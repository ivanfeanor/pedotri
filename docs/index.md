# pedotri

Modern, extensible soil texture classification and pedotransfer functions for Python.

## Why pedotri

- **Vectorized.** numpy-based point-in-polygon — handles hundreds of thousands of points per second.
- **Extensible.** Define your own classifications in TOML. Auto-discovery via `entry_points` lets third-party packages contribute classifications.
- **Multilingual.** Class names available in many languages; users can extend with their own locales.
- **Comprehensive.** Ships with classifications from around the world: USDA, FAO, GEPPA/Jamagne, KA5, HYPRES, Kachinsky, Embrapa, and more.
- **Beyond classification.** Pedotransfer functions (Saxton-Rawls, Wösten / HYPRES), particle-size standard conversions, hierarchy lookups, distance-to-boundary, and a pure-SVG triangle renderer.

See the [Quick start](quickstart.md) to get going.
