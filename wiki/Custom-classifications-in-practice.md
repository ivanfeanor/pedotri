# Custom classifications in practice

When the 14 built-in classifications don't fit your data — a local
soil-survey convention, an in-house quality grouping, or a research-
specific 2-class threshold — pedotri lets you ship your own as a TOML
file and register it at runtime or via package entry points.

The
[Custom-classifications API guide](https://ivanfeanor.github.io/pedotri/classifications/custom/)
covers the full TOML schema. This page focuses on the **decisions
you'll actually make** as you author one.

## When should you write a custom classification?

✅ Good reasons:

- Local soil-survey vocabulary (e.g. a regional refinement of GEPPA).
- Domain-specific binary grouping ("compactible vs. friable" for a
  tillage-planning tool).
- Project-internal hierarchy that aggregates multiple USDA classes.
- Internationalizing a built-in into a locale not yet shipped.

❌ Bad reasons:

- The built-in classification was *almost* right but you wanted to
  shift one polygon boundary by 1 % to match an old colleague's
  preference. The reproducibility cost outweighs the precision gain.
- "I'll just invent classes that match my dataset's empirical
  distribution." That isn't a classification, it's a clustering
  result — keep it as a numpy array of labels.

## Choosing 1-D vs 2-D

Pedotri supports two axis layouts:

- **2-D (sand, clay)** — for triangle-based classifications. The
  conventional choice; vertex coordinates use `(sand %, clay %)`.
- **1-D (single named axis)** — for interval-based classifications
  like Kachinsky. The axis can be `physical_clay`, `clay`, or any
  custom name your survey uses.

Pick 1-D if your community thinks in terms of a *single* particle-size
threshold and treats sand/silt as derived. Pick 2-D for everything
else.

## Defining polygons cleanly

A polygon is a list of `(sand, clay)` vertices in mass-percent space.
Two rules that prevent 90 % of bugs:

1. **Order vertices consistently** — either all clockwise or all
   counter-clockwise. Pedotri's point-in-polygon test handles both,
   but inconsistent ordering makes the polygon hard to read and
   review.
2. **Don't repeat the closing vertex.** Pedotri closes the polygon
   implicitly. A duplicated first/last vertex is harmless but adds
   visual noise.

## Overlapping classes: first-match-wins

When two class polygons overlap (which happens deliberately in
classifications like USDA, where the "silty clay" wedge sits inside
the broader clay region), pedotri attributes a point to the **first
matching class** in the listed order. This makes the order of classes
in your TOML *semantically significant*: more specific classes go
first, more general classes go last.

The same rule applies to 1-D classifications: intervals are evaluated
in declaration order and the first match wins, so abutting intervals
should be ordered to disambiguate boundary values consistently.

## i18n: don't ship only English

Pedotri matches class names through a locale fallback chain
(`fr-FR` → `fr` → `en` → class key). If your classification has a
regional vocabulary, ship at least the regional locale and English —
otherwise non-English users will see your raw class keys
(`heavy_loam_with_carbonates`) instead of human-readable names.

```toml
[[classes]]
key = "heavy_loam"
group = "fine"
names = { en = "heavy loam", fr = "limon lourd", de = "schwerer Lehm" }
vertices = [[10, 35], [20, 50], [5, 50]]
```

The `key` is the stable identifier — it never changes once published.
The `names` are the surface labels — they can be edited freely
(typos, retranslations) without breaking downstream consumers.

## Shipping as a package

For project-internal use, `register_classification(path)` at startup
is fine. For wider distribution, expose your TOML via the
`pedotri.classifications` entry point so users only need to
`pip install your-package`:

```toml
# pyproject.toml of your downstream package
[project.entry-points."pedotri.classifications"]
my_regional = "my_pkg.data:classification_path"
```

Where `my_pkg.data.classification_path` is a string or `Path` pointing
to the TOML file. Pedotri's registry will discover and load it the
first time `pedotri.list_classifications()` runs.

## Testing your custom classification

A handful of property-style assertions catch most authoring bugs:

```python
import pedotri

def test_every_corner_classifies():
    pedotri.register_classification("path/to/my.toml")
    # Three corners of the triangle should map to something
    for sand, clay in [(99, 0), (0, 99), (0, 0)]:
        assert pedotri.classify(sand, clay, "MY") is not None

def test_class_count_matches_toml():
    info = pedotri.get_classification("MY")
    assert len(info.classes) == EXPECTED_COUNT

def test_known_samples():
    # A handful of hand-classified samples from your reference table
    for sand, clay, expected in REFERENCE_SAMPLES:
        assert pedotri.classify(sand, clay, "MY") == expected
```

`REFERENCE_SAMPLES` should be at least 3-5 points per class, with at
least one sample near each class boundary (so a future re-typing of
the polygon coordinates that quietly shifts a boundary will trip the
test). See `tests/test_builtin_classifications.py` for how the
built-in classifications are tested.

## Related

- API guide: [Custom classifications](https://ivanfeanor.github.io/pedotri/classifications/custom/)
- [Classifications-and-when-to-use-them](Classifications-and-when-to-use-them) — pick your closest built-in for inspiration before authoring
- Schema reference: [`pedotri.Classification`](https://ivanfeanor.github.io/pedotri/api/) and `pedotri.TextureClass`
