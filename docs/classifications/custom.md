# Custom classifications

A classification is a TOML file describing axes, classes (polygons or intervals), and localized names. pedotri loads and validates TOML at runtime, so you can add a classification without touching any Python code.

## Loading at runtime

```python
import pedotri

pedotri.register_classification("path/to/my_scheme.toml")
pedotri.classify(40, 30, "MY_SCHEME")
```

`register_classification` accepts a path, raw TOML bytes, a pre-parsed dict, or a `Classification` instance.

## Distributing as a plugin

Third-party packages can ship classifications without users having to call `register_classification`. Add an entry point in the `pedotri.classifications` group of your `pyproject.toml`:

```toml
[project.entry-points."pedotri.classifications"]
MY_SCHEME = "my_package:get_classification"
```

The referenced callable receives no arguments and must return either a `Classification` instance or anything that `load_classification` accepts (path, bytes, or dict). pedotri discovers and registers the classification on first use.

## TOML schema

### Two-axis (sand–clay) classification

```toml
[meta]
key = "MY_SCHEME"
axes = ["sand", "clay"]      # always two of: sand, silt, clay
default_locale = "en"
reference = "Author (year). Title. Journal vol(issue): pages."
url = "https://..."           # optional
description = "Free-form notes about scope and intended use."  # optional

[meta.names]
en = "My scheme"
fr = "Mon schéma"
ru = "Моя схема"

[[class]]
key = "clay"                  # snake_case ASCII identifier
group = "fine"                # optional textural-group tag, e.g. "fine" / "medium" / "coarse"
parent = "clay"               # optional hierarchical parent (must be another class key in this file)
vertices = [
  [0, 100],                   # (sand %, clay %) pairs
  [0, 60],
  [45, 40],
  [45, 55],
]

[class.names]
en = "clay"
fr = "argile"

[[class]]
key = "loam"
group = "medium"
vertices = [[0, 0], [50, 0], [25, 30]]
[class.names]
en = "loam"
fr = "limon"
```

**Rules:**

- `axes` must be a 2-element list when classes use `vertices`. Permitted axis names: `sand`, `silt`, `clay`, `physical_clay`.
- `vertices` must have at least 3 points; closure (joining the last vertex to the first) is implicit.
- Class keys must be unique within a classification.
- The first class to contain a point wins. Edge cases on shared boundaries are deterministically attributed to the earlier-listed class.

### One-axis (interval) classification

```toml
[meta]
key = "MY_AXIS_SCHEME"
axes = ["physical_clay"]      # exactly one axis
default_locale = "ru"
reference = "..."

[meta.names]
ru = "Моя одноосная схема"
en = "My axis scheme"

[[class]]
key = "loose_sand"
group = "coarse"
interval = [0, 5]             # half-open [low, high)
[class.names]
ru = "песок рыхлый"
en = "loose sand"

[[class]]
key = "sandy_loam"
interval = [5, 20]
[class.names]
en = "sandy loam"
```

**Rules:**

- `axes` is a 1-element list when classes use `interval`.
- `interval = [low, high]` is half-open: a value matches when `low <= v < high`. Adjacent classes therefore tile the axis without overlap.

## Validation

`load_classification` validates the document on load. Common errors:

- Missing or wrong-length `axes`.
- A class with both `vertices` and `interval` (or neither).
- Duplicate class keys.
- Polygons with fewer than 3 vertices.
- Intervals where `low >= high`.

All raise `pedotri.ClassificationError` with a message that points at the offending field.

## i18n

Class names and the classification's own name are looked up via a locale-fallback chain:

`fr-CA` → `fr` → `en` → class key

If a user passes a locale that no class declares, the resolver falls all the way back to the class key. So you can ship a classification with English names only and add more locales later without breaking existing users.

## Tips

- Use `group` tags consistently across your classification so plotting backends can colour-code by texture group.
- Keep `key` snake_case and stable — it appears in user data, file paths, and reports.
- The polygon list is order-significant. Place narrow / specific classes earlier than broad ones if their boundaries touch.
