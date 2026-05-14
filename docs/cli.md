# Command-line interface

pedotri installs a `pedotri` script when installed. It uses only the standard library (argparse) so no extras are required.

## Subcommands

### `list`

List registered classifications:

```bash
$ pedotri list
EMBRAPA
FAO
GEPPA
HYPRES
INTERNATIONAL
ISSS
JAMAGNE
KA5
KACHINSKY
USDA
```

### `info`

Show metadata and the full class list for one classification:

```bash
$ pedotri info USDA
USDA — USDA
  axes:    sand, clay
  classes: 12
  locale:  en (default: en)
  ref:     Soil Survey Division Staff (1993). Soil Survey Manual...
  ...

Classes:
  clay                 clay                         polygon (5 vertices)  [fine]
  silty_clay           silty clay                   polygon (3 vertices)  [fine]
  ...
```

Pass `--locale fr` to render the class list in French.

### `classify` — single sample

```bash
$ pedotri classify --sand 60 --clay 20 -c USDA
sandy_clay_loam

$ pedotri classify --sand 60 --clay 20 -c USDA --locale fr
limon argilo-sableux

$ pedotri classify -c KACHINSKY --physical-clay 35
medium_loam
```

### `classify` — batch CSV

Input CSV must have a column for each axis of the chosen classification:

```bash
$ cat samples.csv
sand,clay
60,20
10,70
80,5

$ pedotri classify --csv samples.csv --output results.csv -c USDA
Wrote 3 rows to results.csv.

$ cat results.csv
sand,clay,texture
60,20,sandy_clay_loam
10,70,clay
80,5,loamy_sand
```

If `--output` is omitted, the CSV is printed to stdout.

### `render` — write an SVG

```bash
$ pedotri render -c USDA -o usda.svg --title "USDA Triangle"
Wrote usda.svg.

$ pedotri render -c GEPPA --locale fr --no-grid -o geppa.svg
```

Omit `-o` to print the SVG to stdout (useful for piping into `display` or other tools).

## Exit codes

- `0` — success
- `1` — user-facing error (unknown classification, missing column, etc.)
- `2` — argparse usage error
