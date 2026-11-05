# Grids and windows

Geometry and time are the two things everything else is expressed against, so
both are value objects that validate on construction.

## `Grid`

```python
Grid(crs="EPSG:4326", west=-180.0, north=60.0,
     resolution_x=0.1, resolution_y=0.1, width=3600, height=1200)
```

North-up, axis-aligned, cell areas, one CRS. The origin is the **outer corner
of the north-west cell**, matching the GeoTIFF convention, so no separate
affine transform is needed to describe it.

Derived properties: `shape` (rows, columns, matching NumPy and rasterio),
`east`, `south`.

### Negotiation between what you want and what exists

Three parties have an opinion about geometry, and they are reconciled
explicitly rather than assumed to agree:

| Party | Knows |
| --- | --- |
| the **decoder** | the native grid its provider publishes |
| the **configuration** | the domain an operator wants |
| the **model** | any constraint of its own, such as a minimum size |

Configuration carries **bounds only**. Resolution and CRS come from the
decoder:

```python
grid = config.domain.to_grid(decoder.native_grid)
```

That fails loudly when the two disagree. `Grid.locate` is the check:

```python
row, column = native.locate(requested)
```

It raises with a specific reason rather than quietly resampling:

| Message | Means |
| --- | --- |
| `Grid CRS differs` | the request is in a different coordinate system |
| `Grid resolution differs from the native grid` | no resampling is performed, ever |
| `Grid edges must align with the native grid cells` | the bounds fall mid-cell |
| `Grid starts outside the native grid` | north or west is beyond the provider's extent |
| `Grid extends beyond the native grid` | the requested size runs off the edge |

```{admonition} No silent resampling
:class: important

A requested grid is either a cell-aligned subgrid of the native one or it is
an error. Interpolating to make a request fit would change the data a model
sees while nothing in the output said so.
```

### A worked example

The LATAM domain against the global GSMaP grid:

```python
GSMAP = Grid("EPSG:4326", -180.0, 60.0, 0.1, 0.1, 3600, 1200)
LATAM = Grid("EPSG:4326", -120.0, 33.0, 0.1, 0.1, 970, 880)

GSMAP.locate(LATAM)     # (270, 600)
```

Row 270 because `(60 - 33) / 0.1`, column 600 because `(-120 + 180) / 0.1`.
That arithmetic used to be inlined in the GSMaP reader; a regression test now
pins `locate` to exactly those offsets.

## `WindowSpec`

Which observations one run consumes, as offsets from its run time.

```python
WindowSpec.uniform(timedelta(hours=1), 12)
# offsets: (-11h, -10h, ..., -1h, 0)
```

The invariants, all enforced on construction:

- offsets are **non-positive**, so no run depends on a future observation
- they are unique and increasing
- the last is exactly zero, so the run time is itself an observation

### Why offsets rather than a step and a count

A step and a count can only express a uniform history. Offsets subsume that
and admit a non-uniform one at no cost:

```python
WindowSpec((-timedelta(hours=6), -timedelta(hours=1), timedelta(0)))
```

A model that wants "an hour ago, ten minutes ago, and now" simply says so. One
hour stops being written into the contract, which is what makes a ten-minute
radar product and a six-hourly model equally expressible.

### Resolving a window

```python
window.times(run_time)   # -> (datetime, ...) in chronological order
```

The run time must be timezone aware. Selection is exact: every named time must
be present, or the run raises `DataNotReady` and a scheduled poll waits.

```{warning}
A partial window is **never** silently narrowed to an older complete one. That
would publish a forecast built from the wrong history while looking entirely
successful.
```

## Masks

A cell that is missing in **any** input variable at **any** time in the window
is masked in **every** output. The mask is reduced across the variable axis, so
it broadcasts regardless of how many variables a model emits.

Masked output cells are written as the sink's NoData value, `-9999.0` for
GeoTIFF, and `masked_output_fraction` in the manifest records how much of the
field that was.

```{note}
A provider sentinel such as `-99` is **not** missing data. It is a finite
negative that means zero, and it is clipped by a preprocess step before any
transform. Only non-finite values enter the mask. Confusing the two would turn
ordinary dry pixels into NoData.
```

## Time in outputs

Valid times follow the predictor's declared `step`:

```text
valid_time = run_time + lead * predictor.step
```

Lead numbering, valid times and the mask are computed by the runtime, not by a
model. A model yields bare arrays, so it cannot make them inconsistent with
each other.
