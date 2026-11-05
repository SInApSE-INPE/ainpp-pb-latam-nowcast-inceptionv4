# The ports

All contracts live in `nowcast_runtime.ports` as `typing.Protocol`. Structural
typing means an adapter never imports the framework, which is real dependency
inversion.

```{admonition} Unit-length axes are never squeezed
:class: important

Several shapes carry axes that are length one in every deployment today — the
variable axis on a frame, the member axis on a prediction. That is deliberate.
The contracts admit several input products, auxiliary variables and ensemble
members, and squeezing a unit axis in shared code is exactly what makes
supporting them later expensive.
```

## Value objects

These validate on construction, so an invalid one cannot be passed anywhere.

### `Grid`

A regular, axis-aligned, north-up grid of cell areas in one CRS. The origin is
the outer corner of the north-west cell, matching the GeoTIFF convention, so a
grid is fully described without a separate transform.

```python
Grid(crs="EPSG:4326", west=-180.0, north=60.0,
     resolution_x=0.1, resolution_y=0.1, width=3600, height=1200)
```

`locate(other)` returns the row and column of another grid's origin inside
this one, raising unless it is a cell-aligned subgrid. That is the check every
decoder owes its caller before slicing.

### `WindowSpec`

Which observations a run consumes, as offsets from its run time. Offsets are
non-positive, unique and increasing, so the run time is the last entry and no
run can depend on a future observation.

```python
WindowSpec.uniform(timedelta(hours=1), 12)        # the common case
WindowSpec((-timedelta(hours=6), -timedelta(hours=1), timedelta(0)))
```

Explicit offsets subsume "a fixed step and a count" and admit a non-uniform
history at no cost. This is what removed the assumption that one hour was the
only possible spacing.

### `Frame`

One decoded observation in physical units, with its missing-data mask. Both
arrays are `(variables, height, width)`.

### `Prediction`

One lead, with the mask it inherited from its inputs. Values are
`(members, variables, height, width)`. Carrying the mask here keeps a
prediction self-contained, so a postprocess step and a sink each get
everything they need from one object.

## `Source`

Where observations are listed and how they reach local storage.

```python
@property
def identity(self) -> str: ...            # for the manifest; never credentials
@property
def live(self) -> bool: ...               # does it keep publishing?
def candidates(self, window, run_time) -> Sequence[tuple[str, str]]: ...
def fetch(self, observation) -> Path: ...
def prune(self, ttl) -> int: ...
```

```{important}
A source owns **path syntax** and reports each candidate's bare name, so shared
code never has to guess whether a location is POSIX or native.
```

`live` distinguishes a provider that keeps publishing from a directory of
supplied files. Only a live source is subject to the freshness limit;
rejecting a supplied archive for being old would make reprocessing impossible.

Built-in: `LocalFiles`, `FtpSource`.

## `Decoder`

How a named file becomes a `Frame` on the requested grid. The decoder owns the
file name contract and the native grid, which is why a new input product needs
no change to acquisition or scheduling.

```python
@property
def native_grid(self) -> Grid: ...
@property
def units(self) -> str: ...
def observed_at(self, name: str) -> datetime | None: ...
def read(self, path: Path, grid: Grid) -> Frame: ...
```

Two optional methods, which a conformance suite uses to exercise a whole
deployment with no real data:

```python
def filename(self, observed: datetime) -> str: ...
def synthesize(self, path, values, grid) -> Path: ...
```

Built-in: `RawBinaryDecoder`, plus a `gsmap()` factory that is nothing more
than one configuration of it.

## `Preprocess` and `Postprocess`

Ordered, possibly empty sequences of steps in physical units.

```python
def apply(self, payload): ...                 # Frame, or Prediction
def describe(self) -> Mapping[str, object]: ...
```

`describe()` is recorded in the run manifest, so it must be JSON serializable.
Built-in: `ClipNegative`, `MissingPolicy`, `FlipLatitude`. Ordering is covered
in [Processing stages](stages.md).

## `Transform`

A model's input scaling, and its exact inverse.

```python
def forward(self, values: np.ndarray) -> np.ndarray: ...
def inverse(self, values: np.ndarray) -> np.ndarray: ...
def describe(self) -> Mapping[str, object]: ...
```

Optional. A model consuming physical units declares none and the runtime
applies identity.

```{warning}
A transform is bound by the plugin and **cannot** be overridden by
configuration. Its statistics belong to one checkpoint; a mismatched pair
raises nothing and produces plausible, meaningless output.
```

Built-in: `Identity`, `LogZScore`.

## `Predictor`

A loaded model, ready to infer on one fixed grid.

```python
@property
def window(self) -> WindowSpec: ...
@property
def step(self) -> timedelta: ...
@property
def output_steps(self) -> int: ...
@property
def output_variables(self) -> int: ...
@property
def grid(self) -> Grid: ...
def leads(self, inputs: np.ndarray, steps: int) -> Iterator[np.ndarray]: ...
```

Inputs arrive as `(time, variables, rows, columns)` in model space. `leads`
yields **bare arrays** shaped `(members, variables, rows, columns)`: lead
numbering, valid times and the inherited mask are the runtime's job, so a
model author writes less and cannot make them inconsistent.

It returns an iterator, so a direct model can run a single forward pass while a
recurrent or autoregressive one advances its own state per lead.

`output_variables` is declared rather than inferred from the inputs, because a
model may consume one field and emit several — a rate beside an exceedance
probability, say.

## `Sink`

How one lead is named and written.

```python
@property
def extension(self) -> str: ...
def filename(self, slug, run_time, prediction, member) -> str: ...
def write(self, path, prediction, member, grid, units, tags) -> None: ...
```

A deterministic run omits any member component from the name, so the
single-member case reads exactly as it did before members were admitted.

Built-in: `GeoTiffSink`, one band per variable.

## `ModelPlugin`

Everything the runtime needs from a model. Only six fields are required:

```python
PLUGIN = ModelPlugin(
    name="MyUNet",
    slug="myunet",
    version="0.1.0",
    bundle_roles=("checkpoint",),
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    # optional:
    transform=None, preprocess=None, postprocess=None, library_versions=dict,
)
```

`bundle_roles` is declared rather than fixed, so a model with no normalization
statistics, or one carrying auxiliary static fields, is expressible without
changing the manifest reader.

## How conformance is enforced

Structural typing is checked only if someone runs a type checker, and nobody
runs one on your code. So conformance has three layers:

1. static checking inside this workspace,
2. a structural check when an adapter is resolved,
3. **a conformance suite** that verifies semantics a type cannot express.

The third is where it actually happens. Types do not capture contracts; tests
do. See [Proving conformance](../guides/conformance.md).
