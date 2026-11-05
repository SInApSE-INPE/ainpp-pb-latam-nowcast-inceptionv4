# Adding a new data source

Two separate questions, often confused:

| | |
| --- | --- |
| **Where do files come from?** | a `Source` |
| **What is inside a file?** | a `Decoder` |

They are separate because a single format is usually read from several places
— a local directory while developing, a remote server in production. Keeping
them apart means one decoder serves both.

```{important}
A source owns **path syntax** and reports bare file names. A decoder owns
**name semantics** and answers "is this one of mine, and for when?".

This is not tidiness. Name parsing once lived in shared code that assumed
POSIX separators, so on Windows an entire native path was treated as the file
name, nothing matched, and local acquisition was silently broken.
```

## A new format, without writing a decoder

If your data is a headerless binary block on a regular grid, it is
configuration:

```python
RawBinaryDecoder(
    native_grid=Grid("EPSG:4326", -180.0, 90.0, 0.25, 0.25, 1440, 720),
    naming=TimestampedName(
        pattern=r"sat\.(?P<stamp>\d{12})\.bin",
        stamp_format="%Y%m%d%H%M",
        template="sat.%Y%m%d%H%M.bin",
    ),
    units="K",
    dtype=">f4",        # big-endian
    roll_columns=0,
    gzipped=False,
    suffix=".bin",
)
```

| Parameter | Covers |
| --- | --- |
| `dtype` | byte order and width: `<f4`, `>f4`, `<i2`, … |
| `native_grid` | the extent and resolution the provider publishes |
| `roll_columns` | a longitude origin that is not the array's first column |
| `gzipped`, `suffix` | compression and extension |
| `naming` | the file name contract |

### File names

`pattern` matches, `template` generates. They are different jobs: a pattern
with optional groups accepts more names than it can produce, so both are
declared and the generated name is checked against the pattern.

```python
TimestampedName(
    pattern=r"gsmap_now\.(?P<stamp>\d{8}\.\d{4})(?:_(?P<end>\d{4}))?\.dat\.gz",
    stamp_format="%Y%m%d.%H%M",
    template="gsmap_now.%Y%m%d.%H%M.dat.gz",
    interval=timedelta(hours=1),
    end_format="%H%M",
)
```

The optional `end` group handles a provider that appends the end of its
accumulation window. With `interval` and `end_format` set, that suffix is
**verified** rather than trusted.

```{tip}
One `stamp` group holding the whole timestamp is what makes a sub-hourly
product work. Ten-minute data is a different `stamp_format`, not a different
class.
```

## A format that is not a raw block

Write a decoder. The contract is four members:

```python
class NetcdfDecoder:
    def __init__(self, native_grid, variable, units):
        self._grid, self._variable, self._units = native_grid, variable, units

    @property
    def native_grid(self): return self._grid

    @property
    def units(self): return self._units

    def observed_at(self, name: str) -> datetime | None:
        match = re.fullmatch(r"obs_(\d{12})\.nc", name)
        return (
            datetime.strptime(match[1], "%Y%m%d%H%M").replace(tzinfo=UTC)
            if match else None
        )

    def read(self, path: Path, grid: Grid) -> Frame:
        row, column = self.native_grid.locate(grid)      # validates alignment
        with netCDF4.Dataset(path) as source:
            block = source[self._variable][
                row : row + grid.height, column : column + grid.width
            ]
        values = np.asarray(block, dtype=np.float32)
        missing = ~np.isfinite(values)
        return Frame(values[np.newaxis], missing[np.newaxis],
                     self.units, grid, self.observed_at(path.name))
```

Three obligations:

1. `observed_at` returns `None` for a name that is not yours. Returning a
   guess instead makes an unrelated file look like an observation.
2. `read` calls `native_grid.locate(grid)` **before** slicing. That is the
   check you owe your caller.
3. Return physical units with a mask. A provider sentinel is **not** missing
   data; clip it in a preprocess step.

```{tip}
Add `filename()` and `synthesize()` and the conformance suite will exercise
your whole deployment with no real data. `synthesize` is just `read`
backwards.
```

## A new source

```python
class S3Prefix:
    @property
    def identity(self) -> str:
        return f"s3://{self._bucket}/{self._prefix}"   # never credentials

    @property
    def live(self) -> bool:
        return True

    def candidates(self, window, run_time):
        return [
            (PurePosixPath(key).name, key)
            for key in self._client.list(self._bucket, self._prefix)
        ]

    def fetch(self, observation) -> Path:
        target = self._cache / observation.name
        if not target.exists():
            self._client.download(self._bucket, observation.location, target)
            self._verify(target)
        return target

    def prune(self, ttl) -> int:
        ...
```

Four things to get right:

`identity`
: Goes into every run manifest. **Never** include credentials.

`live`
: `True` for a provider that keeps publishing, `False` for supplied files.
  Only a live source is subject to the freshness limit — getting this wrong
  makes reprocessing an archive impossible.

`candidates`
: Returns `(bare name, location)` pairs. The decoder sees only the name.

`fetch`
: Must return a **fully transferred and structurally valid** file. Verify by
  decoding; a truncated download that decodes as garbage is worse than a
  failed one.

```{warning}
`prune` must only ever delete from directories it created. The built-in FTP
source scopes deletion to namespace directories named with sixteen hex
characters, so a user directory sharing the cache root is never touched.
```

## Wiring it in

Only the composition root changes:

```python
def compose(args):
    config = Config.load(args.config)
    reader = NetcdfDecoder(NATIVE_GRID, "precipitation", "mm/h")
    grid = config.domain.to_grid(reader.native_grid)
    source = (
        LocalFiles(args.input_dir, "*.nc") if args.source == "local"
        else S3Prefix(bucket, prefix, args.cache_dir, verify=lambda p: reader.read(p, grid))
    )
    ...
```

Nothing in the runtime changes. Run conformance; if it passes, the contracts
hold.
