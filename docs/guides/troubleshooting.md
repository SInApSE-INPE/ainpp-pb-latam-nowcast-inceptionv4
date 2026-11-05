# Troubleshooting

Grouped by where the message comes from. Run with `--verbose` for a full
traceback.

## Composition and configuration

These surface **before** any data is touched, which is deliberate.

`Domain extent must be a whole number of native cells`
: Your bounds do not divide evenly by the provider's resolution. Round them to
  the grid rather than letting anything resample.

`Grid edges must align with the native grid cells`
: The bounds fall mid-cell. A 0.1° grid starting at −180 admits −120.0, not
  −120.05.

`Grid extends beyond the native grid` / `Grid starts outside the native grid`
: The domain runs off the provider's extent. Check the native grid declared by
  your decoder against the provider's real coverage.

`Grid CRS differs` / `Grid resolution differs from the native grid`
: No resampling or reprojection is ever performed. Match the provider, or
  write a decoder that regrids deliberately.

`Artifact manifest does not declare the role: X`
: `bundle_roles` names a role your `manifest.json` does not. They must agree.

`Artifact checksum mismatch: X`
: The file does not match its recorded digest — corrupt, truncated, or
  replaced. Re-download. **Do not** regenerate the manifest to make the error
  go away; that defeats the check.

`Configured latitude order differs from the model bundle`
: The bundle states a training orientation and the configuration contradicts
  it. One of them is wrong, and publishing a vertically flipped forecast is
  worse than failing.

## Acquisition

`No observations available yet`
: Nothing in the directory or listing matched the decoder's name pattern. Test
  it directly:

      decoder().observed_at("the_real_filename.bin")

  A `None` means the pattern is wrong. This is the most common first failure
  with a new source.

`Incomplete window; missing: ...`
: One or more required observations are absent. Normal while waiting; a
  problem if it persists. The message names the exact timestamps.

`Latest observation is stale or in the future`
: The newest observation is older than `--max-age-hours`, or your clock is
  wrong. Applies only to a **live** source.

`Ambiguous duplicate observation for ...`
: Two files claim the same timestamp — often a mixture of naming conventions
  in one directory.

`FTP request failed after N attempts`
: Deliberately unspecific: server responses are not logged, because they can
  echo credentials. Test connectivity separately.

## Processing

`Missing fraction N% exceeds the limit`
: More of the field is non-finite than `max_missing_fraction` allows. Raise it
  only with a reason you would defend.

`Missing values at ...; choose an explicit policy`
: `missing_policy` is `error` and there are gaps. Choose `zero_rain` or
  `normalized_zero` consciously.

`Non-finite values after normalization`
: A transform received something it cannot handle — usually a raw sentinel
  that should have been clipped by a preprocess step first. Check stage order.

`Frame values do not match the grid`
: A decoder returned an array of the wrong shape. It must crop to the
  requested grid, not the native one.

## Inference and output

`Model returned (a, b, c) for a lead, expected (x, y, z)`
: What `leads` yielded does not match `output_variables` and the grid. Yielded
  arrays are `(members, variables, rows, columns)`.

`Model produced non-finite predictions`
: NaN or infinity from the network. Usually bad inputs or an untrained
  checkpoint.

`Output values must be finite and nonnegative`
: The sink refuses negatives. If your variable can legitimately be negative,
  that sink is the wrong one.

`Run already exists: ...`
: A complete run is never overwritten. Use a different `--output-dir`, or
  remove the directory deliberately.

```{admonition} Output looks vertically mirrored
:class: warning

Check `model_latitude_order`. The symptom is a forecast that looks plausible
but is flipped about the equator, and nothing raises — the shapes are right.
Compare a known coastline against the input, not against intuition.
```

## Tests

`PermissionError: [WinError 5]` under `%TEMP%\pytest-of-<you>`
: A stale pytest directory with broken permissions. Remove it from an elevated
  terminal, or pass `--basetemp` somewhere writable.

`symlinks unavailable on this platform`
: A skip, not a failure. Windows needs a privilege or Developer Mode.

`reference/ assets are not in the repository`
: A skip. Those tests need the supplied training reference, which is not
  redistributed.

Conformance failing right after a change
: That is the design. The suite runs your real composition root, so the
  failure is about the thing you just changed. Read which check failed before
  changing anything else.

## Containers

`KeyError: getpwuid(): uid not found: 10001`
: A library called `getpass.getuser()` at import time. Set `LOGNAME` in the
  image; a baked passwd entry cannot work for an arbitrary host UID.

`DECRYPTION_FAILED_OR_BAD_RECORD_MAC` while downloading a wheel
: A network problem inside the Docker build network, not a Dockerfile
  problem. The host reaching the same index proves nothing, because the build
  has its own network namespace. Retry; check MTU or VPN settings.

`The lockfile at uv.lock needs to be updated, but --locked was provided`
: The build context does not reproduce the resolution context. Confirm every
  file the lock depends on is copied in, and that the `uv` version in the
  image matches the one that wrote the lock.

## When nothing matches

1. Run with `--verbose`.
2. Run the conformance suite — it isolates a contract violation from a data
   problem.
3. Read the last manifest. `preprocess`, `postprocess` and `normalization`
   record exactly what ran.
4. Reproduce with `--source local` and a handful of files. Removing the
   network halves the search space.
