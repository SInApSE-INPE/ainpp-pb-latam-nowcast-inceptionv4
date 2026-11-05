# Artifacts and manifests

Two different records, easy to confuse:

| | |
| --- | --- |
| **Artifact manifest** | ships with a model bundle; says which files it holds and their checksums |
| **Run manifest** | written beside each published forecast; says what produced it |

## The artifact bundle

A versioned directory holding a checkpoint and whatever else a model needs:

```text
artifacts/v1/
├── manifest.json
├── inceptionv4.pth
├── gsmap_nrt+mvk_log_mean_ainpp-latin-america.npy
└── gsmap_nrt+mvk_log_std_ainpp-latin-america.npy
```

```json
{
  "schema_version": 1,
  "version": "v1",
  "download_base_url": "https://example.org/checkpoints/myunet/v1/",
  "files": {
    "checkpoint": {"name": "inceptionv4.pth", "sha256": "33d03add..."},
    "mean":       {"name": "..._log_mean_....npy", "sha256": "fc60bf59..."},
    "std":        {"name": "..._log_std_....npy",  "sha256": "8e09d7e6..."}
  }
}
```

### Roles are declared by the plugin

```python
bundle_roles = ("checkpoint", "mean", "std")
```

The reader checks every role a plugin declared and only those. A model with no
normalization statistics names only `checkpoint`; a model with auxiliary
static fields names those too. Nothing in the runtime is fixed to a particular
set, which is what lets two very different models share one reader.

Verification happens before anything is loaded, and a mismatch is fatal.

### Artifacts stay out of the image

A container image carries the code, the configuration and the manifest — not
the weights. At startup the service verifies the local bundle and downloads
whatever is missing or invalid:

```bash
uv run <slug>-download-artifacts            # verify and fetch
uv run <slug>-download-artifacts --list     # show URLs without downloading
```

```{important}
**The local manifest is the source of truth; remote metadata is never
trusted.** Downloads are verified against the checksums you shipped, not
against anything the server says.

The consequence is deliberate: publishing a new model version means a new
bundle and a redeploy, not swapping a file on a server. If that becomes
limiting, the answer is signed manifests, not trusting the remote.
```

Keep version prefixes **immutable**. Publish `v2/`; never mutate `v1/`.

## The run manifest

Written beside every published forecast, and the reason a file produced months
ago can still be explained.

```json
{
  "schema_version": 1,
  "status": "complete",
  "created_at": "2026-11-05T16:41:40.560578+00:00",
  "run_time": "2000-01-02T12:00:00+00:00",
  "model": "RainNet",
  "model_version": "0.1.0",
  "bundle_version": "v1",
  "artifacts": {"checkpoint": {"name": "...", "sha256": "..."}},
  "normalization": {"kind": "identity"},
  "preprocess": [{"kind": "missing-policy", "policy": "zero_rain", "...": "..."}],
  "postprocess": [],
  "source": "local:data/raw",
  "units": "mm/h",
  "crs": "EPSG:4326",
  "grid": {"crs": "EPSG:4326", "west": -10.0, "north": 5.0, "...": "..."},
  "device": "auto",
  "versions": {"nowcast-runtime": "0.3.0", "rainnet": "0.1.0", "numpy": "2.x"},
  "input_steps": 3,
  "input_offsets": [-7200.0, -3600.0, 0.0],
  "step_seconds": 3600.0,
  "output_steps": 3,
  "native_output_steps": 3,
  "masked_output_fraction": 0.0,
  "inputs": [{"name": "...", "time": "...", "sha256": "...", "missing_fraction": 0.0}],
  "outputs": [{"name": "...", "valid_time": "...", "lead_hours": 1,
               "member": 0, "sha256": "..."}]
}
```

### Fields worth knowing

`status`
: Always `complete`. A partial run is never published, so the field exists to
  be checked rather than to vary.

`artifacts`, `versions`
: Which weights and which library versions produced this. Reproducing the run
  means reproducing these.

`preprocess`, `postprocess`, `normalization`
: Exactly which steps ran, in order, with their parameters. If you ever wonder
  whether a flip was applied, this answers it.

`source`
: A stable identity such as `ftp://host/now/latest`. **Never credentials.**

`input_offsets`, `step_seconds`
: The window and spacing in seconds, so nothing assumes hours.

`inputs[].sha256`
: A checksum of every observation consumed, so an input that was later
  revised upstream is detectable.

`outputs[].sha256`
: What makes idempotency verifiable rather than assumed. A scheduled poll
  re-checks these before deciding a run is already done.

`masked_output_fraction`
: How much of the field was NoData because of missing inputs. A rising value
  is a data-quality signal worth alerting on.

```{note}
`device` records what was **requested**, not what was resolved. A run that
asked for `auto` records `auto`, even though the model may have chosen CPU.
Resolving that would need the predictor to report its device, which no port
currently requires.
```

## Idempotency

`completed_run` decides whether an issue time is already done:

1. the directory exists and holds a manifest,
2. the model, status, run time, product and horizon all match,
3. the recorded leads are exactly `1..steps`,
4. **every output file still matches its recorded checksum**.

Any inconsistency is an **error**, not a skip. A half-written or tampered run
stops the poll loudly rather than being treated as finished.

## Atomic publication

Outputs are written into a staging directory inside the target's parent, and
the whole directory is renamed into place only after every file **and** the
manifest exist.

```{important}
A crash or a disk error halfway leaves **no directory at all**, rather than a
directory a downstream consumer would happily read as complete. Conformance
checks this by making a write fail and asserting nothing remains.
```

A rename within one filesystem is atomic, which is why staging happens beside
the target rather than in a system temporary directory.
