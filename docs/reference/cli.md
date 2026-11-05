# Command line reference

## Runtime commands

Installed with `nowcast-runtime`, available from the workspace.

### `nowcast-new-model`

Generates a model project that already passes conformance.

```text
usage: nowcast-new-model [-h] --name NAME [--slug SLUG] [--package PACKAGE]
                         [--product PRODUCT] [--units UNITS] [--into INTO]
                         [--force]

options:
  -h, --help         show this help message and exit
  --name NAME        Display name, e.g. MyUNet
  --slug SLUG        Path and filename safe token; defaults from the name
  --package PACKAGE  Python package name; defaults to <slug>_nowcast
  --product PRODUCT  Input product identifier used in file names
  --units UNITS      Physical units of the data
  --into INTO        Target directory; defaults to models/<slug>
  --force            Write into a non-empty directory
```

`--force` writes into a directory that already has files. Existing files it
does not generate are left alone.

### `nowcast-download-artifacts`

Downloads and verifies a bundle against a **local** manifest.

| Option | Meaning |
| --- | --- |
| `--bundle PATH` | bundle directory holding a trusted `manifest.json` |
| `--base-url URL` | override the `download_base_url` in that manifest |
| `--attempts N` | transfer attempts per file, default 3 |
| `--timeout SECONDS` | per-request timeout, default 60 |
| `--list` | print expected URLs and checksums, download nothing |

Files already matching their checksum are skipped. Downloads are atomic: a
file is verified in a temporary location and only then moved into place.

## Model commands

Each model gets three console scripts, named after its slug. For InceptionV4:

| Command | Purpose |
| --- | --- |
| `inceptionv4-nowcast` | run the pipeline once |
| `inceptionv4-serve` | the container entrypoint: verify the bundle, then poll |
| `inceptionv4-download-artifacts` | fetch and verify this model's bundle |

### Running the pipeline

```text
usage: inceptionv4-nowcast [-h] --config CONFIG [--source {local,ftp}]
                           [--input-dir INPUT_DIR] [--cache-dir CACHE_DIR]
                           [--cache-ttl-hours CACHE_TTL_HOURS]
                           [--output-dir OUTPUT_DIR] [--run-time RUN_TIME]
                           [--steps STEPS] [--device {auto,cpu,cuda}]
                           [--threads THREADS] [--max-age-hours MAX_AGE_HOURS]
                           [--verbose] [--scheduled]
```

`--config PATH` *(required)*
: The deployment configuration. Paths inside it resolve relative to the file.

`--source {local,ftp}`
: Where observations come from. `local` requires `--input-dir`.

`--input-dir PATH`
: Directory of supplied observations, searched recursively. **Never pruned.**

`--cache-dir PATH`
: Where transfers are cached. Defaults to a per-user directory under the
  system temporary directory. Only managed subdirectories are ever pruned.

`--cache-ttl-hours N`
: Retention for cached transfers, default 24. Cleanup runs on every FTP
  execution and every scheduled poll, including polls that publish nothing.

`--output-dir PATH`
: Root for published runs. Defaults to `outputs`.

`--run-time TIME`
: Timestamp of the **last** input, timezone required, for example
  `2000-01-02T12:00:00Z`. Without it the latest complete window is used.

`--steps N`
: Number of leads. Defaults to the model's native horizon; asking for more is
  an error.

`--device {auto,cpu,cuda}`
: Passed to the model, which resolves `auto` itself.

`--threads N`
: CPU threads for inference, default 4.

`--max-age-hours N`
: Freshness limit, default 12. Applies only to a **live** source; a supplied
  directory is exempt, so reprocessing an archive is always possible.

`--verbose`
: Debug logging, and a full traceback on failure.

`--scheduled`
: One idempotent poll. Requires `--source ftp` and no `--run-time`.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | published, or nothing to do |
| `1` | the run failed; the message says why |
| `2` | the service could not start, such as an invalid poll interval |

```{note}
`DataNotReady` is **not** a failure. An incomplete or stale window exits `0`
after logging `WAITING`, because waiting for data is normal operation.
```

### Scheduled poll outcomes

A poll logs one of four stable tokens, suitable for monitoring:

| Token | Meaning |
| --- | --- |
| `NEW_DATA` | a new issue time was found and is being processed |
| `UP_TO_DATE` | the latest issue time is already published and verified |
| `WAITING` | the window is incomplete or stale; retry next cycle |
| `BUSY` | another poll holds the lock |

## Environment

| Variable | Used by | Meaning |
| --- | --- | --- |
| `NOWCAST_POLL_INTERVAL_SECONDS` | the service loop | seconds between polls, default 300 |
| `FTP_JAXA_HOST` | the InceptionV4 FTP source | override the default host |
| `FTP_JAXA_USER` | the InceptionV4 FTP source | credentials, required for FTP |
| `FTP_JAXA_PASSWORD` | the InceptionV4 FTP source | credentials, required for FTP |

```{important}
Credentials are read from the environment only. They never appear in a
configuration file, on a command line, in a log, or in a run manifest, and the
source's recorded identity is host and directory alone.
```

A generated model names its own variables after its slug, for example
`MYUNET_FTP_USER`.
