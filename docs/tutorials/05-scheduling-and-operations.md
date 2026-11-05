# Scheduling and operations

The unit of work is **one idempotent poll**. Any scheduler can drive it, which
is why the scheduler itself is not a port.

**From `models/myunet/`:**

```bash
myunet-nowcast --config config/default.json --source ftp --scheduled \
    --output-dir outputs
```

## What a poll does

1. Take the advisory lock, or log `BUSY` and exit.
2. Prune expired cached transfers.
3. List the provider — **cheaply**, one observation, no checkpoint read.
4. Check freshness against `--max-age-hours`.
5. If that issue time is already published and every output still matches its
   checksum, log `UP_TO_DATE` and exit.
6. Otherwise acquire the window, load the model, run, publish atomically.

```{important}
Step 3 before step 6 is deliberate. Composition is two-phase so that a poll
finding nothing new never reads a checkpoint that may be hundreds of
megabytes.
```

## Reading the logs

Four stable tokens, suitable for alerting:

| Token | Meaning | Normal? |
| --- | --- | --- |
| `NEW_DATA` | a new issue time is being processed | yes |
| `UP_TO_DATE` | the latest is already published and verified | yes |
| `WAITING` | the window is incomplete or stale | yes, often |
| `BUSY` | another poll holds the lock | occasionally |

```{note}
`WAITING` is **not** an error and exits `0`. Waiting for data that has not
been published is ordinary operation. Alerting on it will produce noise and
train people to ignore the alerts.
```

## Three ways to drive it

### A systemd timer

The OS owns scheduling, restart and logging, and a one-shot unit cannot drift.

**From `models/myunet/`:**

```bash
uv run python deploy/render_units.py
systemd-analyze verify deploy/generated/myunet.service deploy/generated/myunet.timer
```

The units land in `models/myunet/deploy/generated/`, which is gitignored
because they carry absolute host paths. The generator writes **only** there:
it installs nothing and enables nothing. Regenerate after moving the project.

### The container loop

One artefact, portable, no host configuration. See
[Running in a container](04-running-in-a-container.md).

### An external orchestrator

A Kubernetes `CronJob`, Argo, Airflow — anything that can run the one-shot
command. Set `concurrencyPolicy: Forbid`; the file lock is host scoped and
does not coordinate replicas.

## Choosing an interval

```{admonition} In nowcasting, latency is the product
:class: important

Polling every five minutes for hourly data costs up to five minutes of a
product whose value decays in minutes, and spends about twelve listings an
hour to do it.

Polling more often does not make the provider publish sooner. It only narrows
the gap between publication and detection. If the provider offers a
notification, use it; the polling loop is a fallback, not an ideal.
```

A reasonable middle ground is adaptive polling: frequent during the window
when data usually arrives, relaxed outside it. The latency metric below tells
you where that window is.

## Worth measuring

Two signals carry most of the value:

**Latency from issue time to publication.** The service-level indicator that
matters. `run_time` and `created_at` in each manifest give it to you with no
extra instrumentation.

**Staleness.** *No run published for more than twice the step* is the single
most useful alert. It catches a dead provider, a crash loop, a full disk and
an expired credential, all with one rule.

Also worth watching: `masked_output_fraction` rising, repeated
`IntegrityError`, and polls that never reach `NEW_DATA`.

## Failure modes

| Symptom | Likely cause | Response |
| --- | --- | --- |
| `WAITING` forever | the provider stopped, or names changed | check a listing by hand |
| `BUSY` every cycle | a previous poll hung | the lock releases on death; look for a stuck process |
| `Artifact checksum mismatch` | corrupt or swapped bundle | re-download; do **not** retry blindly |
| `FileExistsError` | reprocessing an existing issue time | intended; use a different output directory |
| `Missing fraction exceeds the limit` | degraded input | raise `max_missing_fraction` only with a reason |

## Cache retention

Transfers are cached under a per-user temporary directory and expire after
`--cache-ttl-hours`, default 24. Cleanup runs on **every** execution and every
scheduled poll, including polls that publish nothing.

```{important}
Only managed namespace directories are pruned. Files supplied through
`--input-dir` are **never** removed, whatever their age.
```

## Backfilling

There is no `--backfill` mode yet. Because runs are idempotent and keyed by
issue time, a loop works:

**From `models/myunet/`:**

```bash
for hour in $(seq 0 23); do
  myunet-nowcast --config config/archive.json --source ftp \
      --run-time "2026-01-15T$(printf %02d $hour):00:00Z" \
      --output-dir outputs || true
done
```

An archive configuration — a second file such as
`models/myunet/config/archive.json` — uses an strftime template in
`remote_directory`,
which requires an explicit `--run-time` and therefore cannot be combined with
`--scheduled`.

```{note}
A first-class `--backfill FROM..TO` would be cheap, since idempotency and
issue-time keying already exist. It is simply not built.
```
