# Ensembles and several variables

A prediction is shaped `(members, variables, rows, columns)`. Both leading
axes are length one for an ordinary deterministic model, and they are never
squeezed in shared code — which is what makes using them a change to your
model alone.

## Several output variables

Declare how many a lead carries, then yield them:

```python
class MyPredictor:
    output_variables = 2            # a rate, and an exceedance probability

    def leads(self, inputs, steps):
        for lead in range(steps):
            rate = self._rate(inputs, lead)            # (rows, columns)
            probability = self._probability(inputs, lead)
            yield np.stack([rate, probability])[np.newaxis]   # (1, 2, rows, cols)
```

```{important}
`output_variables` is **declared**, not inferred from the inputs. A model may
consume one field and emit several, which the pipeline previously rejected
because it checked the output width against the input's.
```

Name the bands so the files are self-describing:

```python
sink = GeoTiffSink(variable_names=("rate", "probability"))
```

Each lead becomes one GeoTIFF with one band per variable:

```text
driver      GTiff
bands       2 ('float32', 'float32')
band name   ('rate', 'probability')
units       ('mm/h', 'mm/h')
```

```{warning}
Band **units** come from the deployment, so every variable in one file is
labelled the same. A rate in mm/h beside a dimensionless probability is
mislabelled. Until per-variable units exist, publish them through separate
sinks, or accept the label and document it.
```

## Ensemble members

```python
class MyEnsemble:
    output_variables = 1

    def leads(self, inputs, steps):
        for lead in range(steps):
            members = [self._member(inputs, lead, seed) for seed in self._seeds]
            yield np.stack(members)          # (members, 1, rows, columns)
```

The runtime writes one file per member per lead, and the sink adds a member
component to the name:

```text
spread_20000102T1200Z_f001_m000_20000102T1300Z.tif
spread_20000102T1200Z_f001_m001_20000102T1300Z.tif
```

```{note}
A deterministic run — one member — **omits** the component entirely, so
single-member names read exactly as they did before members were admitted.
Conformance checks both that members do not collide and that one member stays
unadorned.
```

The run manifest records the member on every output:

```json
{"name": "...", "lead_hours": 1, "member": 1, "valid_time": "...", "sha256": "..."}
```

## Quantiles

Quantiles are members with meaning. Nothing in the contract distinguishes
them, so record the mapping yourself:

```python
QUANTILES = (0.1, 0.5, 0.9)

PLUGIN = ModelPlugin(
    ...,
    library_versions=lambda: {"quantiles": ",".join(str(q) for q in QUANTILES)},
)
```

That lands in the manifest's `versions` block, which is a slightly awkward
home for it. A dedicated field would be better and does not exist yet.

## The mask

The mask is reduced across the input variable axis and broadcast to whatever
shape you emit, so a cell missing in any input at any time is masked in every
member and every variable.

You do not construct it. `leads` yields bare arrays; the runtime attaches the
mask, the lead number and the valid time. That is deliberate — those three
cannot then disagree with each other.

## Cost

Members multiply output files and bytes linearly. Four leads with sixteen
members is sixty-four files per run, each with its own checksum in the
manifest. Check that your consumers and your retention policy expect that
before turning it on.

## A worked example

`packages/nowcast-runtime/tests/test_reference_deployment.py` is a complete
deployment with two members and two output variables, from one input variable,
on a ten-minute step with a non-uniform window. It is the shortest full
example of these features, and it runs on every build.
