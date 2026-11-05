# Processing stages

Preprocessing and postprocessing are ordered, possibly empty sequences of
steps operating in **physical units**.

```{admonition} Absence is the default
:class: tip

A model that consumes physical units and needs no stages declares nothing. The
runtime applies identity and runs no steps. Having no processing is an ordinary
configuration, not a special case to work around.
```

## Why these are ports at all

Not anticipation — three of them already existed as branches inside shared
code before the ports did:

| Was | Is now |
| --- | --- |
| `missing_policy` with three arms | `MissingPolicy`, selected by configuration |
| flipping latitude on input | `FlipLatitude`, contributed by the plugin |
| flipping latitude on output | the same step, at the other end |

Every new model would have added another arm to code that all models shared.
That is the pattern this platform exists to prevent.

## The contract

```python
def apply(self, payload): ...                  # Frame in, Frame out
def describe(self) -> Mapping[str, object]: ...
```

A preprocess step receives a `Frame` and must return one on the **same grid**
at the **same time**. A postprocess step receives a `Prediction` and must
return one with the **same lead** and **valid time**. Conformance checks both.

`describe()` is written into the run manifest, so it must be JSON
serializable. That is how an operator later sees exactly which steps ran:

```json
"preprocess": [
  {"kind": "clip-negative", "floor": 0.0},
  {"kind": "missing-policy", "policy": "normalized_zero",
   "max_fraction": 0.05, "fill": 0.0453},
  {"kind": "flip-latitude"}
],
"postprocess": [{"kind": "flip-latitude"}]
```

## Ordering is fixed, not composed

```text
input:   configured steps, then model steps, then Transform.forward
output:  Transform.inverse, then model steps, then configured steps
```

A configured step therefore always sees physical units **as the provider
published them**. A model step always sees them **as its checkpoint expects**.

Reversing the output order restores the input ordering, so a symmetric pair
such as a latitude flip composes correctly with neither side knowing about the
other.

```{important}
Order is semantically load bearing, so the runtime fixes it rather than letting
a composition choose. Getting it wrong produces output that is subtly wrong
and raises nothing.
```

## Where does a new stage belong?

One rule decides it:

> **What is scientifically coupled to the checkpoint belongs to the plugin.
> What is operationally variable belongs to the configuration.**

Worked through:

| Step | Belongs to | Because |
| --- | --- | --- |
| flipping latitude | the plugin | the orientation the checkpoint was trained on |
| clipping a `-99` sentinel | configuration | a property of the provider, not the model |
| the missing-data policy | configuration | an operator's choice about acceptable gaps |
| log/z-score normalization | the plugin | its statistics belong to one checkpoint |

A plugin contributes stages through optional factories that receive the
verified artifact paths, so a model-side step can be parameterized by its own
bundle:

```python
PLUGIN = ModelPlugin(
    ...,
    preprocess=lambda artifacts: (FlipLatitude(),),
    postprocess=lambda artifacts: (FlipLatitude(),),
)
```

## Built-in stages

### `ClipNegative(floor=0.0)`

Clamps finite negatives to a floor, leaving non-finite cells untouched. A
provider sentinel is a **source** concern, not a format one, which is why it
is a stage rather than something the decoder does.

```{note}
Non-finite cells stay non-finite and remain the mask's business. Clipping does
not decide what is missing.
```

### `MissingPolicy(policy, max_fraction, fill=0.0)`

Bounds how much of a field may be missing and decides what fills the gaps.

| Policy | Behaviour |
| --- | --- |
| `error` | any missing cell is a failure |
| `zero_rain` | gaps become zero in physical units |
| `normalized_zero` | gaps become the value the model reads as neutral |

The fill is an **explicit physical value**, so the step is independent of any
transform. For `normalized_zero`, the composition root resolves it from the
transform, because only the plugin knows what neutral means for its
checkpoint:

```python
fill = transform.neutral() if config.missing_policy == "normalized_zero" else 0.0
```

```{note}
For a log/z-score transform, the neutral value is `expm1(mean)`: a cell filled
with it reads as the training mean. That is not bit-identical to writing zero
into model space; the difference is at float32 resolution and applies only to
cells the mask already marks, whose outputs are published as NoData.
```

### `FlipLatitude()`

Reverses row order, for a model trained south to north. The same step serves
at both ends, and applying it twice is the identity — conformance checks that.

## Transforms

A transform is a stage of its own kind: it moves between physical units and
model space, and it is **optional**.

```python
def forward(self, values): ...    # physical -> model space
def inverse(self, values): ...    # model space -> physical
def describe(self): ...
```

```{warning}
A transform is bound by the plugin and cannot be overridden by configuration.
Its statistics are inseparable from the checkpoint: for this repository's own
model, the inverse has a floor near 0.0453 mm/h purely because of its mean. A
configurable transform would let a mismatched pair run cleanly and produce
plausible nonsense.
```

Built-in: `Identity` (what a model without one receives) and `LogZScore`.

Conformance round-trips a transform over values **after the input stages**,
because that is what it actually receives. Checking the raw field would be
checking the wrong contract — a provider sentinel is still raw at that point,
and no transform is asked to survive one.
