# Proving conformance

Types describe shape. They cannot say that a transform round-trips, that a
stage leaves a frame on its own grid, that nothing partial is ever published,
or that a completed run is recognised rather than redone. The conformance
suite checks those.

## Wiring it up

```python
from nowcast_runtime.testing import ConformanceSuite

from myunet_nowcast.entrypoint import compose, parser


class TestMyUNet(ConformanceSuite):
    product = "my_product"

    def build(self, tmp_path):
        args = parser().parse_args([
            "--config", str(write_config(tmp_path)),
            "--source", "local",
            "--input-dir", str(make_empty_inputs(tmp_path)),
        ])
        deployment, product = compose(args)
        return deployment
```

One hook. `build` returns a `Deployment` whose source is a `LocalFiles`
pointing at an **empty** directory; the suite fills it.

```{important}
It runs your **real composition root**, not a stand-in. What passes here is
the wiring your container uses, which is exactly where a model tends to get
things wrong.
```

A generated project arrives with this already written.

## Where the data comes from

The suite synthesizes its own observations through the deployment's decoder:

```python
decoder.synthesize(directory / decoder.filename(observed), values, grid)
```

`synthesize` is the exact inverse of `read`, so a built-in decoder gets
end-to-end conformance with **no real data at all**. A custom decoder without
those two optional methods still gets every contract-level check; the
end-to-end ones skip with a stated reason.

To use more plausible values, override one method:

```python
    def field(self, index, shape):
        values = super().field(index, shape)
        values[0, 0] = -99.0        # the provider sentinel
        return values
```

## What it checks

**Declarations**

- the plugin's `requires_runtime` is satisfied by the installed runtime
- every declared bundle role resolves to a real file
- the model was loaded for the deployment's grid
- the requested grid fits inside the decoder's native grid
- the window ends at the run time and looks only backwards
- horizon, output width and step are positive
- declared units agree with the decoder's

**Semantics**

- `inverse(forward(x)) ≈ x`, over values **after the input stages**, because
  that is what a transform really receives
- every preprocess step preserves grid, time and shape
- every postprocess step preserves lead, valid time and shape
- every `describe()` is JSON serializable, so a manifest can hold it
- the decoder returns `None` for a name that is not its own
- sink names contain no path separator, and members do not collide

**End to end**

- a full run publishes, and the manifest is complete and serializable
- valid times follow the declared step, not an assumed hour
- a shorter horizon is honoured and a longer one refused
- a completed run is recognised; running again raises `FileExistsError`
- an incomplete window raises `DataNotReady` rather than narrowing
- **a failed write leaves nothing behind at all**

## What it cannot check

```{warning}
Conformance checks the contract. It will not tell you whether your model is
correct, and it will never tell you whether it has any skill.

Two tests remain yours: a **numerical** test comparing a full run against the
same computation done directly, and a written record of the assumptions you
have **not** verified.
```

The numerical one earns its keep. In this repository it caught a loader
returning a module in training mode, where BatchNorm used batch statistics and
the output changed completely while nothing raised.

## Cost

Each test builds a deployment and loads the checkpoint, so a model suite takes
noticeably longer than a unit test — roughly 70 seconds for InceptionV4
against about 20 before. That is the price of running the real composition
root rather than a fake, and it is worth paying. Use `-k` while iterating.

## The reference deployment

`packages/nowcast-runtime/tests/test_reference_deployment.py` is a deployment
deliberately unlike any real one: no transform, no stages, a ten-minute step, a
non-uniform window, two members and two output variables from one input
variable, a single bundle role.

It exists because every one of those paths was admitted by the contracts and
never actually run. Writing it found two defects immediately: the pipeline
validated output width against the **input** variable count, and the file name
contract could not express a sub-hourly observation at all.

```{admonition} It is synthetic, and that matters
:class: important

A synthetic deployment proves the contracts are implementable by something
other than the first model. It does **not** prove they are sufficient for a
second real one. The two-model rule stands until a real second model runs with
no change to the shared runtime.
```
