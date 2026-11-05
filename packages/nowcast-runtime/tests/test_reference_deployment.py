"""A second deployment, deliberately unlike the first, through the conformance suite.

InceptionV4 exercises exactly one path: hourly inputs, a log/z-score transform,
three stages, one deterministic output variable. Everything else ADR 0001
committed the contracts to admitting had never been through the pipeline. This
reference deployment is the opposite on every axis that matters:

- no transform at all, so the runtime must supply identity
- no preprocess and no postprocess stages
- a ten-minute step rather than an hour
- a non-uniform input window
- two ensemble members and two output variables from one input variable
- a single bundle role, with no normalization statistics

It is synthetic, so it does not replace porting a second real model. What it
does is hold every admitted-but-unused path under test before a real model
depends on it, and prove the conformance suite passes for something that is not
InceptionV4.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio

from nowcast_runtime.artifacts import sha256, verify_bundle
from nowcast_runtime.builtin.decoders import RawBinaryDecoder, TimestampedName
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import LocalFiles
from nowcast_runtime.composition import Deployment
from nowcast_runtime.ports import Grid, ModelPlugin, WindowSpec
from nowcast_runtime.testing import ConformanceSuite

STEP = timedelta(minutes=10)
NATIVE = Grid("EPSG:4326", -180.0, 60.0, 0.5, 0.5, 720, 240)
GRID = Grid("EPSG:4326", -10.0, 5.0, 0.5, 0.5, 20, 10)
# Deliberately non-uniform: two recent observations and one an hour back.
WINDOW = WindowSpec((-timedelta(hours=1), -STEP, timedelta(0)))
MEMBERS, VARIABLES = 2, 2
PRODUCT = "synth"
RUN_TIME = datetime(2000, 1, 2, 13, 0, tzinfo=UTC)


def decoder() -> RawBinaryDecoder:
    return RawBinaryDecoder(
        native_grid=NATIVE,
        naming=TimestampedName(
            pattern=r"synth\.(?P<stamp>\d{12})\.bin",
            stamp_format="%Y%m%d%H%M",
            template="synth.%Y%m%d%H%M.bin",
        ),
        units="m/s",
        gzipped=False,
        suffix=".bin",
    )


class Spread:
    """Persistence, widened into members and variables, in physical units."""

    window = WINDOW
    step = STEP
    output_steps = 4
    output_variables = VARIABLES
    grid = GRID

    def leads(self, inputs, steps):
        if inputs.shape != (len(WINDOW), 1, *GRID.shape):
            raise ValueError(f"Unexpected input shape {inputs.shape}")
        last = inputs[-1, 0]
        for lead in range(1, steps + 1):
            fields = np.stack(
                [np.stack([last * member + lead] * VARIABLES) for member in (1.0, 2.0)]
            )
            yield fields.astype(np.float32)


PLUGIN = ModelPlugin(
    name="Spread",
    slug="spread",
    version="0.1.0",
    bundle_roles=("weights",),
    requires_runtime=">=0.3,<0.4",
    load=lambda artifacts, device, grid: Spread(),
)


def build_deployment(tmp_path: Path) -> Deployment:
    """A complete deployment with a synthesized bundle and an empty input directory."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    weights = bundle / "weights.npy"
    np.save(weights, np.zeros(1, dtype=np.float32))
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "v1",
                "files": {"weights": {"name": weights.name, "sha256": sha256(weights)}},
            }
        ),
        encoding="utf-8",
    )
    manifest, artifacts = verify_bundle(bundle, PLUGIN.bundle_roles)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    reader = decoder()
    return Deployment(
        plugin=PLUGIN,
        source=LocalFiles(inputs, reader.glob),
        decoder=reader,
        sink=GeoTiffSink(variable_names=("rate", "probability")),
        grid=GRID,
        bundle=manifest,
        artifacts=artifacts,
    )


class TestReferenceDeployment(ConformanceSuite):
    product = PRODUCT
    run_time = RUN_TIME

    def build(self, tmp_path):
        return build_deployment(tmp_path)

    # The checks below are specific to this deployment, not part of the contract.

    def test_a_plugin_without_a_transform_records_identity(self, deployment, populated, outputs):
        target = self.run(deployment, outputs)
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["normalization"] == {"kind": "identity"}
        assert manifest["preprocess"] == [] and manifest["postprocess"] == []

    def test_a_non_uniform_window_is_recorded_as_offsets(self, deployment, populated, outputs):
        target = self.run(deployment, outputs)
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["input_offsets"] == [-3600.0, -600.0, 0.0]
        assert manifest["step_seconds"] == 600
        assert manifest["input_steps"] == 3

    def test_members_and_variables_both_reach_the_output(
        self, deployment, populated, outputs
    ):
        target = self.run(deployment, outputs)
        manifest = json.loads((target / "manifest.json").read_text())
        assert len(manifest["outputs"]) == Spread.output_steps * MEMBERS
        assert {item["member"] for item in manifest["outputs"]} == {0, 1}

        last = populated[RUN_TIME]
        for item in manifest["outputs"]:
            assert f"_m{item['member']:03d}_" in item["name"]
            with rasterio.open(target / item["name"]) as dataset:
                assert dataset.count == VARIABLES
                assert dataset.descriptions == ("rate", "probability")
                assert dataset.units == ("m/s", "m/s")
                assert dataset.xy(0, 0) == pytest.approx((-9.75, 4.75))
                expected = last * (1.0 if item["member"] == 0 else 2.0) + item["lead_hours"]
                np.testing.assert_allclose(dataset.read(1), expected, atol=1e-6)
                np.testing.assert_allclose(dataset.read(2), expected, atol=1e-6)

    def test_an_output_width_other_than_declared_is_refused(self, deployment, populated, outputs):
        from dataclasses import replace

        class Narrow(Spread):
            def leads(self, inputs, steps):
                for _ in range(steps):
                    yield np.zeros((MEMBERS, VARIABLES + 1, *GRID.shape), dtype=np.float32)

        narrowed = replace(
            deployment,
            plugin=replace(PLUGIN, load=lambda artifacts, device, grid: Narrow()),
        )
        with pytest.raises(ValueError, match="for a lead, expected"):
            self.run(narrowed, outputs)
