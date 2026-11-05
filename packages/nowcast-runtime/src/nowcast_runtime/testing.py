"""A conformance suite a model repository subclasses.

Types cannot express what these contracts actually require: that a transform
round trips, that a stage leaves a frame on its own grid, that nothing partial is
ever published, that a completed run is recognized rather than redone. This is
the third layer of enforcement ADR 0003 promised, and the only one that checks
semantics.

Use it from a model's own tests:

    from nowcast_runtime.testing import ConformanceSuite

    class TestMyModel(ConformanceSuite):
        product = "my_product"

        def build(self, tmp_path):
            return compose_for_conformance(tmp_path)

`build` returns a Deployment whose source is a `LocalFiles` pointing at an empty
directory. The suite fills that directory with observations synthesized through
the deployment's own decoder, so no real data is needed. A deployment whose
decoder cannot synthesize still gets every contract-level check; the
end-to-end ones skip with an explicit reason.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from . import __version__
from .acquisition import DataNotReady
from .composition import Deployment
from .ports import Frame, Prediction

# Far enough in the past that no freshness rule applies, and a round number so a
# failure message is readable. A supplied directory is not live, so age is moot.
RUN_TIME = datetime(2000, 1, 2, 12, 0, tzinfo=UTC)


def _json_safe(payload: object, label: str) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label} is not JSON serializable: {exc}") from exc


class ConformanceSuite:
    """Contract checks every deployment must pass. Subclass, do not edit."""

    product: str = "product"
    run_time: datetime = RUN_TIME

    # ------------------------------------------------------------------ hooks

    def build(self, tmp_path: Path) -> Deployment:
        raise NotImplementedError("A conformance suite must implement build(tmp_path)")

    def field(self, index: int, shape: tuple[int, int]) -> np.ndarray:
        """One synthetic input field. Override for a model needing plausible values."""
        rows = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)[:, None]
        columns = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)[None, :]
        return (rows + columns + index).astype(np.float32)

    # -------------------------------------------------------------- fixtures

    @pytest.fixture
    def deployment(self, tmp_path) -> Deployment:
        return self.build(tmp_path)

    @pytest.fixture
    def loaded(self, deployment):
        return deployment.load()

    @pytest.fixture
    def outputs(self, tmp_path) -> Path:
        return tmp_path / "conformance-outputs"

    @pytest.fixture
    def populated(self, deployment, loaded):
        """Synthesize one full window into the deployment's input directory."""
        decoder, source = deployment.decoder, deployment.source
        if not hasattr(decoder, "synthesize") or not hasattr(decoder, "filename"):
            pytest.skip("This decoder cannot synthesize observations")
        directory = getattr(source, "directory", None)
        if directory is None:
            pytest.skip("Conformance runs against a local source")
        written = {}
        for index, observed in enumerate(loaded.predictor.window.times(self.run_time)):
            values = self.field(index, deployment.grid.shape)
            target = Path(directory) / decoder.filename(observed)
            decoder.synthesize(target, values, deployment.grid)
            written[observed] = values
        return written

    def run(self, deployment, outputs, *extra):
        from . import pipeline

        args = pipeline.parser(deployment.slug, deployment.plugin.name).parse_args(
            ["--config", "unused.json", "--source", "local", "--output-dir", str(outputs), *extra]
        )
        return pipeline.run(args, deployment, self.product)

    # ------------------------------------------------------- declaration only

    def test_plugin_is_compatible_with_the_installed_runtime(self, deployment):
        specifier = deployment.plugin.requires_runtime
        assert specifier, "A plugin must declare the runtime range it needs"
        lower = specifier.split(",")[0].lstrip(">=~^ ")
        installed = tuple(int(part) for part in __version__.split(".")[:2])
        wanted = tuple(int(part) for part in lower.split(".")[:2])
        assert installed >= wanted, f"Runtime {__version__} is older than {specifier}"

    def test_every_declared_bundle_role_resolves(self, deployment):
        for role in deployment.plugin.bundle_roles:
            assert role in deployment.artifacts, f"Role {role} was declared but not resolved"
            assert deployment.artifacts[role].is_file(), f"Role {role} is not a file"

    def test_the_model_was_loaded_for_the_deployment_grid(self, deployment, loaded):
        assert loaded.predictor.grid == deployment.grid

    def test_the_requested_grid_fits_the_decoder_native_grid(self, deployment):
        assert deployment.decoder.native_grid.contains(deployment.grid)

    def test_the_window_ends_at_the_run_time_and_looks_backwards(self, loaded):
        window = loaded.predictor.window
        assert len(window) >= 1
        assert window.times(RUN_TIME)[-1] == RUN_TIME
        assert all(moment <= RUN_TIME for moment in window.times(RUN_TIME))

    def test_the_horizon_and_output_width_are_positive(self, loaded):
        assert loaded.predictor.output_steps >= 1
        assert loaded.predictor.output_variables >= 1
        assert loaded.predictor.step.total_seconds() > 0

    def test_units_are_declared_and_agree_with_the_decoder(self, deployment):
        assert deployment.units
        assert deployment.units == deployment.decoder.units

    # -------------------------------------------------------------- semantics

    def test_the_transform_round_trips(self, deployment, loaded):
        """Over values as the transform really receives them: after the input stages.

        Testing the raw field instead would be testing the wrong contract. A
        provider sentinel is still raw at that point, and a transform is never
        asked to survive one.
        """
        staged = self.staged_frame(deployment, loaded)
        restored = loaded.transform.inverse(loaded.transform.forward(staged.values))
        np.testing.assert_allclose(restored, staged.values, rtol=1e-4, atol=1e-4)

    def test_the_transform_describes_itself_for_the_manifest(self, loaded):
        _json_safe(dict(loaded.transform.describe()), "Transform.describe()")

    def raw_frame(self, deployment) -> Frame:
        values = self.field(0, deployment.grid.shape)[np.newaxis]
        return Frame(
            values,
            np.zeros_like(values, dtype=bool),
            deployment.units,
            deployment.grid,
            self.run_time,
        )

    def staged_frame(self, deployment, loaded) -> Frame:
        frame = self.raw_frame(deployment)
        for stage in deployment.input_stages(loaded.preprocess):
            frame = stage.apply(frame)
        return frame

    def test_every_stage_preserves_what_it_must(self, deployment, loaded):
        grid = deployment.grid
        frame = self.raw_frame(deployment)
        values = frame.values
        for stage in deployment.input_stages(loaded.preprocess):
            result = stage.apply(frame)
            assert result.grid == grid, f"{stage} changed the grid"
            assert result.time == frame.time, f"{stage} changed the observation time"
            assert result.values.shape == values.shape, f"{stage} changed the shape"
            _json_safe(dict(stage.describe()), f"{stage}.describe()")

        width = loaded.predictor.output_variables
        field = np.abs(np.broadcast_to(values[0], (1, width, *grid.shape))).copy()
        prediction = Prediction(
            field, np.zeros_like(field, dtype=bool), 1, self.run_time + loaded.predictor.step
        )
        for stage in deployment.output_stages(loaded.postprocess):
            result = stage.apply(prediction)
            assert result.lead == prediction.lead, f"{stage} changed the lead"
            assert result.valid_time == prediction.valid_time, f"{stage} changed the valid time"
            assert result.values.shape == field.shape, f"{stage} changed the shape"
            _json_safe(dict(stage.describe()), f"{stage}.describe()")

    def test_the_decoder_ignores_a_name_that_is_not_its_own(self, deployment):
        assert deployment.decoder.observed_at("not-an-observation.xyz") is None

    def test_the_sink_names_members_only_when_there_are_several(self, deployment, loaded):
        width = loaded.predictor.output_variables
        shape = (1, width, *deployment.grid.shape)
        single = Prediction(
            np.zeros(shape, np.float32), np.zeros(shape, bool), 1, self.run_time
        )
        name = deployment.sink.filename(deployment.slug, self.run_time, single, 0)
        assert deployment.slug in name
        assert name.endswith(deployment.sink.extension)
        assert "/" not in name and "\\" not in name, "A sink name must not contain a separator"

        many = (2, width, *deployment.grid.shape)
        ensemble = Prediction(
            np.zeros(many, np.float32), np.zeros(many, bool), 1, self.run_time
        )
        first = deployment.sink.filename(deployment.slug, self.run_time, ensemble, 0)
        second = deployment.sink.filename(deployment.slug, self.run_time, ensemble, 1)
        assert first != second, "Members must not collide on one name"

    # ------------------------------------------------------------ end to end

    def test_a_full_run_publishes_a_complete_and_serializable_manifest(
        self, deployment, loaded, populated, outputs
    ):
        target = self.run(deployment, outputs)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        _json_safe(manifest, "The run manifest")
        predictor = loaded.predictor
        assert manifest["status"] == "complete"
        assert manifest["model"] == deployment.plugin.name
        assert manifest["run_time"] == self.run_time.isoformat()
        assert manifest["output_steps"] == predictor.output_steps
        assert manifest["step_seconds"] == predictor.step.total_seconds()
        assert len(manifest["inputs"]) == len(predictor.window)
        for item in manifest["outputs"]:
            path = target / item["name"]
            assert path.is_file(), f"{item['name']} was recorded but not written"
        assert len(manifest["outputs"]) % predictor.output_steps == 0

    def test_valid_times_follow_the_declared_step(self, deployment, loaded, populated, outputs):
        target = self.run(deployment, outputs)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        step = loaded.predictor.step
        for item in manifest["outputs"]:
            expected = self.run_time + item["lead_hours"] * step
            assert item["valid_time"] == expected.isoformat()

    def test_a_shorter_horizon_is_honoured(self, deployment, loaded, populated, outputs):
        if loaded.predictor.output_steps < 2:
            pytest.skip("A single-lead model has no shorter horizon")
        target = self.run(deployment, outputs, "--steps", "1")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["output_steps"] == 1
        assert {item["lead_hours"] for item in manifest["outputs"]} == {1}

    def test_a_longer_horizon_is_refused(self, deployment, loaded, populated, outputs):
        beyond = str(loaded.predictor.output_steps + 1)
        with pytest.raises(ValueError, match="exceeds native horizon"):
            self.run(deployment, outputs, "--steps", beyond)

    def test_a_completed_run_is_recognized_and_never_overwritten(
        self, deployment, populated, outputs
    ):
        from .scheduler import completed_run

        target = self.run(deployment, outputs)
        assert completed_run(target, self.run_time, self.product, None, deployment.plugin.name)
        with pytest.raises(FileExistsError):
            self.run(deployment, outputs)

    def test_an_incomplete_window_waits_rather_than_narrowing(
        self, deployment, loaded, populated, outputs
    ):
        if len(loaded.predictor.window) < 2:
            pytest.skip("A single-observation window cannot be incomplete")
        directory = Path(deployment.source.directory)
        oldest = sorted(directory.iterdir())[0]
        oldest.unlink()
        with pytest.raises(DataNotReady, match="Incomplete window"):
            self.run(deployment, outputs)

    def test_nothing_is_published_when_a_write_fails(
        self, deployment, populated, outputs, monkeypatch
    ):
        def fail(self, path, prediction, member, grid, units, tags):
            raise OSError("conformance: simulated write failure")

        monkeypatch.setattr(type(deployment.sink), "write", fail)
        with pytest.raises(OSError, match="simulated write failure"):
            self.run(deployment, outputs)
        assert not list(outputs.rglob("manifest.json")), "A failed run left a manifest behind"
        leftovers = [path for path in outputs.rglob("*") if path.is_file()]
        assert not leftovers, f"A failed run left files behind: {leftovers}"
