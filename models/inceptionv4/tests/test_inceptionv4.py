import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from inceptionv4_nowcast.entrypoint import NAME, PLUGIN, SLUG, parser, run
from inceptionv4_nowcast.model import InceptionV4Predictor, PrecipitationForecaster, load_module
from nowcast_runtime import scheduler
from nowcast_runtime.artifacts import sha256, verify_bundle
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.transforms import LogZScore
from nowcast_runtime.ports import WindowSpec

PROJECT = Path(__file__).resolve().parents[1]
# The supplied training and inference references are not redistributed here.
requires_reference = pytest.mark.skipif(
    not (PROJECT / "reference").is_dir(), reason="reference/ assets are not in the repository"
)


@requires_reference
def test_reference_equivalence_and_input_contract(module):
    spec = importlib.util.spec_from_file_location(
        "reference_inceptionv4", PROJECT / "reference/inceptionv4.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    original = reference.PrecipitationForecaster(
        input_timesteps=module.input_steps, output_timesteps=module.output_steps, pretrained=False
    ).eval()
    original.load_state_dict(module.state_dict(), strict=True)
    inputs = torch.linspace(-1, 1, 12 * 128 * 128).reshape(1, 12, 128, 128)
    with torch.inference_mode():
        expected = original(inputs)
        actual = module(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(module(inputs.unsqueeze(2)), actual, rtol=0, atol=0)
    assert actual.shape == (1, 6, 1, 128, 128)
    assert actual.min() >= 0
    with pytest.raises(ValueError, match="Expected"):
        module(torch.zeros(1, 6, 128, 128))


def test_predictor_declares_its_window_and_uses_one_forward_pass():
    class CountingModule(torch.nn.Module):
        input_steps = 12
        output_steps = 6
        img_size = (2, 3)

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1, 1).expand(1, 6, 1, 2, 3)

    module = CountingModule()
    grid = pytest.importorskip("nowcast_runtime.ports").Grid(
        "EPSG:4326", 0.0, 1.0, 0.1, 0.1, 3, 2
    )
    predictor = InceptionV4Predictor(module, grid, torch.device("cpu"))
    assert predictor.window == WindowSpec.uniform(timedelta(hours=1), 12)
    assert predictor.step == timedelta(hours=1)
    assert predictor.output_steps == 6

    fields = list(predictor.leads(np.zeros((12, 1, 2, 3), dtype=np.float32), 3))
    assert module.calls == 1
    assert [field.shape for field in fields] == [(1, 1, 2, 3)] * 3
    assert [float(field[0, 0, 0, 0]) for field in fields] == [0, 1, 2]
    with pytest.raises(ValueError, match="one variable per timestep"):
        list(predictor.leads(np.zeros((12, 2, 2, 3), dtype=np.float32), 1))


def test_state_loading_rejects_incompatible_weights(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"wrong.weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="supported InceptionV4"):
        load_module(path, (128, 128))
    with pytest.raises(ValueError, match="dimensions >= 128"):
        PrecipitationForecaster(12, 6, (64, 64))


@requires_reference
def test_bundle_contains_only_inference_weights():
    manifest, paths = verify_bundle(PROJECT / "artifacts/v1", PLUGIN.bundle_roles)
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    assert all(torch.is_tensor(v) for v in state.values())
    assert state["encoder.features_0.conv.weight"].shape[1] == 12
    assert state["outc.weight"].shape[0] == 6
    assert manifest["source_checkpoint"]["sha256"] == sha256(PROJECT / "reference/best_model.pth")


def test_bundle_verification_checks_only_declared_roles():
    manifest, paths = verify_bundle(PROJECT / "artifacts/v1", PLUGIN.bundle_roles)
    assert set(paths) == set(PLUGIN.bundle_roles)
    with pytest.raises(ValueError, match="does not declare the role"):
        verify_bundle(PROJECT / "artifacts/v1", ("checkpoint", "orography"))


def args_for(case, *extra):
    return parser().parse_args(
        [
            "--config",
            str(case["config"]),
            "--source",
            "local",
            "--input-dir",
            str(case["inputs"]),
            "--output-dir",
            str(case["outputs"]),
            "--device",
            "cpu",
            "--threads",
            "2",
            *extra,
        ]
    )


def test_end_to_end_native_horizon_geography_and_scheduled_skip(case, module):
    args = args_for(case)
    assert args.steps is None  # The native horizon comes from the weights.
    target = run(args)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["model"] == NAME
    assert manifest["input_steps"] == 12
    assert manifest["output_steps"] == manifest["native_output_steps"] == 6
    assert manifest["masked_output_fraction"] == 0
    assert manifest["step_seconds"] == 3600
    assert manifest["units"] == "mm/h"
    # The stages the run actually applied are recorded, in order.
    assert [stage["kind"] for stage in manifest["preprocess"]] == [
        "clip-negative",
        "missing-policy",
        "flip-latitude",
    ]
    assert [stage["kind"] for stage in manifest["postprocess"]] == ["flip-latitude"]

    transform = LogZScore(
        manifest["normalization"]["mean"], manifest["normalization"]["std"]
    )
    inputs = transform.forward(case["rain"][:, ::-1, :].copy())
    with torch.inference_mode():
        predicted = module(torch.from_numpy(inputs).unsqueeze(0))[0, :, 0].numpy()
    expected = transform.inverse(predicted)[:, ::-1, :]

    assert len(manifest["outputs"]) == 6
    for index, item in enumerate(manifest["outputs"]):
        path = target / item["name"]
        assert item["member"] == 0
        assert "_m" not in item["name"]  # Deterministic runs omit a member component.
        assert sha256(path) == item["sha256"]
        with rasterio.open(path) as dataset:
            assert dataset.crs.to_epsg() == 4326
            assert dataset.xy(0, 0) == pytest.approx((-119.95, 32.95))
            assert dataset.tags()["lead_hours"] == str(index + 1)
            assert dataset.units == ("mm/h",)
            np.testing.assert_allclose(dataset.read(1), expected[index], atol=1e-6)

    assert scheduler.completed_run(target, case["run_time"], "gsmap_now", None, NAME)
    with pytest.raises(ValueError, match="incompatible"):
        scheduler.completed_run(target, case["run_time"], "gsmap_now", 6, "ARAFNO")
    with scheduler.poll_lock(case["outputs"], "gsmap_now", SLUG) as first:
        with scheduler.poll_lock(case["outputs"], "gsmap_now", "arafno") as second:
            assert first and second
    with pytest.raises(FileExistsError):
        run(args)
    with pytest.raises(ValueError, match="exceeds native horizon"):
        run(args_for(case, "--steps", "7"))


def test_missing_hour_and_failed_export_never_publish(case, module, monkeypatch):
    args = args_for(case)
    first = sorted(case["inputs"].glob("*.gz"))[0]
    absent = first.with_suffix(".hidden")
    first.rename(absent)
    with pytest.raises(ValueError, match="Incomplete window"):
        run(args)
    absent.rename(first)

    original = GeoTiffSink.write

    def fail_second(self, path, prediction, member, grid, units, tags):
        if prediction.lead == 2:
            raise OSError("simulated disk error")
        original(self, path, prediction, member, grid, units, tags)

    monkeypatch.setattr(GeoTiffSink, "write", fail_second)
    with pytest.raises(OSError, match="disk error"):
        run(args)
    assert not list(case["outputs"].rglob("*.tif"))
    assert not list(case["outputs"].rglob("manifest.json"))
