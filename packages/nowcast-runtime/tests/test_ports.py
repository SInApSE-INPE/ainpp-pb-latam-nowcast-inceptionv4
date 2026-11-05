"""The contracts carry real invariants, so they are checked here, not downstream."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from nowcast_runtime.ports import (
    Frame,
    Grid,
    ModelPlugin,
    Postprocess,
    Prediction,
    Predictor,
    Preprocess,
    WindowSpec,
)

# The full GSMaP binary grid and the operational LATAM domain in config/latam.json.
GSMAP = Grid("EPSG:4326", -180.0, 60.0, 0.1, 0.1, 3600, 1200)
LATAM = Grid("EPSG:4326", -120.0, 33.0, 0.1, 0.1, 970, 880)


def test_grid_geometry_matches_the_operational_domain():
    assert LATAM.shape == (880, 970)
    assert LATAM.east == pytest.approx(-23.0)
    assert LATAM.south == pytest.approx(-55.0)


def test_locate_reproduces_the_previously_hardcoded_offsets():
    # read_gsmap computed row0 = (60 - north)/res and col0 = (west + 180)/res.
    assert GSMAP.locate(LATAM) == (270, 600)
    assert GSMAP.contains(LATAM)
    assert GSMAP.locate(GSMAP) == (0, 0)


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        (Grid("EPSG:3857", -120.0, 33.0, 0.1, 0.1, 970, 880), "CRS differs"),
        (Grid("EPSG:4326", -120.0, 33.0, 0.2, 0.2, 485, 440), "resolution differs"),
        (Grid("EPSG:4326", -120.05, 33.0, 0.1, 0.1, 970, 880), "must align"),
        (Grid("EPSG:4326", -120.0, 70.0, 0.1, 0.1, 970, 880), "starts outside"),
        (Grid("EPSG:4326", -120.0, 33.0, 0.1, 0.1, 970, 1100), "extends beyond"),
    ],
)
def test_locate_rejects_grids_that_are_not_aligned_subgrids(grid, message):
    with pytest.raises(ValueError, match=message):
        GSMAP.locate(grid)
    assert not GSMAP.contains(grid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"crs": ""}, "requires a CRS"),
        ({"west": float("nan")}, "must be finite"),
        ({"resolution_x": 0.0}, "resolution must be positive"),
        ({"width": 0}, "must be positive"),
        ({"height": 1.5}, "must be integers"),
    ],
)
def test_grid_rejects_invalid_geometry(kwargs, message):
    base = {
        "crs": "EPSG:4326",
        "west": -120.0,
        "north": 33.0,
        "resolution_x": 0.1,
        "resolution_y": 0.1,
        "width": 970,
        "height": 880,
    }
    with pytest.raises(ValueError, match=message):
        Grid(**{**base, **kwargs})


def test_uniform_window_ends_at_the_run_time_and_reads_backwards():
    window = WindowSpec.uniform(timedelta(hours=1), 12)
    assert len(window) == 12
    assert window.offsets[0] == timedelta(hours=-11)
    assert window.offsets[-1] == timedelta(0)
    run_time = datetime(2000, 1, 2, tzinfo=UTC)
    times = window.times(run_time)
    assert times[-1] == run_time
    assert times[0] == datetime(2000, 1, 1, 13, tzinfo=UTC)


def test_window_admits_a_non_uniform_history():
    window = WindowSpec((timedelta(hours=-6), timedelta(hours=-1), timedelta(0)))
    assert len(window) == 3
    assert window.times(datetime(2000, 1, 2, tzinfo=UTC))[0].hour == 18


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ((), "at least one offset"),
        ((timedelta(hours=1),), "must not follow"),
        ((timedelta(0), timedelta(hours=-1)), "unique and increasing"),
        ((timedelta(0), timedelta(0)), "unique and increasing"),
        ((timedelta(hours=-1),), "must be the run time itself"),
    ],
)
def test_window_rejects_invalid_offsets(offsets, message):
    with pytest.raises(ValueError, match=message):
        WindowSpec(offsets)


@pytest.mark.parametrize(
    ("step", "count", "message"),
    [(timedelta(0), 12, "step must be positive"), (timedelta(hours=1), 0, "count must be")],
)
def test_uniform_window_rejects_invalid_spacing(step, count, message):
    with pytest.raises(ValueError, match=message):
        WindowSpec.uniform(step, count)


def test_window_times_require_an_aware_run_time():
    with pytest.raises(ValueError, match="timezone aware"):
        WindowSpec.uniform(timedelta(hours=1), 2).times(datetime(2000, 1, 2))


def test_frame_keeps_a_variable_axis_even_for_one_product():
    grid = Grid("EPSG:4326", 0.0, 10.0, 0.1, 0.1, 4, 3)
    frame = Frame(
        np.zeros((1, 3, 4), dtype=np.float32),
        np.zeros((1, 3, 4), dtype=bool),
        "mm/h",
        grid,
        datetime(2000, 1, 1, tzinfo=UTC),
    )
    assert frame.variables == 1
    assert frame.values.shape[1:] == grid.shape


@pytest.mark.parametrize(
    ("values", "missing", "units", "message"),
    [
        (np.zeros((3, 4)), np.zeros((3, 4), bool), "mm/h", "variables, height, width"),
        (np.zeros((1, 9, 9)), np.zeros((1, 9, 9), bool), "mm/h", "do not match the grid"),
        (np.zeros((1, 3, 4)), np.zeros((1, 3, 5), bool), "mm/h", "must match its values"),
        (np.zeros((1, 3, 4)), np.zeros((1, 3, 4)), "mm/h", "must be boolean"),
        (np.zeros((1, 3, 4)), np.zeros((1, 3, 4), bool), "", "explicit physical units"),
    ],
)
def test_frame_rejects_inconsistent_payloads(values, missing, units, message):
    grid = Grid("EPSG:4326", 0.0, 10.0, 0.1, 0.1, 4, 3)
    with pytest.raises(ValueError, match=message):
        Frame(values, missing, units, grid, datetime(2000, 1, 1, tzinfo=UTC))


def test_prediction_keeps_member_and_variable_axes_and_carries_its_mask():
    prediction = Prediction(
        np.zeros((1, 1, 3, 4), dtype=np.float32),
        np.zeros((1, 1, 3, 4), dtype=bool),
        1,
        datetime(2000, 1, 1, tzinfo=UTC),
    )
    assert prediction.members == 1
    assert prediction.values.ndim == 4
    assert prediction.missing.shape == prediction.values.shape


@pytest.mark.parametrize(
    ("values", "missing", "lead", "message"),
    [
        (np.zeros((1, 3, 4)), np.zeros((1, 3, 4), bool), 1, "members, variables, height, width"),
        (np.zeros((1, 1, 3, 4)), np.zeros((1, 1, 3, 5), bool), 1, "mask must match"),
        (np.zeros((1, 1, 3, 4)), np.zeros((1, 1, 3, 4)), 1, "mask must be boolean"),
        (np.zeros((1, 1, 3, 4)), np.zeros((1, 1, 3, 4), bool), 0, "starts at one"),
    ],
)
def test_prediction_rejects_invalid_payloads(values, missing, lead, message):
    with pytest.raises(ValueError, match=message):
        Prediction(values, missing, lead, datetime(2000, 1, 1, tzinfo=UTC))


def plugin(**overrides):
    base = {
        "name": "Example",
        "slug": "example",
        "version": "1.0.0",
        "bundle_roles": ("checkpoint",),
        "requires_runtime": ">=0.2,<0.3",
        "load": lambda artifacts, device, grid: None,
    }
    return ModelPlugin(**{**base, **overrides})


def test_a_plugin_needs_no_transform_and_no_stages():
    minimal = plugin()
    assert minimal.transform is None
    assert minimal.preprocess is None
    assert minimal.postprocess is None
    assert minimal.library_versions() == {}


def test_plugin_declares_its_own_bundle_roles():
    assert plugin(bundle_roles=("weights", "mean", "std")).bundle_roles == (
        "weights",
        "mean",
        "std",
    )
    assert plugin().bundle_roles == ("checkpoint",)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "name, version and runtime"),
        ({"requires_runtime": ""}, "name, version and runtime"),
        ({"slug": "has space"}, "path and filename safe"),
        ({"slug": "../escape"}, "path and filename safe"),
        ({"bundle_roles": ()}, "non-empty and unique"),
        ({"bundle_roles": ("mean", "mean")}, "non-empty and unique"),
    ],
)
def test_plugin_rejects_unsafe_declarations(overrides, message):
    with pytest.raises(ValueError, match=message):
        plugin(**overrides)


def test_predictor_protocol_recognizes_a_minimal_implementation():
    class Direct:
        window = WindowSpec.uniform(timedelta(hours=1), 12)
        step = timedelta(hours=1)
        output_steps = 6
        grid = LATAM

        def leads(self, inputs, steps):
            yield from ()

    assert isinstance(Direct(), Predictor)
    assert not isinstance(object(), Predictor)


def test_stage_protocols_recognize_a_plain_object():
    class FlipLatitude:
        def apply(self, payload):
            return payload

        def describe(self):
            return {"axis": "latitude"}

    assert isinstance(FlipLatitude(), Preprocess)
    assert isinstance(FlipLatitude(), Postprocess)
    assert not isinstance(object(), Preprocess)
