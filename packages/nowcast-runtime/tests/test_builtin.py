"""The built-in adapters are what make most adoption configuration, so they are
tested on their own rather than only through a model."""

import gzip
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from nowcast_runtime.builtin.decoders import GSMAP_GRID, RawBinaryDecoder, TimestampedName, gsmap
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import LocalFiles, RetryPolicy
from nowcast_runtime.builtin.stages import ClipNegative, FlipLatitude, MissingPolicy
from nowcast_runtime.builtin.transforms import Identity, LogZScore
from nowcast_runtime.config import Bounds
from nowcast_runtime.ports import Frame, Grid, Prediction, WindowSpec

SMALL = Grid("EPSG:4326", 0.0, 1.0, 0.1, 0.1, 4, 3)


def frame(values, missing=None, units="mm/h"):
    values = np.asarray(values, dtype=np.float32).reshape(1, 3, 4)
    mask = np.zeros_like(values, dtype=bool) if missing is None else np.asarray(missing).reshape(
        1, 3, 4
    )
    return Frame(values, mask, units, SMALL, datetime(2000, 1, 1, tzinfo=UTC))


# --- naming -----------------------------------------------------------------


def test_timestamped_name_reads_the_stamp_and_verifies_an_interval():
    naming = TimestampedName(
        pattern=r"p\.(?P<stamp>\d{8}\.\d{4})(?:_(?P<end>\d{4}))?\.bin",
        stamp_format="%Y%m%d.%H%M",
        template="p.%Y%m%d.%H%M.bin",
        interval=timedelta(hours=1),
        end_format="%H%M",
    )
    assert naming.observed_at("p.20000102.1300.bin") == datetime(2000, 1, 2, 13, tzinfo=UTC)
    assert naming.observed_at("p.20000102.1300_1400.bin") == datetime(2000, 1, 2, 13, tzinfo=UTC)
    assert naming.observed_at("other.bin") is None
    assert naming.filename(datetime(2000, 1, 2, 13, tzinfo=UTC)) == "p.20000102.1300.bin"
    with pytest.raises(ValueError, match="Unexpected interval"):
        naming.observed_at("p.20000102.1300_1700.bin")


def test_timestamped_name_supports_a_sub_hourly_product():
    """A ten-minute product is a different stamp format, not a different class."""
    naming = TimestampedName(
        pattern=r"radar\.(?P<stamp>\d{12})\.bin",
        stamp_format="%Y%m%d%H%M",
        template="radar.%Y%m%d%H%M.bin",
    )
    assert naming.observed_at("radar.200001021310.bin") == datetime(
        2000, 1, 2, 13, 10, tzinfo=UTC
    )
    assert naming.filename(datetime(2000, 1, 2, 13, 10, tzinfo=UTC)) == "radar.200001021310.bin"


def test_timestamped_name_requires_consistent_declarations():
    with pytest.raises(ValueError, match="stamp group"):
        TimestampedName(pattern=r"p\.(?P<date>\d{8})\.bin", stamp_format="%Y%m%d")
    with pytest.raises(ValueError, match="declared together"):
        TimestampedName(
            pattern=r"p\.(?P<stamp>\d{8})\.bin",
            stamp_format="%Y%m%d",
            interval=timedelta(hours=1),
        )
    with pytest.raises(ValueError, match="end group"):
        TimestampedName(
            pattern=r"p\.(?P<stamp>\d{8})\.bin",
            stamp_format="%Y%m%d",
            interval=timedelta(hours=1),
            end_format="%H%M",
        )
    naming = TimestampedName(
        pattern=r"p\.(?P<stamp>\d{8})\.bin", stamp_format="%Y%m%d", template="q.%Y%m%d.bin"
    )
    with pytest.raises(ValueError, match="the pattern rejects"):
        naming.filename(datetime(2000, 1, 2, tzinfo=UTC))


# --- decoding ---------------------------------------------------------------


def test_gsmap_is_expressed_as_configuration_of_the_raw_decoder():
    decoder = gsmap("gsmap_now", GSMAP_GRID)
    assert isinstance(decoder, RawBinaryDecoder)
    assert decoder.units == "mm/h"
    assert decoder.roll_columns == 1800
    assert decoder.native_grid is GSMAP_GRID
    assert decoder.observed_at("gsmap_now.20000102.1300_1400.dat.gz") == datetime(
        2000, 1, 2, 13, tzinfo=UTC
    )
    assert gsmap("gsmap_nrt", GSMAP_GRID).naming.interval is None
    with pytest.raises(ValueError, match="Unknown GSMaP product"):
        gsmap("radar", GSMAP_GRID)


def gsmap_file(directory, stamp, field):
    path = directory / f"gsmap_now.{stamp:%Y%m%d.%H00}.dat.gz"
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write(field.tobytes())
    return path


def test_raw_decoder_rolls_crops_and_reports_non_finite_cells(tmp_path):
    decoder = gsmap("gsmap_now", GSMAP_GRID)
    bounds = Bounds(west=-120.0, south=20.2, east=-107.2, north=33.0)
    grid = bounds.to_grid(GSMAP_GRID)
    assert grid.shape == (128, 128)

    field = np.zeros((1200, 3600), dtype="<f4")
    # Rows start north first; columns are stored half the globe out of phase.
    row, column = GSMAP_GRID.locate(grid)
    patch = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    field[row : row + 128, (column + 1800) % 3600 : (column + 1800) % 3600 + 128] = patch
    field[row, (column + 1800) % 3600] = np.nan
    stamp = datetime(2000, 1, 2, 13, tzinfo=UTC)
    path = gsmap_file(tmp_path, stamp, field)

    decoded = decoder.read(path, grid)
    assert decoded.time == stamp
    assert decoded.units == "mm/h"
    assert decoded.variables == 1
    assert decoded.missing[0, 0, 0] and decoded.missing.sum() == 1
    np.testing.assert_array_equal(decoded.values[0, 1:], patch[1:])


def test_raw_decoder_rejects_truncation_and_foreign_names(tmp_path):
    decoder = gsmap("gsmap_now", GSMAP_GRID)
    grid = Bounds(-120.0, 20.2, -107.2, 33.0).to_grid(GSMAP_GRID)
    short = tmp_path / "gsmap_now.20000102.1300.dat.gz"
    with gzip.open(short, "wb", compresslevel=1) as handle:
        handle.write(b"too short")
    with pytest.raises(ValueError, match="Invalid field size"):
        decoder.read(short, grid)
    foreign = tmp_path / "unrelated.dat.gz"
    foreign.write_bytes(b"")
    with pytest.raises(ValueError, match="Not an observation"):
        decoder.read(foreign, grid)


# --- stages -----------------------------------------------------------------


def test_clip_negative_floors_finite_values_and_keeps_non_finite():
    values = [[-99.0, 1.0, np.nan, -0.5], [0.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]
    result = ClipNegative().apply(frame(values))
    assert result.values[0, 0, 0] == 0.0
    assert result.values[0, 0, 3] == 0.0
    assert np.isnan(result.values[0, 0, 2])
    assert result.values[0, 1, 1] == 2.0


def test_missing_policy_fills_bounds_and_can_refuse():
    values = np.zeros((1, 3, 4), dtype=np.float32)
    missing = np.zeros((1, 3, 4), dtype=bool)
    missing[0, 0, 0] = True
    values[0, 0, 0] = np.nan
    filled = MissingPolicy("zero_rain", 0.5, fill=0.0).apply(frame(values, missing))
    assert filled.values[0, 0, 0] == 0.0
    neutral = MissingPolicy("normalized_zero", 0.5, fill=2.5).apply(frame(values, missing))
    assert neutral.values[0, 0, 0] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="choose an explicit policy"):
        MissingPolicy("error", 0.5).apply(frame(values, missing))
    with pytest.raises(ValueError, match="exceeds the limit"):
        MissingPolicy("zero_rain", 0.0).apply(frame(values, missing))


def test_missing_policy_rejects_an_unknown_name_and_a_bad_limit():
    with pytest.raises(ValueError, match="Unknown missing policy"):
        MissingPolicy("guess", 0.5)
    with pytest.raises(ValueError, match="max_fraction"):
        MissingPolicy("zero_rain", 1.0)


def test_flip_latitude_is_its_own_inverse_for_frames_and_predictions():
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    once = FlipLatitude().apply(frame(values))
    twice = FlipLatitude().apply(once)
    np.testing.assert_array_equal(twice.values, values)
    assert not np.array_equal(once.values, values)

    field = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
    prediction = Prediction(field, np.zeros_like(field, bool), 1, datetime(2000, 1, 1, tzinfo=UTC))
    restored = FlipLatitude().apply(FlipLatitude().apply(prediction))
    np.testing.assert_array_equal(restored.values, field)


# --- transforms -------------------------------------------------------------


def test_identity_is_what_a_model_without_a_transform_receives():
    values = np.array([[0.0, 1.5]], dtype=np.float32)
    np.testing.assert_array_equal(Identity().forward(values), values)
    np.testing.assert_array_equal(Identity().inverse(values), values)
    assert Identity().neutral() == 0.0


def test_log_zscore_round_trips_and_reports_its_neutral_value():
    transform = LogZScore(mean=0.1, std=0.5)
    values = np.array([[0.0, 0.5, 12.0]], dtype=np.float32)
    np.testing.assert_allclose(
        transform.inverse(transform.forward(values)), values, rtol=1e-5, atol=1e-5
    )
    neutral = transform.neutral()
    np.testing.assert_allclose(
        transform.forward(np.array([[neutral]], dtype=np.float32)), 0.0, atol=1e-6
    )


def test_log_zscore_rejects_degenerate_statistics():
    with pytest.raises(ValueError, match="positive finite std"):
        LogZScore(mean=0.1, std=0.0)


# --- sinks ------------------------------------------------------------------


def test_geotiff_names_omit_a_member_component_for_a_deterministic_run():
    sink = GeoTiffSink()
    run_time = datetime(2000, 1, 2, tzinfo=UTC)
    field = np.zeros((1, 1, 3, 4), dtype=np.float32)
    single = Prediction(field, np.zeros_like(field, bool), 3, datetime(2000, 1, 2, 3, tzinfo=UTC))
    assert sink.filename("m", run_time, single, 0) == (
        "m_20000102T0000Z_f003_20000102T0300Z.tif"
    )
    many = np.zeros((2, 1, 3, 4), dtype=np.float32)
    ensemble = Prediction(many, np.zeros_like(many, bool), 3, datetime(2000, 1, 2, 3, tzinfo=UTC))
    assert "_m001_" in sink.filename("m", run_time, ensemble, 1)


def test_geotiff_writes_nodata_where_the_mask_is_set(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    sink = GeoTiffSink()
    field = np.ones((1, 1, 3, 4), dtype=np.float32)
    mask = np.zeros_like(field, dtype=bool)
    mask[0, 0, 2, 3] = True
    prediction = Prediction(field, mask, 1, datetime(2000, 1, 2, 1, tzinfo=UTC))
    path = tmp_path / "out.tif"
    sink.write(path, prediction, 0, SMALL, "mm/h", {"lead_hours": 1})
    with rasterio.open(path) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.units == ("mm/h",)
        assert dataset.nodata == sink.nodata
        assert dataset.read(1)[2, 3] == sink.nodata
        assert dataset.tags()["lead_hours"] == "1"


def test_geotiff_refuses_a_negative_or_non_finite_field(tmp_path):
    sink = GeoTiffSink()
    bad = np.full((1, 1, 3, 4), -1.0, dtype=np.float32)
    prediction = Prediction(bad, np.zeros_like(bad, bool), 1, datetime(2000, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        sink.write(tmp_path / "bad.tif", prediction, 0, SMALL, "mm/h", {})


# --- sources ----------------------------------------------------------------


def test_local_files_reports_bare_names_so_decoders_never_see_a_path(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "gsmap_now.20000102.1300.dat.gz").write_bytes(b"x")
    source = LocalFiles(tmp_path, "*.dat.gz")
    candidates = source.candidates(WindowSpec.uniform(timedelta(hours=1), 1), None)
    assert [name for name, _ in candidates] == ["gsmap_now.20000102.1300.dat.gz"]
    assert source.prune(timedelta(hours=1)) == 0
    assert source.identity.startswith("local:")


def test_local_files_requires_an_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        LocalFiles(tmp_path / "absent")


def test_retry_policy_backs_off_and_is_capped():
    policy = RetryPolicy(60.0, 3, 30.0, 120.0)
    assert [policy.delay(attempt) for attempt in range(4)] == [30.0, 60.0, 120.0, 120.0]
    with pytest.raises(ValueError, match="positive integer"):
        RetryPolicy(60.0, 0, 30.0, 120.0)
    with pytest.raises(ValueError, match="must not be less"):
        RetryPolicy(60.0, 3, 120.0, 30.0)
