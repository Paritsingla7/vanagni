"""ERA5-Land precip de-accumulation (AUDIT.md: daily-max precip leaked
yesterday's rain; a naive first-difference drops the 00-01 UTC hour)."""
import numpy as np
import xarray as xr

from wfpin.ingest.weather import _deaccumulate_precip


def _accumulated(times, hourly_mm):
    """Build ERA5-Land-style `tp` (m): HH = total since 00 UTC that day,
    00 UTC = full total of the previous day."""
    acc, running = [], 0.0
    for t, mm in zip(times, hourly_mm, strict=True):
        running = mm if int(str(t)[11:13]) == 1 else running + mm
        acc.append(running / 1000)
    return xr.DataArray(acc, coords={"time": times}, dims="time")


def test_recovers_every_hour_including_01_utc():
    times = np.arange(
        np.datetime64("2020-01-01T01"), np.datetime64("2020-01-04T01"), np.timedelta64(1, "h")
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 3, len(times))
    out = _deaccumulate_precip(_accumulated(times, truth))
    np.testing.assert_allclose(out.values, truth, atol=1e-9)


def test_gap_is_nan_not_zero():
    times = np.array(
        ["2020-01-01T01", "2020-01-01T02", "2020-01-01T05", "2020-01-01T06"],
        dtype="datetime64[ns]",
    )
    tp = xr.DataArray([0.001, 0.002, 0.004, 0.005], coords={"time": times}, dims="time")
    out = _deaccumulate_precip(tp).values
    assert np.isnan(out[2])  # 05 UTC follows a gap: unknown, not "no rain"
    np.testing.assert_allclose(out[[0, 1, 3]], [1.0, 1.0, 1.0])
