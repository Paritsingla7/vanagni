"""ERA5-Land ingestion via cdsapi, resampled to daily per-cell features.

FR-1.2: hourly reanalysis for the configured bbox + date window, resampled
using all 24 hourly steps (not the 5-synoptic-hour subset the old US
pipeline used) and bucketed by local day (UTC+5:30 fixed shift for India),
not UTC day.
"""
from __future__ import annotations

import calendar
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from wfpin.config import cli_config

IST_OFFSET_H = 5.5  # India is a single fixed UTC+5:30 offset, no DST

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
CDS_WORKERS = 4  # concurrent monthly requests; 6 got requests 'rejected' by CDS
# CDS also rejects requests when the *dataset-wide* queue (all users) is full
# ("Maximum number queued requests for this dataset has been reached"), which
# can last hours. Rejections cost nothing, so retry patiently instead of giving up.
CDS_RETRIES = 60
CDS_BACKOFF_S = (120, 900)  # first wait, cap


def _month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """(year, month) pairs spanning [start, end], inclusive, one request per month."""
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    chunks = []
    y, m = start_d.year, start_d.month
    while (y, m) <= (end_d.year, end_d.month):
        chunks.append((str(y), f"{m:02d}"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return chunks


def prioritise_months(
    months: list[tuple[str, str]], priority_months: list[int]
) -> list[tuple[str, str]]:
    """Months in `priority_months` (calendar months, e.g. the fire season)
    first, chronological within each group. When CDS's queue is the
    bottleneck, the months that feed the evaluation should not wait behind
    off-season ones."""
    return sorted(months, key=lambda ym: (int(ym[1]) not in priority_months, ym))


def _download_month(year: str, month: str, area: list[float], out: Path) -> None:
    """One month, retried with backoff. Never raises: CDS rejects or fails
    individual requests (queue limits, transient errors), and one bad month
    must not take the other workers down -- download_weather reports any
    month still missing at the end, and a re-run resumes from the markers."""
    label = f"{year}-{month}"
    if (out / f"{label}.done").exists():
        return
    for attempt in range(1, CDS_RETRIES + 1):
        try:
            _retrieve_month(year, month, area, out)
            return
        except Exception as exc:  # noqa: BLE001 -- any CDS failure is retryable here
            print(f"[{label}] attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(CDS_BACKOFF_S[0] * attempt, CDS_BACKOFF_S[1]))


def _retrieve_month(year: str, month: str, area: list[float], out: Path) -> None:
    label = f"{year}-{month}"
    marker = out / f"{label}.done"
    n_days = calendar.monthrange(int(year), int(month))[1]
    tmp = out / f"{label}.download"
    print(f"[{label}] requesting...", flush=True)
    cdsapi.Client(quiet=True).retrieve(  # one client per thread; not shared
        "reanalysis-era5-land",
        {
            "variable": VARIABLES,
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": ALL_HOURS,
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(tmp),
    )
    # The new CDS can still return a zip (one .nc per step type, e.g. instant
    # vs accumulated) even when "unarchived" is requested.
    if zipfile.is_zipfile(tmp):
        dest = out / label
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest)
        tmp.unlink()
    else:
        tmp.rename(out / f"{label}.nc")
    marker.write_text("ok")
    print(f"[{label}] done", flush=True)


def download_weather(config: dict, out_dir: str = "data/raw/weather") -> Path:
    """FR-1.2: chunked, resumable monthly ERA5-Land pull for the configured
    bbox + window, all 24 hourly steps. Idempotent via a `.done` marker per
    month, the resumable pattern carried over from the earlier US prototype.

    Months are requested concurrently: CDS queue time dominates each request,
    and the link to CDS is throttled per connection, so a serial loop would
    take most of a day for the full window.
    """
    bbox = config["region"]["bbox"]
    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]
    window = config["window"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    months = prioritise_months(
        _month_chunks(window["start"], window["end"]),
        config.get("weather", {}).get("priority_months", []),
    )
    with ThreadPoolExecutor(max_workers=CDS_WORKERS) as pool:
        list(pool.map(lambda ym: _download_month(*ym, area, out), months))
    missing = [f"{y}-{m}" for y, m in months if not (out / f"{y}-{m}.done").exists()]
    if missing:
        raise RuntimeError(f"ERA5-Land months not downloaded: {missing}")
    return out


def _relative_humidity(t2m_k: xr.DataArray, d2m_k: xr.DataArray) -> xr.DataArray:
    """Magnus formula. ERA5-Land does not ship RH directly (DATA_SOURCES.md #2)."""
    t_c, td_c = t2m_k - 273.15, d2m_k - 273.15
    es = 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))
    e = 6.1094 * np.exp(17.625 * td_c / (td_c + 243.04))
    return (100 * e / es).clip(0, 100)  # Magnus ratio can float-error slightly past 100


def _deaccumulate_precip(tp_m: xr.DataArray) -> xr.DataArray:
    """True per-hour precip (mm, hour ending at each timestamp) from
    ERA5-Land's UTC-day-accumulating `tp`.

    ERA5-Land convention: the value at HH UTC (HH = 01..23) is the total
    since 00 UTC that day, and the value at 00 UTC of day D+1 is the *full*
    total of day D (AUDIT.md's finding: a plain `.resample("1D").max()`
    therefore leaks yesterday's rain into today). So:

    - 01 UTC: the accumulation restarted, the value itself is that hour's rain;
    - every other hour, including 00 UTC: difference from the previous hour.

    A plain first-difference everywhere is wrong at 01 UTC (it subtracts the
    previous day's full total, goes negative, and clipping hides the lost
    hour). Hours whose predecessor is missing (series start, a download gap)
    come out NaN, never a silent 0.
    """
    t = tp_m["time"]
    prev = tp_m.shift(time=1)
    contiguous = (t - t.shift(time=1)) == np.timedelta64(1, "h")
    hourly = xr.where(t.dt.hour == 1, tp_m, xr.where(contiguous, tp_m - prev, np.nan))
    # Float round-off in the accumulation can produce tiny negatives; real rain can't.
    return (hourly * 1000).clip(min=0)  # m -> mm


DAILY_COLUMNS = [
    "t2m_mean",
    "t2m_max",
    "rh_mean",
    "rh_min",
    "wind_mean",
    "wind_max",
    "precip_sum",
    "soil_moisture",
]
HOURS_PER_DAY = 24


def _open_hourly(nc_dir: str) -> xr.Dataset:
    nc_files = sorted(Path(nc_dir).glob("**/*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files under {nc_dir}. Run download_weather first.")
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    ds = ds.rename({"valid_time" if "valid_time" in ds.coords else "time": "time"})
    # CDS sometimes includes these even for a single-model, single-experiment
    # request; the old US pipeline hit the same thing (firepulse.load_weather).
    for extra in ("number", "expver"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)
        elif extra in ds.coords:
            ds = ds.reset_coords(extra, drop=True)
    return ds.sortby("time")


def _nearest_valid_points(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> tuple:
    """(lat_idx, lon_idx) of the nearest ERA5-Land point *that has data* for
    each cell centroid. ERA5-Land is land-only; a plain nearest-neighbour
    `.sel` can land on a masked (all-NaN) point and silently null a cell."""
    valid = ds["t2m"].isel(time=0).notnull().values
    lat_i, lon_i = np.nonzero(valid)
    tree = cKDTree(np.column_stack([ds["latitude"].values[lat_i], ds["longitude"].values[lon_i]]))
    _, nearest = tree.query(np.column_stack([lat, lon]))
    return lat_i[nearest], lon_i[nearest]


def resample_to_daily(nc_dir: str, grid: pd.DataFrame) -> pd.DataFrame:
    """FR-1.2: hourly ERA5-Land -> one row per (cell_id, date), local (IST) day.

    Each grid cell takes the nearest valid ERA5-Land point (~9-11 km) --
    multiple 5 km cells share one weather sample, which is correct, not a
    bug: the weather input is coarser than the fire grid by design
    (INDIA.md 2.5), not something a finer regrid can fix.

    Only complete local days are kept: the window's first and last local
    days straddle the UTC download boundary and would otherwise produce
    max/min/sum over a handful of hours.
    """
    ds = _open_hourly(nc_dir)

    rh = _relative_humidity(ds["t2m"], ds["d2m"])
    wind = np.hypot(ds["u10"], ds["v10"])
    precip = _deaccumulate_precip(ds["tp"])
    soil = ds["swvl1"]
    bad_soil = int((soil < 0).sum())
    if bad_soil:
        print(f"soil_moisture: dropping {bad_soil} negative (fill-value) samples", flush=True)
        soil = soil.where(soil >= 0)  # SCHEMA.md: negative swvl1 is a fill-value bug, not data
    t2m_c = ds["t2m"] - 273.15

    local_date = (ds["time"] + np.timedelta64(int(IST_OFFSET_H * 3600), "s")).dt.floor("D")
    local_date = local_date.rename("local_date")

    def _daily(da: xr.DataArray, agg: str) -> xr.DataArray:
        grp = da.groupby(local_date)
        if agg == "sum":
            # A missing hour makes the day's total unknown, not smaller.
            return grp.sum(min_count=HOURS_PER_DAY)
        return getattr(grp, agg)()

    daily = xr.Dataset(
        {
            "t2m_mean": _daily(t2m_c, "mean"),
            "t2m_max": _daily(t2m_c, "max"),
            "rh_mean": _daily(rh, "mean"),
            "rh_min": _daily(rh, "min"),
            "wind_mean": _daily(wind, "mean"),
            "wind_max": _daily(wind, "max"),
            "precip_sum": _daily(precip, "sum"),
            "soil_moisture": _daily(soil, "mean"),
        }
    )
    hours = pd.Series(local_date.values).value_counts()
    full_days = np.sort(hours[hours == HOURS_PER_DAY].index.values)
    dropped = len(hours) - len(full_days)
    if dropped:
        print(f"dropping {dropped} partial local day(s) at the window edges", flush=True)
    daily = daily.sel(local_date=full_days)

    lat_idx, lon_idx = _nearest_valid_points(
        ds, grid["centroid_lat"].to_numpy(), grid["centroid_lon"].to_numpy()
    )
    sampled = daily.isel(
        latitude=xr.DataArray(lat_idx, dims="cell"), longitude=xr.DataArray(lon_idx, dims="cell")
    ).compute()

    df = sampled.drop_vars(["latitude", "longitude"]).to_dataframe().reset_index()
    df["cell_id"] = grid["cell_id"].to_numpy()[df["cell"].to_numpy()].astype("int32")
    df["date"] = pd.to_datetime(df["local_date"]).dt.date
    df[DAILY_COLUMNS] = df[DAILY_COLUMNS].astype("float32")
    df = df[["cell_id", "date", *DAILY_COLUMNS]]
    return df.sort_values(["cell_id", "date"], ignore_index=True)


def build_weather_daily(
    config: dict,
    raw_dir: str = "data/raw/weather",
    static_path: str = "data/processed/static_layers.parquet",
    out_path: str = "data/processed/weather_daily.parquet",
) -> Path:
    """FR-1.2 end to end: download (idempotent), then resample onto the grid
    from static_layers.parquet -- the grid is owned by static_layers.py, so
    weather never re-derives it."""
    download_weather(config, raw_dir)
    grid = pd.read_parquet(static_path, columns=["cell_id", "centroid_lat", "centroid_lon"])
    df = resample_to_daily(raw_dir, grid)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df['cell_id'].nunique()} cells", flush=True)
    return out


if __name__ == "__main__":
    build_weather_daily(cli_config("FR-1.2: ERA5-Land download + daily per-cell resample"))
