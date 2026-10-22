"""Canadian Forest Fire Weather Index (FWI) System, the operational baseline.

Van Wagner (1987), "Development and structure of the Canadian Forest Fire
Weather Index System". The FWI is the standard fire-danger rating that India's
and many other agencies' practice descends from; it is a fixed formula with no
learned parameters, so it is the baseline every learned stage must beat to
justify itself (AUDIT.md: "if a 1970s index beats the model, that must be
known").

Everything here is vectorised over a leading "point" axis (grid cells or
ERA5 points); only the day axis is a Python loop, because each day's moisture
codes depend on the previous day's.

Inputs are local-noon observations: temperature (C), relative humidity (%),
10 m wind (km/h), and the 24 h rain (mm) up to that noon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wfpin.ingest.weather import (
    _deaccumulate_precip,
    _nearest_valid_points,
    _open_hourly,
    _relative_humidity,
)

# Startup values for the three moisture codes (Van Wagner & Pickett 1985).
FFMC0, DMC0, DC0 = 85.0, 6.0, 15.0

# Day-length adjustment for DMC (Le) and DC (Lf) by month, Jan..Dec. These are
# the original 46 N tables; using them at 28-32 N is the standard
# no-latitude-adjustment variant and is stated as such in the methodology.
DAY_LENGTH_DMC = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0])
DAY_LENGTH_DC = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])

OUTPUT_COLUMNS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]


def _ffmc(f0, t, h, w, rain):
    mo = 147.2 * (101.0 - f0) / (59.5 + f0)
    rf = np.maximum(rain - 0.5, 0.0)
    wet = rain > 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
    gain = np.where(wet, gain, 0.0)
    extra = np.where(mo > 150.0, 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), 0.0)
    mo = np.minimum(mo + gain + np.where(wet, extra, 0.0), 250.0)

    ed = (0.942 * h**0.679 + 11.0 * np.exp((h - 100.0) / 10.0)
          + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
    ew = (0.618 * h**0.753 + 10.0 * np.exp((h - 100.0) / 10.0)
          + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
    sw = np.sqrt(w)
    ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * sw * (1.0 - (h / 100.0) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * t)
    k1 = (0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7)
          + 0.0694 * sw * (1.0 - ((100.0 - h) / 100.0) ** 8))
    kw = k1 * 0.581 * np.exp(0.0365 * t)
    drying = ed + (mo - ed) * 10.0 ** (-kd)
    wetting = ew - (ew - mo) * 10.0 ** (-kw)
    m = np.where(mo > ed, drying, np.where(mo < ew, wetting, mo))
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(p0, t, h, rain, month):
    p0 = np.asarray(p0, dtype=float)
    wet = rain > 1.5
    re = 0.92 * rain - 1.27
    mo = 20.0 + np.exp(5.6348 - p0 / 43.43)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(p0 <= 33.0, 100.0 / (0.5 + 0.3 * p0),
                     np.where(p0 <= 65.0, 14.0 - 1.3 * np.log(p0), 6.2 * np.log(p0) - 17.2))
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(mr - 20.0)
    p0 = np.where(wet, np.maximum(pr, 0.0), p0)
    k = np.where(t > -1.1, 1.894 * (t + 1.1) * (100.0 - h) * DAY_LENGTH_DMC[month - 1] * 1e-6, 0.0)
    return p0 + 100.0 * k


def _dc(d0, t, rain, month):
    wet = rain > 2.8
    rd = 0.83 * rain - 1.27
    qo = 800.0 * np.exp(-d0 / 400.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = 400.0 * np.log(800.0 / (qo + 3.937 * rd))
    d0 = np.where(wet, np.maximum(dr, 0.0), d0)
    v = np.maximum(0.36 * (np.maximum(t, -2.8) + 2.8) + DAY_LENGTH_DC[month - 1], 0.0)
    return d0 + 0.5 * v


def _indices(ffmc, dmc, dc, w):
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    f_w = np.exp(0.05039 * w)
    f_f = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / 4.93e7)
    isi = 0.208 * f_w * f_f
    with np.errstate(divide="ignore", invalid="ignore"):
        bui = np.where(dmc <= 0.4 * dc, 0.8 * dmc * dc / (dmc + 0.4 * dc),
                       dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc))
                       * (0.92 + (0.0114 * dmc) ** 1.7))
    bui = np.where(np.isnan(bui), 0.0, np.maximum(bui, 0.0))
    f_d = np.where(bui <= 80.0, 0.626 * bui**0.809 + 2.0,
                   1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * f_d
    with np.errstate(divide="ignore", invalid="ignore"):
        fwi = np.where(b > 1.0, np.exp(2.72 * (0.434 * np.log(b)) ** 0.647), b)
    return isi, bui, fwi


def step(prev, t, rh, wind_kmh, rain, month) -> dict[str, np.ndarray]:
    """One day. `prev` = (ffmc, dmc, dc) from the previous day. Returns all
    six outputs; the three codes feed the next call."""
    t, h, w, rain = (np.asarray(x, dtype=float) for x in (t, rh, wind_kmh, rain))
    h = np.clip(h, 0.0, 100.0)
    ffmc = _ffmc(np.asarray(prev[0], dtype=float), t, h, w, rain)
    dmc = _dmc(prev[1], t, h, rain, month)
    dc = _dc(np.asarray(prev[2], dtype=float), t, rain, month)
    isi, bui, fwi = _indices(ffmc, dmc, dc, w)
    return {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}


def run_series(
    dates: np.ndarray,
    t: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    rain: np.ndarray,
    restart: tuple[int, int] = (1, 1),
) -> dict[str, np.ndarray]:
    """FWI for every point and day. Arrays are (points, days); `dates` is
    (days,) datetime64. The moisture codes reset to the startup values on
    each year's `restart` (month, day) -- the seasonal restart that stands in
    for snow-melt on a winter-cold, high-altitude state."""
    n_pts, n_days = t.shape
    out = {c: np.full((n_pts, n_days), np.nan) for c in OUTPUT_COLUMNS}
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    prev = (np.full(n_pts, FFMC0), np.full(n_pts, DMC0), np.full(n_pts, DC0))
    for i in range(n_days):
        if (months[i], days[i]) == restart:
            prev = (np.full(n_pts, FFMC0), np.full(n_pts, DMC0), np.full(n_pts, DC0))
        today = step(prev, t[:, i], rh[:, i], wind_kmh[:, i], rain[:, i], months[i])
        known = np.isfinite(t[:, i] + rh[:, i] + wind_kmh[:, i] + rain[:, i])
        for c in OUTPUT_COLUMNS:
            out[c][known, i] = today[c][known]
        # A day with missing inputs yields NaN output, but must not poison the
        # moisture codes for the rest of the season: carry yesterday's forward.
        prev = tuple(
            np.where(known, today[c], prev[j]) for j, c in enumerate(("ffmc", "dmc", "dc"))
        )
    return out


# --- from hourly ERA5-Land to a (cell_id, date) table ----------------------------------

NOON_UTC_HOURS = (6, 7)  # 11:30 and 12:30 IST bracket local noon (UTC+5:30)


def noon_inputs(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Daily local-noon FWI inputs from the hourly ERA5-Land dataset.

    Temperature, RH and wind are the mean of the 06 and 07 UTC hours (the two
    hourly steps either side of 12:00 IST); rain is the 24 hours of
    precipitation ending at the 06 UTC step, i.e. noon-to-noon. A day whose
    24 h window has any missing hour gets NaN rain, never a partial sum.
    """
    at_noon = ds["time"].dt.hour.isin(NOON_UTC_HOURS)
    t_c = ds["t2m"] - 273.15
    obs = {
        "t": t_c,
        "rh": _relative_humidity(ds["t2m"], ds["d2m"]),
        "wind_kmh": np.hypot(ds["u10"], ds["v10"]) * 3.6,
    }
    daily = {k: v.where(at_noon, drop=True).resample(time="1D").mean() for k, v in obs.items()}
    rain24 = _deaccumulate_precip(ds["tp"]).rolling(time=24, min_periods=24).sum()
    at_06 = ds["time"].dt.hour == NOON_UTC_HOURS[0]
    daily["rain"] = rain24.where(at_06, drop=True).resample(time="1D").mean()
    return daily


def build_fwi_daily(
    config: dict,
    raw_dir: str = "data/raw/weather",
    static_path: str = "data/processed/static_layers.parquet",
    out_path: str = "data/processed/fwi_daily.parquet",
) -> Path:
    """FWI System per cell-day -> fwi_daily.parquet (cell_id, date, ffmc, dmc,
    dc, isi, bui, fwi). The index is computed once per ERA5 point and sampled
    onto cells exactly as the weather features are."""
    ds = _open_hourly(raw_dir)
    inputs = {k: v.load() for k, v in noon_inputs(ds).items()}
    days = pd.date_range(
        str(inputs["t"]["time"].min().values)[:10], str(inputs["t"]["time"].max().values)[:10]
    )
    n_lat, n_lon = ds.sizes["latitude"], ds.sizes["longitude"]
    flat = {
        k: v.reindex(time=days).transpose("latitude", "longitude", "time").values
        .reshape(n_lat * n_lon, len(days))
        for k, v in inputs.items()
    }
    restart = tuple(config.get("fwi", {}).get("restart", [1, 1]))
    result = run_series(days.to_numpy().astype("datetime64[D]"), flat["t"], flat["rh"],
                        flat["wind_kmh"], flat["rain"], restart)

    grid = pd.read_parquet(static_path, columns=["cell_id", "centroid_lat", "centroid_lon"])
    lat_idx, lon_idx = _nearest_valid_points(
        ds, grid["centroid_lat"].to_numpy(), grid["centroid_lon"].to_numpy()
    )
    point = lat_idx * n_lon + lon_idx  # row-major (latitude, longitude)
    frames = []
    for j, cell in enumerate(grid["cell_id"].to_numpy()):
        frames.append(pd.DataFrame({
            "cell_id": np.int32(cell), "date": days,
            **{c: result[c][point[j]].astype("float32") for c in OUTPUT_COLUMNS},
        }))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["fwi"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df['cell_id'].nunique()} cells", flush=True)
    return out


if __name__ == "__main__":
    from wfpin.config import cli_config

    build_fwi_daily(cli_config("FWI System baseline from ERA5-Land noon observations"))
