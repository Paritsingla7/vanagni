"""Next-day wildfire risk: leakage-safe panel construction + fire-weather features.

The task: for a ~25 km grid cell, given everything known at the end of day *d*,
predict whether FIRMS detects a fire in that cell on day *d+1*.

Everything here is shared by `scripts/train_forecast.py`, the notebook, and the
tests, so there is exactly one definition of the panel and the features.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Region/window defaults: California + Western US, bounded by the FIRMS archive.
AREA_N, AREA_W, AREA_S, AREA_E = 42, -125, 32, -114
DATE_START, DATE_END = "2018-01-01", "2020-06-30"
CELL_DEG = 0.25
UTC_OFFSET_H = -8  # Pacific standard time; see local_day() below

FIRE_FEATS = ["fire_today", "fire_roll3", "fire_roll7", "fire_roll14",
              "days_since_fire", "frp_mean", "neighbor_roll7"]
WX_FEATS = ["t2m_max", "rh_min", "wind_max", "precip", "soil_moist",
            "t2m_max_3d", "rh_min_3d", "wind_max_3d", "precip_3d",
            "soil_moist_3d", "precip_7d"]
GEO_FEATS = ["cell_lat", "cell_lon", "doy_sin", "doy_cos"]
FEATURES = FIRE_FEATS + WX_FEATS + GEO_FEATS

NO_FIRE_SENTINEL = 400.0  # days_since_fire cap; > the 2.5 yr window, finite for non-tree models


def snap(values: np.ndarray | pd.Series, cell_deg: float = CELL_DEG) -> np.ndarray:
    """Snap coordinates to the lower-left corner of their grid cell."""
    return np.floor(np.asarray(values, dtype=float) / cell_deg) * cell_deg


def _days_since(had_fire: pd.Series) -> np.ndarray:
    """Days since the most recent fire in this cell; 0 on a fire day."""
    arr = had_fire.to_numpy()
    idx = np.arange(len(arr))
    last = np.maximum.accumulate(np.where(arr == 1, idx, -1))
    return np.where(last < 0, NO_FIRE_SENTINEL, np.minimum(idx - last, NO_FIRE_SENTINEL))


def load_detections(csv_path: Path, cell_deg: float = CELL_DEG) -> pd.DataFrame:
    """FIRMS CSV -> per cell-day detection counts, clipped to the region/window."""
    fires = pd.read_csv(csv_path, usecols=["latitude", "longitude", "acq_date", "frp"])
    fires["acq_date"] = pd.to_datetime(fires["acq_date"])
    fires = fires[
        fires["latitude"].between(AREA_S, AREA_N)
        & fires["longitude"].between(AREA_W, AREA_E)
        & fires["acq_date"].between(DATE_START, DATE_END)
    ].copy()
    fires["cell_lat"] = snap(fires["latitude"], cell_deg)
    fires["cell_lon"] = snap(fires["longitude"], cell_deg)
    return (
        fires.groupby(["cell_lat", "cell_lon", "acq_date"])
        .agg(fire_count=("frp", "size"), frp_mean=("frp", "mean"))
        .reset_index()
    )


def build_panel(daily: pd.DataFrame, cells: pd.DataFrame,
                start: str = DATE_START, end: str = DATE_END) -> pd.DataFrame:
    """Dense cell x day panel with the label and all fire-history features.

    `cells` is the full grid to score, not just cells that have burned — a cell
    that has never burned is the one a deployed model most needs to handle, and
    excluding it inflates the base rate.

    Label is day d+1. Features are day-d state and windows ending at day d, so
    nothing from the future reaches the feature matrix.
    """
    all_days = pd.DataFrame({"acq_date": pd.date_range(start, end, freq="D")})
    panel = (
        cells.merge(all_days, how="cross")
        .merge(daily, on=["cell_lat", "cell_lon", "acq_date"], how="left")
    )
    panel[["fire_count", "frp_mean"]] = panel[["fire_count", "frp_mean"]].fillna(0.0)
    panel["fire_today"] = (panel["fire_count"] > 0).astype(np.int8)
    panel = panel.sort_values(["cell_lat", "cell_lon", "acq_date"]).reset_index(drop=True)

    # The panel is dense and sorted per cell, so positional rolling == time rolling.
    g = panel.groupby(["cell_lat", "cell_lon"])["fire_today"]
    panel["fire_next_day"] = g.shift(-1)
    for w in (3, 7, 14):
        panel[f"fire_roll{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
    panel["days_since_fire"] = g.transform(_days_since)

    # Neighbour pressure: the 8 surrounding cells' own trailing-7-day activity on
    # the same day. Every input is already past-only, so this adds no leakage.
    own = panel[["cell_lat", "cell_lon", "acq_date", "fire_roll7"]]
    neighbour = np.zeros(len(panel), dtype=np.float32)
    keys = panel[["cell_lat", "cell_lon", "acq_date"]]
    offsets = [(dx, dy) for dx in (-CELL_DEG, 0, CELL_DEG)
               for dy in (-CELL_DEG, 0, CELL_DEG) if (dx, dy) != (0, 0)]
    for dx, dy in offsets:
        shifted = own.rename(columns={"fire_roll7": "n"}).copy()
        shifted["cell_lat"] -= dx
        shifted["cell_lon"] -= dy
        merged = keys.merge(shifted, on=["cell_lat", "cell_lon", "acq_date"], how="left")
        neighbour += merged["n"].fillna(0.0).to_numpy(dtype=np.float32)
    panel["neighbor_roll7"] = neighbour
    return panel


def load_weather(nc_dir: Path, cell_deg: float = CELL_DEG,
                 utc_offset_h: int = UTC_OFFSET_H) -> pd.DataFrame:
    """ERA5-Land NetCDFs -> daily fire-weather features per grid cell.

    Two alignment details that a naive daily resample gets wrong:

    1. *Local days.* FIRMS `acq_date` is UTC, but the western-US burn peak is
       mid-afternoon local = 21-23 UTC. Resampling weather on UTC days puts the
       hot part of the afternoon in the wrong bucket. Instantaneous fields are
       shifted by `utc_offset_h` before resampling so a "day" is a local day.
    2. *Accumulated precipitation.* ERA5 `tp` accumulates from 00 UTC, so the
       00:00 sample carries the *previous* day's total. Taking a plain daily max
       over all samples therefore leaks yesterday's rain into today. Hour 0 is
       dropped and the daily total is the last in-day accumulation.
    """
    nc_files = sorted(glob.glob(str(Path(nc_dir) / "**" / "*.nc"), recursive=True))
    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files under {nc_dir}. Run scripts/download_era5.py first."
        )
    ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
    ds = ds.rename({"valid_time" if "valid_time" in ds.coords else "time": "time"})
    for extra in ("number", "expver"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)
        elif extra in ds.coords:
            ds = ds.reset_coords(extra, drop=True)

    t2m_c = ds["t2m"] - 273.15
    d2m_c = ds["d2m"] - 273.15
    a, b = 17.625, 243.04  # Magnus coefficients over water
    rh = (100 * np.exp((a * d2m_c) / (b + d2m_c)) / np.exp((a * t2m_c) / (b + t2m_c))).clip(0, 100)
    wind = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    soil = ds["swvl1"].clip(min=0)  # ERA5 fill values leak small negatives

    inst = xr.Dataset({"t2m_max": t2m_c, "rh_min": rh, "wind_max": wind, "soil_moist": soil})
    inst = inst.assign_coords(time=inst["time"] + np.timedelta64(utc_offset_h, "h"))
    daily = xr.Dataset({
        "t2m_max": inst["t2m_max"].resample(time="1D").max(),
        "rh_min": inst["rh_min"].resample(time="1D").min(),
        "wind_max": inst["wind_max"].resample(time="1D").max(),
        "soil_moist": inst["soil_moist"].resample(time="1D").mean(),
    })

    tp = (ds["tp"] * 1000.0).where(ds["time"].dt.hour != 0)  # mm, minus the carry-over sample
    daily["precip"] = tp.resample(time="1D").max()

    wx = daily.to_dataframe().reset_index().dropna(subset=["t2m_max"])
    wx["cell_lat"] = snap(wx["latitude"], cell_deg)
    wx["cell_lon"] = snap(wx["longitude"], cell_deg)
    wx = wx.rename(columns={"time": "acq_date"})
    wx["acq_date"] = pd.to_datetime(wx["acq_date"]).dt.normalize()
    cols = ["t2m_max", "rh_min", "wind_max", "precip", "soil_moist"]
    return wx.groupby(["cell_lat", "cell_lon", "acq_date"])[cols].mean().reset_index()


def join_features(panel: pd.DataFrame, wx_cell: pd.DataFrame) -> pd.DataFrame:
    """Attach weather to the panel and derive antecedent-dryness + seasonality.

    Left join, not inner: the panel must stay dense per cell or the rolling
    windows below silently span gaps (pandas `.rolling` counts rows, not days).
    Rows without weather are dropped only after the windows are computed.
    """
    df = panel.merge(wx_cell, on=["cell_lat", "cell_lon", "acq_date"], how="left")
    df = df.sort_values(["cell_lat", "cell_lon", "acq_date"]).reset_index(drop=True)

    g = df.groupby(["cell_lat", "cell_lon"])
    for col in ["t2m_max", "rh_min", "wind_max", "precip", "soil_moist"]:
        df[f"{col}_3d"] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["precip_7d"] = g["precip"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    doy = df["acq_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    return df.dropna(subset=["fire_next_day", "t2m_max"]).reset_index(drop=True)


def build_dataset(firms_csv: Path, nc_dir: Path, cell_deg: float = CELL_DEG) -> pd.DataFrame:
    """FIRMS CSV + ERA5 directory -> model-ready frame. The full pipeline."""
    daily = load_detections(firms_csv, cell_deg)
    wx_cell = load_weather(nc_dir, cell_deg)
    # Score every land cell the weather covers, not only cells that have burned.
    cells = wx_cell[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    return join_features(build_panel(daily, cells), wx_cell)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split on time, never at random — a random split leaks the future."""
    cutoff = df["acq_date"].quantile(train_frac)
    return df[df["acq_date"] <= cutoff], df[df["acq_date"] > cutoff], cutoff


def precision_at_k(y_true, prob, k_frac: float = 0.01) -> float:
    """Share of true fires among the top k_frac riskiest cells.

    The operational metric: an agency can inspect a fixed number of cells per
    day, so what matters is the hit rate of the shortlist — not accuracy, and
    not precision at an arbitrary 0.5 threshold.
    """
    y_true = np.asarray(y_true)
    k = max(1, int(len(y_true) * k_frac))
    top = np.argsort(prob)[::-1][:k]
    return float(y_true[top].mean())
