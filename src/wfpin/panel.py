"""Dense, leakage-safe cell x day panel construction.

FR-3.1-3.5: every cell, every day in the window, quiet days included as
explicit negatives. Label is a qualifying fire detected on local day d+1;
every feature comes from day d or earlier. No sentinel values reach the
feature matrix.

Rolling windows are computed on a (cell x day) array whose day axis *is*
the date (column = days since window start, by construction), never by
positional `.rolling()` over rows of a joined frame -- so a missing row
cannot silently shift a window (AUDIT.md's 4,560-dropped-rows failure).

    python -m wfpin.panel [--config config/uttarakhand.yaml]
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import uniform_filter

from wfpin.config import cli_config
from wfpin.ingest.weather import DAILY_COLUMNS as WEATHER_COLUMNS
from wfpin.static_layers import PROCESSED_DIR, RAW_DIR, UTM_CRS, WGS84_CRS, lattice

IST = pd.Timedelta(hours=5, minutes=30)
DAYS_SINCE_CAP = 365  # SCHEMA.md §4: capped, NaN when the cell has no history at all

STATIC_FEATURE_COLUMNS = [
    "elevation_m", "slope_deg", "aspect_deg", "northness", "eastness", "landcover_class",
    "frac_forest", "frac_shrub_grass", "frac_cropland", "frac_built_up", "frac_other",
    "dist_road_m", "dist_settlement_m",
]


# --- FR-3.2/3.5: which detections count -----------------------------------------------


def local_day(acq_date: pd.Series, acq_time: pd.Series) -> pd.Series:
    """FIRMS UTC date + HHMM (int, not zero-padded) -> local (IST) calendar day,
    the same day convention the weather is bucketed by (FR-1.2)."""
    hh, mm = np.divmod(acq_time.astype(int).to_numpy(), 100)
    utc = pd.to_datetime(acq_date) + pd.to_timedelta(hh, "h") + pd.to_timedelta(mm, "m")
    return (utc + IST).dt.normalize()


def vegetation_share(landcover_path: Path, codes: list[int], footprint_px: int):
    """Share of natural-vegetation pixels within a footprint_px window around
    every pixel (over pixels with data), plus the raster's transform."""
    with rasterio.open(landcover_path) as src:
        lc, transform = src.read(1), src.transform
    valid = (lc > 0).astype("float32")
    veg = np.isin(lc, codes).astype("float32")
    covered = np.maximum(uniform_filter(valid, footprint_px), 1e-6)
    share = uniform_filter(veg, footprint_px) / covered
    return share, transform


def persistent_hotspots(
    det: pd.DataFrame, bin_m: float, reference: tuple[str, str], min_years: int
) -> pd.Series:
    """True for detections in a ~bin_m spot that, *within the reference
    period only*, was detected outside its fire season in >= min_years
    distinct years -- a kiln/factory signature. Using a fixed pre-test
    reference keeps the mask a static layer: later detections never relabel
    earlier ones."""
    bins = pd.Series(
        list(zip(np.floor(det["x"] / bin_m), np.floor(det["y"] / bin_m), strict=True)),
        index=det.index,
    )
    ref = det["date"].between(*reference) & ~det["in_season"]
    years = det.loc[ref, "date"].dt.year.groupby(bins[ref]).nunique()
    flagged = set(years[years >= min_years].index)
    return bins.map(lambda b: b in flagged)


def qualify_detections(
    fires: pd.DataFrame,
    static: pd.DataFrame,
    state: gpd.GeoDataFrame,
    landcover_path: Path,
    config: dict,
    min_vegetation_share: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """FR-3.5: raw FIRMS rows -> qualifying detections with cell_id and IST
    date, plus a step-by-step count of what each filter removed (written to
    the panel manifest for the Methodology screen)."""
    lab = config["label"]
    share_min = (
        lab["min_vegetation_share"] if min_vegetation_share is None else min_vegetation_share
    )
    classes = config["static_layers"]["worldcover"]["classes"]
    veg_codes = [code for name in lab["vegetation_classes"] for code in classes[name]]
    lat = lattice(config)

    steps = {"raw": len(fires)}
    det = fires[fires["satellite"].astype(str).isin(lab["satellites"])].copy()
    steps["satellite"] = len(det)

    x, y = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True).transform(
        det["longitude"].to_numpy(), det["latitude"].to_numpy()
    )
    det["x"], det["y"] = x, y
    inside = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=UTM_CRS).within(state.union_all())
    det = det[inside.to_numpy()]
    steps["in_state"] = len(det)

    det = det[det["type"].isin(lab["fire_type"]) & det["confidence"].isin(lab["confidence"])]
    steps["type_and_confidence"] = len(det)

    share, transform = vegetation_share(landcover_path, veg_codes, lab["footprint_px"])
    r, c = rasterio.transform.rowcol(transform, det["x"].to_numpy(), det["y"].to_numpy())
    det = det.assign(veg_share=share[np.asarray(r), np.asarray(c)])
    det = det[det["veg_share"] >= share_min]
    steps["vegetation_footprint"] = len(det)

    col = np.floor((det["x"] - lat.west_m) / lat.res_m).astype(int)
    row = np.floor((det["y"] - lat.south_m) / lat.res_m).astype(int)
    det = det.assign(cell_id=(row * lat.n_cols + col).astype("int32"))
    det["date"] = local_day(det["acq_date"], det["acq_time"])
    season = static.set_index("cell_id")[["season_start_doy", "season_end_doy"]]
    det = det[det["cell_id"].isin(season.index)]
    doy = det["date"].dt.dayofyear
    det["in_season"] = doy.between(
        det["cell_id"].map(season["season_start_doy"]), det["cell_id"].map(season["season_end_doy"])
    )
    hot = persistent_hotspots(
        det, lab["persistent_hotspot_bin_m"], tuple(lab["persistent_hotspot_reference"]),
        lab["persistent_hotspot_min_offseason_years"],
    )
    det = det[~hot]
    steps["not_persistent_hotspot"] = len(det)
    return det[["cell_id", "date", "frp", "veg_share"]].reset_index(drop=True), steps


def fire_days(detections: pd.DataFrame) -> pd.DataFrame:
    """Qualifying detections -> one row per (cell_id, date) that had a fire."""
    return (
        detections.groupby(["cell_id", "date"], as_index=False)
        .size()
        .rename(columns={"size": "n_detections"})
    )


# --- FR-3.1/3.3/3.4: the dense panel ---------------------------------------------------


def _windowed_count(fire: np.ndarray, window: int) -> np.ndarray:
    """Fire days in the `window` days ending at each day, inclusive."""
    cum = np.concatenate([np.zeros((fire.shape[0], 1), np.int32), fire.cumsum(axis=1)], axis=1)
    end = np.arange(1, fire.shape[1] + 1)
    return (cum[:, end] - cum[:, np.maximum(end - window, 0)]).astype("int16")


def _neighbour_sum(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sum over each cell's 8 lattice neighbours (cells absent from the
    panel contribute 0), via a padded (row, col, day) array."""
    r, c = rows - rows.min() + 1, cols - cols.min() + 1
    grid = np.zeros((r.max() + 2, c.max() + 2, values.shape[1]), dtype="float32")
    grid[r, c] = values
    total = np.zeros_like(values, dtype="float32")
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                total += grid[r + dr, c + dc]
    return total


def build_panel(
    static: pd.DataFrame,
    fire_daily: pd.DataFrame,
    start: str,
    end: str,
    weather: pd.DataFrame | None = None,
    fwi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dense cell x day panel over every cell in `static` and every day in
    [start, end]. `fire_daily` holds (cell_id, date) fire days; only those
    feed history features and the label. `weather` (cell_id, date, ...) is
    left-joined as-of `date` only. `fwi` (cell_id, date, fwi) is the
    operational-index baseline: one same-day column, in no model's features."""
    static = static.sort_values("cell_id").reset_index(drop=True)
    days = pd.date_range(start, end, freq="D")
    n_cells, n_days = len(static), len(days)
    cell_idx = pd.Index(static["cell_id"])

    fire = np.zeros((n_cells, n_days), dtype=np.int32)
    fd = fire_daily.assign(date=pd.to_datetime(fire_daily["date"]))
    fd = fd[fd["date"].between(days[0], days[-1])]
    fd = fd[fd["cell_id"].isin(cell_idx)]
    fire[cell_idx.get_indexer(fd["cell_id"]), (fd["date"] - days[0]).dt.days.to_numpy()] = 1

    count_7d = _windowed_count(fire, 7)
    count_30d = _windowed_count(fire, 30)
    day_no = np.arange(n_days)
    last = np.maximum.accumulate(np.where(fire == 1, day_no, -1), axis=1)
    since = np.where(last >= 0, np.minimum(day_no - last, DAYS_SINCE_CAP), np.nan)
    since = since.astype("float32")
    rows, cols = static["grid_row"].to_numpy(), static["grid_col"].to_numpy()
    pressure = _neighbour_sum(count_7d, rows, cols)
    label = np.full((n_cells, n_days), np.nan, dtype="float32")
    label[:, :-1] = fire[:, 1:]  # the ONLY place day d+1 is read

    doy = np.tile(days.dayofyear.to_numpy(), n_cells)
    start_doy = np.repeat(static["season_start_doy"].to_numpy(), n_days)
    end_doy = np.repeat(static["season_end_doy"].to_numpy(), n_days)

    panel = pd.DataFrame(
        {
            "cell_id": np.repeat(static["cell_id"].to_numpy(), n_days),
            "date": np.tile(days.to_numpy(), n_cells),
            "state_id": np.repeat(static["state_id"].astype(str).to_numpy(), n_days),
            "label_fire_next_day": pd.array(label.ravel(), dtype="boolean"),
            "fire_today": fire.ravel().astype(bool),
            "fire_count_7d": count_7d.ravel(),
            "fire_count_30d": count_30d.ravel(),
            "days_since_last_fire": since.ravel(),
            "has_fire_history": (last >= 0).ravel(),
            "neighbor_fire_pressure_7d": pressure.ravel(),
            "days_since_season_onset": (doy - start_doy).astype("int16"),
            "in_season": (doy >= start_doy) & (doy <= end_doy),
        }
    )
    for col in STATIC_FEATURE_COLUMNS:
        panel[col] = np.repeat(static[col].to_numpy(), n_days)
    panel["state_id"] = panel["state_id"].astype("category")
    panel["landcover_class"] = pd.Categorical(
        panel["landcover_class"], categories=static["landcover_class"].cat.categories
    )

    if weather is not None:
        w = weather.assign(date=pd.to_datetime(weather["date"]))
        panel = panel.merge(w[["cell_id", "date", *WEATHER_COLUMNS]], on=["cell_id", "date"],
                            how="left", validate="one_to_one")
    if fwi is not None:
        f = fwi[["cell_id", "date", "fwi"]].assign(date=pd.to_datetime(fwi["date"]))
        panel = panel.merge(f, on=["cell_id", "date"], how="left", validate="one_to_one")
    return panel


def scored_cells(static: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The scored population: cells with enough natural vegetation to burn."""
    veg_cols = [f"frac_{name}" for name in config["label"]["vegetation_classes"]]
    keep = static[veg_cols].sum(axis=1) >= config["label"]["min_cell_vegetation"]
    return static[keep]


def build_panel_from_files(
    config: dict,
    min_vegetation_share: float | None = None,
    out_path: Path | None = None,
) -> Path:
    """Phase 2 end to end -> panel.parquet + panel_manifest.json. History
    features and neighbour pressure use every in-state cell's fires; only
    the scored population's rows are written."""
    static = pd.read_parquet(PROCESSED_DIR / "static_layers.parquet")
    fires = pd.read_csv(RAW_DIR / "fire/firms_viirs_uttarakhand.csv")
    state = gpd.read_file(RAW_DIR / "osm/osm_state.gpkg")
    weather = pd.read_parquet(PROCESSED_DIR / "weather_daily.parquet")
    fwi = None
    if any(spec == "fwi" for spec in config["ablation"]["stages"].values()):
        fwi_path = PROCESSED_DIR / "fwi_daily.parquet"
        if not fwi_path.exists():
            raise FileNotFoundError(f"{fwi_path} missing -- run `python -m wfpin.fwi` first")
        fwi = pd.read_parquet(fwi_path, columns=["cell_id", "date", "fwi"])

    detections, steps = qualify_detections(
        fires, static, state, PROCESSED_DIR / "landcover_utm.tif", config, min_vegetation_share
    )
    window = config["window"]
    panel = build_panel(
        static, fire_days(detections), window["start"], window["end"], weather, fwi
    )
    population = scored_cells(static, config)["cell_id"]
    panel = panel[panel["cell_id"].isin(population)].reset_index(drop=True)

    default = min_vegetation_share is None
    share = config["label"]["min_vegetation_share"] if default else min_vegetation_share
    name = "panel.parquet" if default else f"panel_veg{round(share * 100)}.parquet"
    out = out_path or PROCESSED_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)

    season = panel[panel["in_season"] & panel["label_fire_next_day"].notna()]
    manifest = {
        "panel": out.name,
        "window": window,
        "label": {**config["label"], "min_vegetation_share": share},
        "detection_filter_steps": steps,
        "cells_in_state": int(len(static)),
        "cells_scored": int(population.size),
        "rows": int(len(panel)),
        "in_season_base_rate": float(season["label_fire_next_day"].astype(float).mean()),
        "in_season_positive_cell_days": int(season["label_fire_next_day"].sum()),
        "weather_missing_rows": int(panel[WEATHER_COLUMNS[0]].isna().sum()),
    }
    out.with_name(out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return out


if __name__ == "__main__":
    build_panel_from_files(cli_config("FR-3.1-3.5: build the dense leakage-safe panel"))
