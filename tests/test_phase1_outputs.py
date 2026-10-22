"""EXECUTION_PLAN Phase 1 done-condition, checked against the real built
outputs (skipped until they exist): every FR-2.1-2.5 layer keyed by
cell_id, weather keyed by (cell_id, date), zero null season windows."""

import pandas as pd
import pytest

from wfpin.config import load_config, paths

P = paths(load_config())
STATIC = P.processed / "static_layers.parquet"
WEATHER = P.processed / "weather_daily.parquet"
FIRE = P.raw / "fire/firms_viirs.csv"

STATIC_REQUIRED = [
    "cell_id", "grid_row", "grid_col", "in_state_frac", "state_id", "centroid_lat",
    "centroid_lon", "elevation_m", "slope_deg", "aspect_deg", "northness", "eastness",
    "landcover_class", "frac_forest", "frac_shrub_grass", "frac_cropland", "frac_built_up",
    "frac_other", "dist_road_m", "dist_settlement_m", "district", "season_start_doy",
    "season_end_doy", "calendar_source",
]


@pytest.mark.skipif(not STATIC.exists(), reason="static layers not built yet")
def test_static_layers_complete():
    df = pd.read_parquet(STATIC)
    assert set(STATIC_REQUIRED) <= set(df.columns)
    assert df["cell_id"].is_unique
    assert (df["state_id"] == "UK").all()
    nulls = df[STATIC_REQUIRED].isna().sum()
    assert not nulls.any(), f"null static values:\n{nulls[nulls > 0]}"
    assert (df["calendar_source"].str.len() > 0).all()
    assert df["district"].nunique() == 13  # Uttarakhand's districts
    fracs = df.filter(regex="^frac_").sum(axis=1)
    assert fracs.between(0.999, 1.001).all()


@pytest.mark.skipif(not (WEATHER.exists() and STATIC.exists()), reason="weather not built yet")
def test_weather_daily_dense_and_sane():
    w = pd.read_parquet(WEATHER)
    cells = pd.read_parquet(STATIC, columns=["cell_id"])["cell_id"]
    assert not w.duplicated(["cell_id", "date"]).any()
    assert set(w["cell_id"]) == set(cells)
    per_cell = w.groupby("cell_id")["date"].nunique()
    assert per_cell.nunique() == 1, "every cell must cover the same days"
    assert w.drop(columns=["cell_id", "date"]).notna().all().all()
    assert (w["soil_moisture"] >= 0).all() and (w["precip_sum"] >= 0).all()
    assert w["rh_min"].between(0, 100).all()
    assert (w["t2m_max"] >= w["t2m_mean"]).all()


@pytest.mark.skipif(not FIRE.exists(), reason="fire not ingested yet")
def test_fire_raw_has_both_satellites_and_window():
    f = pd.read_csv(FIRE, usecols=["acq_date", "satellite", "type"])
    assert set(f["satellite"].astype(str)) == {"N", "N20"}
    assert f["acq_date"].min() >= "2017-01-01" and f["acq_date"].max() <= "2024-12-31"
    noaa20_first = f.loc[f["satellite"].astype(str) == "N20", "acq_date"].min()
    assert noaa20_first >= "2018-04-01"
