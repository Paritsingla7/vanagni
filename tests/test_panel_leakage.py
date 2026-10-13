"""FR-6.1: leakage guarantee for panel.py.

Synthetic fixture first (a 3x3-cell lattice), then the same truncation check
against the real inputs as an integration test. The core property: a panel
built from fires truncated at day c must match the full panel on every
non-label column for every date <= c -- i.e. nothing at date d can depend on
anything after d. The label is the only column allowed to see d+1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfpin.panel import DAYS_SINCE_CAP, build_panel

START, END = "2020-01-01", "2020-03-31"
DAYS = pd.date_range(START, END)
CENTRE = 4  # cell_id of the middle cell in the 3x3 lattice
LABEL = "label_fire_next_day"


def _static() -> pd.DataFrame:
    rows, cols = np.divmod(np.arange(9), 3)
    return pd.DataFrame(
        {
            "cell_id": np.arange(9, dtype="int32"),
            "grid_row": rows.astype("int16"),
            "grid_col": cols.astype("int16"),
            "state_id": pd.Categorical(["UK"] * 9),
            "season_start_doy": np.int16(46),
            "season_end_doy": np.int16(166),
            "elevation_m": np.linspace(500, 2500, 9, dtype="float32"),
            "slope_deg": np.float32(20), "aspect_deg": np.float32(180),
            "northness": -1.0, "eastness": 0.0,
            "landcover_class": pd.Categorical(["forest"] * 9, categories=["forest", "other"]),
            "frac_forest": np.float32(0.8), "frac_shrub_grass": np.float32(0.1),
            "frac_cropland": np.float32(0.05), "frac_built_up": np.float32(0.0),
            "frac_other": np.float32(0.05),
            "dist_road_m": np.arange(9, dtype="float32") * 100,
            "dist_settlement_m": np.arange(9, dtype="float32") * 200,
        }
    )


def _fires(spec: dict[int, list[int]]) -> pd.DataFrame:
    """{cell_id: [day offsets]} -> fire_daily frame."""
    rows = [(c, DAYS[d]) for c, offsets in spec.items() for d in offsets]
    return pd.DataFrame(rows, columns=["cell_id", "date"])


FIRES = {CENTRE: [5, 6, 20, 50], 0: [10, 60], 8: [30]}


def _panel(spec=FIRES, weather=None):
    return build_panel(_static(), _fires(spec), START, END, weather)


def _cell(panel, cell_id):
    return panel[panel["cell_id"] == cell_id].set_index("date")


def test_label_is_tomorrows_fire():
    p = _cell(_panel(), CENTRE)
    assert (p[LABEL].iloc[:-1].to_numpy() == p["fire_today"].iloc[1:].to_numpy()).all()
    assert p[LABEL].iloc[-1] is pd.NA  # no tomorrow inside the window
    assert bool(p.loc[DAYS[4], LABEL]) and not bool(p.loc[DAYS[4], "fire_today"])


@pytest.mark.parametrize("cutoff", [4, 5, 19, 20, 29, 49, 75])
def test_truncating_the_future_changes_nothing_before_it(cutoff):
    full = _panel()
    truncated = _panel({c: [d for d in ds if d <= cutoff] for c, ds in FIRES.items()})
    key = ["cell_id", "date"]
    a = full[full["date"] <= DAYS[cutoff]].sort_values(key).reset_index(drop=True)
    b = truncated[truncated["date"] <= DAYS[cutoff]].sort_values(key).reset_index(drop=True)
    features = [c for c in full.columns if c != LABEL]
    pd.testing.assert_frame_equal(a[features], b[features])


def test_moving_a_fire_later_changes_no_earlier_feature():
    """FR-6.1 as originally worded: move the day-20 fire to day 40."""
    moved = {**FIRES, CENTRE: [5, 6, 40, 50]}
    a, b = _panel(), _panel(moved)
    early_a, early_b = a[a["date"] < DAYS[20]], b[b["date"] < DAYS[20]]
    features = [c for c in a.columns if c != LABEL]
    pd.testing.assert_frame_equal(early_a[features].reset_index(drop=True),
                                  early_b[features].reset_index(drop=True))


def test_rolling_counts_are_days_not_rows():
    p = _cell(_panel({CENTRE: [5]}), CENTRE)
    assert p["fire_count_7d"].loc[DAYS[5]:DAYS[11]].eq(1).all()
    assert p.loc[DAYS[12], "fire_count_7d"] == 0
    assert p.loc[DAYS[4], "fire_count_7d"] == 0
    assert p.loc[DAYS[34], "fire_count_30d"] == 1 and p.loc[DAYS[35], "fire_count_30d"] == 0


def test_neighbour_pressure_counts_neighbours_not_self():
    p = _panel({CENTRE: [5]})
    on_day = p[p["date"] == DAYS[5]].set_index("cell_id")["neighbor_fire_pressure_7d"]
    assert on_day[CENTRE] == 0
    assert (on_day.drop(CENTRE) == 1).all()  # all 8 others are the centre's neighbours


def test_no_sentinel_values():
    p = _cell(_panel({CENTRE: [5]}), CENTRE)
    assert p["days_since_last_fire"].loc[: DAYS[4]].isna().all()
    assert not p["has_fire_history"].loc[: DAYS[4]].any()
    assert p.loc[DAYS[5], "days_since_last_fire"] == 0
    assert p.loc[DAYS[9], "days_since_last_fire"] == 4
    assert p["days_since_last_fire"].max() <= DAYS_SINCE_CAP


def test_days_since_cap():
    long = build_panel(_static(), _fires({CENTRE: [0]}), "2020-01-01", "2021-06-30")
    cell = _cell(long, CENTRE)
    assert cell["days_since_last_fire"].max() == DAYS_SINCE_CAP


def test_weather_joins_same_day_only():
    from wfpin.ingest.weather import DAILY_COLUMNS
    w = pd.DataFrame([(c, d) for c in range(9) for d in DAYS], columns=["cell_id", "date"])
    for col in DAILY_COLUMNS:
        w[col] = w["date"].dt.dayofyear.astype("float32")  # value encodes its own date
    p = _panel(weather=w)
    assert (p["t2m_max"] == p["date"].dt.dayofyear).all()


# --- integration: the same truncation property on the real inputs ----------------------

REAL = [
    Path("data/processed/static_layers.parquet"),
    Path("data/raw/fire/firms_viirs_uttarakhand.csv"),
    Path("data/processed/landcover_utm.tif"),
    Path("data/raw/osm/osm_state.gpkg"),
]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason="real Phase 1 inputs not built")
def test_real_panel_truncation_is_leak_free():
    import geopandas as gpd

    from wfpin.config import load_config
    from wfpin.panel import fire_days, qualify_detections

    config = load_config()
    static = pd.read_parquet(REAL[0])
    fires, state = pd.read_csv(REAL[1]), gpd.read_file(REAL[3])
    det, _ = qualify_detections(fires, static, state, REAL[2], config)
    daily = fire_days(det)
    start, end, cutoff = "2021-01-01", "2021-06-30", pd.Timestamp("2021-04-15")
    full = build_panel(static, daily, start, end)
    trunc = build_panel(static, daily[daily["date"] <= cutoff], start, end)
    features = [c for c in full.columns if c != LABEL]
    a = full.loc[full["date"] <= cutoff, features].reset_index(drop=True)
    b = trunc.loc[trunc["date"] <= cutoff, features].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert full[LABEL].sum() > 0
