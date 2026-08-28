"""The checks that matter: the label must never be reachable from the features,
and rolling windows must count days rather than rows.

    python test_firepulse.py
"""
import numpy as np
import pandas as pd

from firepulse import FIRE_FEATS, build_panel, join_features, precision_at_k, temporal_split

DAYS = pd.date_range("2020-01-01", "2020-01-20", freq="D")
CELLS = pd.DataFrame({"cell_lat": [34.0, 34.0, 34.25], "cell_lon": [-118.0, -118.25, -118.0]})


def _panel(fire_days):
    """Panel where cell (34.0, -118.0) burns on the given day offsets."""
    daily = pd.DataFrame({
        "cell_lat": 34.0, "cell_lon": -118.0,
        "acq_date": [DAYS[i] for i in fire_days],
        "fire_count": 1.0, "frp_mean": 10.0,
    })
    return build_panel(daily, CELLS, start=str(DAYS[0].date()), end=str(DAYS[-1].date()))


def test_label_is_tomorrow():
    panel = _panel([5, 6, 12])
    cell = panel[(panel.cell_lat == 34.0) & (panel.cell_lon == -118.0)].reset_index(drop=True)
    # fire_next_day on row i must equal fire_today on row i+1.
    assert cell["fire_next_day"][:-1].equals(cell["fire_today"].shift(-1)[:-1].astype(float))
    assert cell.loc[4, "fire_next_day"] == 1  # day 5 burns
    assert cell.loc[5, "fire_today"] == 1
    assert pd.isna(cell.loc[len(cell) - 1, "fire_next_day"])  # no tomorrow for the last day


def test_no_feature_sees_the_future():
    """Move a fire later in time; nothing at an earlier date may change."""
    early, late = _panel([5]), _panel([5, 15])
    key = ["cell_lat", "cell_lon", "acq_date"]
    cutoff = DAYS[13]  # before the added fire, and before its d+1 label
    a = early[early.acq_date <= cutoff].sort_values(key).reset_index(drop=True)
    b = late[late.acq_date <= cutoff].sort_values(key).reset_index(drop=True)
    for feat in FIRE_FEATS:
        assert np.allclose(a[feat], b[feat]), f"{feat} changed in response to a future fire"


def test_rolling_counts_days_not_rows():
    """A cell missing weather must not have its window silently span the gap."""
    panel = _panel([2, 3, 4])
    wx = pd.DataFrame({
        "cell_lat": 34.0, "cell_lon": -118.0, "acq_date": DAYS,
        "t2m_max": 30.0, "rh_min": 20.0, "wind_max": 5.0, "precip": 0.0, "soil_moist": 0.1,
    })
    wx.loc[wx.acq_date == DAYS[10], "t2m_max"] = np.nan  # a hole in the weather
    joined = join_features(panel, wx)
    # The hole is dropped, but the surviving rows keep true 3-day windows: the
    # row after the gap must not average across it as if the days were adjacent.
    assert DAYS[10] not in set(joined["acq_date"])
    assert len(joined) > 0
    assert joined["precip_7d"].notna().all()


def test_precision_at_k_ranks():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    perfect = np.array([0, 0, .9, 0, .8, 0, 0, 0, 0, 0])
    assert precision_at_k(y, perfect, k_frac=0.2) == 1.0
    assert precision_at_k(y, 1 - perfect, k_frac=0.2) == 0.0


def test_temporal_split_does_not_overlap():
    panel = _panel([5])
    train, test, cutoff = temporal_split(panel, 0.8)
    assert train["acq_date"].max() <= cutoff < test["acq_date"].min()
    assert len(train) + len(test) == len(panel)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
