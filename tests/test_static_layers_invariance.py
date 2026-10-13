"""FR-6.2: static layers never vary by date.

Road/settlement distance, land cover, terrain, and the fire-calendar signal
must be constant across every date for a given cell, and equal to that
cell's row in static_layers.parquet -- the layers most likely to get
silently re-joined wrong.
"""
from pathlib import Path

import pandas as pd
import pytest

from wfpin.panel import STATIC_FEATURE_COLUMNS, build_panel

PANEL = Path("data/processed/panel.parquet")
STATIC = Path("data/processed/static_layers.parquet")


def _assert_invariant(panel: pd.DataFrame, static: pd.DataFrame) -> None:
    per_cell = panel.groupby("cell_id", observed=True)[STATIC_FEATURE_COLUMNS].nunique(dropna=False)
    varying = per_cell.columns[(per_cell > 1).any()].tolist()
    assert not varying, f"static columns vary by date: {varying}"
    first = panel.groupby("cell_id", observed=True)[STATIC_FEATURE_COLUMNS].first()
    expected = static.set_index("cell_id").loc[first.index, STATIC_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(first.astype(str), expected.astype(str), check_dtype=False)
    # The calendar signal: onset offset moves exactly one day per day.
    onset = panel.sort_values(["cell_id", "date"]).groupby("cell_id")["days_since_season_onset"]
    steps = onset.diff().dropna()
    doy = panel.sort_values(["cell_id", "date"])["date"].dt.dayofyear
    year_wraps = doy.diff().loc[steps.index] < 0
    assert (steps[~year_wraps] == 1).all()


def test_static_invariance_synthetic():
    from tests.test_panel_leakage import END, START, _fires, _static

    static = _static()
    _assert_invariant(build_panel(static, _fires({4: [3]}), START, END), static)


@pytest.mark.skipif(not (PANEL.exists() and STATIC.exists()), reason="real panel not built")
def test_static_invariance_real_panel():
    cols = ["cell_id", "date", "days_since_season_onset", *STATIC_FEATURE_COLUMNS]
    _assert_invariant(pd.read_parquet(PANEL, columns=cols), pd.read_parquet(STATIC))
