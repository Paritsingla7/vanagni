"""Phase 3: stage -> feature resolution, rolling-origin fold boundaries,
fold-local burn rate (FR-4.4), and an end-to-end train run on synthetic data."""
import json

import numpy as np
import pandas as pd
import pytest

from wfpin.config import load_config
from wfpin.features import CELL_BURN_RATE, FAMILIES, LABEL, add_fold_features, stage_features
from wfpin.panel import build_panel
from wfpin.train import folds, run

CONFIG = load_config()


def test_stage_feature_sets():
    assert stage_features(CONFIG, "1") is None
    s3, s4a, s4b = (set(stage_features(CONFIG, s)) for s in ("3", "4a", "4b"))
    assert s4b - s4a == set(FAMILIES["human"])  # 4b vs 4a isolates human features
    assert s4a - s3 == set(FAMILIES["fuel_terrain"])
    no_weather = set(stage_features(CONFIG, "4b-weather"))
    assert no_weather == s4b - set(FAMILIES["weather"])
    assert CELL_BURN_RATE in set(stage_features(CONFIG, "2"))
    for banned in ("cell_id", "centroid_lat", "centroid_lon", "grid_row", "grid_col"):
        assert banned not in s4b  # never let the model memorise cell identity


def test_unknown_family_is_an_error():
    cfg = {"ablation": {"stages": {"x": ["history", "typo"]}}}
    with pytest.raises(KeyError):
        stage_features(cfg, "x")


def _synthetic_panel(seasons=range(2017, 2025), n_cells=9, seed=0):
    from tests.test_panel_leakage import _static

    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{min(seasons)}-01-01", f"{max(seasons)}-12-31")
    spring = days[(days.dayofyear >= 46) & (days.dayofyear <= 166)]
    fires = pd.DataFrame(
        {"cell_id": rng.integers(0, n_cells, 400), "date": rng.choice(spring, 400)}
    ).drop_duplicates()
    panel = build_panel(_static(), fires, str(days[0].date()), str(days[-1].date()))
    for col in FAMILIES["weather"]:
        panel[col] = rng.normal(size=len(panel)).astype("float32")
    return panel


def test_folds_never_train_on_the_test_season_or_later():
    panel = _synthetic_panel()
    seen = []
    for season, train, test in folds(panel, CONFIG):
        assert train["date"].dt.year.max() < season
        assert (test["date"].dt.year == season).all() and test["in_season"].all()
        assert train[LABEL].notna().all() and test[LABEL].notna().all()
        seen.append(season)
    assert seen == CONFIG["evaluation"]["test_seasons"]


def test_burn_rate_ignores_test_labels():
    panel = _synthetic_panel()
    season, train, test = next(folds(panel, CONFIG))
    flipped = test.assign(**{LABEL: ~test[LABEL].astype(bool)})
    _, a = add_fold_features(train, test)
    _, b = add_fold_features(train, flipped)
    pd.testing.assert_series_equal(a[CELL_BURN_RATE], b[CELL_BURN_RATE])


def test_end_to_end_writes_loadable_artifacts(tmp_path):
    import xgboost as xgb

    cfg = {**CONFIG, "model": {**CONFIG["model"], "n_estimators": 5},
           "evaluation": {**CONFIG["evaluation"], "test_seasons": [2023, 2024]}}
    log = run(cfg, _synthetic_panel(), models_dir=tmp_path / "m", predictions_dir=tmp_path / "p")
    stages = list(cfg["ablation"]["stages"])
    assert len(log) == len(stages) * 2
    for stage in stages:
        for season in (2023, 2024):
            name = f"stage_{stage}_season_{season}"
            preds = pd.read_parquet(tmp_path / "p" / f"{name}.parquet")
            assert preds["score"].between(0, 1).all() and len(preds) > 0
            meta = json.loads((tmp_path / "m" / f"{name}_meta.json").read_text())
            if meta["features"] is not None:
                booster = xgb.XGBClassifier()
                booster.load_model(tmp_path / "m" / f"{name}.json")
                assert booster.n_features_in_ == len(meta["features"])
    # Resumable: a second run skips everything.
    assert run(cfg, _synthetic_panel(), models_dir=tmp_path / "m",
               predictions_dir=tmp_path / "p") == []
