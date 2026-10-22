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
    assert stage_features(CONFIG, "fwi") is None  # a rule stage, not a model
    s3, s4a, s4b = (set(stage_features(CONFIG, s)) for s in ("3", "4a", "4b"))
    assert s4b - s4a == set(FAMILIES["human"])  # 4b vs 4a isolates human features
    assert s4a - s3 == set(FAMILIES["fuel_terrain"])
    no_weather = set(stage_features(CONFIG, "4b-weather"))
    assert no_weather == s4b - set(FAMILIES["weather"])
    assert CELL_BURN_RATE in set(stage_features(CONFIG, "2"))
    # The FWI is visible to exactly one learned stage, "3f"; no main family contains it.
    sees_fwi = {
        s for s in CONFIG["ablation"]["stages"] if "fwi" in (stage_features(CONFIG, s) or [])
    }
    assert sees_fwi == {"3f"}
    assert [f for f, cols in FAMILIES.items() if "fwi" in cols] == ["fwi_index"]
    assert set(stage_features(CONFIG, "3f")) - set(stage_features(CONFIG, "2")) == {"fwi"}
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
    panel["fwi"] = rng.random(len(panel)).astype("float32")
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


def test_rule_stage_missing_scores_fail_loudly(tmp_path):
    cfg = {**CONFIG, "model": {**CONFIG["model"], "n_estimators": 3},
           "evaluation": {**CONFIG["evaluation"], "test_seasons": [2024]}}
    panel = _synthetic_panel()
    panel.loc[panel["date"].dt.year == 2024, "fwi"] = np.nan
    with pytest.raises(ValueError, match="no 'fwi' value"):
        run(cfg, panel, stages=["fwi"], models_dir=tmp_path / "m", predictions_dir=tmp_path / "p")


# --- spatial holdout -------------------------------------------------------------------

def _grid_static(n=12):
    rows, cols = np.divmod(np.arange(n * n), n)
    return pd.DataFrame({"cell_id": np.arange(n * n, dtype="int32"),
                         "grid_row": rows.astype("int16"), "grid_col": cols.astype("int16")})


def test_spatial_folds_keep_blocks_whole_and_are_balanced():
    from wfpin.train import spatial_folds

    cfg = {**CONFIG, "evaluation": {**CONFIG["evaluation"],
           "spatial_holdout": {"block_km": 20, "n_folds": 3, "seed": 0}}}  # 4x4-cell blocks
    static = _grid_static()
    fold = spatial_folds(cfg, static)
    assert set(fold.index) == set(static["cell_id"]) and fold.notna().all()
    block = (static["grid_row"] // 4) * 10 + (static["grid_col"] // 4)
    assert (fold.groupby(block.to_numpy()).nunique() == 1).all()  # a block never straddles folds
    assert fold.nunique() == 3
    assert fold.equals(spatial_folds(cfg, static))  # deterministic


def test_split_spatial_holds_out_whole_cells_and_blanks_their_burn_rate():
    from wfpin.train import split_spatial

    panel = _synthetic_panel()
    season, train_all, test_all = next(folds(panel, CONFIG))
    held = [0, 1, 2]
    train, test = split_spatial(train_all, test_all, held)
    assert not train["cell_id"].isin(held).any() and set(test["cell_id"]) <= set(held)
    _, test_f = add_fold_features(train, test)
    assert test_f[CELL_BURN_RATE].isna().all()  # held-out places have no history of their own


def test_run_spatial_scores_every_test_row_exactly_once(tmp_path):
    from wfpin.train import run_spatial

    cfg = {**CONFIG, "model": {**CONFIG["model"], "n_estimators": 3},
           "evaluation": {**CONFIG["evaluation"], "test_seasons": [2024],
                          "spatial_holdout": {"block_km": 5, "n_folds": 3, "seed": 0}}}
    panel = _synthetic_panel()
    cell_fold = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2], index=np.arange(9))
    run_spatial(cfg, panel, cell_fold, models_dir=tmp_path / "m", predictions_dir=tmp_path / "p")
    _, _, full_test = next(f for f in folds(panel, cfg) if f[0] == 2024)
    for stage in cfg["ablation"]["stages"]:
        pred = pd.read_parquet(tmp_path / "p" / f"stage_{stage}_season_2024_spatial.parquet")
        assert len(pred) == len(full_test)
        assert not pred.duplicated(["cell_id", "date"]).any()
