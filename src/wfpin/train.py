"""Staged ablation training, one model per stage per rolling-origin fold.

FR-4.1-4.5: stages from config `ablation.stages` (persistence, +history,
+weather, +fuel/terrain, +human proximity, and the 4b-minus-weather check).
For each test season S in `evaluation.test_seasons`, every stage trains on
all earlier seasons and scores season S. Stage 1 has no learned parameters
but writes its predictions through the same path, so FR-5.4's comparison is
apples to apples. Every stage x fold serializes a loadable artifact, and its
test-season predictions -- evaluate.py scores those files, it never re-fits.

    python -m wfpin.train [--stage 4a --stage 4b] [--panel panel_veg30.parquet --tag _veg30]
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import xgboost as xgb

from wfpin.config import cli_args
from wfpin.features import CELL_BURN_RATE, FAMILIES, LABEL, add_fold_features, stage_features
from wfpin.static_layers import PROCESSED_DIR

MODELS_DIR = PROCESSED_DIR / "models"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
KEY_COLUMNS = ["cell_id", "date", LABEL, "in_season", "fire_today"]


def folds(panel: pd.DataFrame, config: dict) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """FR-5.1 rolling origin: (season, train, test). Train = labelled rows of
    every season before S (in-season only if configured); test = labelled
    in-season rows of S. Seasons are calendar years -- the FR-2.4 window
    (Feb 15 - Jun 15) sits inside one year."""
    labelled = panel[panel[LABEL].notna()]
    year = labelled["date"].dt.year
    in_season_only = config["ablation"]["train_in_season_only"]
    pool = labelled[labelled["in_season"]] if in_season_only else labelled
    pool_year = pool["date"].dt.year
    for season in config["evaluation"]["test_seasons"]:
        train = pool[pool_year < season]
        test = labelled[(year == season) & labelled["in_season"]]
        if train.empty or test.empty:
            raise ValueError(f"season {season}: empty train ({len(train)}) or test ({len(test)})")
        yield season, train, test


def fit_stage(train: pd.DataFrame, features: list[str], model_cfg: dict) -> xgb.XGBClassifier:
    """FR-4.2: XGBoost with scale_pos_weight from *this* training split."""
    y = train[LABEL].astype(int)
    pos = int(y.sum())
    model = xgb.XGBClassifier(
        **model_cfg,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=(len(y) - pos) / max(pos, 1),
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(train[features].astype("float32"), y)
    return model


def _artifact_name(stage: str, season: int, tag: str) -> str:
    return f"stage_{stage}_season_{season}{tag}"


def run(
    config: dict,
    panel: pd.DataFrame,
    stages: list[str] | None = None,
    tag: str = "",
    models_dir: Path = MODELS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    force: bool = False,
) -> list[dict]:
    """Train + score every requested stage on every fold. Resumable: a
    stage x fold whose predictions already exist is skipped unless `force`."""
    stages = stages or list(config["ablation"]["stages"])
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for season, train, test in folds(panel, config):
        train, test = add_fold_features(train, test)
        for stage in stages:
            name = _artifact_name(stage, season, tag)
            pred_path = predictions_dir / f"{name}.parquet"
            if pred_path.exists() and not force:
                print(f"[{name}] predictions exist, skipping", flush=True)
                continue
            features = stage_features(config, stage)
            entry = {
                "stage": stage, "season": season, "tag": tag, "features": features,
                "train_rows": len(train), "train_positives": int(train[LABEL].sum()),
                "test_rows": len(test), "test_positives": int(test[LABEL].sum()),
            }
            if features is None:  # stage 1: the persistence rule
                scores = test["fire_today"].to_numpy(dtype="float32")
                (models_dir / f"{name}.json").write_text(
                    json.dumps({"rule": "score = fire_today", "stage": stage, "season": season})
                )
            else:
                model = fit_stage(train, features, config["model"])
                model.save_model(models_dir / f"{name}.json")
                scores = model.predict_proba(test[features].astype("float32"))[:, 1]
                entry["importance_gain"] = dict(
                    zip(features, map(float, model.feature_importances_), strict=True)
                )
            out = test[["cell_id", "date", LABEL]].assign(score=scores.astype("float32"))
            out.to_parquet(pred_path, index=False)
            (models_dir / f"{name}_meta.json").write_text(json.dumps(entry, indent=2))
            print(f"[{name}] train {len(train):,} rows / test {len(test):,} rows done", flush=True)
            log.append(entry)
    return log


def load_panel(path: Path) -> pd.DataFrame:
    feature_cols = sorted({c for cols in FAMILIES.values() for c in cols} - {CELL_BURN_RATE})
    return pd.read_parquet(path, columns=KEY_COLUMNS + feature_cols)


def _add_args(parser) -> None:
    parser.add_argument("--stage", action="append", help="repeatable; default = all stages")
    parser.add_argument("--panel", default="panel.parquet", help="file under data/processed/")
    parser.add_argument("--tag", default="", help="suffix for artifacts, e.g. _veg30")
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")


if __name__ == "__main__":
    cfg, args = cli_args("FR-4.1-4.5: train every ablation stage per fold", _add_args)
    run(cfg, load_panel(PROCESSED_DIR / args.panel), args.stage, args.tag, force=args.force)
