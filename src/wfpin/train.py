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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from wfpin.config import cli_args, paths
from wfpin.features import (
    CELL_BURN_RATE,
    FAMILIES,
    LABEL,
    RULES,
    add_fold_features,
    rule_column,
    stage_features,
)

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


def score_stage(config: dict, stage: str, train: pd.DataFrame, test: pd.DataFrame):
    """Fit (if the stage learns) and score one stage on one train/test split.
    Returns (scores, meta, model); model is None for a rule stage."""
    features = stage_features(config, stage)
    meta = {
        "stage": stage, "features": features,
        "train_rows": len(train), "train_positives": int(train[LABEL].sum()),
        "test_rows": len(test), "test_positives": int(test[LABEL].sum()),
    }
    if features is None:  # a rule stage (persistence, FWI): no model to fit
        column = rule_column(config, stage)
        if column not in test.columns:
            raise KeyError(f"stage {stage!r} scores by {column!r}, missing from the panel")
        scores = test[column].to_numpy(dtype="float32")
        if np.isnan(scores).any():
            raise ValueError(f"stage {stage!r}: {int(np.isnan(scores).sum())} test rows "
                             f"have no {column!r} value -- rebuild its inputs first")
        meta["rule"] = f"score = {column}"
        return scores, meta, None
    model = fit_stage(train, features, config["model"])
    gain = map(float, model.feature_importances_)
    meta["importance_gain"] = dict(zip(features, gain, strict=True))
    return model.predict_proba(test[features].astype("float32"))[:, 1], meta, model


def run(
    config: dict,
    panel: pd.DataFrame,
    stages: list[str] | None = None,
    tag: str = "",
    models_dir: Path | None = None,
    predictions_dir: Path | None = None,
    force: bool = False,
) -> list[dict]:
    """Train + score every requested stage on every fold. Resumable: a
    stage x fold whose predictions already exist is skipped unless `force`."""
    stages = stages or list(config["ablation"]["stages"])
    models_dir = models_dir or paths(config).models
    predictions_dir = predictions_dir or paths(config).predictions
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
            scores, meta, model = score_stage(config, stage, train, test)
            if model is not None:
                model.save_model(models_dir / f"{name}.json")
            else:
                (models_dir / f"{name}.json").write_text(
                    json.dumps({"rule": meta["rule"], "stage": stage, "season": season})
                )
            test[["cell_id", "date", LABEL]].assign(score=scores.astype("float32")).to_parquet(
                pred_path, index=False
            )
            entry = {**meta, "season": season, "tag": tag}
            (models_dir / f"{name}_meta.json").write_text(json.dumps(entry, indent=2))
            print(f"[{name}] train {len(train):,} rows / test {len(test):,} rows done", flush=True)
            log.append(entry)
    return log


# --- spatial holdout: does the model generalise to places it has never seen? ------------

SPATIAL_TAG = "_spatial"


def spatial_folds(config: dict, static: pd.DataFrame) -> pd.Series:
    """cell_id -> spatial fold. Cells are grouped into square blocks of
    `block_km` (so neighbouring cells, which share weather and terrain, never
    straddle train and test), and whole blocks are dealt to `n_folds` folds in
    a seeded random order."""
    cfg = config["evaluation"]["spatial_holdout"]
    k = max(1, round(cfg["block_km"] / config["grid"]["resolution_km"]))
    block = (static["grid_row"] // k).astype(int) * 10_000 + (static["grid_col"] // k).astype(int)
    blocks = np.unique(block)
    order = np.random.default_rng(cfg["seed"]).permutation(len(blocks))
    fold_of_block = dict(zip(blocks[order], np.arange(len(blocks)) % cfg["n_folds"], strict=True))
    return pd.Series(block.map(fold_of_block).to_numpy(), index=static["cell_id"].to_numpy(),
                     name="spatial_fold")


def split_spatial(train: pd.DataFrame, test: pd.DataFrame, held_cells) -> tuple:
    """Train without the held-out cells, test on only them. The fold-local
    burn rate is then NaN for every held-out cell -- by design: the point is
    to see whether the static features transfer to places with no history of
    their own, instead of memorising which cells burn."""
    return train[~train["cell_id"].isin(held_cells)], test[test["cell_id"].isin(held_cells)]


def run_spatial(
    config: dict,
    panel: pd.DataFrame,
    cell_fold: pd.Series,
    stages: list[str] | None = None,
    models_dir: Path | None = None,
    predictions_dir: Path | None = None,
    force: bool = False,
) -> list[dict]:
    """Rolling-origin x spatial-block holdout: for each test season and each
    spatial fold, train on earlier seasons using only cells *outside* the
    fold, and score the fold's cells. The pooled predictions (each cell
    scored once, by a model that never saw its block) are written per
    stage x season under the `_spatial` tag, ready for evaluate.py."""
    stages = stages or list(config["ablation"]["stages"])
    models_dir = models_dir or paths(config).models
    predictions_dir = predictions_dir or paths(config).predictions
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for season, train_all, test_all in folds(panel, config):
        pending = [s for s in stages if force or not (
            predictions_dir / f"{_artifact_name(s, season, SPATIAL_TAG)}.parquet").exists()]
        pooled: dict[str, list[pd.DataFrame]] = {s: [] for s in pending}
        sizes = []
        for fold in sorted(cell_fold.unique()):
            held = cell_fold.index[cell_fold == fold]
            train, test = split_spatial(train_all, test_all, held)
            if test.empty:
                continue
            train, test = add_fold_features(train, test)
            sizes.append({"fold": int(fold), "train_rows": len(train), "test_rows": len(test)})
            for stage in pending:
                scores, _, model = score_stage(config, stage, train, test)
                if model is not None:
                    model.save_model(
                        models_dir / f"{_artifact_name(stage, season, SPATIAL_TAG)}_fold{fold}.json"
                    )
                pooled[stage].append(
                    test[["cell_id", "date", LABEL]].assign(score=scores.astype("float32"))
                )
            print(f"[spatial season {season} fold {fold}] {len(test):,} test rows", flush=True)
        for stage, frames in pooled.items():
            name = _artifact_name(stage, season, SPATIAL_TAG)
            pd.concat(frames).to_parquet(predictions_dir / f"{name}.parquet", index=False)
            (models_dir / f"{name}_meta.json").write_text(json.dumps(
                {"stage": stage, "season": season, "tag": SPATIAL_TAG, "folds": sizes}, indent=2))
            log.append({"stage": stage, "season": season})
    return log


def load_panel(path: Path) -> pd.DataFrame:
    feature_cols = sorted({c for cols in FAMILIES.values() for c in cols} - {CELL_BURN_RATE})
    available = set(pq.read_schema(path).names)
    rule_cols = [c for c in RULES.values() if c in available and c not in KEY_COLUMNS]
    return pd.read_parquet(path, columns=KEY_COLUMNS + feature_cols + rule_cols)


def _add_args(parser) -> None:
    parser.add_argument("--stage", action="append", help="repeatable; default = all stages")
    parser.add_argument("--panel", default="panel.parquet", help="file under data/processed/")
    parser.add_argument("--tag", default="", help="suffix for artifacts, e.g. _veg30")
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    parser.add_argument("--spatial", action="store_true",
                        help="spatial-block holdout instead of the standard rolling origin")


if __name__ == "__main__":
    cfg, args = cli_args("FR-4.1-4.5: train every ablation stage per fold", _add_args)
    processed = paths(cfg).processed
    data = load_panel(processed / args.panel)
    if args.spatial:
        static = pd.read_parquet(processed / "static_layers.parquet",
                                 columns=["cell_id", "grid_row", "grid_col"])
        run_spatial(cfg, data, spatial_folds(cfg, static), args.stage, force=args.force)
    else:
        run(cfg, data, args.stage, args.tag, force=args.force)
