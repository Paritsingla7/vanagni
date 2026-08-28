"""Train the next-day fire-risk model and write artifacts + metrics.

Runs the ablation the project exists to answer: does weather predict next-day
fire beyond what fire history alone already tells you? Compares persistence,
fire-history-only XGBoost, and weather-enriched XGBoost on a temporal holdout.

    python scripts/train_forecast.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from firepulse import (  # noqa: E402
    FEATURES, FIRE_FEATS, GEO_FEATS, WX_FEATS,
    build_dataset, precision_at_k, temporal_split,
)

FIRMS_CSV = REPO / "archive" / "fire_archive_M6_156000.csv"
ERA5_NC_DIR = REPO / "era5" / "nc"
MODEL_DIR = REPO / "frontend" / "model"
DATA_DIR = REPO / "frontend" / "data"
K_FRAC = 0.01  # shortlist size: the riskiest 1% of cells


def fit(train, test, feats, y_train):
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=neg / max(pos, 1), random_state=42, verbosity=0,
    )
    model.fit(train[feats].astype(np.float32), y_train)
    return model, model.predict_proba(test[feats].astype(np.float32))[:, 1]


def score(name, y_true, prob):
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        f"precision_at_{K_FRAC:.0%}": precision_at_k(y_true, prob, K_FRAC),
    }


def main():
    for path in (FIRMS_CSV, ERA5_NC_DIR):
        if not path.exists():
            sys.exit(f"Missing {path}. See the README data setup section.")

    print("Building panel + weather features ...", flush=True)
    df = build_dataset(FIRMS_CSV, ERA5_NC_DIR)
    train, test, cutoff = temporal_split(df)
    y_train = train["fire_next_day"].astype(int)
    y_test = test["fire_next_day"].astype(int)
    base_rate = float(y_test.mean())
    print(f"{len(df):,} cell-days | train {len(train):,} (<= {cutoff.date()}) "
          f"| test {len(test):,} | base rate {base_rate:.4f}", flush=True)

    results = [score("persistence", y_test, test["fire_today"].astype(float))]

    feats_hist = FIRE_FEATS + GEO_FEATS
    print("Fitting fire-history model ...", flush=True)
    _, prob_hist = fit(train, test, feats_hist, y_train)
    results.append(score("xgb_fire_history", y_test, prob_hist))

    print("Fitting weather-enriched model ...", flush=True)
    model_full, prob_full = fit(train, test, FEATURES, y_train)
    results.append(score("xgb_weather", y_test, prob_full))

    table = pd.DataFrame(results).set_index("model").round(4)
    print("\n" + table.to_string())
    lift = table.loc["xgb_weather", "pr_auc"] - table.loc["xgb_fire_history", "pr_auc"]
    print(f"\nWeather PR-AUC lift over fire-history-only: {lift:+.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dump(model_full, MODEL_DIR / "forecast_xgb.joblib")
    payload = {
        "task": "next-day fire detection per 0.25 deg cell (label = day d+1)",
        "region": {"north": 42, "west": -125, "south": 32, "east": -114},
        "window": {"start": "2018-01-01", "end": "2020-06-30"},
        "rows": int(len(df)),
        "cells": int(df[["cell_lat", "cell_lon"]].drop_duplicates().shape[0]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_cutoff": str(cutoff.date()),
        "test_base_rate": base_rate,
        "features": FEATURES,
        "weather_features": WX_FEATS,
        "results": results,
        "weather_pr_auc_lift": float(lift),
        "importances": dict(
            sorted(
                zip(FEATURES, (float(v) for v in model_full.feature_importances_)),
                key=lambda kv: kv[1], reverse=True,
            )
        ),
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (DATA_DIR / "forecast_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {MODEL_DIR / 'forecast_xgb.joblib'} and {DATA_DIR / 'forecast_metrics.json'}")


if __name__ == "__main__":
    main()
