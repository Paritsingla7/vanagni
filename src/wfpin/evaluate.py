"""Rolling-origin evaluation of every ablation stage, plus the artifacts the
demo serves.

FR-5.1-5.5: scores the test-season predictions train.py wrote (never
re-fits). Headline metrics are PR-AUC and per-day precision@top-k (k from
config), each with a block-bootstrap 95% CI, the base rate, and the positive
count beside it. Stage-to-stage deltas use a *paired* bootstrap -- both
stages are re-scored on the same resampled weeks, so shared noise cancels.
No threshold-0.5 precision/recall/F1 anywhere.

Writes data/processed/ablation.json, methodology.json, and risk/ (per-date
risk + ground truth for the map, FR-7.1).

    python -m wfpin.evaluate [--config config/uttarakhand.yaml] [--skip-tests]
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from wfpin.config import cli_args, paths
from wfpin.features import FAMILIES, LABEL, family_of

RISK_STAGE = "4b"  # the map shows the full model (API_CONTRACT.md /api/risk)
LEAKAGE_TESTS = {
    "temporal_leakage": "tests/test_panel_leakage.py",
    "static_layer_invariance": "tests/test_static_layers_invariance.py",
}


# --- metrics ---------------------------------------------------------------------------


def precision_at_k_by_day(pred: pd.DataFrame, k: int) -> pd.Series:
    """FR-5.2: per test day, the share of true fires among that day's top-k
    cells, averaged later over days. Ties at the k-th score are resolved in
    expectation (random tie-breaking): the persistence rule scores most
    cells exactly 0 or 1, so a plain sort would make its precision depend on
    row order."""

    def one_day(day: pd.DataFrame) -> float:
        s, y = day["score"].to_numpy(), day[LABEL].to_numpy(dtype=float)
        kk = min(k, len(s))
        threshold = np.sort(s)[::-1][kk - 1]
        above, tied = s > threshold, s == threshold
        return (y[above].sum() + (kk - above.sum()) * y[tied].mean()) / kk

    return pred.groupby("date").apply(one_day, include_groups=False)


def _block_samples(dates: pd.Series, block_days: int, n: int, rng) -> list[np.ndarray]:
    """Row-index arrays for n block-bootstrap resamples: the test season cut
    into block_days-long blocks, drawn with replacement. Fire days cluster
    in runs, so resampling single days would understate the noise."""
    block = ((dates - dates.min()).dt.days // block_days).to_numpy()
    rows = [np.flatnonzero(block == b) for b in np.unique(block)]
    return [np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
            for _ in range(n)]


def _ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def season_scores(preds: dict[str, pd.DataFrame], k: int, n_boot: int, block_days: int,
                  rng) -> dict[str, dict]:
    """Point metrics + bootstrap draws for every stage of one season. All
    stages must cover exactly the same test rows -- the draws are paired."""
    ordered = {
        s: p.sort_values(["date", "cell_id"]).reset_index(drop=True) for s, p in preds.items()
    }
    ref = next(iter(ordered.values()))
    for stage, p in ordered.items():
        if not p[["cell_id", "date"]].equals(ref[["cell_id", "date"]]):
            raise ValueError(f"stage {stage} predictions cover different rows than the others")
    samples = _block_samples(ref["date"], block_days, n_boot, rng)
    day_index = pd.Index(sorted(ref["date"].unique()))
    day_of_row = day_index.get_indexer(ref["date"])

    out = {}
    for stage, p in ordered.items():
        y, s = p[LABEL].to_numpy(dtype=int), p["score"].to_numpy()
        daily = precision_at_k_by_day(p, k).reindex(day_index).to_numpy()
        boot_ap = np.array([average_precision_score(y[i], s[i]) for i in samples])
        boot_pk = np.array([_resampled_day_mean(daily, day_of_row, i) for i in samples])
        out[stage] = {
            "pr_auc": float(average_precision_score(y, s)),
            "precision_at_k": float(daily.mean()),
            "base_rate": float(y.mean()),
            "n_positive": int(y.sum()),
            "n_days": len(day_index),
            "_boot_pr_auc": boot_ap,
            "_boot_precision_at_k": boot_pk,
        }
    return out


def _resampled_day_mean(daily: np.ndarray, day_of_row: np.ndarray, rows: np.ndarray) -> float:
    """Mean per-day precision over a resample, each drawn block's days
    counted as many times as the block was drawn."""
    days, counts = np.unique(day_of_row[rows], return_counts=True)
    # Every row of a day is drawn together (blocks are whole days), so a
    # day's row count / its size = how many times that day was drawn.
    sizes = np.bincount(day_of_row, minlength=len(daily))[days]
    return float(np.average(daily[days], weights=counts / sizes))


# --- assembling ablation.json ------------------------------------------------------------


def _load_predictions(stage: str, season: int, tag: str, predictions_dir: Path) -> pd.DataFrame:
    path = predictions_dir / f"stage_{stage}_season_{season}{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m wfpin.train` first")
    return pd.read_parquet(path)


def _public(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def evaluate(config: dict, stages: list[str] | None = None, tag: str = "",
             predictions_dir: Path | None = None, seed: int = 0) -> dict:
    ev = config["evaluation"]
    stages = stages or list(config["ablation"]["stages"])
    predictions_dir = predictions_dir or paths(config).predictions
    rng = np.random.default_rng(seed)
    per_season = {}
    for season in ev["test_seasons"]:
        preds = {s: _load_predictions(s, season, tag, predictions_dir) for s in stages}
        per_season[season] = season_scores(preds, ev["precision_at_k"], ev["bootstrap_samples"],
                                           ev["bootstrap_block_days"], rng)

    stage_entries = []
    for stage in stages:
        rows = []
        for season, scores in per_season.items():
            m = scores[stage]
            rows.append({"season": season, **_public(m),
                         "pr_auc_ci": _ci(m["_boot_pr_auc"]),
                         "precision_at_k_ci": _ci(m["_boot_precision_at_k"])})
        # Seasons are independent, so the across-season mean's draws are the
        # mean of each season's draws.
        mean_ap, mean_pk = (
            np.mean([per_season[s][stage][key] for s in per_season], axis=0)
            for key in ("_boot_pr_auc", "_boot_precision_at_k")
        )
        stage_entries.append({
            "stage": stage,
            "families": config["ablation"]["stages"][stage],
            "per_season": rows,
            "mean": {
                "pr_auc": float(np.mean([r["pr_auc"] for r in rows])),
                "pr_auc_ci": _ci(mean_ap),
                "precision_at_k": float(np.mean([r["precision_at_k"] for r in rows])),
                "precision_at_k_ci": _ci(mean_pk),
                "n_positive": int(sum(r["n_positive"] for r in rows)),
            },
        })

    comparisons = []
    for comp in ev["comparisons"]:
        a, b = comp["a"], comp["b"]
        if a not in stages or b not in stages:
            continue
        entry = {"name": comp["name"], "a": a, "b": b}
        for metric in ("pr_auc", "precision_at_k"):
            point = [per_season[s][a][metric] - per_season[s][b][metric] for s in per_season]
            key = f"_boot_{metric}"
            draws = np.mean([per_season[s][a][key] - per_season[s][b][key] for s in per_season],
                            axis=0)
            ci = _ci(draws)
            entry[f"delta_{metric}"] = float(np.mean(point))
            entry[f"delta_{metric}_ci"] = ci
            entry[f"delta_{metric}_per_season"] = dict(
                zip(map(str, per_season), point, strict=True)
            )
            # FR-5.5: a CI spanning zero is "no detectable lift", not a small win.
            entry[f"{metric}_verdict"] = (
                "no detectable difference" if ci[0] <= 0 <= ci[1]
                else ("a better" if ci[0] > 0 else "b better")
            )
        comparisons.append(entry)

    return {"stages": stage_entries, "comparisons": comparisons}


def importance_by_family(stage: str, seasons: list[int], models_dir: Path) -> dict:
    """Share of XGBoost gain per feature family, averaged over folds
    (SCREENS.md §2's optional chart: which *family* earns its keep)."""
    shares = []
    for season in seasons:
        meta = json.loads((models_dir / f"stage_{stage}_season_{season}_meta.json").read_text())
        gain = pd.Series(meta["importance_gain"])
        fam = gain.groupby(gain.index.map(family_of)).sum()
        shares.append(fam / fam.sum())
    mean = pd.concat(shares, axis=1).mean(axis=1)
    # Only families this stage's model actually had, in the registry's order.
    return mean.reindex([f for f in FAMILIES if f in mean.index]).to_dict()


def build_ablation(config: dict, predictions_dir: Path | None = None,
                   models_dir: Path | None = None) -> dict:
    ev = config["evaluation"]
    predictions_dir = predictions_dir or paths(config).predictions
    models_dir = models_dir or paths(config).models
    result = evaluate(config, predictions_dir=predictions_dir)
    sensitivity = []
    for tag in ev.get("sensitivity_tags", []):
        try:
            sens = evaluate(config, stages=["4a", "4b"], tag=tag, predictions_dir=predictions_dir)
        except FileNotFoundError:
            sensitivity.append({"tag": tag, "status": "not_run"})
            continue
        comp = next(c for c in sens["comparisons"] if c["name"] == "human_vs_fuel")
        sensitivity.append({"tag": tag, "status": "ok",
                            **{k: v for k, v in comp.items() if k.startswith("delta_")}})
    spatial = {"status": "not_configured"}
    if "spatial_holdout" in ev:
        spatial = {"status": "not_run"}
        try:
            spatial = {"status": "ok", "block_km": ev["spatial_holdout"]["block_km"],
                       "n_folds": ev["spatial_holdout"]["n_folds"],
                       **evaluate(config, tag="_spatial", predictions_dir=predictions_dir)}
        except FileNotFoundError:
            pass  # `python -m wfpin.train --spatial` hasn't been run: say so, don't fake it
    return {
        "generated_at": date.today().isoformat(),
        "k": ev["precision_at_k"],
        "test_seasons": ev["test_seasons"],
        "bootstrap": {"samples": ev["bootstrap_samples"], "block_days": ev["bootstrap_block_days"]},
        "label": config["label"],
        **result,
        "importance_by_family": importance_by_family(RISK_STAGE, ev["test_seasons"], models_dir),
        "sensitivity": sensitivity,
        "spatial_holdout": spatial,
    }


# --- methodology.json + per-date risk ---------------------------------------------------


def run_leakage_tests() -> dict:
    """FR-7.3: the methodology page states each leakage test's real
    pass/fail and run date -- it is never allowed to omit a failure."""
    results = {}
    for name, path in LEAKAGE_TESTS.items():
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", path],
                              capture_output=True, text=True)
        results[name] = {"status": "pass" if proc.returncode == 0 else "fail",
                         "last_run": date.today().isoformat(),
                         "summary": proc.stdout.strip().splitlines()[-1] if proc.stdout else ""}
    return results


def build_methodology(config: dict, leakage: dict | None) -> dict:
    p = paths(config)
    fire = json.loads((p.raw / "fire/manifest.json").read_text())
    panel = json.loads((p.processed / "panel_manifest.json").read_text())
    static = pd.read_parquet(p.processed / "static_layers.parquet",
                             columns=["cell_id", "in_state_frac"])
    weather = p.processed / "weather_daily.parquet"
    weather_dates = pd.read_parquet(weather, columns=["date"])["date"]
    b = config["region"]["bbox"]
    return {
        "generated_at": date.today().isoformat(),
        "data_sources": [
            {"name": "FIRMS VIIRS (Suomi-NPP, standard processing)", "window": fire["window"],
             "downloaded_at": fire["downloaded_at"], "rows_raw": fire["row_count"]},
            {"name": "ERA5-Land hourly", "window": {"start": str(weather_dates.min()),
                                                   "end": str(weather_dates.max())}},
            {"name": "ESA WorldCover", "version": "v200 (2021)"},
            {"name": "SRTM", "version": "SRTM GL1 (30 m)"},
            {"name": "OpenStreetMap", "source": config["static_layers"]["osm"]["pbf_url"]},
        ],
        "grid": {"resolution_km": config["grid"]["resolution_km"], "cells_in_state": len(static),
                 "cells_scored": panel["cells_scored"],
                 "bbox": [b["west"], b["south"], b["east"], b["north"]]},
        "label": {"definition": "a qualifying Suomi-NPP detection in the cell on local (IST) "
                                "day d+1", "filter_steps": panel["detection_filter_steps"],
                  "in_season_base_rate": panel["in_season_base_rate"]},
        "leakage_tests": leakage or {n: {"status": "not_run"} for n in LEAKAGE_TESTS},
        "known_limitations": [
            "Weather resolution (ERA5-Land, ~9-11 km) is coarser than the 5 km fire grid: "
            "neighbouring cells share one weather sample.",
            "Uttarakhand only: no test of generalisation to another state's fire regime.",
            "The fire calendar is one state-wide window (Feb 15 - Jun 15); no district-level "
            "source was found, so the human-calendar part of the hypothesis is not tested.",
            "Road/settlement distances come from today's OpenStreetMap applied to 2017-2024; "
            "roads built in the window (e.g. the Char Dham highway) are treated as always present.",
            "The OSM extract stops at the state border, so distances in border cells are upper "
            "bounds.",
            "Model scores are not calibrated probabilities (class weighting inflates them); "
            "they are meant for ranking cells, which is how precision@k uses them.",
            "WorldCover (2021) has a single tree-cover class: chir pine is not distinguished "
            "from other forest.",
            "The FWI baseline uses the original 46 N day-length tables (no latitude adjustment), "
            "local-noon ERA5-Land values (mean of the 11:30 and 12:30 IST hours) and an annual "
            "January 1 restart of its moisture codes; an operational system would tune these.",
        ],
    }


def write_risk_files(config: dict, predictions_dir: Path | None = None,
                     out_dir: Path | None = None) -> int:
    """FR-7.1: one small JSON per test date ({cell_id: [risk, fire_next_day]})
    plus grid.geojson once; the API joins them per request, no scoring."""
    predictions_dir = predictions_dir or paths(config).predictions
    out_dir = out_dir or paths(config).risk
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    cells = set()
    ranges = []
    for season in config["evaluation"]["test_seasons"]:
        pred = _load_predictions(RISK_STAGE, season, "", predictions_dir)
        cells |= set(pred["cell_id"])
        ranges.append({"season": season, "start": pred["date"].min().date().isoformat(),
                       "end": pred["date"].max().date().isoformat()})
        for day, g in pred.groupby("date"):
            payload = {"date": day.date().isoformat(), "stage": RISK_STAGE, "season": season,
                       "cells": {str(c): [round(float(r), 5), bool(y)] for c, r, y in
                                 zip(g["cell_id"], g["score"], g[LABEL], strict=True)}}
            (out_dir / f"{day.date().isoformat()}.json").write_text(json.dumps(payload))
            n += 1
    # The API's 400 (outside the test window) vs 404 (inside it, file missing).
    (out_dir / "index.json").write_text(json.dumps(
        {"stage": RISK_STAGE, "k": config["evaluation"]["precision_at_k"], "ranges": ranges}))
    grid = gpd.read_parquet(paths(config).processed / "grid.geoparquet")
    grid[grid["cell_id"].isin(cells)].to_file(out_dir / "grid.geojson", driver="GeoJSON")
    return n


def _add_args(parser) -> None:
    parser.add_argument("--skip-tests", action="store_true",
                        help="don't run the leakage tests (methodology shows not_run)")


if __name__ == "__main__":
    cfg, args = cli_args("FR-5.1-5.5 + FR-7: evaluate all stages, write demo artifacts", _add_args)
    ablation = build_ablation(cfg)
    processed = paths(cfg).processed
    (processed / "ablation.json").write_text(json.dumps(ablation, indent=2))
    leakage = None if args.skip_tests else run_leakage_tests()
    (processed / "methodology.json").write_text(
        json.dumps(build_methodology(cfg, leakage), indent=2)
    )
    print(f"wrote ablation.json, methodology.json, {write_risk_files(cfg)} risk files", flush=True)
