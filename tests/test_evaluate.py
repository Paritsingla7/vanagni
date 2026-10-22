"""Phase 4: tie-aware precision@k, paired bootstrap, and an end-to-end
train -> evaluate run on synthetic data."""
import numpy as np
import pandas as pd

from wfpin.config import load_config
from wfpin.evaluate import (
    _resampled_day_mean,
    evaluate,
    importance_by_family,
    precision_at_k_by_day,
    season_scores,
)
from wfpin.features import FAMILIES, LABEL

CONFIG = load_config()


def _day(scores, labels, day="2024-04-01"):
    return pd.DataFrame({"cell_id": range(len(scores)), "date": pd.Timestamp(day),
                         "score": scores, LABEL: labels})


def test_precision_at_k_resolves_ties_in_expectation():
    # Three cells tied at the top for two slots, one of them a fire: 2 * 1/3 / 2.
    p = precision_at_k_by_day(_day([1, 1, 1, 0], [1, 0, 0, 1]), k=2)
    assert np.isclose(p.iloc[0], 1 / 3)
    # No ties: exact.
    assert precision_at_k_by_day(_day([0.9, 0.8, 0.1, 0.0], [1, 1, 0, 0]), 2).iloc[0] == 1.0
    assert precision_at_k_by_day(_day([0.0, 0.1, 0.8, 0.9], [1, 1, 0, 0]), 2).iloc[0] == 0.0


def test_resampled_day_mean_counts_repeated_days():
    daily = np.array([0.0, 1.0])
    day_of_row = np.array([0, 0, 1, 1])
    assert _resampled_day_mean(daily, day_of_row, np.arange(4)) == 0.5
    # Day 1 drawn twice, day 0 once -> 2/3.
    assert np.isclose(_resampled_day_mean(daily, day_of_row, np.array([0, 1, 2, 3, 2, 3])), 2 / 3)


def test_identical_stages_have_zero_paired_delta():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-02-15", "2024-06-15")
    pred = pd.DataFrame([(c, d) for d in days for c in range(30)], columns=["cell_id", "date"])
    pred["score"] = rng.random(len(pred))
    pred[LABEL] = rng.random(len(pred)) < 0.05
    out = season_scores({"a": pred, "b": pred.copy()}, k=5, n_boot=50, block_days=7, rng=rng)
    assert np.array_equal(out["a"]["_boot_pr_auc"], out["b"]["_boot_pr_auc"])
    assert np.array_equal(out["a"]["_boot_precision_at_k"], out["b"]["_boot_precision_at_k"])


def test_end_to_end_train_then_evaluate(tmp_path):
    from tests.test_train import _synthetic_panel
    from wfpin.train import run

    seasons = [2023, 2024]
    cfg = {**CONFIG, "model": {**CONFIG["model"], "n_estimators": 5},
           "evaluation": {**CONFIG["evaluation"], "test_seasons": seasons,
                          "bootstrap_samples": 30, "precision_at_k": 3}}
    run(cfg, _synthetic_panel(), models_dir=tmp_path / "m", predictions_dir=tmp_path / "p")
    result = evaluate(cfg, predictions_dir=tmp_path / "p")

    assert [s["stage"] for s in result["stages"]] == list(cfg["ablation"]["stages"])
    for entry in result["stages"]:
        assert [r["season"] for r in entry["per_season"]] == seasons
        for r in entry["per_season"]:
            assert 0 <= r["pr_auc"] <= 1 and 0 <= r["precision_at_k"] <= 1
            assert r["pr_auc_ci"][0] <= r["pr_auc_ci"][1]
            assert r["n_positive"] > 0
    names = {c["name"] for c in result["comparisons"]}
    assert {"human_vs_fuel", "cost_of_weather"} <= names
    for c in result["comparisons"]:
        assert c["pr_auc_verdict"] in {"no detectable difference", "a better", "b better"}

    shares = importance_by_family("4b", seasons, models_dir=tmp_path / "m")
    assert set(shares) == set(FAMILIES) - {"fwi_index"}  # stage 4b never sees the FWI feature
    assert np.isclose(sum(shares.values()), 1)


def test_ablation_reports_spatial_holdout_only_when_it_was_run(tmp_path):
    from tests.test_train import _synthetic_panel
    from wfpin.evaluate import build_ablation
    from wfpin.train import run, run_spatial

    cfg = {**CONFIG, "model": {**CONFIG["model"], "n_estimators": 3},
           "evaluation": {**CONFIG["evaluation"], "test_seasons": [2023, 2024],
                          "bootstrap_samples": 20, "precision_at_k": 3,
                          "spatial_holdout": {"block_km": 5, "n_folds": 3, "seed": 0}}}
    panel, m, p = _synthetic_panel(), tmp_path / "m", tmp_path / "p"
    run(cfg, panel, models_dir=m, predictions_dir=p)

    assert build_ablation(cfg, predictions_dir=p, models_dir=m)["spatial_holdout"] == {
        "status": "not_run"}

    unconfigured = {**cfg, "evaluation": {k: v for k, v in cfg["evaluation"].items()
                                          if k != "spatial_holdout"}}
    assert build_ablation(unconfigured, predictions_dir=p, models_dir=m)["spatial_holdout"] == {
        "status": "not_configured"}

    run_spatial(cfg, panel, pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2], index=np.arange(9)),
                models_dir=m, predictions_dir=p)
    spatial = build_ablation(cfg, predictions_dir=p, models_dir=m)["spatial_holdout"]
    assert spatial["status"] == "ok" and spatial["n_folds"] == 3
    assert {s["stage"] for s in spatial["stages"]} == set(cfg["ablation"]["stages"])


def test_date_gaps_reports_missing_runs_between_first_and_last_day():
    from wfpin.evaluate import date_gaps

    days = list(pd.date_range("2023-11-28", "2023-11-30")) + list(
        pd.date_range("2023-12-04", "2023-12-05")) + [pd.Timestamp("2023-12-09")]
    assert date_gaps(pd.Series(days)) == [
        {"start": "2023-12-01", "end": "2023-12-03"}, {"start": "2023-12-06", "end": "2023-12-08"}]
    assert date_gaps(pd.Series(pd.date_range("2023-01-01", "2023-01-10"))) == []
