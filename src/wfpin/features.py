"""Feature families and the stage -> feature-list mapping for the ablation.

A family is a named set of panel columns; a stage (config `ablation.stages`)
is a list of families. train.py and evaluate.py only ever ask this module
"which columns does stage X use", so adding, removing, or regrouping a
family is a one-line change here or in config -- never duplicated logic.

`cell_burn_rate_train` is the one fold-local feature (FR-4.4): it is not a
panel column, it is derived from a fold's training rows by `add_fold_features`.
"""
from __future__ import annotations

import pandas as pd

from wfpin.ingest.weather import DAILY_COLUMNS

LABEL = "label_fire_next_day"
CELL_BURN_RATE = "cell_burn_rate_train"

# Rule stages have no learned parameters: they score each row by one panel
# column, and go through exactly the same evaluation path as the models
# (FR-5.4). They are baselines, never features -- `fwi` is deliberately in no
# family, so no model can see it.
RULES: dict[str, str] = {"persistence": "fire_today", "fwi": "fwi"}

FAMILIES: dict[str, list[str]] = {
    "history": [
        "fire_count_7d",
        "fire_count_30d",
        "days_since_last_fire",
        "has_fire_history",
        "neighbor_fire_pressure_7d",
        "days_since_season_onset",
        CELL_BURN_RATE,
    ],
    "weather": list(DAILY_COLUMNS),
    "fuel_terrain": [
        "elevation_m",
        "slope_deg",
        "northness",
        "eastness",
        "frac_forest",
        "frac_shrub_grass",
        "frac_other",
    ],
    "human": ["dist_road_m", "dist_settlement_m", "frac_cropland", "frac_built_up"],
}


def rule_column(config: dict, stage: str) -> str | None:
    """The panel column a rule stage scores by; None for a learned stage."""
    spec = config["ablation"]["stages"][stage]
    return RULES[spec] if isinstance(spec, str) else None


def stage_features(config: dict, stage: str) -> list[str] | None:
    """Columns a stage trains on, in family order; None for a rule stage."""
    families = config["ablation"]["stages"][stage]
    if isinstance(families, str):
        if families not in RULES:
            raise KeyError(f"stage {stage!r}: unknown rule {families!r}; known: {sorted(RULES)}")
        return None
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise KeyError(f"stage {stage!r} names unknown families: {sorted(unknown)}")
    return [col for fam in families for col in FAMILIES[fam]]


def family_of(column: str) -> str:
    return next(fam for fam, cols in FAMILIES.items() if column in cols)


def add_fold_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FR-4.4: per-cell long-run burn rate from the fold's *training* rows
    only, attached to both sides. Computed here, never in panel.py, because
    a rate that saw the test season would leak its label."""
    rate = train.groupby("cell_id")[LABEL].mean().astype("float32").rename(CELL_BURN_RATE)
    return train.join(rate, on="cell_id"), test.join(rate, on="cell_id")
