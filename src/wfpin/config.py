"""Shared config + environment loading for every pipeline step.

`config/<region>.yaml` is the single source of region/window/grid/k values
(ARCHITECTURE.md §3, "config-driven, not hardcoded"); secrets come from the
environment, optionally seeded from a repo-root `.env` (gitignored).
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config/uttarakhand.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    """Where one region's data lives. Everything a pipeline step reads or
    writes is under `raw` (downloaded, untouched) or `processed` (derived), so
    a second region can never overwrite the first."""

    raw: Path
    processed: Path

    @property
    def models(self) -> Path:
        return self.processed / "models"

    @property
    def predictions(self) -> Path:
        return self.processed / "predictions"

    @property
    def risk(self) -> Path:
        return self.processed / "risk"


def paths(config: dict) -> Paths:
    """`paths.raw` / `paths.processed` from the config, else `data/<region>/...`.
    A region may pin an existing layout explicitly (Uttarakhand does)."""
    cfg = config.get("paths", {})
    root = Path("data") / config["region"]["name"]
    return Paths(raw=Path(cfg.get("raw", root / "raw")),
                 processed=Path(cfg.get("processed", root / "processed")))


def region_crs(config: dict) -> str:
    """Projected CRS the cell lattice is built in. `region.crs` if set, else
    the UTM zone of the bbox centre. A projected, metre-based CRS keeps cells
    the same size, which a raw lat/lon grid does not -- but a bbox wider than
    one 6-degree UTM zone distorts true cell area by up to ~1% at its edges,
    so such regions should pin an equal-area CRS (e.g. ESRI:102028, Asia South
    Albers), where a projected 5 km cell is exactly 25 km2 everywhere."""
    bbox = config["region"]["bbox"]
    if "crs" in config["region"]:
        return config["region"]["crs"]
    if bbox["east"] - bbox["west"] > 6:
        warnings.warn(
            f"region {config['region']['name']!r} spans {bbox['east'] - bbox['west']:.1f} degrees "
            "of longitude, wider than one UTM zone: true cell areas will vary by up to ~1%. "
            "Set region.crs to an equal-area CRS (e.g. ESRI:102028).",
            stacklevel=2,
        )
    lon, lat = (bbox["west"] + bbox["east"]) / 2, (bbox["south"] + bbox["north"]) / 2
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def load_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE loader; never overrides a variable already set."""
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def cli_args(description: str, add_args=None) -> tuple[dict, argparse.Namespace]:
    """Common CLI entry for every `python -m wfpin.<step>` script: `--config`
    plus any step-specific flags `add_args(parser)` registers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    load_dotenv()
    return load_config(args.config), args


def cli_config(description: str) -> dict:
    """`cli_args` for steps that take no flags beyond `--config`."""
    return cli_args(description)[0]
