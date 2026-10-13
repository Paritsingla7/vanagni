"""Shared config + environment loading for every pipeline step.

`config/<region>.yaml` is the single source of region/window/grid/k values
(ARCHITECTURE.md §3, "config-driven, not hardcoded"); secrets come from the
environment, optionally seeded from a repo-root `.env` (gitignored).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config/uttarakhand.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
