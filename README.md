# Vanagni

Next-day forest-fire risk for Uttarakhand, India — backtest-only research
demo. Predicts, for a grid cell, whether a fire will be detected there
tomorrow, using fire history, weather, and human-proximity/land-use
features (roads, settlements, land cover).

**The hypothesis this exists to test:** in a fire regime that is 90–95%
human-caused, do human-activity and land-use-adjacency features contribute
more predictive lift than weather alone?

## Status

Planning complete, build not yet started. Full spec (requirements,
architecture, screens, API/data contracts, execution plan) exists as local
working docs, not published here.

## What's in this repo right now

The four files below are carried over from an earlier, unrelated US-focused
project, kept only for their *patterns* — nothing in the eventual `src/`
package will import them:

- `firepulse.py` / `test_firepulse.py` — leakage-safe panel construction and
  its test style (label is always day *d+1*, no feature sees the future)
- `scripts/download_era5.py` — resumable, idempotent chunked download
  pattern for weather reanalysis via the CDS API
- `scripts/train_forecast.py` — staged ablation runner pattern

These will be retired once their patterns are ported into this project's
own structure.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Licence

MIT — see `LICENSE`.
