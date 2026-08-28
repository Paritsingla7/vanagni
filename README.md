# Vanagni

Next-day forest-fire risk for Uttarakhand, India — backtest-only research
demo. Predicts, for a grid cell, whether a fire will be detected there
tomorrow, using fire history, weather, and human-proximity/land-use
features (roads, settlements, land cover).

**The hypothesis this exists to test:** in a fire regime that is 90–95%
human-caused, do human-activity and land-use-adjacency features contribute
more predictive lift than weather alone?

## Status

Phase 0 (scaffolding) done. Phase 1 (ingestion + static layers) in
progress. Full spec (requirements, architecture, screens, API/data
contracts, execution plan) exists as local working docs, not published
here.

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
conda env create -f environment.yml
conda activate vanagni
pip install -e .
```

`requirements.txt` is kept as a plain-pip fallback, but conda-forge is the
primary path — several deps here (`rasterio`, `geopandas`) bind to GDAL,
which conda handles more reliably than pip wheels. Verified installing
clean under Python 3.12 (3.14 has no prebuilt geospatial wheels yet).

## Licence

MIT — see `LICENSE`.
