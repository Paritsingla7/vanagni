# Vanagni

**Next-day forest-fire risk for Uttarakhand and Odisha, India: a backtested, leakage-tested research project.**

For every 5 km grid cell, the model predicts whether a satellite will detect a vegetation fire
there *tomorrow*, from the cell's fire history, weather, fuel and terrain, and its proximity to
people (roads, settlements, cropland). Every prediction is scored against what actually burned.

> **The question it tests.** In a fire regime that is reportedly 90–95% human-caused, do human-proximity
> features add more predictive lift than weather?
>
> **The answer, on six held-out fire seasons (2021–2026): no.** Weather added **+0.035** PR-AUC;
> human-proximity features added **+0.003**, all of it from the first two seasons (nothing in
> 2023–2026), and indistinguishable from zero on places the model never trained on.
>
> **Repeated in a second state, Odisha (four seasons, 2021–2024): still no.** Weather added
> **+0.022**; human proximity added **+0.0004**, indistinguishable from zero. On unseen places it
> was a small +0.006 against weather's +0.044.
> The full write-up is in [REPORT.md](REPORT.md).

![Risk map: model ranking against the fires that followed](assets/map.png)

*Risk map for 2024-04-20. Cells are coloured by risk rank within the day, the 20 highest-ranked
cells are outlined, and orange dots are the fires detected the next day (11 of the top 20 burned).*

![Ablation: what each feature family adds, with uncertainty](assets/ablation.png)

## What was built

| | |
|---|---|
| **Leakage-safe panel** | One row per cell and day (6.8 M rows across the 1,950 scored cells); the label is a qualifying fire on day *d+1*, and every feature is computed from day *d* or earlier. Windows are array-based, never positional, so a missing day cannot shift one. |
| **Honest label** | Raw satellite detections are filtered to vegetation fires: FIRMS static-source flag, confidence, ≥ 50 % natural vegetation in the 375 m footprint, no repeat hotspots (kilns, factories). 397,346 raw detections become 104,440 qualifying ones. |
| **Staged ablation** | Persistence → fire history → + weather → + fuel & terrain → + human proximity, plus a "full model minus weather" run so credit does not depend on the order features are added. |
| **Operational baseline** | The Canadian Fire Weather Index, implemented from the published equations and checked against the published reference case, as a fixed-formula rule and as a model feature. |
| **Rigour** | Rolling-origin evaluation over six seasons, 7-day block-bootstrap confidence intervals on every metric and every stage difference, a spatial-block holdout (whole 50 km blocks held out), label-threshold sensitivity runs, and automated leakage tests that gate the Methodology page. |
| **Demo** | A read-only Flask API and four plain-JS screens (About, Risk Map, Ablation, Methodology) over pre-computed results. Nothing trains or scores on request. |
| **A second region** | `config/odisha.yaml` plus a shared `config/defaults.yaml`: Odisha is built, modelled and evaluated by the same pipeline, with a fire season that wraps the new year. See [Scaling](#scaling). |

## Results at a glance

Uttarakhand, mean over test seasons 2021–2026 (26,209 positive test cell-days; base rate 0.55–3.3 % per season):

| Stage | PR-AUC (95 % CI) |
|---|---|
| Persistence (fire today → fire tomorrow) | 0.075 (0.058–0.088) |
| Fire Weather Index, as a rule | 0.038 (0.029–0.045) |
| + fire history | 0.145 (0.111–0.167) |
| + weather | 0.180 (0.140–0.203) |
| + fuel & terrain | 0.187 (0.147–0.210) |
| + human proximity | 0.190 (0.149–0.214) |
| full model minus weather | 0.158 (0.123–0.181) |

Intervals reflect day-to-day noise within seasons, not season-to-season variation; with six seasons,
treat small differences with caution. Precision@20 is 0.12–0.13 for every learned stage (persistence
0.103), and the stage-to-stage differences in it are within its intervals.
Odisha, mean over test seasons 2021–2024 (60,720 positive test cell-days; base rate 0.6–1.5 % per season):

| Stage | PR-AUC (95 % CI) |
|---|---|
| Persistence | 0.069 (0.048–0.082) |
| Fire Weather Index, as a rule | 0.035 (0.023–0.047) |
| + fire history | 0.184 (0.140–0.211) |
| + weather | 0.206 (0.161–0.229) |
| + fuel & terrain | 0.217 (0.171–0.242) |
| + human proximity | 0.218 (0.170–0.242) |
| full model minus weather | 0.191 (0.146–0.219) |

[REPORT.md](REPORT.md) has the full tables, the spatial holdout, the sensitivity runs and the limitations.

## Quick start

```bash
conda env create -f environment.yml      # conda-forge: rasterio/geopandas bind to GDAL
conda activate vanagni                   # must be active: the DEM download shells out to GDAL tools
pip install -e .
make lint test                           # ruff + the test suite, leakage tests included
```

The data is not in the repo (it is large and reproducible). To rebuild everything for Uttarakhand:

```bash
cp .env.example .env                     # add FIRMS_MAP_KEY (free, firms.modaps.eosdis.nasa.gov/api)
# ~/.cdsapirc with a free Copernicus CDS key (cds.climate.copernicus.eu)
make pipeline                            # ingest → static → weather → fwi → panel → train → evaluate
make serve                               # http://localhost:5000
```

`make serve` and `make site` show every region in `REGIONS` (default: Uttarakhand and Odisha) behind
a region switch; the landing page tells the first region's story.

Every step is idempotent and resumable (`make -n pipeline` lists them in order). The slow step is
`weather`: ERA5-Land requests wait hours in Copernicus's queue and are capped per account, so a full
window takes days. Everything else is minutes to about an hour. `make weather-resample` uses whatever
months are already on disk.

## Repository layout

```
config/            defaults.yaml (shared) + one file per region + fire-calendar CSVs
src/wfpin/
  ingest/          fire.py (FIRMS), weather.py (ERA5-Land, resumable, queue-aware)
  static_layers.py grid, terrain, land cover, OSM roads/settlements/districts, calendar join
  fwi.py           Canadian Fire Weather Index
  panel.py         the dense, leakage-safe cell × day panel and the label filter
  features.py      feature families and the stage → features registry
  train.py         per-stage, per-season models; spatial-block holdout
  evaluate.py      metrics, bootstrap CIs, methodology and map artifacts
  serve/           read-only Flask API + static frontend
tests/             leakage, static-invariance, FWI reference values, config, API, ...
Makefile           the pipeline, one target per step
```

## Design principles

* **Config-driven.** Region, window, grid, label rules, stages and model settings live in YAML.
  A region file states only what is regional (bbox, CRS, data paths, season calendar, OSM source);
  those keys exist in no default, so a region cannot silently inherit another's.
* **Leakage is tested, not asserted.** `tests/test_panel_leakage.py` truncates the future and checks no
  earlier feature changes, on synthetic data and on the real inputs. It gates the Methodology page.
* **Say what the numbers are.** Model scores are not calibrated probabilities, so the map shows rank;
  intervals say what they do and do not cover; limitations are listed on the page that shows the result.

## Scaling

A second state is a config file, not new code. `config/odisha.yaml` (37 lines) builds Odisha's
6,608-cell grid, terrain, land cover, roads and districts with the same code, in its own
`data/odisha/` directory and an equal-area projection. Getting there required making paths and the
projection config-driven, computing the SRTM tile cap from the bbox, and handling land-cover tiles
that do not exist over open water.

Odisha's fire season runs November to June and so wraps the new year; the panel, rolling-origin folds
and FWI restart handle that (a season is labelled by the year it ends in, and the FWI restarts on
Oct 1). Odisha has been modelled and evaluated end to end over four seasons (2021–2024), and the
finding holds there (see [Results at a glance](#results-at-a-glance)). Next: live scoring from a
weather forecast instead of reanalysis, district-level human-ignition calendars, and calibrated
probabilities.

## Limitations

Backtest only; two states; six seasons in Uttarakhand and four in Odisha; ERA5-Land (about 9–11 km) is coarser than the 5 km grid;
the fire calendar is one state-wide window, so the *timing* half of the human-ignition hypothesis is
untested; OSM roads are today's, applied to 2017–2026; land cover cannot separate chir pine from other
forest. See [REPORT.md](REPORT.md) for the complete list.

## Licence

MIT, see [LICENSE](LICENSE).
