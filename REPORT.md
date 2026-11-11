# Vanagni: does knowing where people are predict forest fire better than weather?

A backtested study of next-day forest-fire risk in Uttarakhand, India, 2017–2026.

## 1. Summary

India's forest fires are widely reported to be overwhelmingly human-caused (90–95 % is the figure
usually cited). If that is true, features describing human activity (distance to roads and
settlements, cropland adjacency) ought to predict *where fire will be detected tomorrow* at
least as well as weather does. This project tests that, as a falsifiable claim, on 5 km grid
cells over six held-out fire seasons.

**Result: not supported.** Over seasons 2021–2026, adding weather to a fire-history model
raised PR-AUC by **+0.035** (95 % CI 0.024–0.042). Adding human-proximity features on top of
history, weather, fuel and terrain raised it by **+0.003** (95 % CI 0.001–0.004). That
small lift came entirely from the first two seasons (+0.015 in 2021, +0.005 in 2022); in the
four seasons since it was −0.001, −0.001, +0.000 and −0.002. It was **not detectable on places the
model had never trained on** (−0.002, CI −0.005 to +0.002), and did not change with the label's
vegetation threshold. Weather's lift was about thirteen times larger.

Other findings: a fixed-formula operational index (the Canadian Fire Weather Index) is a weak
predictor on its own (PR-AUC 0.038, below the persistence rule's 0.075) because it rates
regional dryness and cannot rank cells; learned models beat it by +0.14 to +0.15. Inside one
model, raw weather variables still beat the index (+0.017). The models transfer reasonably to
unseen places when they have weather, and degrade sharply when they do not.

The negative result is the finding. The hypothesis was stated before the evaluation, the
evaluation was built to be able to refute it, and the leakage that would have flattered the
result is tested for, not assumed away.

## 2. Data

| Layer | Source | Notes |
|---|---|---|
| Fire detections | NASA FIRMS VIIRS 375 m, Suomi-NPP, standard processing, 2017–2026 (seasons through mid-2026) | One satellite only: NOAA-20's archive starts 2018-04, and mixing satellites roughly doubles overpass density partway through the window. |
| Weather | ERA5-Land hourly (Copernicus CDS), ~9–11 km | 2 m temperature and dewpoint, 10 m wind, precipitation, soil moisture. All 24 hours; local (IST) day buckets; precipitation de-accumulated correctly. |
| Terrain | SRTM 30 m | Elevation, slope, aspect (as north-ness and east-ness; a circular mean, not an arithmetic one). |
| Land cover | ESA WorldCover v200 (2021), 10 m | Area fractions of forest, shrub/grass, cropland, built-up, other per cell. |
| Roads, settlements, districts | OpenStreetMap (state extract) | Exact distance to the nearest road line and settlement. |
| Fire season | Hand-compiled calendar | One state-wide window (Feb 15 – Jun 15) applied to all 13 districts; no district-level source was found. |

The grid is 2,314 cells of 5 km × 5 km (1,950 with enough vegetation to score), built in the
state's UTM zone so every cell has the same true area. ERA5-Land is coarser than the grid:
neighbouring cells share one weather sample (a stated limitation, not a bug).

## 3. Method

**The label.** `fire_next_day` is 1 if a *qualifying* detection falls in the cell on the next
local (IST) day. Raw detections are not forest fires: they include crop-residue burning,
kilns, factories and fires in the neighbouring states that the bounding box also covers. They are
filtered in layers: inside the state boundary; FIRMS's own static-source flag; nominal or
high confidence; at least 50 % natural vegetation in the 375 m footprint (a footprint share, not
a point lookup); and not a persistent hotspot (a spot detected outside the season in several
reference years, computed from a period that ends before the first test season, so future
detections cannot relabel the past). 397,346 raw detections become 104,440 qualifying ones.
Of the detections the vegetation step removes, 94 % sit on cropland, peaking in the
October–November and April–May residue-burning seasons, which is exactly the contamination that
would have flattered a human-proximity hypothesis.

**Features, in families.** *History*: fire counts over 7 and 30 days, days since the last fire,
neighbour-cell pressure over the 8 surrounding cells, days since season onset, and a per-cell
long-run burn rate computed from each fold's training seasons only. *Weather*: daily mean/max
temperature, mean/min relative humidity, mean/max wind, precipitation, soil moisture. *Fuel and
terrain*: elevation, slope, north-ness, east-ness, vegetation fractions. *Human proximity*:
distance to road and settlement, cropland and built-up fractions. Cell identity (id, coordinates)
is never a feature.

**Stages.** Each stage adds one family to the previous: persistence (a rule: fire today, fire
tomorrow) → fire history → + weather → + fuel & terrain → + human proximity. Stage 4b versus 4a is
the hypothesis test. Because sequential stages credit whichever family comes first, a
"full model minus weather" run measures what *removing* weather costs, and a
"history + Fire Weather Index" stage asks whether raw weather adds anything beyond the standard
index. All learned stages use one fixed XGBoost configuration with `scale_pos_weight` set from the
fold's training split; there is no per-stage tuning, so differences come from features, not knobs.

**Evaluation.** Rolling origin: for each test season 2021–2026, train on every earlier season and
score that one, on in-season days only. Headline metrics are PR-AUC and precision@20 (per day, over
that day's cells, averaged over days; ties broken in expectation), reported beside the base rate and
the positive count. Every metric and every stage-to-stage difference carries a 7-day block-bootstrap
95 % interval (1,000 resamples; paired for differences). No threshold-0.5 precision, recall or F1.
A second evaluation holds out whole 50 km blocks of cells, so each is scored by a model that never saw
its neighbourhood and has no burn history of its own.

**Leakage controls.** An automated test builds the panel from fires truncated at a cutoff and
asserts that no non-label feature at or before the cutoff changes, on synthetic data and on the
real inputs. A second test asserts that static layers never vary by date. The Methodology page
reports both tests' real pass/fail.

## 4. Results

### 4.1 The ablation (mean over 2021–2026)

| Stage | PR-AUC (95 % CI) | Precision@20 |
|---|---|---|
| Persistence | 0.075 (0.058–0.088) | 0.103 |
| Fire Weather Index (rule) | 0.038 (0.029–0.045) | 0.010 |
| + fire history | 0.145 (0.111–0.167) | 0.119 |
| + weather | 0.180 (0.140–0.203) | 0.123 |
| History + FWI (one index instead of raw weather) | 0.163 (0.126–0.187) | 0.121 |
| + fuel & terrain | 0.187 (0.147–0.210) | 0.129 |
| + human proximity | 0.190 (0.149–0.214) | 0.128 |
| Full model minus weather | 0.158 (0.123–0.181) | 0.130 |

26,209 positive test cell-days; base rate 3.3 %, 2.2 %, 0.9 %, 2.7 %, 0.55 % and 1.45 % in
2021–2026. A learned model's precision@20 is about 0.12–0.13 against a base rate of 1–2 %, meaning
roughly two to three of the twenty highest-ranked cells burn the next day versus under one at random.
The differences in precision@20 between learned stages are small relative to its intervals
(roughly 0.10–0.15), so PR-AUC, not precision@20, is what separates the stages. The full model's
PR-AUC by season (0.239, 0.241, 0.069, 0.247, 0.143, 0.199) tracks how much burned: the two
low-fire seasons, 2023 and 2025, are the hardest.

### 4.2 Stage-to-stage differences (change in PR-AUC)

| Question | Δ PR-AUC (95 % CI) |
|---|---|
| Fire history over persistence | +0.070 (0.052–0.082) |
| Weather over history | +0.035 (0.024–0.042) |
| Fuel & terrain over weather | +0.007 (0.004–0.009) |
| **Human proximity over fuel & terrain (the hypothesis test)** | **+0.003 (0.001–0.004)** |
| Cost of removing weather from the full model | +0.032 (0.021–0.038) |
| Standard FWI over history | +0.019 (0.012–0.023) |
| Raw weather over the FWI, inside the same model | +0.017 (0.010–0.022) |
| Full model over the FWI rule | +0.152 (0.119–0.170) |

### 4.3 Why the human-proximity result should not be read as a small win

The interval above captures day-to-day noise within each season only. Per season, the human-proximity
change was **+0.015 (2021), +0.005 (2022), −0.001 (2023), −0.001 (2024), +0.000 (2025) and −0.002
(2026)**: the two seasons added since the first, four-season write-up show nothing. Three further checks agree that it is not a robust effect:

* **Unseen places** (spatial-block holdout, 50 km blocks, 5 folds): human proximity changes PR-AUC
  by **−0.002** (95 % CI −0.005 to +0.002), no detectable difference, while weather still adds
  **+0.049** (0.035–0.057).
* **Label threshold**: at 30 % and 70 % vegetation share instead of 50 %, the human-proximity
  change is +0.0002 (−0.002 to +0.002) and −0.0017 (−0.003 to +0.001).
* **Model's own use of features** (share of gain, full model): history 73.5 %, weather 16.0 %,
  fuel & terrain 7.1 %, human proximity 3.5 %.

### 4.4 Spatial holdout, all stages (PR-AUC)

| Persistence | FWI rule | + history | + weather | History + FWI | + fuel & terrain | + human | Full − weather |
|---|---|---|---|---|---|---|---|
| 0.075 | 0.038 | 0.110 | 0.159 | 0.127 | 0.170 | 0.168 | 0.129 |

Models with weather transfer to unseen places with a modest drop (full model 0.190 → 0.168). The model
*without* weather drops further (0.158 → 0.129) because it must lean on static place features, which is
what location memorisation looks like. Fire history loses its per-cell burn rate on held-out cells
(0.145 → 0.110), which is expected.

## 5. Interpretation

1. **Weather, not human proximity, carries the next-day signal.** Fire history carries most of the
   model's gain (73.5 %) and the persistence rule alone is a strong baseline (fires persist), weather is the
   largest addition, and human-proximity features are a rounding error that does not survive unseen places.
2. **That does not mean people do not cause the fires.** The study shows that *static* proximity to
   roads and settlements adds little *next-day* ranking power once weather, fuel and fire history are
   known. Most human-caused ignition is plausibly decided by *timing* (harvest, litter clearing,
   collection seasons), which the data here cannot express: the fire calendar is one state-wide
   window. The timing half of the hypothesis is untested, not refuted.
3. **The operational index is a floor, not a competitor.** The Fire Weather Index loses to simple
   persistence because it does not distinguish cells; a learned model that ranks cells and uses
   weather beats it by +0.14 to +0.15, and still adds +0.017 beyond it when the index is available as
   a feature.

## 6. Threats to validity

* **Six seasons, one state.** Season-to-season variation is not in the bootstrap intervals. 2023 and
  2025 were low-fire seasons (base rates 0.9 % and 0.55 %).
* **The label is any vegetation fire seen by one satellite**, not a fire of known cause. It misses
  fires hidden by cloud or between overpasses and cannot separate human from natural ignition.
* **Weather is coarser than the grid** (≈ 9–11 km vs 5 km): neighbouring cells share a sample.
* **Road and settlement distances use today's OpenStreetMap** for 2017–2026, so roads built in the window
  (the Char Dham highway, for example) are treated as always present; the OSM extract stops at the state
  border, so border-cell distances are upper bounds.
* **Land cover cannot separate chir pine from other forest** (one tree-cover class).
* **Scores are not calibrated probabilities** (class weighting inflates them); the demo shows rank.
* **One model family, one fixed configuration.** No per-stage tuning was done, on purpose. A different
  learner could weigh features differently.

## 7. Quality notes: what the checks caught

These are included because they show the process, not because they were comfortable.

* **Precipitation.** ERA5-Land accumulates rainfall from 00 UTC. A plain first difference loses the
  01 UTC hour every day (negative, then clipped). Found by testing against a real month of data (13.38 mm
  at 00 UTC, 0.296 mm at 01 UTC); the test now pins it.
* **Fire Weather Index.** The first implementation used a wrong constant in the duff-moisture-code drying term (10⁻⁴ for 10⁻⁶).
  It was caught by the published reference case, which all six outputs now match.
* **Label contamination.** The padded bounding box is twice the state's area, and 38 % of the satellite's
  detections in it fell outside the state (Uttar Pradesh, Haryana, Himachal, Nepal), including crop-burning
  plains. The grid clip and label filter remove that before it can flatter the hypothesis.
* **Map scale.** The first map coloured by each day's maximum and labelled scores as "probabilities".
  Neither was defensible; it now colours by rank within the day and says the scores are uncalibrated.
* **Basemap.** A free map-tile provider turned out to require an API key and stamped "API KEY REQUIRED"
  across the map; found only by looking at the rendered page.

## 8. Reproducing this

`make pipeline` (see the README) rebuilds everything from public data. Randomness is fixed: model seed 42,
bootstrap seed 0, spatial-fold seed 0. The slow step is the ERA5-Land download, which waits in
Copernicus's queue for hours to days. All thresholds, stage definitions and evaluation settings are in
`config/defaults.yaml`; the Uttarakhand-specific settings are in `config/uttarakhand.yaml`.

## 9. Scaling and next steps

The pipeline is region-agnostic: `config/odisha.yaml` builds Odisha's grid (6,608 cells), terrain,
land cover, roads and 30 districts with the same code, in `data/odisha/`, using an equal-area projection
because the state spans two UTM zones. Not yet done, and the natural next work, in order:

1. **Odisha weather and modelling**, the real test of whether the finding generalises. Odisha's
   fire season runs November to June and wraps the new year; the panel, folds and FWI now handle that
   (a season is labelled by the year it ends in, and the FWI restarts on Oct 1), and the weather
   download is under way.
2. **District-level human-ignition calendars**, to test the timing half of the hypothesis.
3. **Calibrated probabilities**, so scores can be read as risk rather than rank.
4. **Live daily scoring** from a weather *forecast* (reanalysis is not available at prediction time),
   reporting the drop in skill honestly, with a public scorecard of yesterday's hits.
