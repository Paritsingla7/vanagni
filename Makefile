# Vanagni pipeline, one target per step, in dependency order.
#
#   make pipeline                              # everything for Uttarakhand
#   make pipeline CONFIG=config/odisha.yaml    # the same pipeline for another region
#
# Every step is idempotent and resumable: downloads skip finished chunks, training skips
# finished stage x season predictions (use FORCE=1 to retrain).
#
# The slow step is `weather`: ERA5-Land requests wait hours in Copernicus's queue, and the
# download is capped per account, so a full window takes days. Everything else is minutes
# (static layers, panel) to about an hour (evaluation's bootstrap).

CONFIG ?= config/uttarakhand.yaml
PY     ?= python
FORCE  ?=
FORCE_FLAG := $(if $(FORCE),--force,)

.PHONY: pipeline ingest static weather weather-resample fwi panel panel-sensitivity train train-sensitivity \
        train-spatial evaluate serve test lint

pipeline: ingest static weather fwi panel panel-sensitivity train train-sensitivity \
          train-spatial evaluate

# FR-1.1  FIRMS VIIRS detections (needs FIRMS_MAP_KEY in .env, or a bulk CSV under raw/fire/manual/)
ingest:
	$(PY) -m wfpin.ingest.fire --config $(CONFIG)

# FR-2.1-2.5  terrain, land cover, OSM roads/settlements/districts, grid, fire calendar
static:
	$(PY) -m wfpin.static_layers --config $(CONFIG)

# FR-1.2  ERA5-Land download (hours-to-days: CDS queue) and daily per-cell resample
weather:
	$(PY) -m wfpin.ingest.weather --config $(CONFIG)

# Same, but only resample the months already on disk (no new CDS requests)
weather-resample:
	$(PY) -m wfpin.ingest.weather --config $(CONFIG) --no-download

# Operational baseline: Canadian Fire Weather Index from the same ERA5 hourly data
fwi:
	$(PY) -m wfpin.fwi --config $(CONFIG)

# FR-3.1-3.5  the dense, leakage-safe cell x day panel
panel:
	$(PY) -m wfpin.panel --config $(CONFIG)

# Label-threshold robustness (FR-3.5): the same panel at 30% and 70% vegetation share
panel-sensitivity:
	$(PY) -m wfpin.panel --config $(CONFIG) --min-vegetation-share 0.3
	$(PY) -m wfpin.panel --config $(CONFIG) --min-vegetation-share 0.7

# FR-4.1-4.5  every ablation stage, per rolling-origin test season
train:
	$(PY) -m wfpin.train --config $(CONFIG) $(FORCE_FLAG)

train-sensitivity:
	$(PY) -m wfpin.train --config $(CONFIG) --panel panel_veg30.parquet --tag _veg30 \
	    --stage 4a --stage 4b $(FORCE_FLAG)
	$(PY) -m wfpin.train --config $(CONFIG) --panel panel_veg70.parquet --tag _veg70 \
	    --stage 4a --stage 4b $(FORCE_FLAG)

# Does the model transfer to places it never trained on? (whole 50 km blocks held out)
train-spatial:
	$(PY) -m wfpin.train --config $(CONFIG) --spatial $(FORCE_FLAG)

# FR-5.1-5.5 + FR-7  metrics with bootstrap CIs, methodology (runs the leakage tests), map data
evaluate:
	$(PY) -m wfpin.evaluate --config $(CONFIG)

serve:
	$(PY) -m wfpin.serve.api --config $(CONFIG)

test:
	$(PY) -m pytest -q

lint:
	ruff check .
